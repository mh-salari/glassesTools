"""Cast raw Tobii data into common format.

The output directory will contain:
    - frame_timestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gaze_data.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera,
                    and 3D coordinates in the coordinate frame of the glasses (may need rotation/translation
                    to represent in world camera's coordinate frame)
    - calibration.xml: info about the camera intrinsics and transform glasses coordinate system to
                       camera coordinate system
"""

import datetime
import gzip
import json
import math
import pathlib
import shutil

import cv2
import numpy as np
import pandas as pd

from .. import data_files, naming, timestamps, video_utils
from ..eyetracker import EyeTracker
from ..recording import Recording


def preprocess_data(
    output_dir: str | pathlib.Path | None = None,
    source_dir: str | pathlib.Path | None = None,
    rec_info: Recording | None = None,
    copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
) -> Recording:
    """Run all preprocessing steps on Tobii Glasses 3 data and store in output_dir.

    Copies the scene video and gaze data, reads camera calibration from
    ``recording.g3``, and formats gaze data from the decompressed
    ``gazedata`` JSON file.

    Args:
        output_dir: Working directory where the imported recording will be placed.
        source_dir: Path to the raw recording. Falls back to ``rec_info.source_directory``.
        rec_info: Optional pre-populated recording metadata.
        copy_scene_video: Whether to copy the scene video to output_dir.
        source_dir_as_relative_path: Store source_dir as a relative path in rec_info.

    Returns:
        The populated Recording object written to output_dir.

    """
    from . import _store_data, check_folders

    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, EyeTracker.Tobii_Glasses_3)
    print(f"processing: {source_dir.name} -> {output_dir}")

    # check and copy needed files to the output directory
    print("  Check and copy raw data...")
    # check tobii recording and get export directory
    if rec_info is not None:
        check_recording(source_dir, rec_info)
    else:
        rec_info = get_recording_info(source_dir)
        if rec_info is None:
            raise RuntimeError(
                f"The folder {source_dir} is not recognized as a {EyeTracker.Tobii_Glasses_3.value} recording."
            )

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()

    # copy the raw data to the output directory
    src_vid, dest_vid = copy_tobii_recording(source_dir, output_dir, copy_scene_video)
    if dest_vid:
        rec_info.scene_video_file = dest_vid.name
    else:
        rec_info.scene_video_file = src_vid.name

    # prep the copied data...
    print("  Getting camera calibration...")
    scene_video_dimensions = get_camera_from_json(source_dir, output_dir)
    print("  Prepping gaze data...")
    gaze_df, frame_timestamps = format_gaze_data(output_dir, scene_video_dimensions, rec_info)

    _store_data(
        output_dir, gaze_df, frame_timestamps, rec_info, source_dir_as_relative_path=source_dir_as_relative_path
    )

    return rec_info


def get_recording_info(input_dir: str | pathlib.Path) -> Recording | None:
    """Return recording info for a Tobii Glasses 3 recording directory.

    Reads ``recording.g3`` for name, duration, and start time, the
    ``participant`` file for participant name, and system info files
    (``RuVersion``, ``HuSerial``, ``RuSerial``) from the meta-folder.

    Args:
        input_dir: Path to the Tobii Glasses 3 recording directory.

    Returns:
        A Recording object, or None if ``recording.g3`` is missing.

    """
    input_dir = pathlib.Path(input_dir)
    rec_info = Recording(source_directory=input_dir, eye_tracker=EyeTracker.Tobii_Glasses_3)

    # get recording info
    file = input_dir / "recording.g3"
    if not file.is_file():
        return None
    with pathlib.Path(file).open("rb") as j:
        r_info = json.load(j)
    rec_info.name = r_info["name"]
    rec_info.duration = float(r_info["duration"] * 1000)  # in seconds, convert to ms
    time_string = r_info["created"]
    if time_string[-1:] == "Z":
        # change Z suffix to +00:00 for ISO 8601 format that datetime understands
        time_string = time_string[:-1] + "+00:00"
    rec_info.start_time = timestamps.Timestamp(int(datetime.datetime.fromisoformat(time_string).timestamp()))

    # get participant info (if available)
    file = input_dir / r_info["meta-folder"] / "participant"
    if file.is_file():
        with pathlib.Path(file).open("rb") as j:
            p_info = json.load(j)
        rec_info.participant = p_info["name"]

    # get system info
    rec_info.firmware_version = (input_dir / r_info["meta-folder"] / "RuVersion").read_text()
    rec_info.glasses_serial = (input_dir / r_info["meta-folder"] / "HuSerial").read_text()
    rec_info.recording_unit_serial = (input_dir / r_info["meta-folder"] / "RuSerial").read_text()

    return rec_info


def check_recording(input_dir: str | pathlib.Path, rec_info: Recording) -> None:
    """Validate that rec_info matches the actual recording on disk.

    Re-reads recording info and compares participant, duration, start time,
    firmware version, and serial numbers.

    Args:
        input_dir: Path to the Tobii Glasses 3 recording directory.
        rec_info: Recording metadata to validate.

    Raises:
        ValueError: If any field in rec_info doesn't match the actual recording.

    """
    actual_rec_info = get_recording_info(input_dir)
    if actual_rec_info is None or rec_info.name != actual_rec_info.name:
        raise ValueError(f'A recording with the name "{rec_info.name}" was not found in the folder {input_dir}.')

    # make sure caller did not mess with rec_info
    if rec_info.participant != actual_rec_info.participant:
        raise ValueError(
            f'A recording with the participant "{rec_info.participant}" was not found in the folder {input_dir}.'
        )
    if rec_info.duration != actual_rec_info.duration:
        raise ValueError(
            f'A recording with the duration "{rec_info.duration}" was not found in the folder {input_dir}.'
        )
    if rec_info.start_time.value != actual_rec_info.start_time.value:
        raise ValueError(
            f'A recording with the start_time "{rec_info.start_time.display}" was not found in the folder {input_dir}.'
        )
    if rec_info.firmware_version != actual_rec_info.firmware_version:
        raise ValueError(
            f'A recording with the firmware_version "{rec_info.firmware_version}" was not found in the folder {input_dir}.'
        )
    if rec_info.glasses_serial != actual_rec_info.glasses_serial:
        raise ValueError(
            f'A recording with the glasses_serial "{rec_info.glasses_serial}" was not found in the folder {input_dir}.'
        )
    if rec_info.recording_unit_serial != actual_rec_info.recording_unit_serial:
        raise ValueError(
            f'A recording with the recording_unit_serial "{rec_info.recording_unit_serial}" was not found in the folder {input_dir}.'
        )


def copy_tobii_recording(
    input_dir: pathlib.Path, output_dir: pathlib.Path, copy_scene_video: bool
) -> tuple[pathlib.Path, pathlib.Path | None]:
    """Copy scene video and decompress gaze data to output dir.

    Copies ``scenevideo.mp4`` and decompresses ``gazedata.gz`` into the
    output directory.

    Args:
        input_dir: Source recording directory.
        output_dir: Destination directory.
        copy_scene_video: If True, copy the video file; otherwise just
            return the source path.

    Returns:
        A tuple of (source video path, destination video path or None).

    """
    src_file = input_dir / "scenevideo.mp4"
    if copy_scene_video:
        dest_file = output_dir / f"{naming.scene_camera_video_fname_stem}.mp4"
        shutil.copy2(str(src_file), str(dest_file))
    else:
        dest_file = None

    # Unzip the gaze data file
    for f in ["gazedata.gz"]:
        with (
            gzip.open(str(input_dir / f)) as zip_file,
            pathlib.Path(output_dir / pathlib.Path(f).stem).open("wb") as unzipped_file,
        ):
            shutil.copyfileobj(zip_file, unzipped_file)

    return src_file, dest_file


def get_camera_from_json(input_dir: str | pathlib.Path, output_dir: str | pathlib.Path) -> np.ndarray:
    """Read camera calibration from the ``recording.g3`` JSON file.

    Extracts scene camera intrinsics (focal length, principal point,
    distortion coefficients, position, rotation), builds an OpenCV-style
    camera matrix, and writes the calibration XML.

    Args:
        input_dir: Source recording directory containing ``recording.g3``.
        output_dir: Destination directory for the calibration XML.

    Returns:
        The scene camera resolution as a numpy array ``[width, height]``.

    """
    with pathlib.Path(input_dir / "recording.g3").open("rb") as f:
        r_info = json.load(f)

    camera = r_info["scenecamera"]["camera-calibration"]

    # rename some fields, ensure they are numpy arrays
    camera["focalLength"] = np.array(camera.pop("focal-length"))
    camera["principalPoint"] = np.array(camera.pop("principal-point"))
    camera["radialDistortion"] = np.array(camera.pop("radial-distortion"))
    camera["tangentialDistortion"] = np.array(camera.pop("tangential-distortion"))

    camera["position"] = np.array(camera["position"])
    camera["resolution"] = np.array(camera["resolution"])
    camera["rotation"] = np.array(camera["rotation"])

    # turn into camera matrix and distortion coefficients as used by OpenCV
    camera["cameraMatrix"] = np.identity(3)
    camera["cameraMatrix"][0, 0] = camera["focalLength"][0]
    camera["cameraMatrix"][0, 1] = camera["skew"]
    camera["cameraMatrix"][1, 1] = camera["focalLength"][1]
    camera["cameraMatrix"][0, 2] = camera["principalPoint"][0]
    camera["cameraMatrix"][1, 2] = camera["principalPoint"][1]

    camera["distCoeff"] = np.zeros(5)
    camera["distCoeff"][:2] = camera["radialDistortion"][:2]
    camera["distCoeff"][2:4] = camera["tangentialDistortion"]
    camera["distCoeff"][4] = camera["radialDistortion"][2]

    # store to file
    fs = cv2.FileStorage(output_dir / naming.scene_camera_calibration_fname, cv2.FILE_STORAGE_WRITE)
    for key, value in camera.items():
        fs.write(name=key, val=value)
    fs.release()

    return camera["resolution"]


def format_gaze_data(
    input_dir: str | pathlib.Path, scene_video_dimensions: list[int], rec_info: Recording
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and process Tobii Glasses 3 gaze data from the gazedata file.

    Parses the JSON gaze data, extracts frame timestamps from the scene
    video, and assigns frame indices to each gaze sample.

    Args:
        input_dir: Directory containing the decompressed ``gazedata`` file.
        scene_video_dimensions: Scene camera resolution ``[width, height]``.
        rec_info: Recording metadata for locating the scene video.

    Returns:
        A tuple of (gaze DataFrame indexed by timestamp, frame timestamps
        DataFrame).

    """
    # convert the json file to pandas dataframe
    df = json2df(input_dir / "gazedata", scene_video_dimensions)

    # read video file, create array of frame timestamps
    frame_timestamps = video_utils.get_frame_timestamps_from_video(rec_info.get_scene_video_path())

    # use the frame timestamps to assign a frame number to each data point
    frame_idx = video_utils.timestamps_to_frame_number(df.index, frame_timestamps["timestamp"].to_numpy())
    df.insert(0, "frame_idx", frame_idx["frame_idx"])

    return df, frame_timestamps


def json2df(json_file: str | pathlib.Path, scene_video_dimensions: list[int]) -> pd.DataFrame:
    """Parse Tobii Glasses 3 gazedata JSON into a pandas DataFrame.

    Each line in the file is a JSON object. Non-gaze entries are dropped.
    Gaze origin, direction, pupil diameter, 2D and 3D gaze positions are
    extracted per eye. Timestamps are converted from seconds to
    milliseconds. The gazedata file is deleted after parsing.

    Args:
        json_file: Path to the decompressed ``gazedata`` file.
        scene_video_dimensions: Scene camera resolution ``[width, height]``
            for converting normalized gaze coordinates to pixels.

    Returns:
        A DataFrame indexed by timestamp (ms) with gaze data columns.

    """
    # file contains one JSON object per line; wrap into a JSON array for parsing
    entries = json.loads("[" + pathlib.Path(json_file).read_text(encoding="utf-8").replace("\n", ",")[:-1] + "]")

    # json no longer needed, remove
    json_file.unlink(missing_ok=True)

    # turn gaze data into data frame
    df_r = pd.json_normalize(entries)
    # convert timestamps from s to ms and set as index
    df_r.loc[:, "timestamp"] *= 1000.0
    df_r = df_r.set_index("timestamp")
    # drop anything thats not gaze
    df_r = df_r.drop(df_r[df_r.type != "gaze"].index)
    # manipulate data frame to expand columns as needed
    df = pd.DataFrame([], index=df_r.index)

    def expander(a: list, n: int) -> list:
        return [[math.nan] * n if not isinstance(x, list) else x for x in a]

    # monocular gaze data
    for eye in ("left", "right"):
        if "data.eye" + eye + ".gazeorigin" not in df_r.columns:
            continue  # no data at all for this eye
        which_eye = eye[:1]
        df[data_files.get_column_labels("gaze_ori_" + which_eye, 3)] = pd.DataFrame(
            expander(df_r["data.eye" + eye + ".gazeorigin"].tolist(), 3), index=df_r.index
        )
        df[data_files.get_column_labels("gaze_dir_" + which_eye, 3)] = pd.DataFrame(
            expander(df_r["data.eye" + eye + ".gazedirection"].tolist(), 3), index=df_r.index
        )
        df["pup_diam_" + which_eye] = df_r["data.eye" + eye + ".pupildiameter"]

    # binocular gaze data
    df[data_files.get_column_labels("gaze_pos_3d", 3)] = pd.DataFrame(
        expander(df_r["data.gaze3d"].tolist(), 3), index=df_r.index
    )
    df[data_files.get_column_labels("gaze_pos_vid", 2)] = pd.DataFrame(
        expander(df_r["data.gaze2d"].tolist(), 2), index=df_r.index
    )
    # convert normalized [0,1] gaze coordinates to pixel coordinates
    df.loc[:, "gaze_pos_vid_x"] *= scene_video_dimensions[0]
    df.loc[:, "gaze_pos_vid_y"] *= scene_video_dimensions[1]

    return df
