"""Cast raw AdHawk MindLink data into common format."""

import csv
import datetime
import json
import pathlib
import shutil

import cv2
import numpy as np
import pandas as pd

from .. import naming, timestamps, video_utils
from ..eyetracker import EyeTracker
from ..recording import Recording


def preprocess_data(
    output_dir: str | pathlib.Path | None = None,
    source_dir: str | pathlib.Path | None = None,
    rec_info: Recording | None = None,
    copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
    cam_cal_file: str | pathlib.Path | None = None,
) -> Recording:
    """Run all preprocessing steps on AdHawk MindLink data and store in output_dir.

    Args:
        output_dir: Working directory where the imported recording will be placed.
        source_dir: Path to the raw recording. Falls back to ``rec_info.source_directory``.
        rec_info: Optional pre-populated recording metadata.
        copy_scene_video: Whether to copy the scene video to output_dir.
        source_dir_as_relative_path: Store source_dir as a relative path in rec_info.
        cam_cal_file: Path to an external camera calibration file. If None,
            hardcoded calibration values from AdHawk are used.

    Returns:
        The populated Recording object written to output_dir.

    """
    from . import _store_data, check_folders

    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, EyeTracker.AdHawk_MindLink)
    print(f"processing: {source_dir.name} -> {output_dir}")

    # check and copy needed files to the output directory
    print("  Check and copy raw data...")
    # check adhawk recording and get export directory
    if rec_info is not None:
        check_recording(source_dir, rec_info)
    else:
        rec_info = get_recording_info(source_dir)
        if rec_info is None:
            raise RuntimeError(
                f"The folder {source_dir} is not recognized as a {EyeTracker.AdHawk_MindLink.value} recording."
            )

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()

    # copy the raw data to the output directory
    src_vid, dest_vid = copy_adhawk_recording(source_dir, output_dir, copy_scene_video)
    if dest_vid:
        rec_info.scene_video_file = dest_vid.name
    else:
        rec_info.scene_video_file = src_vid.name

    # prep the copied data...
    print("  Getting camera calibration...")
    if cam_cal_file is not None:
        shutil.copyfile(str(cam_cal_file), str(output_dir / naming.scene_camera_calibration_fname))
        scene_video_dimensions = np.array([1280, 720])
    else:
        print("    !! No camera calibration provided! Defaulting to hardcoded")
        scene_video_dimensions = get_camera_hardcoded(output_dir)
    print("  Prepping gaze data...")
    gaze_df, frame_timestamps = format_gaze_data(source_dir, scene_video_dimensions, rec_info)

    _store_data(
        output_dir, gaze_df, frame_timestamps, rec_info, source_dir_as_relative_path=source_dir_as_relative_path
    )

    return rec_info


def get_recording_info(input_dir: str | pathlib.Path) -> Recording | None:
    """Return recording info for an AdHawk MindLink recording directory.

    Reads ``meta_data.json`` for duration and participant, and the first
    gaze sample's UTC timestamp for the recording start time.

    Args:
        input_dir: Path to the AdHawk recording directory.

    Returns:
        A Recording object, or None if ``meta_data.json`` is not found.

    """
    input_dir = pathlib.Path(input_dir)
    rec_info = Recording(source_directory=input_dir, eye_tracker=EyeTracker.AdHawk_MindLink)

    # get recording info
    rec_info.name = input_dir.name

    file = input_dir / "meta_data.json"
    if not file.is_file():
        return None
    with pathlib.Path(file).open("rb") as j:
        r_info = json.load(j)
    rec_info.duration = float(r_info["manifest"]["recording_length_ms"])
    rec_info.participant = r_info["user_profile"]["name"]
    # get recording start time by reading UTC time associated with first gaze sample
    gaze_entry = get_meta_entry(input_dir, "gaze")
    file = input_dir / gaze_entry["file_name"]
    with pathlib.Path(file).open(encoding="utf-8") as read_obj:
        csv_reader = csv.DictReader(read_obj)
        sample = next(csv_reader)
    time_string = sample["UTC_Time"]
    if time_string[-1:] == "Z":
        # change Z suffix (if any) to +00:00 for ISO 8601 format that datetime understands
        time_string = time_string[:-1] + "+00:00"
    rec_info.start_time = timestamps.Timestamp(int(datetime.datetime.fromisoformat(time_string).timestamp()))

    return rec_info


def get_meta(input_dir: str | pathlib.Path, key: str | None = None) -> dict | None:
    """Read meta_data.json and return the full dict or a specific top-level key.

    Args:
        input_dir: Path to the AdHawk recording directory.
        key: If given, return only this key from the JSON. Otherwise return
            the entire dict.

    Returns:
        The parsed JSON dict (or a sub-dict), or None if the file doesn't exist.

    """
    file = input_dir / "meta_data.json"
    if not file.is_file():
        return None
    with pathlib.Path(file).open("rb") as j:
        r_info = json.load(j)

    if key:
        return r_info[key]
    return r_info


def get_meta_entry(input_dir: str | pathlib.Path, entry_name: str) -> dict | None:
    """Return the manifest entry whose type matches entry_name.

    Args:
        input_dir: Path to the AdHawk recording directory.
        entry_name: The entry type to find (e.g., ``"gaze"``, ``"video"``,
            ``"pupil_position"``).

    Returns:
        The matching manifest entry dict, or None if not found.

    """
    manifest = get_meta(input_dir, key="manifest")
    entry = None
    for e in manifest["entries"]:
        if e["type"].lower() == entry_name:
            entry = e
            break
    return entry


def check_recording(input_dir: str | pathlib.Path, rec_info: Recording) -> None:
    """Validate that rec_info matches the actual recording on disk.

    Re-reads recording info from the directory and compares name, participant,
    duration, and start time.

    Args:
        input_dir: Path to the AdHawk recording directory.
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


def copy_adhawk_recording(
    input_dir: pathlib.Path, output_dir: pathlib.Path, copy_scene_video: bool
) -> tuple[pathlib.Path, pathlib.Path | None]:
    """Copy the scene video from input_dir to output_dir.

    Looks up the video filename from the manifest metadata.

    Args:
        input_dir: Source recording directory.
        output_dir: Destination directory.
        copy_scene_video: If True, copy the video file; otherwise return
            the source path without copying.

    Returns:
        A tuple of (source video path, destination video path or None).

    """
    # figure out what the video file is called
    vid_entry = get_meta_entry(input_dir, "video")
    src_file = input_dir / vid_entry["file_name"]
    if copy_scene_video:
        dest_file = output_dir / f"{naming.scene_camera_video_fname_stem}.mp4"
        shutil.copy2(str(src_file), str(dest_file))
    else:
        dest_file = None

    return src_file, dest_file


def get_camera_hardcoded(output_dir: str | pathlib.Path) -> np.ndarray:
    """Write hardcoded camera calibration to an OpenCV XML file.

    Uses calibration values provided by AdHawk (1280x720), including
    camera position and rotation relative to the glasses frame.

    Args:
        output_dir: Directory where the calibration XML file is written.

    Returns:
        The scene camera resolution as a 2-element array ``[width, height]``.

    """
    # turn into camera matrix and distortion coefficients as used by OpenCV
    camera = {}
    camera["cameraMatrix"] = np.array([
        [8.6175611023603130e02, 0.0, 6.4220317156609758e02],
        [0.0, 8.6411314484767183e02, 3.4611059418088462e02],
        [0.0, 0.0, 1.0],
    ])
    camera["distCoeff"] = np.array([
        6.4704736326069179e-01,
        6.9842325204621162e01,
        -3.8446374749176787e-03,
        -6.5685769622407693e-03,
        3.3239962207009803e01,
        5.0824354805695138e-01,
        6.9018441628550974e01,
        3.1191976852198923e01,
    ])
    camera["resolution"] = np.array([1280, 720])
    camera["position"] = np.array([-0.0685, 0.0152028, 0.00340752]) * 1000  # our positions are in mm, not m
    camera["rotation"] = cv2.Rodrigues(np.radians(np.array([12.000000000000043, 0.0, 0.0])))[0]

    # store to file
    fs = cv2.FileStorage(output_dir / naming.scene_camera_calibration_fname, cv2.FILE_STORAGE_WRITE)
    for key, value in camera.items():
        fs.write(name=key, val=value)
    fs.release()

    return camera["resolution"]


def format_gaze_data(
    input_dir: str | pathlib.Path, scene_video_dimensions: list[int], rec_info: Recording
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and process AdHawk gaze data CSV files.

    Parses gaze data, extracts frame timestamps from the scene video,
    and returns both.

    Args:
        input_dir: Path to the AdHawk recording directory.
        scene_video_dimensions: Scene camera resolution ``[width, height]``.
        rec_info: Recording metadata for locating the scene video.

    Returns:
        A tuple of (gaze DataFrame indexed by timestamp, frame timestamps DataFrame).

    """
    df = csv2df(input_dir, scene_video_dimensions)

    # read video file, create array of frame timestamps
    frame_timestamps = video_utils.get_frame_timestamps_from_video(rec_info.get_scene_video_path())

    return df, frame_timestamps


def csv2df(input_dir: str | pathlib.Path, scene_video_dimensions: list[int]) -> pd.DataFrame:
    """Parse AdHawk gaze and pupil position CSVs into a single DataFrame.

    Reads the gaze CSV and pupil position CSV (from the manifest), merges
    them on timestamp, converts to milliseconds in video time, scales
    normalized gaze coordinates to pixels, and transforms coordinates
    from AdHawk's system (Y-up, Z-backward) to the common system
    (Y-down, Z-forward).

    Args:
        input_dir: Path to the AdHawk recording directory.
        scene_video_dimensions: Scene camera resolution ``[width, height]``
            for converting normalized gaze coordinates to pixels.

    Returns:
        A DataFrame indexed by timestamp (ms) with gaze data columns.

    """
    vid_entry = get_meta_entry(input_dir, "video")
    gaze_entry = get_meta_entry(input_dir, "gaze")

    file = input_dir / gaze_entry["file_name"]
    df = pd.read_csv(file)

    # prepare data frame
    # remove unneeded columns
    df = df.drop(
        columns=["Screen_X", "Screen_Y", "UTC_Time", "Image_One_Degree_X", "Image_One_Degree_Y"], errors="ignore"
    )  # drop these columns if they exist

    # rename and reorder columns
    lookup = {
        "Timestamp": "timestamp",
        "Frame_Index": "frame_idx",
        "Image_X": "gaze_pos_vid_x",
        "Image_Y": "gaze_pos_vid_y",
        "Gaze_X_Left": "gaze_dir_l_x",
        "Gaze_Y_Left": "gaze_dir_l_y",
        "Gaze_Z_Left": "gaze_dir_l_z",
        "Gaze_X_Right": "gaze_dir_r_x",
        "Gaze_Y_Right": "gaze_dir_r_y",
        "Gaze_Z_Right": "gaze_dir_r_z",
        "Gaze_X": "gaze_pos_3d_x",
        "Gaze_Y": "gaze_pos_3d_y",
        "Gaze_Z": "gaze_pos_3d_z",
    }
    df = df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    idx.extend([x for x in df.columns if x not in idx])  # append columns not in lookup
    df = df[idx]

    # get gaze vector origins
    pp_entry = get_meta_entry(input_dir, "pupil_position")
    file = input_dir / pp_entry["file_name"]
    df_p = pd.read_csv(file)
    df_p = df_p.drop(columns=["UTC_Time"], errors="ignore")  # drop these columns if they exist
    # rename and reorder columns
    lookup = {
        "Timestamp": "timestamp",
        "Pupil_Pos_X_Left": "gaze_ori_l_x",
        "Pupil_Pos_Y_Left": "gaze_ori_l_y",
        "Pupil_Pos_Z_Left": "gaze_ori_l_z",
        "Pupil_Pos_X_Right": "gaze_ori_r_x",
        "Pupil_Pos_Y_Right": "gaze_ori_r_y",
        "Pupil_Pos_Z_Right": "gaze_ori_r_z",
    }
    df_p = df_p.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df_p.columns]
    idx.extend([x for x in df_p.columns if x not in idx])  # append columns not in lookup
    df_p = df_p[idx]

    # merge
    df = df.merge(df_p, on="timestamp")

    # convert timestamps from s to ms and set as index
    df.loc[:, "timestamp"] *= 1000.0
    # set first gaze timestamp to 0 and express gaze timestamps in video time
    df.loc[:, "timestamp"] -= df.loc[0, "timestamp"] - (
        gaze_entry["attribute"]["start_time_ms"] - vid_entry["attribute"]["start_time_ms"]
    )
    # remove data with negative timestamps
    df = df[df.timestamp >= 0]
    df = df.set_index("timestamp")

    # binocular gaze data
    df.loc[:, "gaze_pos_vid_x"] *= scene_video_dimensions[0]
    df.loc[:, "gaze_pos_vid_y"] *= scene_video_dimensions[1]

    # adhawk positive z is backward, ours is forward
    df.loc[:, "gaze_ori_l_z"] = -df.loc[:, "gaze_ori_l_z"]
    df.loc[:, "gaze_dir_l_z"] = -df.loc[:, "gaze_dir_l_z"]
    df.loc[:, "gaze_ori_r_z"] = -df.loc[:, "gaze_ori_r_z"]
    df.loc[:, "gaze_dir_r_z"] = -df.loc[:, "gaze_dir_r_z"]
    df.loc[:, "gaze_pos_3d_z"] = -df.loc[:, "gaze_pos_3d_z"]

    # adhawk positive y is upward, ours is downward
    df.loc[:, "gaze_ori_l_y"] = -df.loc[:, "gaze_ori_l_y"]
    df.loc[:, "gaze_dir_l_y"] = -df.loc[:, "gaze_dir_l_y"]
    df.loc[:, "gaze_ori_r_y"] = -df.loc[:, "gaze_ori_r_y"]
    df.loc[:, "gaze_dir_r_y"] = -df.loc[:, "gaze_dir_r_y"]
    df.loc[:, "gaze_pos_3d_y"] = -df.loc[:, "gaze_pos_3d_y"]

    # adhawk gaze pos is in m, ours is in mm
    # NB: gaze ori is in mm!
    df.loc[:, "gaze_pos_3d_x"] *= 1000
    df.loc[:, "gaze_pos_3d_y"] *= 1000
    df.loc[:, "gaze_pos_3d_z"] *= 1000

    return df
