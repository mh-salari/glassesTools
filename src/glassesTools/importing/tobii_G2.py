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
import struct

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
) -> Recording:
    """Run all preprocessing steps on Tobii Glasses 2 data and store in output_dir."""
    from . import _store_data, check_folders

    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, EyeTracker.Tobii_Glasses_2)
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
                f"The folder {source_dir} is not recognized as a {EyeTracker.Tobii_Glasses_2.value} recording."
            )

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()

    # copy the raw data to the output directory
    src_vid, dest_vid = copy_tobii_recording(source_dir, output_dir, copy_scene_video)
    if dest_vid:
        rec_info.scene_video_file = dest_vid.name
    else:
        rec_info.scene_video_file = src_vid.parent.parent.name + "/" + src_vid.parent.name + "/" + src_vid.name

    # prep the copied data...
    print("  Getting camera calibration...")
    scene_video_dimensions = get_camera_from_tslv(output_dir)
    print("  Prepping gaze data...")
    gaze_df, frame_timestamps = format_gaze_data(output_dir, scene_video_dimensions, rec_info)

    _store_data(
        output_dir, gaze_df, frame_timestamps, rec_info, source_dir_as_relative_path=source_dir_as_relative_path
    )

    return rec_info


def get_recording_info(input_dir: str | pathlib.Path) -> Recording | None:
    """Return recording info for the given directory, or None if not a valid recording."""
    input_dir = pathlib.Path(input_dir)
    rec_info = Recording(source_directory=input_dir, eye_tracker=EyeTracker.Tobii_Glasses_2)

    # get participant info
    file = input_dir / "participant.json"
    if not file.is_file():
        return None
    with pathlib.Path(file).open(encoding="utf-8") as j:
        i_info = json.load(j)
    rec_info.participant = i_info["pa_info"]["Name"]

    # get recording info
    file = input_dir / "recording.json"
    if not file.is_file():
        return None
    with pathlib.Path(file).open(encoding="utf-8") as j:
        i_info = json.load(j)
    rec_info.name = i_info["rec_info"]["Name"]
    rec_info.duration = float(i_info["rec_length"] * 1000)  # in seconds, convert to ms
    time_string = i_info["rec_created"]
    if time_string[-4:].isdigit() and time_string[-5:-4] == "+":
        # add hour:minute separator for ISO 8601 format that datetime understands
        time_string = time_string[:-2] + ":" + time_string[-2:]
    rec_info.start_time = timestamps.Timestamp(int(datetime.datetime.fromisoformat(time_string).timestamp()))

    # get system info
    file = input_dir / "sysinfo.json"
    if not file.is_file():
        return None
    with pathlib.Path(file).open(encoding="utf-8") as j:
        i_info = json.load(j)

    rec_info.firmware_version = i_info["servicemanager_version"]
    rec_info.glasses_serial = i_info["hu_serial"]
    rec_info.recording_unit_serial = i_info["ru_serial"]

    # we got a valid recording and at least some info if we got here
    # return what we've got
    return rec_info


def check_recording(input_dir: str | pathlib.Path, rec_info: Recording) -> None:
    """Validate that rec_info matches the actual recording on disk."""
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
    """Copy the relevant files from the specified input dir to the specified output dir."""
    # Copy relevent files to new directory
    input_dir = input_dir / "segments" / "1"
    src_file = input_dir / "fullstream.mp4"
    if copy_scene_video:
        dest_file = output_dir / f"{naming.scene_camera_video_fname_stem}.mp4"
        shutil.copy2(str(src_file), str(dest_file))
    else:
        dest_file = None

    # Unzip the gaze data and tslv files
    for f in ["livedata.json.gz", "et.tslv.gz"]:
        with (
            gzip.open(str(input_dir / f)) as zip_file,
            pathlib.Path(output_dir / pathlib.Path(f).stem).open("wb") as unzipped_file,
        ):
            shutil.copyfileobj(zip_file, unzipped_file)

    return src_file, dest_file


def get_camera_from_tslv(input_dir: str | pathlib.Path) -> np.ndarray:
    """Read binary TSLV file until camera calibration information is retrieved."""
    with pathlib.Path(str(input_dir / "et.tslv")).open("rb") as f:
        # first look for camera item (TSLV type==300)
        while True:
            tslv_type = struct.unpack("h", f.read(2))[0]
            _status = struct.unpack("h", f.read(2))[0]
            payload_length = struct.unpack("i", f.read(4))[0]
            payload_length_padded = math.ceil(payload_length / 4) * 4
            if tslv_type != 300:
                # skip payload
                f.read(payload_length_padded)
            else:
                break

        # read info about camera
        camera = {}
        camera["id"] = struct.unpack("b", f.read(1))[0]
        camera["location"] = struct.unpack("b", f.read(1))[0]
        f.read(2)  # skip padding
        camera["position"] = np.array(struct.unpack("3f", f.read(4 * 3)))
        camera["rotation"] = np.reshape(struct.unpack("9f", f.read(4 * 9)), (3, 3))
        camera["focalLength"] = np.array(struct.unpack("2f", f.read(4 * 2)))
        camera["skew"] = struct.unpack("f", f.read(4))[0]
        camera["principalPoint"] = np.array(struct.unpack("2f", f.read(4 * 2)))
        camera["radialDistortion"] = np.array(struct.unpack("3f", f.read(4 * 3)))
        camera["tangentialDistortion"] = np.array(
            struct.unpack("3f", f.read(4 * 3))[:-1]
        )  # drop last element (always zero), since there are only two tangential distortion parameters
        camera["resolution"] = np.array(struct.unpack("2h", f.read(2 * 2)))

    # turn into camera matrix and distortion coefficients as used by OpenCV
    camera["cameraMatrix"] = np.identity(3)
    camera["cameraMatrix"][0, 0] = camera["focalLength"][0]
    camera["cameraMatrix"][0, 1] = camera["skew"]
    camera["cameraMatrix"][1, 1] = camera["focalLength"][1]
    camera["cameraMatrix"][0, 2] = camera["principalPoint"][0]
    camera["cameraMatrix"][1, 2] = camera["principalPoint"][1]

    camera["distCoeff"] = np.zeros(5)
    camera["distCoeff"][:2] = camera["radialDistortion"][:2]
    camera["distCoeff"][2:4] = camera["tangentialDistortion"][:2]
    camera["distCoeff"][4] = camera["radialDistortion"][2]

    # store to file
    fs = cv2.FileStorage(input_dir / naming.scene_camera_calibration_fname, cv2.FILE_STORAGE_WRITE)
    for key, value in camera.items():
        fs.write(name=key, val=value)
    fs.release()

    # tslv no longer needed, remove
    (input_dir / "et.tslv").unlink(missing_ok=True)

    return camera["resolution"]


def format_gaze_data(
    input_dir: str | pathlib.Path, scene_video_dimensions: list[int], rec_info: Recording
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load livedata.json.

    Format to get the gaze coordinates w/r/t world camera, and timestamps for every frame of video.

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps

    """
    # convert the json file to pandas dataframe
    df = json2df(input_dir / "livedata.json", scene_video_dimensions)

    # get array of frame timestamps for video file
    frame_timestamps = video_utils.get_frame_timestamps_from_video(rec_info.get_scene_video_path())

    # use the frame timestamps to assign a frame number to each data point
    frame_idx = video_utils.timestamps_to_frame_number(df.index, frame_timestamps["timestamp"].to_numpy())
    df.insert(0, "frame_idx", frame_idx["frame_idx"])

    # build the formatted dataframe
    df.index.name = "timestamp"

    # return the gaze data df and frame time stamps array
    return df, frame_timestamps


def json2df(json_file: str | pathlib.Path, scene_video_dimensions: list[int]) -> pd.DataFrame:
    """Convert the livedata.json file to a pandas dataframe"""
    # dicts to store sync points
    vts_sync: list[tuple[int, int]] = []  # scene video timestamp sync
    gaze_data: dict[int, dict[str, float]] = {}  # gaze data

    entries = json.loads("[" + pathlib.Path(json_file).read_text(encoding="utf-8").replace("\n", ",")[:-1] + "]")

    # loop over all lines in json file, each line represents unique json object
    for entry in entries:
        # if non-zero status (error), ensure data found in packet is marked as missing
        is_error = False
        if entry["s"] != 0:
            is_error = True

        # a number of different dictKeys are possible, respond accordingly
        if (
            "vts" in entry
        ):  # "vts" key signfies a scene video timestamp (first frame, first keyframe, and ~1/min afterwards)
            vts_sync.append((entry["ts"], entry["vts"] if not is_error else math.nan))
            continue

        # contains eye or gaze position data
        if not any(x in entry for x in ["eye", "gp", "gp3"]):
            # ignore anything else
            continue
        if entry["ts"] not in gaze_data:
            gaze_data[entry["ts"]] = {}

        if "eye" in entry:
            # this json object contains "eye" data (e.g. pupil info)
            which_eye = entry["eye"][:1]
            if "pc" in entry:
                # origin of gaze vector is the pupil center
                gaze_data[entry["ts"]]["gaze_ori_" + which_eye + "_x"] = entry["pc"][0] if not is_error else math.nan
                gaze_data[entry["ts"]]["gaze_ori_" + which_eye + "_x"] = entry["pc"][0] if not is_error else math.nan
                gaze_data[entry["ts"]]["gaze_ori_" + which_eye + "_y"] = entry["pc"][1] if not is_error else math.nan
                gaze_data[entry["ts"]]["gaze_ori_" + which_eye + "_z"] = entry["pc"][2] if not is_error else math.nan
            elif "pd" in entry:
                gaze_data[entry["ts"]]["pup_diam_" + which_eye] = entry["pd"] if not is_error else math.nan
            elif "gd" in entry:
                gaze_data[entry["ts"]]["gaze_dir_" + which_eye + "_x"] = entry["gd"][0] if not is_error else math.nan
                gaze_data[entry["ts"]]["gaze_dir_" + which_eye + "_y"] = entry["gd"][1] if not is_error else math.nan
                gaze_data[entry["ts"]]["gaze_dir_" + which_eye + "_z"] = entry["gd"][2] if not is_error else math.nan

        # otherwise it contains gaze position data
        elif "gp" in entry:
            gaze_data[entry["ts"]]["gaze_pos_vid_x"] = (
                entry["gp"][0] * scene_video_dimensions[0] if not is_error else math.nan
            )
            gaze_data[entry["ts"]]["gaze_pos_vid_y"] = (
                entry["gp"][1] * scene_video_dimensions[1] if not is_error else math.nan
            )
        elif "gp3" in entry:
            gaze_data[entry["ts"]]["gaze_pos_3d_x"] = entry["gp3"][0] if not is_error else math.nan
            gaze_data[entry["ts"]]["gaze_pos_3d_y"] = entry["gp3"][1] if not is_error else math.nan
            gaze_data[entry["ts"]]["gaze_pos_3d_z"] = entry["gp3"][2] if not is_error else math.nan

    # find out t0 for video in gaze time
    vts_sync = np.array(vts_sync)
    t0 = max([vts_sync[vts_sync[:, 1] == 0, 0]])

    # put gaze data into dataframe, convert timestamps from us to ms
    df = pd.DataFrame.from_dict(gaze_data, orient="index")
    df.index = (df.index - t0) / 1000.0

    # json no longer needed, remove
    json_file.unlink(missing_ok=True)

    # return the dataframe
    return df
