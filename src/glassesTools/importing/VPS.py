"""Cast raw Viewpointsystem VPS 19 data into common format.

The output directory will contain:
    - frame_timestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gaze_data.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera,
                    and 3D coordinates in the coordinate frame of the glasses (may need rotation/translation
                    to represent in world camera's coordinate frame)
"""

import datetime
import json
import pathlib
import shutil

import pandas as pd

from .. import naming, timestamps, video_utils
from ..eyetracker import EyeTracker
from ..recording import Recording


def preprocess_data(
    output_dir: str | pathlib.Path | None = None,
    device: str | EyeTracker | None = None,
    source_dir: str | pathlib.Path | None = None,
    rec_info: Recording | None = None,
    copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
    cam_cal_file: str | pathlib.Path | None = None,
) -> Recording:
    """Run all preprocessing steps on VPS data and store in output_dir."""
    from . import _store_data, check_device, check_folders

    device, rec_info, _ = check_device(device, rec_info)
    if device not in {EyeTracker.VPS_19, EyeTracker.VPS_Lite}:
        raise ValueError(
            f"Provided device ({rec_info.eye_tracker.value}) is not a {EyeTracker.VPS_19.value} or a {EyeTracker.VPS_Lite.value}."
        )
    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, device)
    print(f"processing: {source_dir.name} -> {output_dir}")

    # check and copy needed files to the output directory
    print("  Check and copy raw data...")
    # check tobii recording and get export directory
    if rec_info is not None:
        check_recording(device, source_dir, rec_info)
    else:
        rec_info = get_recording_info(source_dir)
        if rec_info is None:
            raise RuntimeError(f"The folder {source_dir} is not recognized as a {device.value} recording.")

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()

    # copy the raw data to the output directory
    copy_vps_recording(device, source_dir, output_dir, rec_info, copy_scene_video)

    # prep the copied data...
    print("  Getting camera calibration...")
    if cam_cal_file is not None:
        shutil.copyfile(cam_cal_file, output_dir / naming.scene_camera_calibration_fname)
    else:
        print("    !! No camera calibration provided!")
    print("  Prepping gaze data...")
    gaze_df, frame_timestamps = format_gaze_data(source_dir, rec_info)

    _store_data(
        output_dir, gaze_df, frame_timestamps, rec_info, source_dir_as_relative_path=source_dir_as_relative_path
    )

    return rec_info


def get_recording_info(input_dir: str | pathlib.Path, device: EyeTracker) -> list[Recording] | None:
    """Return recording info for the given directory, or None if not a valid recording."""
    input_dir = pathlib.Path(input_dir)

    if device == EyeTracker.VPS_19:
        vid_exts = [".mkv", ".mp4"]
    elif device == EyeTracker.VPS_Lite:
        vid_exts = [".mp4"]
    # recordings are identified as a tsv and an mkv file with the
    # same name
    rec_infos: list[Recording] = []
    for r in input_dir.glob("*.tsv"):
        if not any(r.with_suffix(ext).is_file() for ext in vid_exts):
            continue
        # get more info
        with pathlib.Path(r).open(encoding="utf-8") as f:
            lines = []
            for _ in range(5):
                lines.append(f.readline())
        u_info = json.loads(
            lines[4].removeprefix("# system").removeprefix(":")
        )  # info about the system, remove : separately as it seems only VPS 19 and not VPS Lite has it
        # a VPS 19 recording will have a 'Smart Unit' entry
        if device == EyeTracker.VPS_19 and "Smart Unit" not in u_info:
            continue
        # a VPS Lite recording will have a 'Lite Unit' entry
        if device == EyeTracker.VPS_Lite and "Lite Unit" not in u_info:
            continue
        rec_infos.append(Recording(source_directory=input_dir, eye_tracker=device))
        rec_infos[-1].name = r.stem
        time_string = lines[2][len("# Recording start: ") :].strip()
        rec_infos[-1].start_time = timestamps.Timestamp(int(datetime.datetime.fromisoformat(time_string).timestamp()))
        rec_infos[-1].glasses_serial = u_info["glasses"]
        rec_infos[-1].recording_unit_serial = (
            u_info["Smart Unit"] if device == EyeTracker.VPS_19 else u_info["Lite Unit"]
        )
        rec_infos[-1].firmware_version = u_info["operating system"]

    # should return None if no valid recordings found
    return rec_infos or None


def check_recording(device: EyeTracker, input_dir: str | pathlib.Path, rec_info: Recording) -> bool:
    """Check that the expected recording files exist on disk."""
    input_dir = pathlib.Path(input_dir)

    if device == EyeTracker.VPS_19:
        vid_exts = [".mkv", ".mp4"]
    elif device == EyeTracker.VPS_Lite:
        vid_exts = [".mp4"]

    def _raise_if_file_doesnt_exist(file: str | pathlib.Path) -> None:
        if not (input_dir / file).is_file():
            raise RuntimeError(f"Recording {rec_info.name} not found: {file} file not found in {input_dir}.")

    _raise_if_file_doesnt_exist(f"{rec_info.name}.tsv")
    if not any((input_dir / f"{rec_info.name}{ext}").is_file() for ext in vid_exts):
        raise RuntimeError(
            f"Recording {rec_info.name} not found: no {rec_info.name}.ext video file found in {input_dir} where ext is one of the expected extensions: {vid_exts}."
        )

    return True


def copy_vps_recording(
    device: EyeTracker, input_dir: pathlib.Path, output_dir: pathlib.Path, rec_info: Recording, copy_scene_video: bool
) -> None:
    """Copy the relevant files from the specified input dir to the specified output dir."""
    # Copy relevant files to new directory
    if device == EyeTracker.VPS_19:
        src_file = input_dir / f"{rec_info.name}.mkv"
        if not src_file.is_file():
            src_file = input_dir / f"{rec_info.name}.mp4"
    elif device == EyeTracker.VPS_Lite:
        src_file = input_dir / f"{rec_info.name}.mp4"

    if copy_scene_video:
        dest_file = output_dir / f"{naming.scene_camera_video_fname_stem}{src_file.suffix}"
        shutil.copy2(src_file, dest_file)
    else:
        dest_file = None

    if dest_file:
        rec_info.scene_video_file = dest_file.name
    else:
        rec_info.scene_video_file = src_file.name


def format_gaze_data(input_dir: str | pathlib.Path, rec_info: Recording) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load gazedata tsv file.

    Format to get the gaze coordinates w/r/t world camera, and timestamps for every frame of video.

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps

    """
    # convert the json file to pandas dataframe
    df = gaze2df(input_dir / f"{rec_info.name}.tsv")

    # read video file, create array of frame timestamps
    frame_timestamps = video_utils.get_frame_timestamps_from_video(rec_info.get_scene_video_path())

    # return the gaze data df and frame time stamps array
    return df, frame_timestamps


def gaze2df(gaze_file: str | pathlib.Path) -> pd.DataFrame:
    """Convert the .tsv file to a pandas dataframe"""
    df = pd.read_csv(gaze_file, delimiter="\t", comment="#", index_col=False)

    # get time sync info
    t0 = df["FrontTimeStamp"].iloc[0] - df["MediaTimeStamp"].iloc[0]

    # rename and reorder columns
    lookup = {
        "GazeTimeStamp": "timestamp",
        "MediaFrameIndex": "frame_idx",
        "Gaze2dX": "gaze_pos_vid_x",
        "Gaze2dY": "gaze_pos_vid_y",
        "PupilCenterLeftX": "gaze_ori_l_x",
        "PupilCenterLeftY": "gaze_ori_l_y",
        "PupilCenterLeftZ": "gaze_ori_l_z",
        "GazeLeftX": "gaze_dir_l_x",
        "GazeLeftY": "gaze_dir_l_y",
        "GazeLeftZ": "gaze_dir_l_z",
        "PupilCenterRightX": "gaze_ori_r_x",
        "PupilCenterRightY": "gaze_ori_r_y",
        "PupilCenterRightZ": "gaze_ori_r_z",
        "GazeRightX": "gaze_dir_r_x",
        "GazeRightY": "gaze_dir_r_y",
        "GazeRightZ": "gaze_dir_r_z",
        "PupilDiaLeft": "pup_diam_l",
        "PupilDiaRight": "pup_diam_r",
    }
    df = df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    df = df[idx]

    # drop useless gaze frames
    df = df.dropna(axis=0, subset="timestamp")

    # set timestamps t0 to start of video, convert from s to ms and set as index
    df.loc[:, "timestamp"] -= t0
    df.loc[:, "timestamp"] *= 1000.0
    df = df.set_index("timestamp")

    # return the dataframe
    return df
