"""Cast Argus Science ETVision data into common format.

The output directory will contain:
    - frame_timestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gaze_data.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera,
                    and 3D coordinates in the coordinate frame of the glasses (may need rotation/translation
                    to represent in world camera's coordinate frame)
"""

import datetime
import pathlib
import shutil

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
    """Run all preprocessing steps on Argus ETVision data and store in output_dir."""
    from . import _store_data, check_folders

    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, EyeTracker.Argus_ETVision)
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
                f"The folder {source_dir} is not recognized as a {EyeTracker.Argus_ETVision.value} recording."
            )

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()

    # copy the raw data to the output directory
    copy_et_vision_recording(source_dir, output_dir, rec_info, copy_scene_video)

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


def get_recording_info(input_dir: str | pathlib.Path) -> list[Recording] | None:
    """Return recording info for the given directory, or None if not a valid recording."""
    input_dir = pathlib.Path(input_dir)
    # recordings are identified as a tsv and an mkv file with the
    # same name
    rec_infos: list[Recording] = []
    for r in input_dir.glob("*.csv"):
        if not r.with_name(f"{r.stem}_Scene.wmv").is_file():
            continue
        rec_infos.append(Recording(source_directory=input_dir, eye_tracker=EyeTracker.Argus_ETVision))
        rec_infos[-1].name = r.stem
        # get more info from first line
        with pathlib.Path(r).open(encoding="utf-8") as f:
            line = f.readline()
        for s in line.strip().split(","):
            if "ETVision" in s:
                rec_infos[-1].firmware_version = s[len("ETVision: ") :].strip()
            elif "Start_Recording" in s:
                time_string = s[len("Start_Recording: ") :].strip()
                rec_infos[-1].start_time = timestamps.Timestamp(
                    int(datetime.datetime.fromisoformat(time_string).timestamp())
                )

    # should return None if no valid recordings found
    return rec_infos or None


def check_recording(input_dir: str | pathlib.Path, rec_info: Recording) -> bool:
    """Check that the expected recording files exist on disk."""
    for suff in (".csv", "_Scene.wmv"):
        file = f"{rec_info.name}{suff}"
        if not (input_dir / file).is_file():
            raise RuntimeError(f"Recording {rec_info.name} not found: {file} file not found in {input_dir}.")

    return True


def copy_et_vision_recording(
    input_dir: pathlib.Path, output_dir: pathlib.Path, rec_info: Recording, copy_scene_video: bool
) -> None:
    """Copy the relevant files from the specified input dir to the specified output dir."""
    # Copy relevant files to new directory
    src_file = input_dir / f"{rec_info.name}_Scene.wmv"

    if copy_scene_video:
        dest_file = output_dir / f"{naming.scene_camera_video_fname_stem}.wmv"
        shutil.copy2(src_file, dest_file)
    else:
        dest_file = None

    if dest_file:
        rec_info.scene_video_file = dest_file.name
    else:
        rec_info.scene_video_file = src_file.name


def format_gaze_data(input_dir: str | pathlib.Path, rec_info: Recording) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load gaze data csv file.

    Format to get the gaze coordinates w/r/t world camera, and timestamps for every frame of video.

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps

    """
    # convert the json file to pandas dataframe
    df = gaze2df(pathlib.Path(input_dir) / f"{rec_info.name}.csv")

    # read video file, create array of frame timestamps
    frame_timestamps = video_utils.get_frame_timestamps_from_video(rec_info.get_scene_video_path())

    # use the frame timestamps to assign a frame number to each data point
    frame_idx = video_utils.timestamps_to_frame_number(df.index, frame_timestamps["timestamp"].to_numpy())
    df.insert(0, "frame_idx", frame_idx["frame_idx"])

    # return the gaze data df and frame time stamps array
    return df, frame_timestamps


def gaze2df(gaze_file: str | pathlib.Path) -> pd.DataFrame:
    """Convert the .tsv file to a pandas dataframe"""
    df = pd.read_csv(gaze_file, index_col=False, skiprows=1)

    # rename and reorder columns
    # TODO: verg_gaze_coord_x verg_gaze_coord_y verg_gaze_coord_z left_eye_location_x right_eye_location_x left_eye_location_y right_eye_location_y left_eye_location_z right_eye_location_z left_gaze_dir_x right_gaze_dir_x left_gaze_dir_y right_gaze_dir_y left_gaze_dir_z right_gaze_dir_z
    lookup = {
        "start_of_record": "timestamp",
        "horz_gaze_coord": "gaze_pos_vid_x",
        "vert_gaze_coord": "gaze_pos_vid_y",
        "left_pupil_diam": "pup_diam_l",
        "right_pupil_diam": "pup_diam_r",
    }
    df = df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    df = df[idx]

    # convert timestamps from s to ms and set as index
    df.loc[:, "timestamp"] *= 1000.0
    df = df.set_index("timestamp")

    # return the dataframe
    return df
