"""Cast Meta Aria Gen 1 data into common format."""

import json
import pathlib
import shutil

import pandas as pd

from .. import naming, timestamps, video_utils
from ..eyetracker import EyeTracker
from ..recording import Recording


def import_data(
    output_dir: str | pathlib.Path | None = None,
    source_dir: str | pathlib.Path | None = None,
    rec_info: Recording | None = None,
    copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
) -> Recording:
    """Run all preprocessing steps on Meta Aria Gen 1 data and store in output_dir."""
    from . import _store_data, check_folders

    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, EyeTracker.Meta_Aria_Gen_1)
    print(f"processing: {source_dir.name} -> {output_dir}")

    # check and copy needed files to the output directory
    print("  Check and copy raw data...")
    # check recording and get export directory
    if rec_info is not None:
        check_recording(source_dir, rec_info)
    else:
        rec_info = get_recording_info(source_dir)
        if rec_info is None:
            raise RuntimeError(
                f"The folder {source_dir} is not recognized as a {EyeTracker.Meta_Aria_Gen_1.value} recording."
            )

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()

    # copy the raw data to the output directory
    src_vid, dest_vid, gaze_data, frame_timestamps = copy_recording(source_dir, output_dir, rec_info, copy_scene_video)
    if dest_vid:
        rec_info.scene_video_file = dest_vid.name
    else:
        rec_info.scene_video_file = src_vid.name

    _store_data(
        output_dir, gaze_data, frame_timestamps, rec_info, source_dir_as_relative_path=source_dir_as_relative_path
    )

    return rec_info


def get_recording_info(input_dir: str | pathlib.Path) -> Recording | None:
    """Return recording info for the given directory, or None if not a valid recording."""
    input_dir = pathlib.Path(input_dir)
    rec_info = Recording(source_directory=input_dir, eye_tracker=EyeTracker.Meta_Aria_Gen_1)

    # check expected files are present
    for f in ("metadata.json", "worldCamera.mp4", "gaze.tsv", "calibration.xml"):
        if not (input_dir / f).is_file():
            print(
                f"This directory does not contain a valid {EyeTracker.Meta_Aria_Gen_1.value} recording export. The {f} file is not found in the input directory {input_dir}. Make sure you run the meta_aria_gen1_exporter.py script on the recording's vrs file"
            )
            return None

    with pathlib.Path(input_dir / "metadata.json").open(encoding="utf-8") as f:
        metadata = json.load(f)
    rec_info.name = metadata["name"]
    rec_info.scene_camera_serial = metadata["scene_camera_serial"]
    rec_info.duration = float(metadata["duration"] / 1000)  # in us, convert to ms
    rec_info.glasses_serial = metadata["glasses_serial"]
    rec_info.start_time = timestamps.Timestamp(metadata["start_time"])
    rec_info.scene_video_file = "worldCamera.mp4"

    # we got a valid recording
    # return what we've got
    return rec_info


def check_recording(input_dir: str | pathlib.Path, rec_info: Recording) -> None:
    """Validate that rec_info matches the actual recording on disk."""
    actual_rec_info = get_recording_info(input_dir)

    if actual_rec_info is None or rec_info.name != actual_rec_info.name:
        raise ValueError(f'A recording with the name "{rec_info.name}" was not found in the folder {input_dir}.')

    # make sure caller did not mess with rec_info
    if rec_info.eye_tracker != actual_rec_info.eye_tracker:
        raise ValueError(
            f'A recording for a "{rec_info.eye_tracker.value}" device was not found in the folder {input_dir}.'
        )


def copy_recording(
    input_dir: pathlib.Path, output_dir: pathlib.Path, rec_info: Recording, copy_scene_video: bool
) -> tuple[pathlib.Path, pathlib.Path | None, pd.DataFrame, pd.DataFrame]:
    """Copy recording files to output dir and return video paths, gaze data, and frame timestamps."""
    gaze_file = input_dir / "gaze.tsv"
    if not gaze_file.is_file():
        raise RuntimeError(f"The {gaze_file} file is not found in the input directory {input_dir}")
    gaze_data = pd.read_csv(gaze_file, sep="\t", index_col="timestamp")

    vid_src_file = input_dir / "worldCamera.mp4"
    if not vid_src_file.is_file():
        raise RuntimeError(f"The {vid_src_file} file is not found in the input directory {input_dir}")
    if copy_scene_video:
        vid_dest_file = output_dir / f"{naming.scene_camera_video_fname_stem}.mp4"
        shutil.copy2(vid_src_file, vid_dest_file)
    else:
        vid_dest_file = None

    # get video timestamps
    frame_timestamps = video_utils.get_frame_timestamps_from_video(rec_info.get_scene_video_path())

    # convert gaze timestamps from us to ms
    gaze_data = gaze_data.reset_index()
    gaze_data.loc[:, "timestamp"] /= 1000.0
    # add frame_idx for each gaze sample
    frame_idx = video_utils.timestamps_to_frame_number(
        gaze_data.loc[:, "timestamp"].values, frame_timestamps["timestamp"].values
    )
    gaze_data.insert(1, "frame_idx", frame_idx["frame_idx"].values)
    gaze_data = gaze_data.set_index("timestamp")

    cal_file = input_dir / "calibration.xml"
    if not cal_file.is_file():
        raise RuntimeError(f"The {cal_file} file is not found in the input directory {input_dir}")
    shutil.copy2(cal_file, output_dir / naming.scene_camera_calibration_fname)

    return vid_src_file, vid_dest_file, gaze_data, frame_timestamps
