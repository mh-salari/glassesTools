"""Copy data already in common format into a glassesTools recording.

Name of recording will be the name of the folder that is imported.
"""

import logging
import pathlib
import shutil

import pandas as pd

from .. import gaze_headref, naming, video_utils
from ..eyetracker import EyeTracker
from ..recording import Recording

logger = logging.getLogger(__name__)


def import_data(
    output_dir: str | pathlib.Path | None = None,
    source_dir: str | pathlib.Path | None = None,
    rec_info: Recording | None = None,
    device_name: str | None = None,
    copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
    cam_cal_file: str | pathlib.Path | None = None,
) -> Recording:
    """Import data already in the common format into a glassesTools recording.

    Unlike device-specific importers, this copies files that are already
    in the expected layout (``gaze_data.tsv``, ``worldCamera.mp4``, and
    optionally ``frame_timestamps.tsv`` and ``calibration.xml``). Missing
    frame timestamps or frame indices are generated automatically.

    Args:
        output_dir: Working directory where the imported recording will be placed.
        source_dir: Path to the source directory containing common-format files.
        rec_info: Optional pre-populated recording metadata.
        device_name: Custom device name for this Generic recording.
        copy_scene_video: Whether to copy the scene video to output_dir.
        source_dir_as_relative_path: Store source_dir as a relative path in rec_info.
        cam_cal_file: Path to an external camera calibration file.

    Returns:
        The populated Recording object written to output_dir.

    """
    from . import _store_data, check_folders

    output_dir, source_dir, rec_info, device_name = check_folders(
        output_dir, source_dir, rec_info, EyeTracker.Generic, device_name
    )
    print(f"processing: {source_dir.name} -> {output_dir}")

    # check and copy needed files to the output directory
    print("  Check and copy raw data...")
    # check recording and get export directory
    if rec_info is not None:
        check_recording(source_dir, rec_info, device_name)
    else:
        rec_info = get_recording_info(source_dir, device_name)
        if rec_info is None:
            raise RuntimeError(f"The folder {source_dir} is not recognized as a {EyeTracker.Generic.value} recording.")

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()

    # copy the raw data to the output directory
    src_vid, dest_vid, got_frame_ts, got_cal = copy_generic_recording(
        source_dir, output_dir, copy_scene_video, cam_cal_file
    )
    if dest_vid:
        rec_info.scene_video_file = dest_vid.name
    else:
        rec_info.scene_video_file = src_vid.name

    if not got_frame_ts:
        frame_timestamps = video_utils.get_frame_timestamps_from_video(rec_info.get_scene_video_path())
    else:
        frame_timestamps = None

    # check gaze data has a frame index column, and if not, make one
    gaze_data = pd.read_csv(output_dir / naming.gaze_data_fname, delimiter="\t")
    if "frame_idx" not in gaze_data.columns:
        print("    !! No frame index column found in gaze data, adding one based on timestamps...")
        if frame_timestamps is None:
            frame_timestamps = pd.read_csv(
                output_dir / naming.frame_timestamps_fname, delimiter="\t", index_col="frame_idx"
            )
        # make frame index column by matching timestamps
        frame_idx = video_utils.timestamps_to_frame_number(
            gaze_data.loc[:, "timestamp"].values, frame_timestamps["timestamp"].to_numpy()
        )
        gaze_data.insert(1, "frame_idx", frame_idx["frame_idx"].values)
        # store back
        gaze_data.to_csv(output_dir / naming.gaze_data_fname, sep="\t", index=False, na_rep="nan")

    if not got_cal:
        print("    !! No camera calibration provided!")

    if not rec_info.duration:
        # if duration not known, fill it
        # make a reasonable estimate of duration
        gaze = gaze_headref.read_dict_from_file(output_dir / naming.gaze_data_fname)[0]
        framets = frame_timestamps  # local copy to not accidentally trigger any overwriting in _store_data() below
        if framets is None:
            framets = pd.read_csv(output_dir / naming.frame_timestamps_fname, delimiter="\t", index_col="frame_idx")
        gt0 = gaze[min(gaze.keys())][0].timestamp_ori
        gte = gaze[max(gaze.keys())][-1].timestamp_ori
        rec_info.duration = float(round(max(gte - gt0, framets.timestamp.iloc[-1])))

    _store_data(output_dir, None, frame_timestamps, rec_info, source_dir_as_relative_path=source_dir_as_relative_path)

    return rec_info


def get_recording_info(input_dir: str | pathlib.Path, device_name: str | None = None) -> Recording | None:
    """Return recording info for a generic recording directory.

    If a recording info JSON exists, it is loaded and validated. Otherwise
    a new Recording is created from the directory name. The directory must
    contain ``worldCamera.mp4`` and ``gaze_data.tsv``.

    Args:
        input_dir: Path to the generic recording directory.
        device_name: Expected device name to match against the recording.

    Returns:
        A Recording object, or None if the directory is not a valid recording
        or doesn't match the expected device name.

    """
    input_dir = pathlib.Path(input_dir)

    rec_info_fname = input_dir / Recording.default_json_file_name
    if rec_info_fname.is_file():
        rec_info = Recording.load_from_json(rec_info_fname)
        if rec_info.eye_tracker != EyeTracker.Generic:
            logger.info('Not a "%s" eye tracker recording in %s', EyeTracker.Generic.value, input_dir)
            return None
        if rec_info.eye_tracker_name != device_name:
            logger.info('No recording for device "%s" found in %s', device_name, input_dir)
            return None
        # override input_dir to make sure its set correctly
        rec_info.source_directory = input_dir
    else:
        rec_info = Recording(source_directory=input_dir, eye_tracker=EyeTracker.Generic, eye_tracker_name=device_name)
        # get recording info
        rec_info.name = input_dir.name

    # check expected files are present
    for f in ("worldCamera.mp4", "gaze_data.tsv"):
        if not (input_dir / f).is_file():
            logger.info("Missing %s in %s — not a valid generic recording for %s", f, input_dir, device_name)
            return None

    return rec_info


def check_recording(input_dir: str | pathlib.Path, rec_info: Recording, device_name: str | None = None) -> None:
    """Validate that rec_info matches the actual recording on disk.

    Args:
        input_dir: Path to the generic recording directory.
        rec_info: Recording metadata to validate.
        device_name: Expected device name.

    Raises:
        ValueError: If the recording name or device name doesn't match.

    """
    actual_rec_info = get_recording_info(input_dir, device_name)

    if actual_rec_info is None or rec_info.name != actual_rec_info.name:
        raise ValueError(f'A recording with the name "{rec_info.name}" was not found in the folder {input_dir}.')

    # make sure caller did not mess with rec_info
    if rec_info.eye_tracker_name != actual_rec_info.eye_tracker_name:
        raise ValueError(
            f'A recording for a "{rec_info.eye_tracker_name}" device was not found in the folder {input_dir}.'
        )


def copy_generic_recording(
    input_dir: pathlib.Path, output_dir: pathlib.Path, copy_scene_video: bool, cam_cal_file: str | pathlib.Path | None
) -> tuple[pathlib.Path, pathlib.Path | None, bool, bool]:
    """Copy gaze data, video, frame timestamps, and calibration files to output dir.

    Copies all available common-format files from input_dir, using
    cam_cal_file if provided, otherwise falling back to ``calibration.xml``
    in the input directory.

    Args:
        input_dir: Source recording directory.
        output_dir: Destination directory.
        copy_scene_video: Whether to copy the scene video file.
        cam_cal_file: Optional path to an external camera calibration file.

    Returns:
        A tuple of (source video path, destination video path or None,
        whether frame timestamps were found, whether calibration was found).

    Raises:
        RuntimeError: If the gaze data or video file is not found.

    """
    gaze_file = input_dir / "gaze_data.tsv"
    if not gaze_file.is_file():
        raise RuntimeError(f"The {gaze_file} file is not found in the input directory {input_dir}")
    shutil.copy2(gaze_file, output_dir / naming.gaze_data_fname)

    vid_src_file = input_dir / "worldCamera.mp4"
    if not vid_src_file.is_file():
        raise RuntimeError(f"The {vid_src_file} file is not found in the input directory {input_dir}")
    if copy_scene_video:
        vid_dest_file = output_dir / f"{naming.scene_camera_video_fname_stem}.mp4"
        shutil.copy2(vid_src_file, vid_dest_file)
    else:
        vid_dest_file = None

    frame_ts_file = input_dir / "frame_timestamps.tsv"
    got_frame_ts = frame_ts_file.is_file()
    if got_frame_ts:
        shutil.copy2(frame_ts_file, output_dir / naming.frame_timestamps_fname)

    if cam_cal_file is not None:
        shutil.copy2(cam_cal_file, output_dir / naming.scene_camera_calibration_fname)
        got_cal = True
    else:
        cal_file = input_dir / "calibration.xml"
        got_cal = cal_file.is_file()
        if got_cal:
            shutil.copy2(cal_file, output_dir / naming.scene_camera_calibration_fname)

    return vid_src_file, vid_dest_file, got_frame_ts, got_cal
