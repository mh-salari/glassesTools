"""Cast raw SeeTrue data into common format.

The output directory will contain:
    - frame_timestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gaze_data.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera
"""

import pathlib
import shutil
from fractions import Fraction

import av
import cv2
import numpy as np
import pandas as pd

from .. import naming
from ..eyetracker import EyeTracker
from ..recording import Recording


def preprocess_data(
    output_dir: str | pathlib.Path,
    source_dir: str | pathlib.Path | None = None,
    rec_info: Recording | None = None,
    _copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
    cam_cal_file: str | pathlib.Path | None = None,
) -> Recording:
    """Run all preprocessing steps on SeeTrue STONE data and store in output_dir.

    SeeTrue recordings must be transcoded with ffmpeg (scene frames are
    individual JPEGs that need to be assembled into a video).

    Args:
        output_dir: Working directory where the imported recording will be placed.
        source_dir: Path to the raw recording. Falls back to ``rec_info.source_directory``.
        rec_info: Optional pre-populated recording metadata. If not provided,
            the first recording found in source_dir is used.
        _copy_scene_video: Ignored; SeeTrue scene video is always transcoded.
        source_dir_as_relative_path: Store source_dir as a relative path in rec_info.
        cam_cal_file: Path to an external camera calibration file. If None,
            hardcoded calibration values from SeeTrue are used.

    Returns:
        The populated Recording object written to output_dir.

    Raises:
        RuntimeError: If ffmpeg is not found on the system PATH, or if
            source_dir contains no valid SeeTrue recordings.
        ValueError: If rec_info specifies a recording not found in source_dir.

    """
    from . import _store_data, check_folders
    # NB: _copy_scene_video input argument is ignored, SeeTrue recordings must be transcoded with ffmpeg to be useful

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on path. ffmpeg is required for importing SeeTrue recordings. Cannot continue"
        )
    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, EyeTracker.SeeTrue_STONE)
    print(f"processing: {source_dir.name} -> {output_dir}")

    # check and copy needed files to the output directory
    print("  Check and copy raw data...")
    if rec_info is not None:
        if not check_recording(source_dir, rec_info):
            raise ValueError(f'A recording with the name "{rec_info.name}" was not found in the folder {source_dir}.')
    else:
        rec_infos = get_recording_info(source_dir)
        if rec_infos is None:
            raise RuntimeError(f"The folder {source_dir} does not contain SeeTrue STONE recordings.")
        rec_info = rec_infos[
            0
        ]  # take first, arbitrarily. If anything else wanted, user should call this function with a correct rec_info themselves

    # make output dirs
    if not output_dir.is_dir():
        output_dir.mkdir()

    # prep the data
    # NB: gaze data and scene video prep are intertwined, status messages are output inside this function
    gaze_df, frame_timestamps = copy_see_true_recording(source_dir, output_dir, rec_info)

    print("  Getting camera calibration...")
    if cam_cal_file is not None:
        shutil.copyfile(str(cam_cal_file), str(output_dir / naming.scene_camera_calibration_fname))
    else:
        print("    !! No camera calibration provided! Defaulting to hardcoded")
        get_camera_hardcoded(output_dir)

    _store_data(
        output_dir, gaze_df, frame_timestamps, rec_info, source_dir_as_relative_path=source_dir_as_relative_path
    )

    return rec_info


def get_recording_info(input_dir: str | pathlib.Path) -> list[Recording] | None:
    """Return recording info for all SeeTrue STONE recordings in input_dir.

    Recordings are identified by matching pairs of ``EyeData_*.csv`` files
    and ``ScenePics_*`` directories sharing the same sequence number.

    Args:
        input_dir: Path to the SeeTrue recording directory.

    Returns:
        A list of Recording objects, or None if no valid recordings are found.

    """
    input_dir = pathlib.Path(input_dir)
    rec_infos = []

    # NB: a SeeTrue directory may contain multiple recordings

    # get recordings. These are indicated by the sequence number in both EyeData.csv and ScenePics folder names
    for r in input_dir.glob("*.csv"):
        if not str(r.name).startswith("EyeData"):
            # print(f"file {r.name} not recognized as a recording (wrong name, should start with 'EyeData'), skipping")
            continue

        # get sequence number
        _, recording = r.stem.split("_")

        # check there is a matching scenevideo
        scene_vid_dir = r.parent / ("ScenePics_" + recording)
        if not scene_vid_dir.is_dir():
            # print(f"folder {scene_vid_dir} not found, meaning there is no scene video for this recording, skipping")
            continue

        rec_infos.append(Recording(source_directory=input_dir, eye_tracker=EyeTracker.SeeTrue_STONE))
        rec_infos[-1].participant = input_dir.name
        rec_infos[-1].name = recording

    # should return None if no valid recordings found
    return rec_infos or None


def check_recording(input_dir: str | pathlib.Path, rec_info: Recording) -> bool:
    """Check that the folder contains the required EyeData CSV and ScenePics directory.

    Args:
        input_dir: Path to the SeeTrue recording directory.
        rec_info: Recording metadata identifying which recording to check.

    Returns:
        True if both the gaze data CSV and scene frames directory exist.

    """
    # check we have an exported gaze data file
    file = f"EyeData_{rec_info.name}.csv"
    if not (input_dir / file).is_file():
        return False

    # check we have an exported scene video
    file = f"ScenePics_{rec_info.name}"
    return (input_dir / file).is_dir()


def copy_see_true_recording(
    input_dir: pathlib.Path, output_dir: pathlib.Path, rec_info: Recording
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble scene video from JPEG frames and process gaze data.

    Reads individual scene frames, fills gaps with black frames, assembles
    them into an MP4 video using PyAV with correct per-frame timing, and
    aligns gaze timestamps so that the first frame starts at time zero.

    Args:
        input_dir: Source recording directory containing ScenePics and EyeData.
        output_dir: Destination directory for the assembled video.
        rec_info: Recording metadata (updated with the scene video filename).

    Returns:
        A tuple of (gaze DataFrame indexed by timestamp, frame timestamps DataFrame).

    Raises:
        RuntimeError: If the scene video could not be created.

    """
    # get scene video dimensions by interrogating a frame in scene_vid_dir
    scene_vid_dir = input_dir / ("ScenePics_" + rec_info.name)
    frame = next(scene_vid_dir.glob("*.jpeg"))
    h, w, _ = cv2.imread(frame).shape

    # prep gaze data and get video frame timestamps from it
    print("  Prepping gaze data...")
    file = f"EyeData_{rec_info.name}.csv"
    gaze_df, frame_timestamps = format_gaze_data(input_dir / file, [w, h])

    # make scene video
    print("  Prepping scene video...")
    # 1. see if there are frames missing, insert black frames if so
    frames = []
    for f in scene_vid_dir.glob("*.jpeg"):
        _, fr = f.stem.split("_")
        frames.append(int(fr))
    frames = sorted(frames)

    # 2. see if framenumbers are as expected from the gaze data file
    # get average ifi
    ifi = np.mean(np.diff(frame_timestamps.index))
    # 2.1 remove frame timestamps that are before the first frame for which we have an image
    frame_timestamps = frame_timestamps.drop(frame_timestamps[frame_timestamps.frame_idx < frames[0]].index)
    # 2.2 remove frame timestamps that are beyond last frame for which we have an image
    frame_timestamps = frame_timestamps.drop(frame_timestamps[frame_timestamps.frame_idx > frames[-1]].index)
    # 2.3 add frame timestamps for images we have before first eye data
    if frames[0] < frame_timestamps.iloc[0, :].to_numpy()[0]:
        n_frames = frame_timestamps.iloc[0, :].to_numpy()[0] - frames[0]
        t0 = frame_timestamps.index[0]
        f0 = frame_timestamps.iloc[0, :].to_numpy()[0]
        for f in range(-1, -(n_frames + 1), -1):
            frame_timestamps.loc[t0 + f * ifi] = f0 + f
        frame_timestamps = frame_timestamps.sort_index()
    # 2.4 add frame timestamps for images we have after last eye data
    if frames[-1] > frame_timestamps.iloc[-1, :].to_numpy()[0]:
        n_frames = frames[-1] - frame_timestamps.iloc[-1, :].to_numpy()[0]
        t0 = frame_timestamps.index[-1]
        f0 = frame_timestamps.iloc[-1, :].to_numpy()[0]
        for f in range(1, n_frames + 1):
            frame_timestamps.loc[t0 + f * ifi] = f0 + f
        frame_timestamps = frame_timestamps.sort_index()
    # 2.5 check if holes, fill
    black_frames = []
    frame_delta = np.diff(frames)
    if np.any(frame_delta > 1):
        # frames images missing, add them (NB: if timestamps also missing, thats dealt with below)
        idx_gaps = np.argwhere(frame_delta > 1).flatten()  # idx_gaps is last idx before each gap
        fr_gaps = np.array(frames)[idx_gaps].flatten()
        n_frames = frame_delta[idx_gaps].flatten()
        for b, x in zip(fr_gaps + 1, n_frames, strict=True):
            for y in range(x - 1):
                black_frames.append(b + y)

        # make black frame
        black_im = np.zeros((h, w, 3), np.uint8)  # black image
        for f in black_frames:
            # store black frame to file
            cv2.imwrite(scene_vid_dir / f"frame_{f:d}.jpeg", black_im)
            frames.append(f)
        frames = sorted(frames)

    # 3. make into video
    # 3.1 find unique scene camera frames and their timestamps
    first_frame = frame_timestamps["frame_idx"].min()
    first_frame_ts = frame_timestamps["frame_idx"].idxmin()
    frame_timestamps = frame_timestamps.reset_index().groupby("frame_idx").first().reset_index()
    # 3.2 make concat filter input file
    concat_file = output_dir / "concat_input.txt"
    with pathlib.Path(concat_file).open("w", encoding="utf-8") as f:
        f.writelines("ffconcat version 1.0\n")
        fnames = (f"frame_{f}.jpeg" for f in frame_timestamps["frame_idx"].to_numpy())
        f.writelines(f"file '{scene_vid_dir / fn}'\n" for fn in fnames)

    # 3.3 determine frame pts and durations
    ifis = np.diff(frame_timestamps["timestamp"].to_numpy())
    ifis = np.append(ifis, [np.median(ifis)])
    durs = ifis / 1000
    pts_time = np.cumsum(np.append([0], durs))

    # 3.4 read frames through concat filter, output to mp4 with the right pts and dur
    out_file = output_dir / f"{naming.scene_camera_video_fname_stem}.mp4"
    ts = 900000
    with av.open(concat_file, "r", format="concat", options={"safe": "0"}) as inp:
        in_stream = inp.streams.video[0]
        with av.open(out_file, "w", format="mp4") as out:
            out_stream = out.add_stream("libx264")
            out_stream.width = (
                in_stream.codec_context.width
            )  # Set frame width to be the same as the width of the input stream
            out_stream.height = (
                in_stream.codec_context.height
            )  # Set frame height to be the same as the height of the input stream
            out_stream.pix_fmt = (
                in_stream.codec_context.pix_fmt
            )  # Copy pixel format from input stream to output stream
            out_stream.time_base = Fraction(1, ts)

            for frame_idx, frame in enumerate(inp.decode(in_stream)):
                frame.pts = np.round(pts_time[frame_idx] / out_stream.time_base)
                frame.dts = np.round(pts_time[frame_idx] / out_stream.time_base)
                frame.duration = np.round(durs[frame_idx] / out_stream.time_base)
                frame.time_base = out_stream.time_base

                packet = out_stream.encode(frame)
                out.mux(packet)

            # Flush the encoder
            packet = out_stream.encode(None)
            out.mux(packet)

    # 3.5 clean up
    # check for success
    concat_file.unlink(missing_ok=True)
    if out_file.is_file():
        rec_info.scene_video_file = out_file.name
    else:
        raise RuntimeError("Error making a scene video out of the SeeTrues frames")

    # delete the black frames we added, if any
    for f in black_frames:
        if (scene_vid_dir / f"frame_{f:d}.jpeg").is_file():
            (scene_vid_dir / f"frame_{f:d}.jpeg").unlink(missing_ok=True)

    # 4. fix up frame idxs and timestamps in gaze and video data
    # prep the gaze timestamps
    gaze_df.index -= first_frame_ts
    # overwrite the video frames, now we have one video frame per gaze sample
    gaze_df["frame_idx"] -= first_frame

    # Also fix video timestamps
    frame_timestamps = frame_timestamps.drop(columns=["frame_idx"])
    frame_timestamps.index.name = "frame_idx"
    frame_timestamps["timestamp"] -= first_frame_ts

    return gaze_df, frame_timestamps


def get_camera_hardcoded(output_dir: str | pathlib.Path) -> None:
    """Write hardcoded camera calibration to an OpenCV XML file.

    Uses calibration values provided by SeeTrue (640x480, f=495).

    Args:
        output_dir: Directory where the calibration XML file is written.

    """
    # turn into camera matrix and distortion coefficients as used by OpenCV
    camera = {}
    camera["cameraMatrix"] = np.array([[495, 0, 300], [0, 495, 255], [0, 0, 1]], dtype=np.float64)
    camera["distCoeff"] = np.array([-0.55, 0.4, 0, 0, -0.2])
    camera["resolution"] = np.array([640, 480])

    # store to file
    fs = cv2.FileStorage(output_dir / naming.scene_camera_calibration_fname, cv2.FILE_STORAGE_WRITE)
    for key, value in camera.items():
        fs.write(name=key, val=value)
    fs.release()


def format_gaze_data(
    input_file: str | pathlib.Path, scene_video_dimensions: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and process the SeeTrue gaze data CSV file.

    Parses the semicolon-delimited gaze data, converts normalized gaze
    coordinates to pixel coordinates, and extracts per-frame timestamps.

    Args:
        input_file: Path to the ``EyeData_*.csv`` file.
        scene_video_dimensions: Scene camera resolution ``[width, height]``.

    Returns:
        A tuple of (gaze DataFrame indexed by timestamp, frame timestamps DataFrame).

    """
    df = gazedata2df(input_file, scene_video_dimensions)

    # get time stamps for scene picture numbers
    frame_timestamps = pd.DataFrame(df["frame_idx"])

    return df, frame_timestamps


def gazedata2df(text_file: str | pathlib.Path, scene_video_dimensions: list[int]) -> pd.DataFrame:
    """Parse a SeeTrue EyeData CSV file into a pandas DataFrame.

    Reads the semicolon-delimited file, renames columns to the common naming
    scheme, converts pupil area (sq mm) to diameter (mm), and scales
    normalized gaze coordinates to pixel values.

    Args:
        text_file: Path to the ``EyeData_*.csv`` file.
        scene_video_dimensions: Scene camera resolution ``[width, height]``
            for converting normalized gaze coordinates to pixels.

    Returns:
        A DataFrame indexed by timestamp with gaze data columns.

    """
    df = pd.read_table(text_file, sep=";", index_col=False)
    df.columns = df.columns.str.strip()

    # rename and reorder columns
    lookup = {
        "Timestamp": "timestamp",
        "Scene picture number": "frame_idx",
        "Gazepoint X": "gaze_pos_vid_x",
        "Gazepoint Y": "gaze_pos_vid_y",
        "Pupil area (left), sq mm": "pup_diam_l",
        "Pupil area (right), sq mm": "pup_diam_r",
    }
    df = df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    df = df[idx]

    # pupil area to diameter
    df["pup_diam_l"] = 2 * np.sqrt(df["pup_diam_l"].to_numpy() / np.pi)
    df["pup_diam_r"] = 2 * np.sqrt(df["pup_diam_r"].to_numpy() / np.pi)

    # set timestamps as index
    df = df.set_index("timestamp")

    # turn gaze locations into pixel data with origin in top-left
    df["gaze_pos_vid_x"] *= scene_video_dimensions[0]
    df["gaze_pos_vid_y"] *= scene_video_dimensions[1]

    return df
