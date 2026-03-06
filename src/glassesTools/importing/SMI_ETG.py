"""Cast raw SMI ETG data into common format.

The output directory will contain:
    - frame_timestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gaze_data.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera,
                    and 3D coordinates in the coordinate frame of the glasses (may need rotation/translation
                    to represent in world camera's coordinate frame)
    - calibration.xml: info about the camera intrinsics and transform glasses coordinate system to
                       camera coordinate system
"""

import configparser
import math
import pathlib
import shutil
from io import StringIO

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .. import naming, video_utils
from ..eyetracker import EyeTracker
from ..recording import Recording


def preprocess_data(
    output_dir: str | pathlib.Path,
    source_dir: str | pathlib.Path | None = None,
    rec_info: Recording | None = None,
    copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
) -> Recording:
    """Run all preprocessing steps on SMI ETG data and store in output_dir."""
    from . import _store_data, check_folders

    # NB: copy_scene_video input argument might be ignored. If ffmpeg is present, it will be used to transcode the scene camera video
    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, EyeTracker.SMI_ETG)
    print(f"processing: {source_dir.name} -> {output_dir}")

    # check and copy needed files to the output directory
    print("  Check and copy raw data...")
    if rec_info is not None:
        if not check_recording(source_dir, rec_info, use_return=True):
            raise ValueError(
                f'A recording with the name "{rec_info.name}" was not found in the folder {source_dir}. Check that the name is correct and make sure that you export the scene video and gaze data using BeGaze as described in the glassesValidator manual.'
            )
    else:
        rec_infos = get_recording_info(source_dir)
        if rec_infos is None:
            raise RuntimeError(
                f"The folder {source_dir} does not contain SMI ETG recordings prepared for glassesValidator. If this is an SMI recording folder, you may not have run the required gaze data and scene video exports yet. See the glassesValidator manual for which exports you should perform with BeGaze first as well as the file naming scheme."
            )
        rec_info = rec_infos[
            0
        ]  # take first, arbitrarily. If anything else wanted, user should call this function with a correct rec_info themselves

    # make output dirs
    if not output_dir.is_dir():
        output_dir.mkdir()

    # copy the raw data to the output directory
    check_recording(source_dir, rec_info)
    copy_smi_recordings(source_dir, output_dir, rec_info, copy_scene_video)

    # prep the data
    print("  Getting camera calibration...")
    scene_video_dimensions = get_camera_from_file(source_dir, output_dir)
    print("  Prepping gaze data...")
    gaze_df, frame_timestamps = format_gaze_data(source_dir, rec_info, scene_video_dimensions)

    _store_data(
        output_dir, gaze_df, frame_timestamps, rec_info, source_dir_as_relative_path=source_dir_as_relative_path
    )

    return rec_info


def get_recording_info(input_dir: str | pathlib.Path) -> list[Recording] | None:
    """Return recording info for the given directory, or None if not a valid recording."""
    input_dir = pathlib.Path(input_dir)

    # NB: can be multiple recordings in an SMI folder

    # but first see this is a recording folder at all
    if not (input_dir / "codec1.bin").is_file():
        return None
    cam_info = read_smi_cam_info_file(input_dir)
    serial = cam_info.get("MiiSimulation", "DeviceSerialNumber")

    # get recordings. We expect the user to rename their exports to have the same format
    # as the other files in a project directory. So e.g., data exported from 001-2-recording.idf
    # for the corresponding 001-2-recording.avi, should be named 001-2-recording.txt. The
    # exported video should be called 001-2-export.avi
    rec_infos = []
    for r in input_dir.glob("*-export.avi"):
        rec_infos.append(Recording(source_directory=input_dir, eye_tracker=EyeTracker.SMI_ETG))
        rec_infos[-1].participant = input_dir.name
        rec_infos[-1].name = str(r.name)[: -len("-export.avi")]
        rec_infos[-1].glasses_serial = serial

    # should return None if no valid recordings found
    return rec_infos or None


def check_recording(input_dir: str | pathlib.Path, rec_info: str | pathlib.Path, use_return: bool = False) -> bool:
    """Check that the folder contains the required BeGaze exports."""
    # check we have an exported gaze data file
    file = rec_info.name + "-export.avi"
    if not (input_dir / file).is_file():
        if use_return:
            return False
        raise RuntimeError(
            f"{file} file not found in {input_dir}. Make sure you export the scene video using BeGaze as described in the glassesValidator manual."
        )

    # check we have an exported scene video
    file = rec_info.name + "-recording.txt"
    if not (input_dir / file).is_file():
        if use_return:
            return False
        raise RuntimeError(
            f"{file} file not found in {input_dir}. Make sure you export the gaze data using BeGaze as described in the glassesValidator manual."
        )

    return True


def copy_smi_recordings(
    input_dir: pathlib.Path, output_dir: pathlib.Path, rec_info: Recording, copy_scene_video: bool
) -> None:
    """Copy the relevant files from the specified input dir to the specified output dirs."""
    # Copy relevant files to new directory
    src_file = input_dir / f"{rec_info.name}-export.avi"

    if copy_scene_video:
        dest_file = output_dir / f"{naming.scene_camera_video_fname_stem}.avi"
        shutil.copy2(src_file, dest_file)
    else:
        dest_file = None

    if dest_file:
        rec_info.scene_video_file = dest_file.name
    else:
        rec_info.scene_video_file = src_file.name


def read_smi_cam_info_file(input_dir: str | pathlib.Path) -> configparser.ConfigParser:
    """Read and parse the SMI codec1.bin camera info file."""
    cam_info_str = pathlib.Path(input_dir / "codec1.bin").read_text(encoding="utf-8").replace("## ", "")

    cam_info = configparser.ConfigParser(converters={"nparray": lambda x: np.fromstring(x, sep="\t")})
    cam_info.read_string(cam_info_str)
    return cam_info


def get_camera_from_file(input_dir: str | pathlib.Path, output_dir: str | pathlib.Path) -> np.ndarray:
    """Read camera calibration from information file."""
    cam_info = read_smi_cam_info_file(input_dir)

    camera = {}
    camera["FOV"] = cam_info.getfloat("MiiSimulation", "SceneCamFOV")
    camera["resolution"] = np.array([
        cam_info.getint("MiiSimulation", "SceneCamWidth"),
        cam_info.getint("MiiSimulation", "SceneCamHeight"),
    ])
    camera["sensorOffsets"] = np.array([
        cam_info.getfloat("MiiSimulation", "SceneCamSensorOffsetX"),
        cam_info.getfloat("MiiSimulation", "SceneCamSensorOffsetY"),
    ])

    camera["radialDistortion"] = cam_info.getnparray("MiiSimulation", "SceneCamRadialDistortion")
    camera["tangentialDistortion"] = cam_info.getnparray("MiiSimulation", "SceneCamTangentialDistortion")

    camera["position"] = cam_info.getnparray("MiiSimulation", "SceneCamPos")
    camera["eulerAngles"] = np.array([
        cam_info.getfloat("MiiSimulation", "SceneCamOrX"),
        cam_info.getfloat("MiiSimulation", "SceneCamOrY"),
        cam_info.getfloat("MiiSimulation", "SceneCamOrZ"),
    ])

    # now turn these fields into focal length and principal point
    # 1. FOV is horizontal FOV of camera, given resolution we can compute
    # focal length. We assume vertical focal length is the same, best we can do
    fl = camera["resolution"][0] / (2 * math.tan(camera["FOV"] / 2 / 180 * math.pi))
    camera["focalLength"] = np.array([fl, fl])
    # 2. sensor offsets seem to be relative to center of sensor
    camera["principalPoint"] = camera["resolution"] / 2.0 + camera["sensorOffsets"]
    # 3. turn euler angles into rotation matrix (180-Rz because poster space has positive X rightward and positive Y downward)
    camera["rotation"] = Rotation.from_euler(
        "XYZ", [camera["eulerAngles"][0], camera["eulerAngles"][1], 180 - camera["eulerAngles"][2]], degrees=True
    ).as_matrix()

    # turn into camera matrix and distortion coefficients as used by OpenCV
    camera["cameraMatrix"] = np.identity(3)
    camera["cameraMatrix"][0, 0] = camera["focalLength"][0]
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
    input_dir: str | pathlib.Path, rec_info: Recording, scene_video_dimensions: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load gazedata file.

    Format to get the gaze coordinates w.r.t. world camera, and timestamps for
    every frame of video.

    Returns:
        - formatted dataframe with cols for timestamp, frame_idx, and gaze data
        - np array of frame timestamps

    """
    # convert the text file to pandas dataframe
    file = rec_info.name + "-recording.txt"
    df = gazedata2df(input_dir / file, scene_video_dimensions)

    # read video file, create array of frame timestamps
    frame_timestamps = video_utils.get_frame_timestamps_from_video(rec_info.get_scene_video_path())

    # SMI frame counter seems to be of the format HH:MM:SS:FR, where HH:MM:SS is a normal
    # hour, minute, second timecode, and FR is a frame number for within that second. The
    # frame number is zero-based, and runs ranges from 0 to FS-1 where FS is the frame
    # rate of the camera (e.g. 24 Hz). Convert their frame counter to a normal counter
    df_fr = pd.DataFrame(
        df["Frame"].apply(lambda x: [int(y) for y in x.split(":")]).to_list(),
        columns=["hour", "minute", "second", "frame"],
    )
    frame_rate = df_fr.frame.max() + 1
    df.insert(
        0, "frame_idx", (((df_fr.hour * 60 + df_fr.minute) * 60 + df_fr.second) * frame_rate + df_fr.frame).to_numpy()
    )
    # NB: seems we can get frame numbers in the data which are beyond the length of the video. so be it
    df = df.drop(columns=["Frame"])
    # NB: it seems the SMI export doesn't strictly follow their own timecode, but uses the first
    # gaze data point for the first frame of the video, which is then also timestamped with the timecode
    # of that first frame. Subtracting min so that the frame_idx starts at 0 for the data also empirically
    # seems to line up with the SMI export (most of the time, sadly seems to vary a little between videos)
    df.frame_idx -= df.frame_idx.min()

    # return the gaze data df and frame time stamps array
    return df, frame_timestamps


def gazedata2df(text_file: str | pathlib.Path, _scene_video_dimensions: list[int]) -> pd.DataFrame:
    """Convert the gazedata file to a pandas dataframe."""
    text_data = pathlib.Path(text_file).read_text(encoding="utf-8")

    df = pd.read_table(StringIO(text_data), comment="#", index_col=False)

    # prepare data frame
    # remove unneeded columns
    df = df.drop(columns=["Type", "Trial", "Aux1"], errors="ignore")  # drop these columns if they exist
    # if we have monocular point of regard data, check if its not identical to binocular
    if "L POR X [px]" in df.columns:
        x_equal = np.all(np.logical_or(df["L POR X [px]"] == df["B POR X [px]"], np.isnan(df["L POR X [px]"])))
        y_equal = np.all(np.logical_or(df["L POR Y [px]"] == df["B POR Y [px]"], np.isnan(df["L POR Y [px]"])))
        if x_equal and y_equal:
            df = df.drop(columns=["L POR X [px]", "L POR Y [px]"])
    if "R POR X [px]" in df.columns:
        x_equal = np.all(np.logical_or(df["R POR X [px]"] == df["B POR X [px]"], np.isnan(df["R POR X [px]"])))
        y_equal = np.all(np.logical_or(df["R POR Y [px]"] == df["B POR Y [px]"], np.isnan(df["R POR Y [px]"])))
        if x_equal and y_equal:
            df = df.drop(columns=["R POR X [px]", "R POR Y [px]"])

    # rename and reorder columns
    lookup = {
        "Time": "timestamp",
        "L EPOS X": "gaze_ori_l_x",
        "L EPOS Y": "gaze_ori_l_y",
        "L EPOS Z": "gaze_ori_l_z",
        "L Pupil Diameter [mm]": "pup_diam_l",
        "L GVEC X": "gaze_dir_l_x",
        "L GVEC Y": "gaze_dir_l_y",
        "L GVEC Z": "gaze_dir_l_z",
        "R EPOS X": "gaze_ori_r_x",
        "R EPOS Y": "gaze_ori_r_y",
        "R EPOS Z": "gaze_ori_r_z",
        "R Pupil Diameter [mm]": "pup_diam_r",
        "R GVEC X": "gaze_dir_r_x",
        "R GVEC Y": "gaze_dir_r_y",
        "R GVEC Z": "gaze_dir_r_z",
        "B POR X [px]": "gaze_pos_vid_x",
        "B POR Y [px]": "gaze_pos_vid_y",
        "L Dia X [px]": "pup_diam_l_px_x",
        "L Dia Y [px]": "pup_diam_l_px_y",
        "R Dia X [px]": "pup_diam_r_px_x",
        "R Dia Y [px]": "pup_diam_r_px_y",
    }
    df = df.rename(columns=lookup)
    # reorder
    idx = [lookup[k] for k in lookup if lookup[k] in df.columns]
    idx.extend([x for x in df.columns if x not in idx])  # append columns not in lookup
    df = df[idx]

    # convert timestamps from us to ms and set as index
    df = df.astype({"timestamp": "float"})
    df["timestamp"] /= 1000.0
    # NB: SMI gaze timestamps are arbitrary, and not expressed in scene video time. I have experimented
    # with various methods for aligning them, but this does not appear to be possible. Either the scene
    # video or the gaze timestamps are fake. As example, I have a recording with a 24 Hz scene camera and
    # a 60 Hz eye movement signal. Timestamps of both the video and the eye movement are nicely very
    # close to 24 Hz and 60 Hz, yet SMI's frame column (see comments in format_gaze_data above) for this
    # recording sometimes has frames with only one sample associated, and other frames with five samples.
    # This cannot be rescued. So, since the timestamp is arbitrary anyway, subtract so that the first
    # time is zero. Users are strongly recommended to use gazeMapper's VOR sync to better align SMI eye
    # tracker data with the scene video.
    df["timestamp"] -= df["timestamp"].iloc[0]
    df = df.set_index("timestamp")

    # return the dataframe
    return df
