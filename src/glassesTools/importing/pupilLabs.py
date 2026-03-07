"""Cast raw pupil labs (Core, Invisible and Neon) data into common format.

The output directory will contain:
    - frame_timestamps.tsv: frame number and corresponding timestamps for each frame in video
    - worldCamera.mp4: the video from the point-of-view scene camera on the glasses
    - gaze_data.tsv: gaze data, where all 2D gaze coordinates are represented w/r/t the world camera,
                    and 3D coordinates in the coordinate frame of the glasses (may need rotation/translation
                    to represent in world camera's coordinate frame)
    - calibration.xml: info about the camera intrinsics
"""

import json
import pathlib
import re
import shutil
import typing
from urllib.request import urlopen

import cv2
import msgpack
import numpy as np
import pandas as pd

from .. import naming, timestamps, video_utils
from ..eyetracker import EyeTracker
from ..recording import Recording


def preprocess_data(
    output_dir: str | pathlib.Path,
    device: str | EyeTracker | None = None,
    source_dir: str | pathlib.Path | None = None,
    rec_info: Recording | None = None,
    copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
) -> Recording:
    """Run all preprocessing steps on Pupil Labs data and store in output_dir.

    Handles Pupil Core, Pupil Invisible, and Pupil Neon recordings, as well
    as Pupil Cloud exports. Determines the export type, copies the scene
    video, extracts camera calibration, and formats gaze data.

    Args:
        output_dir: Working directory where the imported recording will be placed.
        device: Eye tracker device type (Pupil Core, Invisible, or Neon).
        source_dir: Path to the raw recording. Falls back to ``rec_info.source_directory``.
        rec_info: Optional pre-populated recording metadata.
        copy_scene_video: Whether to copy the scene video to output_dir.
        source_dir_as_relative_path: Store source_dir as a relative path in rec_info.

    Returns:
        The populated Recording object written to output_dir.

    Raises:
        ValueError: If the device is not a supported Pupil Labs eye tracker.

    """
    from . import _store_data, check_device, check_folders

    device, rec_info, _ = check_device(device, rec_info)
    if device not in {EyeTracker.Pupil_Core, EyeTracker.Pupil_Invisible, EyeTracker.Pupil_Neon}:
        raise ValueError(
            f"Provided device ({device.value}) is not a {EyeTracker.Pupil_Core.value}, a {EyeTracker.Pupil_Invisible.value} or a {EyeTracker.Pupil_Neon.value}."
        )
    output_dir, source_dir, rec_info, _ = check_folders(output_dir, source_dir, rec_info, device)
    print(f"processing: {source_dir.name} -> {output_dir}")

    # check and copy needed files to the output directory
    print("  Check and copy raw data...")
    # check pupil recording and get export directory
    export_file, is_cloud_export = check_pupil_recording(source_dir)
    if rec_info is not None:
        check_recording(source_dir, rec_info)
    else:
        rec_info = get_recording_info(source_dir, device)
        if rec_info is None:
            raise RuntimeError(f"The folder {source_dir} is not recognized as a {device.value} recording.")

    # make output dir
    if not output_dir.is_dir():
        output_dir.mkdir()

    # find world video and copy if wanted
    if is_cloud_export:
        scene_vid = list(source_dir.glob("*.mp4"))
        if len(scene_vid) != 1:
            raise RuntimeError(
                f"Scene video missing or more than one found for Pupil Cloud export in folder {source_dir}"
            )
        src_vid = scene_vid[0]
    else:
        src_vid = source_dir / "world.mp4"
    if copy_scene_video:
        dest_vid = output_dir / f"{naming.scene_camera_video_fname_stem}.mp4"
        shutil.copy2(str(src_vid), str(dest_vid))
        rec_info.scene_video_file = dest_vid.name
    else:
        rec_info.scene_video_file = src_vid.name
    print(f"  Input data copied to: {output_dir}")

    # get camera cal
    print("  Getting camera calibration...")
    if is_cloud_export:
        scene_video_dimensions = get_camera_cal_from_cloud_export(source_dir, output_dir, rec_info)
    else:
        match rec_info.eye_tracker:
            case EyeTracker.Pupil_Core:
                scene_video_dimensions = get_camera_from_msg_pack(source_dir, output_dir)
            case EyeTracker.Pupil_Invisible | EyeTracker.Pupil_Neon:
                if (source_dir / "calibration.bin").is_file():
                    scene_video_dimensions = get_camera_cal_from_bin_file(source_dir, output_dir, rec_info)
                else:
                    scene_video_dimensions = get_camera_cal_from_online(source_dir, output_dir, rec_info)

    # get gaze data and video frame timestamps
    print("  Prepping gaze data...")
    if is_cloud_export:
        gaze_df, frame_timestamps = format_gaze_data_cloud_export(source_dir, export_file)
    else:
        # Pupil player or Neon player export
        gaze_df, frame_timestamps = format_gaze_data_pupil_player(
            source_dir, export_file, scene_video_dimensions, rec_info
        )

    _store_data(
        output_dir, gaze_df, frame_timestamps, rec_info, source_dir_as_relative_path=source_dir_as_relative_path
    )

    return rec_info


def check_pupil_recording(input_dir: str | pathlib.Path) -> tuple[pathlib.Path, bool]:
    """Verify the recording folder has been exported and locate the gaze data file.

    Checks for either a Pupil/Neon Player export (``info.player.json`` +
    ``exports/`` folder with ``gaze_positions*.csv``) or a Pupil Cloud
    export (``info.json`` + ``gaze.csv``).

    Args:
        input_dir: Path to the recording directory.

    Returns:
        A tuple of (path to the gaze data file, whether this is a Cloud export).

    Raises:
        RuntimeError: If the folder has no valid export.

    """
    # check we have an info.player.json file
    if not (input_dir / "info.player.json").is_file():
        # possibly a pupil cloud export
        if not (input_dir / "info.json").is_file() or not (input_dir / "gaze.csv").is_file():
            raise RuntimeError(
                f"Neither the info.player.json file nor the info.json and gaze.csv files are found for {input_dir}. Either export from Pupil Cloud or, if the folder contains raw sensor data, open the recording in Pupil Player/Neon Player and run an export before importing."
            )
        return input_dir / "gaze.csv", True

    # check we have an export in the input dir
    input_exp_dir = input_dir / "exports"
    if not input_exp_dir.is_dir():
        raise RuntimeError(
            f"no exports folder for {input_dir}. Perform a recording export in Pupil Player/Neon Player before importing."
        )

    # get latest export in that folder that contain a gaze position file
    gp_files = sorted(input_exp_dir.rglob("*gaze_positions*.csv"))
    if not gp_files:
        raise RuntimeError(
            f"There are no exports in the folder {input_exp_dir}. Perform a recording export in Pupil Player/Neon Player before importing."
        )

    return gp_files[-1], False


def get_recording_info(input_dir: str | pathlib.Path, device: EyeTracker) -> Recording | None:
    """Return recording info for the given directory, or None if not valid.

    Reads device-specific JSON files to extract recording name, duration,
    start time, serial numbers, and participant info. Supports both
    Pupil/Neon Player exports and Pupil Cloud exports.

    Args:
        input_dir: Path to the recording directory.
        device: Expected eye tracker device type.

    Returns:
        A Recording object, or None if the directory doesn't contain a
        valid recording for the specified device.

    """
    input_dir = pathlib.Path(input_dir)
    rec_info = Recording(source_directory=input_dir, eye_tracker=device)

    if (input_dir / "info.player.json").is_file():
        # Pupil player export
        match device:
            case EyeTracker.Pupil_Core:
                # check this is not an invisible recording
                file = input_dir / "info.invisible.json"
                if file.is_file():
                    return None

                file = input_dir / "info.player.json"
                if not file.is_file():
                    return None
                with pathlib.Path(file).open(encoding="utf-8") as j:
                    i_info = json.load(j)
                rec_info.name = i_info["recording_name"]
                rec_info.start_time = timestamps.Timestamp(
                    int(i_info["start_time_system_s"])
                )  # UTC in seconds, keep second part
                rec_info.duration = float(i_info["duration_s"] * 1000)  # in seconds, convert to ms
                rec_info.recording_software_version = i_info["recording_software_version"]

                # get user name, if any
                user_info_file = input_dir / "user_info.csv"
                if user_info_file.is_file():
                    df = pd.read_csv(user_info_file)
                    name_row = df["key"].str.contains("name")
                    if any(name_row) and not pd.isna(df[name_row].value).to_numpy()[0]:
                        rec_info.participant = df.loc[name_row, "value"].to_numpy()[0]

            case EyeTracker.Pupil_Invisible:
                file = input_dir / "info.invisible.json"
                if not file.is_file():
                    return None
                with pathlib.Path(file).open(encoding="utf-8") as j:
                    i_info = json.load(j)
                rec_info.name = i_info["template_data"]["recording_name"]
                rec_info.recording_software_version = i_info["app_version"]
                rec_info.start_time = timestamps.Timestamp(
                    int(i_info["start_time"] // 1000000000)
                )  # UTC in nanoseconds, keep second part
                rec_info.duration = float(i_info["duration"] / 1000000)  # in nanoseconds, convert to ms
                rec_info.glasses_serial = i_info["glasses_serial_number"]
                rec_info.recording_unit_serial = i_info["android_device_id"]
                rec_info.scene_camera_serial = i_info["scene_camera_serial_number"]
                # get participant name
                file = input_dir / "wearer.json"
                if file.is_file():
                    wearer_id = i_info["wearer_id"]
                    with pathlib.Path(file).open(encoding="utf-8") as j:
                        i_info = json.load(j)
                    if wearer_id == i_info["uuid"]:
                        rec_info.participant = i_info["name"]

            case EyeTracker.Pupil_Neon:
                file = input_dir / "info.neon.json"
                if not file.is_file():
                    return None
                with pathlib.Path(file).open(encoding="utf-8") as j:
                    i_info = json.load(j)
                rec_info.name = i_info["template_data"]["recording_name"]
                rec_info.recording_software_version = i_info["app_version"]
                rec_info.start_time = timestamps.Timestamp(
                    int(i_info["start_time"] // 1000000000)
                )  # UTC in nanoseconds, keep second part
                rec_info.duration = float(i_info["duration"] / 1000000)  # in nanoseconds, convert to ms
                rec_info.glasses_serial = i_info["module_serial_number"]
                rec_info.recording_unit_serial = i_info["android_device_id"]
                # get participant name
                file = input_dir / "wearer.json"
                if file.is_file():
                    wearer_id = i_info["wearer_id"]
                    with pathlib.Path(file).open(encoding="utf-8") as j:
                        i_info = json.load(j)
                    if wearer_id == i_info["uuid"]:
                        rec_info.participant = i_info["name"]

            case _:
                print(f"Device {device} unknown")
                return None
    else:
        # pupil cloud export, for either Pupil Invisible or Pupil Neon
        if device == EyeTracker.Pupil_Core:
            return None

        # raw sensor data also contain an info.json (checked below), so checking
        # that file is not enough to see if this is a Cloud Export. Check gaze.csv
        # presence
        if not (input_dir / "gaze.csv").is_file():
            return None

        file = input_dir / "info.json"
        if not file.is_file():
            return None
        with pathlib.Path(file).open(encoding="utf-8") as j:
            i_info = json.load(j)

        # check this is for the expected device
        is_neon = "Neon" in i_info["android_device_name"] or "frame_name" in i_info
        if (device == EyeTracker.Pupil_Invisible and is_neon) or (device == EyeTracker.Pupil_Neon and not is_neon):
            return None

        rec_info.name = i_info["template_data"]["recording_name"]
        rec_info.recording_software_version = i_info["app_version"]
        rec_info.start_time = timestamps.Timestamp(
            int(i_info["start_time"] // 1000000000)
        )  # UTC in nanoseconds, keep second part
        rec_info.duration = float(i_info["duration"] / 1000000)  # in nanoseconds, convert to ms
        if is_neon:
            rec_info.glasses_serial = i_info["module_serial_number"]
        else:
            rec_info.glasses_serial = i_info["glasses_serial_number"]
            rec_info.scene_camera_serial = i_info["scene_camera_serial_number"]
        rec_info.recording_unit_serial = i_info["android_device_id"]
        if is_neon:
            rec_info.firmware_version = (
                f"{i_info['pipeline_version']} ({i_info['firmware_version'][0]}.{i_info['firmware_version'][1]})"
            )
        else:
            rec_info.firmware_version = i_info["pipeline_version"]
        rec_info.participant = i_info["wearer_name"]

    return rec_info


def check_recording(input_dir: str | pathlib.Path, rec_info: Recording) -> None:
    """Validate that rec_info matches the actual recording on disk.

    Re-reads recording info from the directory and compares name, duration,
    start time, software version, and device-specific serial numbers.

    Args:
        input_dir: Path to the recording directory.
        rec_info: Recording metadata to validate.

    Raises:
        ValueError: If any field in rec_info doesn't match the actual recording.

    """
    actual_rec_info = get_recording_info(input_dir, rec_info.eye_tracker)
    if actual_rec_info is None or rec_info.name != actual_rec_info.name:
        raise ValueError(f'A recording with the name "{rec_info.name}" was not found in the folder {input_dir}.')

    # make sure caller did not mess with rec_info
    if rec_info.duration != actual_rec_info.duration:
        raise ValueError(
            f'A recording with the duration "{rec_info.duration}" was not found in the folder {input_dir}.'
        )
    if rec_info.start_time.value != actual_rec_info.start_time.value:
        raise ValueError(
            f'A recording with the start_time "{rec_info.start_time.display}" was not found in the folder {input_dir}.'
        )
    if rec_info.recording_software_version != actual_rec_info.recording_software_version:
        raise ValueError(
            f'A recording with the recording_software_version "{rec_info.recording_software_version}" was not found in the folder {input_dir}.'
        )

    # for invisible and neon recordings we have a bit more info
    if rec_info.eye_tracker in {EyeTracker.Pupil_Invisible, EyeTracker.Pupil_Neon}:
        if rec_info.glasses_serial != actual_rec_info.glasses_serial:
            raise ValueError(
                f'A recording with the glasses_serial "{rec_info.glasses_serial}" was not found in the folder {input_dir}.'
            )
        if rec_info.recording_unit_serial != actual_rec_info.recording_unit_serial:
            raise ValueError(
                f'A recording with the recording_unit_serial "{rec_info.recording_unit_serial}" was not found in the folder {input_dir}.'
            )
        if (
            rec_info.eye_tracker == EyeTracker.Pupil_Invisible
            and rec_info.scene_camera_serial != actual_rec_info.scene_camera_serial
        ):
            raise ValueError(
                f'A recording with the scene_camera_serial "{rec_info.scene_camera_serial}" was not found in the folder {input_dir}.'
            )
        if (
            rec_info.participant is not None or actual_rec_info.participant is not None
        ) and rec_info.participant != actual_rec_info.participant:
            raise ValueError(
                f'A recording with the participant "{rec_info.participant}" was not found in the folder {input_dir}.'
            )


def get_camera_from_msg_pack(input_dir: str | pathlib.Path, output_dir: str | pathlib.Path) -> list[int]:
    """Read Pupil Core camera calibration from a msgpack intrinsics file.

    Reads ``world.intrinsics``, renames fields to OpenCV conventions,
    and writes the result as an XML calibration file.

    Args:
        input_dir: Source recording directory containing ``world.intrinsics``.
        output_dir: Destination directory for the calibration XML.

    Returns:
        The scene camera resolution as a numpy array ``[width, height]``.

    """
    cam_info = get_cam_info(input_dir / "world.intrinsics")

    # rename some fields, ensure they are numpy arrays
    cam_info["cameraMatrix"] = np.array(cam_info.pop("camera_matrix"))
    cam_info["distCoeff"] = np.array(cam_info.pop("dist_coefs")).flatten()
    cam_info["resolution"] = np.array(cam_info["resolution"])

    # store to file
    store_camera_calibration(cam_info, output_dir)

    return cam_info["resolution"]


def get_camera_cal_from_bin_file(
    input_dir: str | pathlib.Path, output_dir: str | pathlib.Path, rec_info: Recording
) -> list[int]:
    """Read camera calibration from the binary ``calibration.bin`` file.

    The binary format differs between Pupil Invisible (3x3 extrinsics)
    and Pupil Neon (4x4 extrinsics with additional eye camera data).

    Args:
        input_dir: Source recording directory containing ``calibration.bin``.
        output_dir: Destination directory for the calibration XML.
        rec_info: Recording metadata (used to determine device type and
            scene video path for resolution lookup).

    Returns:
        The scene camera resolution as a numpy array ``[width, height]``.

    """
    if rec_info.eye_tracker == EyeTracker.Pupil_Invisible:
        cal = np.fromfile(
            input_dir / "calibration.bin",
            np.dtype([
                ("serial", "5a"),
                ("scene_camera_matrix", "(3,3)d"),
                ("scene_distortion_coefficients", "8d"),
                ("scene_extrinsics_affine_matrix", "(3,3)d"),
            ]),
        )
        extrinsics_dim = (3, 3)
    elif rec_info.eye_tracker == EyeTracker.Pupil_Neon:
        cal = np.fromfile(
            input_dir / "calibration.bin",
            np.dtype([
                ("version", "u1"),
                ("serial", "6a"),
                ("scene_camera_matrix", "(3,3)d"),
                ("scene_distortion_coefficients", "8d"),
                ("scene_extrinsics_affine_matrix", "(4,4)d"),
                ("right_camera_matrix", "(3,3)d"),
                ("right_distortion_coefficients", "8d"),
                ("right_extrinsics_affine_matrix", "(4,4)d"),
                ("left_camera_matrix", "(3,3)d"),
                ("left_distortion_coefficients", "8d"),
                ("left_extrinsics_affine_matrix", "(4,4)d"),
                ("crc", "u4"),
            ]),
        )
        extrinsics_dim = (4, 4)

    cam_info = {}
    cam_info["serial_number"] = str(cal["serial"])
    cam_info["cameraMatrix"] = cal["scene_camera_matrix"].reshape((3, 3))
    cam_info["distCoeff"] = cal["scene_distortion_coefficients"].reshape((8, 1))
    cam_info["extrinsic"] = cal["scene_extrinsics_affine_matrix"].reshape(extrinsics_dim)

    # get resolution from the local intrinsics file or scene video
    cam_info["resolution"] = get_scene_camera_resolution(input_dir, rec_info)

    # store to xml file
    store_camera_calibration(cam_info, output_dir)

    return cam_info["resolution"]


def get_camera_cal_from_online(
    input_dir: str | pathlib.Path, output_dir: str | pathlib.Path, rec_info: Recording
) -> list[int]:
    """Download camera calibration from the Pupil Labs cloud API.

    Uses the device serial number to fetch calibration data from
    ``api.cloud.pupil-labs.com``. Falls back to this when no local
    ``calibration.bin`` is available.

    Args:
        input_dir: Source recording directory (for resolution lookup).
        output_dir: Destination directory for the calibration XML.
        rec_info: Recording metadata (provides serial number and device type).

    Returns:
        The scene camera resolution as a numpy array ``[width, height]``.

    Raises:
        RuntimeError: If the API request fails.

    """
    if rec_info.eye_tracker == EyeTracker.Pupil_Invisible:
        serial = rec_info.scene_camera_serial
    elif rec_info.eye_tracker == EyeTracker.Pupil_Neon:
        serial = rec_info.glasses_serial
    url = f"https://api.cloud.pupil-labs.com/v2/hardware/{serial}/calibration.v1?json"

    cam_info = json.loads(urlopen(url).read())
    if cam_info["status"] != "success":
        raise RuntimeError(f"Camera calibration could not be loaded, response: {cam_info['message']}")

    cam_info = cam_info["result"]

    # rename some fields, ensure they are numpy arrays
    cam_info["cameraMatrix"] = np.array(cam_info.pop("camera_matrix"))
    cam_info["distCoeff"] = np.array(cam_info.pop("dist_coefs")).flatten()
    cam_info["rotation"] = np.reshape(np.array(cam_info.pop("rotation_matrix")), (3, 3))

    # get resolution from the local intrinsics file or scene video
    cam_info["resolution"] = get_scene_camera_resolution(input_dir, rec_info)

    # store to xml file
    store_camera_calibration(cam_info, output_dir)

    return cam_info["resolution"]


def get_camera_cal_from_cloud_export(
    input_dir: str | pathlib.Path, output_dir: str | pathlib.Path, rec_info: Recording
) -> list[int] | None:
    """Read camera calibration from a Pupil Cloud export's ``scene_camera.json``.

    Handles both ``dist_coefs`` and ``distortion_coefficients`` key names
    for backward compatibility with different Cloud export versions.

    Args:
        input_dir: Source recording directory containing ``scene_camera.json``.
        output_dir: Destination directory for the calibration XML.
        rec_info: Recording metadata (for scene video resolution lookup).

    Returns:
        The scene camera resolution as a numpy array, or None if
        ``scene_camera.json`` is not found.

    """
    file = input_dir / "scene_camera.json"
    if not file.is_file():
        return None
    with pathlib.Path(file).open(encoding="utf-8") as j:
        cam_info = json.load(j)

    cam_info["cameraMatrix"] = np.array(cam_info.pop("camera_matrix"))
    if "dist_coefs" in cam_info:
        cam_info["distCoeff"] = np.array(cam_info.pop("dist_coefs")).flatten()
    else:
        cam_info["distCoeff"] = np.array(cam_info.pop("distortion_coefficients")).flatten()

    # get resolution from the scene video
    cam_info["resolution"] = get_scene_camera_resolution(input_dir, rec_info)

    # store to xml file
    store_camera_calibration(cam_info, output_dir)

    return cam_info["resolution"]


def store_camera_calibration(cam_info: dict[str, typing.Any], output_dir: str | pathlib.Path) -> None:
    """Write camera calibration dict to an OpenCV FileStorage XML file.

    Args:
        cam_info: Dictionary of calibration parameters (e.g., ``cameraMatrix``,
            ``distCoeff``, ``resolution``).
        output_dir: Directory where the calibration XML file is written.

    """
    fs = cv2.FileStorage(output_dir / naming.scene_camera_calibration_fname, cv2.FILE_STORAGE_WRITE)
    for key, value in cam_info.items():
        fs.write(name=key, val=value)
    fs.release()


def get_cam_info(cam_info_file: str | pathlib.Path) -> dict:
    """Read camera info from a msgpack intrinsics file.

    The file contains entries keyed by resolution strings like ``"(1280, 720)"``.
    Exactly one such entry must be present.

    Args:
        cam_info_file: Path to the ``.intrinsics`` msgpack file.

    Returns:
        The camera info dict for the single resolution entry found.

    Raises:
        RuntimeError: If zero or more than one resolution entry is found.

    """
    with pathlib.Path(cam_info_file).open("rb") as f:
        cam_info = msgpack.unpack(f)

    # get keys which denote a camera resolution
    rex = re.compile(r"^\(\d+, \d+\)$")

    keys = [k for k in cam_info if rex.match(k)]
    if len(keys) != 1:
        raise RuntimeError("No camera intrinsics or intrinsics for more than one camera found")
    return cam_info[keys[0]]


def get_scene_camera_resolution(input_dir: str | pathlib.Path, rec_info: Recording) -> np.ndarray:
    """Get scene camera resolution from intrinsics file or video.

    Tries ``world.intrinsics`` first; falls back to reading the scene
    video dimensions via OpenCV.

    Args:
        input_dir: Recording directory that may contain ``world.intrinsics``.
        rec_info: Recording metadata for locating the scene video.

    Returns:
        The resolution as a 2-element numpy array ``[width, height]``.

    Raises:
        RuntimeError: If the scene video cannot be opened.

    """
    if (input_dir / "world.intrinsics").is_file():
        return np.array(get_cam_info(input_dir / "world.intrinsics")["resolution"])

    cap = cv2.VideoCapture(rec_info.get_scene_video_path())
    if not cap.isOpened():
        raise RuntimeError(f"Could not open scene video to determine resolution: {rec_info.get_scene_video_path()}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return np.array([width, height])


def format_gaze_data_pupil_player(
    input_dir: str | pathlib.Path,
    export_file: str | pathlib.Path,
    scene_video_dimensions: list[int],
    rec_info: Recording,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Format gaze data from a Pupil Player or Neon Player export.

    Reads gaze positions, extracts frame timestamps from the scene video,
    corrects frame indices for any missing video frames using
    ``world_lookup.npy`` or ``world_timestamps.npy``, and aligns gaze
    timestamps to video time.

    Args:
        input_dir: Recording directory containing ``world_lookup.npy`` or
            ``world_timestamps.npy``.
        export_file: Path to the exported ``gaze_positions*.csv`` file.
        scene_video_dimensions: Scene camera resolution ``[width, height]``.
        rec_info: Recording metadata for locating the scene video.

    Returns:
        A tuple of (gaze DataFrame indexed by timestamp, frame timestamps
        DataFrame).

    """
    df = read_gaze_data_pupil_player(export_file, scene_video_dimensions, rec_info)

    # get timestamps for the scene video
    frame_ts = video_utils.get_frame_timestamps_from_video(rec_info.get_scene_video_path())

    # check pupil-labs' frame timestamps because we may need to correct
    # frame indices in case of holes in the video
    # also need this to correctly timestamp gaze samples
    if (input_dir / "world_lookup.npy").is_file():
        ft = pd.DataFrame(np.load(str(input_dir / "world_lookup.npy")))
        ft["frame_idx"] = ft.index
        ft.loc[ft["container_idx"] == -1, "container_frame_idx"] = -1
        # add to gaze data if gaze data doesn't have frame timestamps
        if "frame_idx" not in df:
            frame_idx = video_utils.timestamps_to_frame_number(
                df.loc[:, "timestamp"].values, ft["timestamp"].to_numpy() * 1000.0
            )
            df.insert(1, "frame_idx", frame_idx["frame_idx"].values)
        # check if some frame_idx needs to be adjusted for frames actually in the scene video
        needs_adjust = not ft["frame_idx"].equals(ft["container_frame_idx"])
        # prep for later clean up
        to_drop = [x for x in ft.columns if x not in {"frame_idx", "timestamp"}]
        # do further adjustment that may be needed
        if needs_adjust:
            # not all video frames were encoded into the video file. Need to adjust
            # frame_idx in the gaze data to match actual video file
            temp = df.merge(ft, on="frame_idx")
            temp["frame_idx"] = temp["container_frame_idx"]
            temp = temp.rename(columns={"timestamp_x": "timestamp"})
            to_drop.append("timestamp_y")
            df = temp.drop(columns=to_drop)
    else:
        ft = pd.DataFrame()
        ft["timestamp"] = np.load(str(input_dir / "world_timestamps.npy")) * 1000.0
        ft.index.name = "frame_idx"
        # check there are no gaps in the video file
        if df["frame_idx"].max() > ft.index.max():
            raise RuntimeError(
                "It appears there are frames missing in the scene video, but the file world_lookup.npy that would be needed to deal with that is missing. You can generate it by opening the recording in pupil player."
            )

    # set t=0 to video start time
    t0 = ft["timestamp"].iloc[0] * 1000 - frame_ts["timestamp"].iloc[0]
    df.loc[:, "timestamp"] -= t0

    # set timestamps as index for gaze
    df = df.set_index("timestamp")

    return df, frame_ts


def read_gaze_data_pupil_player(
    file: str | pathlib.Path, scene_video_dimensions: list[int], rec_info: Recording
) -> pd.DataFrame:
    """Read and process gaze data from a Pupil/Neon Player export CSV.

    Reads gaze positions, optionally joins pupil diameter data from
    ``pupil_positions.csv``, renames columns to the common naming scheme,
    marks low-confidence samples as NaN, and converts timestamps and
    gaze coordinates to the expected units and coordinate system.

    Args:
        file: Path to the ``gaze_positions*.csv`` export file.
        scene_video_dimensions: Scene camera resolution ``[width, height]``
            for converting normalized gaze coordinates to pixels.
        rec_info: Recording metadata (determines device-specific processing).

    Returns:
        A DataFrame with gaze data columns and timestamp column (not yet
        set as index).

    """
    is_core = rec_info.eye_tracker is EyeTracker.Pupil_Core

    file = pathlib.Path(file)
    df = pd.read_csv(file)
    pupil_pos_file = file.with_name("pupil_positions.csv")
    has_pupil_pos = pupil_pos_file.is_file() and "base_data" in df

    # drop columns with nothing in them
    df = df.dropna(how="all", axis=1)
    # if there is a pupil positions file, get pupil diameters from there
    if has_pupil_pos:

        def parse_base_data(sample: pd.Series) -> dict:
            out = {}
            bd = sample["base_data"].split(" ")
            for b in bd:
                parts = b.split("-")
                out[f"pup_ts_{parts[1]}"] = float(parts[0])
            return out

        pupil_pos_ts = df.apply(parse_base_data, axis="columns", result_type="expand")
        df_pup = pd.read_csv(pupil_pos_file)
        df_pup = df_pup.dropna(subset="diameter_3d")
        df_pup = df_pup[["pupil_timestamp", "eye_id", "diameter_3d"]]
        eyes = ["l", "r"]
        for e in [0, 1]:
            if not any(df_pup["eye_id"] == e) or f"pup_ts_{e}" not in pupil_pos_ts:
                continue
            diam = pupil_pos_ts.merge(
                df_pup[df_pup["eye_id"] == e], how="left", left_on=f"pup_ts_{e}", right_on="pupil_timestamp"
            )
            df[f"pup_diam_{eyes[e]}"] = diam["diameter_3d"]
    df = df.drop(columns=["base_data"], errors="ignore")  # drop these columns if they exist

    # rename and reorder columns
    lookup = {
        "gaze_timestamp": "timestamp",
        "world_index": "frame_idx",
        "eye_center1_3d_x": "gaze_ori_l_x",
        "eye_center1_3d_y": "gaze_ori_l_y",
        "eye_center1_3d_z": "gaze_ori_l_z",
        "pup_diam_l": "pup_diam_l",
        "gaze_normal1_x": "gaze_dir_l_x",
        "gaze_normal1_y": "gaze_dir_l_y",
        "gaze_normal1_z": "gaze_dir_l_z",
        "eye_center0_3d_x": "gaze_ori_r_x",  # NB: if monocular setup filming left eye, this is the left eye
        "eye_center0_3d_y": "gaze_ori_r_y",
        "eye_center0_3d_z": "gaze_ori_r_z",
        "pup_diam_r": "pup_diam_r",
        "gaze_normal0_x": "gaze_dir_r_x",
        "gaze_normal0_y": "gaze_dir_r_y",
        "gaze_normal0_z": "gaze_dir_r_z",
        "norm_pos_x": "gaze_pos_vid_x",
        "norm_pos_y": "gaze_pos_vid_y",
        "gaze_point_3d_x": "gaze_pos_3d_x",
        "gaze_point_3d_y": "gaze_pos_3d_y",
        "gaze_point_3d_z": "gaze_pos_3d_z",
        # for Neon player exports:
        "timestamp [ns]": "timestamp",
        "gaze x [px]": "gaze_pos_vid_x",
        "gaze y [px]": "gaze_pos_vid_y",
    }
    df = df.rename(columns=lookup)
    # reorder
    seen = set()
    idx = [v for k in lookup if (v := lookup[k]) in df.columns and not (v in seen or seen.add(v))]
    idx.extend([x for x in df.columns if x not in idx])  # append columns not in lookup
    df = df[idx]

    # mark data with insufficient confidence as missing.
    # for pupil core, pupil labs recommends a threshold of 0.6,
    # for the pupil invisible its a binary signal, and
    # confidence 0 should be excluded
    if "confidence" in df:
        conf_thresh = 0.6 if is_core else 0
        todo = [x for x in idx if x in lookup.values()]
        to_remove = df.confidence <= conf_thresh
        for c in todo[2:]:
            df.loc[to_remove, c] = np.nan

    # fix timestamps
    if rec_info.eye_tracker == EyeTracker.Pupil_Neon:
        # these are UTC or some kind of time format. Need them relative to recording start
        # that info is found in export_info.csv
        df["timestamp"] -= df["timestamp"].iloc[0]
        ei = pd.read_csv(file.with_name("export_info.csv"))
        time_range = ei[ei["key"] == "Absolute Time Range"]["value"].iloc[0]
        time_range = [float(x) * 1000.0 for x in time_range.split("-")]  # s -> ms
        # convert timestamps from ns to ms
        df = df.astype({"timestamp": "float"})
        df["timestamp"] /= 1000.0 * 1000.0
        # now correct for starting point gotten from export_info.csv
        df["timestamp"] += time_range[0]
        # NB: gaze positions are already in pixels on the scene camera video
    else:
        # convert timestamps from s to ms
        df.loc[:, "timestamp"] *= 1000.0
        # turn gaze locations into pixel data with origin in top-left
        df.loc[:, "gaze_pos_vid_x"] *= scene_video_dimensions[0]
        df.loc[:, "gaze_pos_vid_y"] = (1 - df.loc[:, "gaze_pos_vid_y"]) * scene_video_dimensions[
            1
        ]  # turn origin from bottom-left to top-left

    return df


def format_gaze_data_cloud_export(
    input_dir: str | pathlib.Path, export_file: str | pathlib.Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Format gaze data from a Pupil Cloud export.

    Reads gaze data and frame timestamps from the Cloud export, converts
    from nanoseconds to milliseconds with t=0 at video start, and assigns
    frame indices to each gaze sample.

    Args:
        input_dir: Recording directory containing ``world_timestamps.csv``.
        export_file: Path to the ``gaze.csv`` Cloud export file.

    Returns:
        A tuple of (gaze DataFrame indexed by timestamp, frame timestamps
        DataFrame).

    """
    df = read_gaze_data_cloud_export(export_file)

    frame_timestamps = pd.read_csv(input_dir / "world_timestamps.csv")
    frame_timestamps = frame_timestamps.rename(columns={"timestamp [ns]": "timestamp"})
    frame_timestamps = frame_timestamps.drop(columns=[x for x in frame_timestamps.columns if x != "timestamp"])
    frame_timestamps["frame_idx"] = frame_timestamps.index
    frame_timestamps = frame_timestamps.set_index("frame_idx")

    # set t=0 to video start time
    t0_ns = frame_timestamps["timestamp"].iloc[0]
    df["timestamp"] -= t0_ns
    frame_timestamps["timestamp"] -= t0_ns
    df["timestamp"] /= 1000000.0  # convert timestamps from ns to ms
    frame_timestamps["timestamp"] /= 1000000.0

    # set timestamps as index for gaze
    df = df.set_index("timestamp")

    # use the frame timestamps to assign a frame number to each data point
    frame_idx = video_utils.timestamps_to_frame_number(df.index, frame_timestamps["timestamp"].to_numpy())
    df.insert(0, "frame_idx", frame_idx["frame_idx"])

    return df, frame_timestamps


def read_gaze_data_cloud_export(file: str | pathlib.Path) -> pd.DataFrame:
    """Read and format gaze data from a Pupil Cloud export CSV file.

    Renames columns to the common naming scheme, optionally merges 3D eye
    state data from ``3d_eye_states.csv``, and marks samples where the
    tracker is not worn or during blinks as NaN.

    Args:
        file: Path to the ``gaze.csv`` Cloud export file.

    Returns:
        A DataFrame with gaze data columns and a ``timestamp`` column
        (in nanoseconds, not yet converted or set as index).

    """
    df = pd.read_csv(file)

    # rename and reorder columns
    lookup = {
        "timestamp [ns]": "timestamp",
        "gaze x [px]": "gaze_pos_vid_x",
        "gaze y [px]": "gaze_pos_vid_y",
        "pupil diameter left [mm]": "pup_diam_l",
        "pupil diameter right [mm]": "pup_diam_r",
        "eyeball center left x [mm]": "gaze_ori_l_x",
        "eyeball center left y [mm]": "gaze_ori_l_y",
        "eyeball center left z [mm]": "gaze_ori_l_z",
        "eyeball center right x [mm]": "gaze_ori_r_x",
        "eyeball center right y [mm]": "gaze_ori_r_y",
        "eyeball center right z [mm]": "gaze_ori_r_z",
        "optical axis left x": "gaze_dir_l_x",
        "optical axis left y": "gaze_dir_l_y",
        "optical axis left z": "gaze_dir_l_z",
        "optical axis right x": "gaze_dir_r_x",
        "optical axis right y": "gaze_dir_r_y",
        "optical axis right z": "gaze_dir_r_z",
    }
    df = df.drop(columns=[x for x in df.columns if x not in lookup and x not in {"worn", "blink id"}])
    df = df.rename(columns=lookup)

    # check if there is an eye states file
    eye_state_file = file.parent / "3d_eye_states.csv"
    if eye_state_file.exists():
        df_eye = pd.read_csv(eye_state_file)
        df_eye = df_eye.drop(columns=[x for x in df_eye.columns if x not in lookup])
        df_eye = df_eye.rename(columns=lookup)
        df = df.join(df_eye.set_index("timestamp"), on="timestamp")

    # mark data where eye tracker is not worn or during blink as missing
    todo = [lookup[k] for k in lookup if lookup[k] in df.columns and lookup[k] != "timestamp"]
    to_remove = df.worn == 0
    if "blink id" in df:
        to_remove = np.logical_or(to_remove, df["blink id"] > 0)
    for c in todo:
        df.loc[to_remove, c] = np.nan

    # remove last columns we don't need anymore
    df = df.drop(columns=[x for x in df.columns if x in {"worn", "blink id"}])

    return df
