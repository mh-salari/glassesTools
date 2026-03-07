"""Unified import interface for all supported eye tracker devices.

Provides a single entry point (``do_import``) that dispatches to
device-specific importers, plus per-device convenience wrappers and
input-validation helpers. Imported recordings are normalized to a
common directory layout with gaze data, frame timestamps, and a
recording-info JSON file.

"""

import os
import pathlib

import pandas as pd
import polars as pl

from .. import eyetracker, naming
from ..eyetracker import EyeTracker
from ..recording import Recording
from .adhawk_mindlink import preprocess_data as adhawk_mindlink
from .argus_ETVision import preprocess_data as argus_ETVision
from .generic import import_data as generic
from .meta_aria_gen1 import import_data as meta_aria_gen1
from .SeeTrue_STONE import preprocess_data as SeeTrue_STONE
from .SMI_ETG import preprocess_data as SMI_ETG
from .tobii_G2 import preprocess_data as tobii_G2
from .tobii_G3 import preprocess_data as tobii_G3


def pupil_core(
    output_dir: str | pathlib.Path,
    source_dir: str | pathlib.Path | None = None,
    rec_info: Recording | None = None,
    copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
) -> Recording:
    """Import and preprocess a Pupil Core recording.

    Args:
        output_dir: Working directory where the imported recording will be placed.
        source_dir: Path to the raw recording. Falls back to ``rec_info.source_directory``.
        rec_info: Optional pre-populated recording metadata.
        copy_scene_video: Whether to copy the scene video to output_dir.
        source_dir_as_relative_path: Store source_dir as a relative path in rec_info.

    Returns:
        The populated Recording object written to output_dir.

    """
    from .pupilLabs import preprocess_data

    return preprocess_data(
        output_dir, EyeTracker.Pupil_Core, source_dir, rec_info, copy_scene_video, source_dir_as_relative_path
    )


def pupil_invisible(
    output_dir: str | pathlib.Path,
    source_dir: str | pathlib.Path | None = None,
    rec_info: Recording | None = None,
    copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
) -> Recording:
    """Import and preprocess a Pupil Invisible recording.

    Args:
        output_dir: Working directory where the imported recording will be placed.
        source_dir: Path to the raw recording. Falls back to ``rec_info.source_directory``.
        rec_info: Optional pre-populated recording metadata.
        copy_scene_video: Whether to copy the scene video to output_dir.
        source_dir_as_relative_path: Store source_dir as a relative path in rec_info.

    Returns:
        The populated Recording object written to output_dir.

    """
    from .pupilLabs import preprocess_data

    return preprocess_data(
        output_dir, EyeTracker.Pupil_Invisible, source_dir, rec_info, copy_scene_video, source_dir_as_relative_path
    )


def pupil_neon(
    output_dir: str | pathlib.Path,
    source_dir: str | pathlib.Path | None = None,
    rec_info: Recording | None = None,
    copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
) -> Recording:
    """Import and preprocess a Pupil Neon recording.

    Args:
        output_dir: Working directory where the imported recording will be placed.
        source_dir: Path to the raw recording. Falls back to ``rec_info.source_directory``.
        rec_info: Optional pre-populated recording metadata.
        copy_scene_video: Whether to copy the scene video to output_dir.
        source_dir_as_relative_path: Store source_dir as a relative path in rec_info.

    Returns:
        The populated Recording object written to output_dir.

    """
    from .pupilLabs import preprocess_data

    return preprocess_data(
        output_dir, EyeTracker.Pupil_Neon, source_dir, rec_info, copy_scene_video, source_dir_as_relative_path
    )


def VPS_19(
    output_dir: str | pathlib.Path,
    source_dir: str | pathlib.Path | None = None,
    rec_info: Recording | None = None,
    copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
    cam_cal_file: str | pathlib.Path | None = None,
) -> Recording:
    """Import and preprocess a Viewpointsystem VPS 19 recording.

    Args:
        output_dir: Working directory where the imported recording will be placed.
        source_dir: Path to the raw recording. Falls back to ``rec_info.source_directory``.
        rec_info: Optional pre-populated recording metadata.
        copy_scene_video: Whether to copy the scene video to output_dir.
        source_dir_as_relative_path: Store source_dir as a relative path in rec_info.
        cam_cal_file: Path to an external camera calibration file.

    Returns:
        The populated Recording object written to output_dir.

    """
    from .VPS import preprocess_data

    return preprocess_data(
        output_dir,
        EyeTracker.VPS_19,
        source_dir,
        rec_info,
        copy_scene_video,
        source_dir_as_relative_path,
        cam_cal_file=cam_cal_file,
    )


def VPS_Lite(
    output_dir: str | pathlib.Path,
    source_dir: str | pathlib.Path | None = None,
    rec_info: Recording | None = None,
    copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
    cam_cal_file: str | pathlib.Path | None = None,
) -> Recording:
    """Import and preprocess a Viewpointsystem VPS Lite recording.

    Args:
        output_dir: Working directory where the imported recording will be placed.
        source_dir: Path to the raw recording. Falls back to ``rec_info.source_directory``.
        rec_info: Optional pre-populated recording metadata.
        copy_scene_video: Whether to copy the scene video to output_dir.
        source_dir_as_relative_path: Store source_dir as a relative path in rec_info.
        cam_cal_file: Path to an external camera calibration file.

    Returns:
        The populated Recording object written to output_dir.

    """
    from .VPS import preprocess_data

    return preprocess_data(
        output_dir,
        EyeTracker.VPS_Lite,
        source_dir,
        rec_info,
        copy_scene_video,
        source_dir_as_relative_path,
        cam_cal_file=cam_cal_file,
    )


def get_recording_info(
    source_dir: str | pathlib.Path, device: str | EyeTracker, device_name: str | None = None
) -> list[Recording]:
    """Retrieve recording metadata from a source directory for a given device type.

    Dispatches to the device-specific ``get_recording_info`` implementation,
    which inspects the source directory structure and returns one or more
    Recording objects describing the recordings found.

    Args:
        source_dir: Path to the raw recording directory to inspect.
        device: Eye tracker type (enum or string name).
        device_name: Custom device name, only used for ``EyeTracker.Generic``.

    Returns:
        A list of Recording objects found in source_dir, or None if the
        device-specific handler returned nothing.

    Raises:
        RuntimeError: If the device type is not supported.

    """
    source_dir = pathlib.Path(source_dir)
    device = eyetracker.string_to_enum(device)

    rec_info = None
    match device:
        case EyeTracker.AdHawk_MindLink:
            from .adhawk_mindlink import get_recording_info

            rec_info = get_recording_info(source_dir)
        case EyeTracker.Argus_ETVision:
            from .argus_ETVision import get_recording_info

            rec_info = get_recording_info(source_dir)
        case EyeTracker.Generic:
            from .generic import get_recording_info

            rec_info = get_recording_info(source_dir, device_name)
        case EyeTracker.Meta_Aria_Gen_1:
            from .meta_aria_gen1 import get_recording_info

            rec_info = get_recording_info(source_dir)
        case EyeTracker.Pupil_Core:
            from .pupilLabs import get_recording_info

            rec_info = get_recording_info(source_dir, device)
        case EyeTracker.Pupil_Invisible:
            from .pupilLabs import get_recording_info

            rec_info = get_recording_info(source_dir, device)
        case EyeTracker.Pupil_Neon:
            from .pupilLabs import get_recording_info

            rec_info = get_recording_info(source_dir, device)
        case EyeTracker.SeeTrue_STONE:
            from .SeeTrue_STONE import get_recording_info

            rec_info = get_recording_info(source_dir)
        case EyeTracker.SMI_ETG:
            from .SMI_ETG import get_recording_info

            rec_info = get_recording_info(source_dir)
        case EyeTracker.Tobii_Glasses_2:
            from .tobii_G2 import get_recording_info

            rec_info = get_recording_info(source_dir)
        case EyeTracker.Tobii_Glasses_3:
            from .tobii_G3 import get_recording_info

            rec_info = get_recording_info(source_dir)
        case EyeTracker.VPS_19:
            from .VPS import get_recording_info

            rec_info = get_recording_info(source_dir, device)
        case EyeTracker.VPS_Lite:
            from .VPS import get_recording_info

            rec_info = get_recording_info(source_dir, device)
        case _:
            raise RuntimeError(f'Not implemented for "{device.value}", contact developer')

    if rec_info is not None and not isinstance(rec_info, list):
        rec_info = [rec_info]
    return rec_info


def do_import(
    output_dir: str | pathlib.Path | None = None,
    source_dir: str | pathlib.Path | None = None,
    device: str | EyeTracker | None = None,
    rec_info: Recording | None = None,
    copy_scene_video: bool = True,
    source_dir_as_relative_path: bool = False,
    cam_cal_file: str | pathlib.Path | None = None,
    device_name: str | None = None,
) -> Recording:
    """Unified front end for importing a recording from any supported device.

    Validates inputs, resolves the device type, and dispatches to the
    appropriate device-specific importer.

    Args:
        output_dir: Working directory where the imported recording will be placed.
            Must match ``rec_info.working_directory`` if both are provided.
        source_dir: Path to the raw recording directory.
        device: Eye tracker type (enum or string). Inferred from rec_info if not given.
        rec_info: Optional pre-populated recording metadata.
        copy_scene_video: Whether to copy the scene video to output_dir.
        source_dir_as_relative_path: Store source_dir as a relative path in rec_info.
        cam_cal_file: Path to an external camera calibration file. Only honored
            for AdHawk MindLink, Argus ETVision, SeeTrue STONE, Generic, and VPS devices.
        device_name: Custom device name, only used for ``EyeTracker.Generic``.

    Returns:
        The populated Recording object written to output_dir.

    Raises:
        ValueError: If rec_info is a list instead of a single Recording.
        RuntimeError: If the device type is not supported or cannot be determined.

    """
    if rec_info is not None and isinstance(rec_info, list):
        raise ValueError(
            'You should provide a single Recording to this functions "rec_info" input argument, not a list of Recordings.'
        )
    device, rec_info, device_name = check_device(device, rec_info, device_name)
    source_dir, rec_info = check_source_dir(source_dir, rec_info)
    output_dir, rec_info = check_output_dir(output_dir, rec_info)

    # do the actual import/pre-process
    match device:
        case EyeTracker.AdHawk_MindLink:
            rec_info = adhawk_mindlink(
                output_dir,
                source_dir,
                rec_info,
                copy_scene_video=copy_scene_video,
                source_dir_as_relative_path=source_dir_as_relative_path,
                cam_cal_file=cam_cal_file,
            )
        case EyeTracker.Argus_ETVision:
            rec_info = argus_ETVision(
                output_dir,
                source_dir,
                rec_info,
                copy_scene_video=copy_scene_video,
                source_dir_as_relative_path=source_dir_as_relative_path,
                cam_cal_file=cam_cal_file,
            )
        case EyeTracker.Generic:
            rec_info = generic(
                output_dir,
                source_dir,
                rec_info,
                device_name,
                copy_scene_video=copy_scene_video,
                source_dir_as_relative_path=source_dir_as_relative_path,
                cam_cal_file=cam_cal_file,
            )
        case EyeTracker.Meta_Aria_Gen_1:
            rec_info = meta_aria_gen1(
                output_dir,
                source_dir,
                rec_info,
                copy_scene_video=copy_scene_video,
                source_dir_as_relative_path=source_dir_as_relative_path,
            )
        case EyeTracker.Pupil_Core:
            rec_info = pupil_core(
                output_dir,
                source_dir,
                rec_info,
                copy_scene_video=copy_scene_video,
                source_dir_as_relative_path=source_dir_as_relative_path,
            )
        case EyeTracker.Pupil_Invisible:
            rec_info = pupil_invisible(
                output_dir,
                source_dir,
                rec_info,
                copy_scene_video=copy_scene_video,
                source_dir_as_relative_path=source_dir_as_relative_path,
            )
        case EyeTracker.Pupil_Neon:
            rec_info = pupil_neon(
                output_dir,
                source_dir,
                rec_info,
                copy_scene_video=copy_scene_video,
                source_dir_as_relative_path=source_dir_as_relative_path,
            )
        case EyeTracker.SeeTrue_STONE:
            rec_info = SeeTrue_STONE(
                output_dir,
                source_dir,
                rec_info,
                copy_scene_video=copy_scene_video,
                source_dir_as_relative_path=source_dir_as_relative_path,
                cam_cal_file=cam_cal_file,
            )
        case EyeTracker.SMI_ETG:
            rec_info = SMI_ETG(
                output_dir,
                source_dir,
                rec_info,
                copy_scene_video=copy_scene_video,
                source_dir_as_relative_path=source_dir_as_relative_path,
            )
        case EyeTracker.Tobii_Glasses_2:
            rec_info = tobii_G2(
                output_dir,
                source_dir,
                rec_info,
                copy_scene_video=copy_scene_video,
                source_dir_as_relative_path=source_dir_as_relative_path,
            )
        case EyeTracker.Tobii_Glasses_3:
            rec_info = tobii_G3(
                output_dir,
                source_dir,
                rec_info,
                copy_scene_video=copy_scene_video,
                source_dir_as_relative_path=source_dir_as_relative_path,
            )
        case EyeTracker.VPS_19:
            rec_info = VPS_19(
                output_dir,
                source_dir,
                rec_info,
                copy_scene_video=copy_scene_video,
                source_dir_as_relative_path=source_dir_as_relative_path,
                cam_cal_file=cam_cal_file,
            )
        case EyeTracker.VPS_Lite:
            rec_info = VPS_Lite(
                output_dir,
                source_dir,
                rec_info,
                copy_scene_video=copy_scene_video,
                source_dir_as_relative_path=source_dir_as_relative_path,
                cam_cal_file=cam_cal_file,
            )
        case _:
            raise RuntimeError(f'Not implemented for "{device.value}", contact developer')

    return rec_info


def check_source_dir(
    source_dir: str | pathlib.Path, rec_info: Recording | None
) -> tuple[pathlib.Path, Recording | None]:
    """Validate and resolve source_dir, cross-checking with rec_info if provided.

    Args:
        source_dir: Path to the raw recording directory, or None to use rec_info.
        rec_info: Optional recording metadata to cross-check against source_dir.

    Returns:
        A tuple of (resolved source_dir, rec_info with source_directory set).

    Raises:
        ValueError: If both source_dir and rec_info.source_directory are set but differ.
        RuntimeError: If neither source_dir nor rec_info is provided.

    """
    if source_dir is not None:
        source_dir = pathlib.Path(source_dir)
        if (
            rec_info is not None
            and rec_info.source_directory
            and pathlib.Path(rec_info.source_directory) != source_dir
        ):
            raise ValueError(
                f"The provided source_dir ({source_dir}) does not equal the source directory set in rec_info ({rec_info.source_directory})."
            )
    elif rec_info is None:
        raise RuntimeError('Either the "input_dir" or the "rec_info" input argument should be set.')
    else:
        source_dir = pathlib.Path(rec_info.source_directory)

    if rec_info is not None and not rec_info.source_directory:
        rec_info.source_directory = source_dir
    return source_dir, rec_info


def check_output_dir(
    output_dir: str | pathlib.Path, rec_info: Recording | None
) -> tuple[pathlib.Path, Recording | None]:
    """Validate and resolve output_dir, ensuring it is empty if it exists.

    Args:
        output_dir: Working directory for the imported recording, or None to use rec_info.
        rec_info: Optional recording metadata to cross-check against output_dir.

    Returns:
        A tuple of (resolved output_dir, rec_info with working_directory set).

    Raises:
        ValueError: If both output_dir and rec_info.working_directory are set but differ.
        RuntimeError: If neither output_dir nor rec_info is provided, or if output_dir
            already exists and is not empty.

    """
    if output_dir is not None:
        output_dir = pathlib.Path(output_dir)
        if (
            rec_info is not None
            and rec_info.working_directory
            and pathlib.Path(rec_info.working_directory) != output_dir
        ):
            raise ValueError(
                f"The provided output_dir ({output_dir}) does not equal the working directory set in rec_info ({rec_info.working_directory})."
            )
    elif rec_info is None:
        raise RuntimeError('Either the "output_dir" or the "rec_info" input argument should be set.')
    else:
        output_dir = pathlib.Path(rec_info.working_directory)

    if output_dir.is_dir():
        with os.scandir(output_dir) as it:
            if any(it):
                raise RuntimeError(f"Output directory ({output_dir}) already exists and is not empty. Cannot use.")

    if rec_info is not None and not rec_info.working_directory:
        rec_info.working_directory = output_dir
    return output_dir, rec_info


def check_folders(
    output_dir: str | pathlib.Path,
    source_dir: str | pathlib.Path,
    rec_info: Recording | None,
    device: EyeTracker,
    device_name: str | None = None,
) -> tuple[pathlib.Path, pathlib.Path, Recording | None, str | None]:
    """Validate and resolve both source and output directories, plus device consistency.

    Checks that rec_info's device matches the expected device, then delegates
    to ``check_source_dir`` and ``check_output_dir`` for path validation.

    Args:
        output_dir: Working directory for the imported recording.
        source_dir: Path to the raw recording directory.
        rec_info: Optional recording metadata to validate against.
        device: Expected eye tracker type.
        device_name: Expected device name (for Generic devices).

    Returns:
        A tuple of (resolved output_dir, resolved source_dir, rec_info, device_name).

    Raises:
        ValueError: If rec_info's eye tracker or device name doesn't match the expected values.

    """
    if rec_info is not None and rec_info.eye_tracker:
        if rec_info.eye_tracker != device:
            raise ValueError(
                f"Provided rec_info is for a device ({rec_info.eye_tracker.value}) that is not a {device.value}. Cannot use."
            )
        if device_name is not None and rec_info.eye_tracker_name != device_name:
            raise ValueError(
                f'Provided rec_info is for a {rec_info.eye_tracker.value} device with the name "{rec_info.eye_tracker_name}" that is not the expected value ({device_name}). Cannot use.'
            )
        if not device_name:
            device_name = rec_info.eye_tracker_name

    source_dir, rec_info = check_source_dir(source_dir, rec_info)
    output_dir, rec_info = check_output_dir(output_dir, rec_info)
    return output_dir, source_dir, rec_info, device_name


def check_device(
    device: str | EyeTracker | None, rec_info: Recording | None, device_name: str | None = None
) -> tuple[EyeTracker, Recording | None, str | None]:
    """Resolve and validate the device type from device and/or rec_info.

    At least one of ``device`` or ``rec_info.eye_tracker`` must be set. If both
    are provided they must agree. For Generic devices, ``device_name`` must be
    set (either directly or via rec_info).

    Args:
        device: Eye tracker type (enum or string), or None to infer from rec_info.
        rec_info: Optional recording metadata containing eye tracker info.
        device_name: Custom device name, only used for ``EyeTracker.Generic``.

    Returns:
        A tuple of (resolved EyeTracker, rec_info, device_name).

    Raises:
        RuntimeError: If the device type cannot be determined, or if device_name
            constraints for Generic devices are not met.
        ValueError: If device and rec_info.eye_tracker disagree, or if device_name
            and rec_info.eye_tracker_name disagree.

    """
    if device is None and (rec_info is None or not rec_info.eye_tracker):
        raise RuntimeError(
            'Either the "device" or the eye_tracker field of the "rec_info" input argument should be set.'
        )
    if device is not None:
        device = eyetracker.string_to_enum(device)
    if rec_info is not None and rec_info.eye_tracker:
        if device is not None:
            if rec_info.eye_tracker != device:
                raise ValueError(
                    f"Provided device ({device.value}) did not match device specified in rec_info ({rec_info.eye_tracker.value}). Provide matching values or do not provide the device input argument."
                )
        else:
            device = eyetracker.string_to_enum(rec_info.eye_tracker)

    if device_name is not None:
        if device != EyeTracker.Generic:
            raise RuntimeError(
                f"The device_name parameter should not be set for devices other than a {EyeTracker.Generic.value} device, but it was set."
            )
        if rec_info is not None and device_name != rec_info.eye_tracker_name:
            raise RuntimeError(
                f"Provided device_name ({device_name}) did not match the device name specified in rec_info ({rec_info.eye_tracker_name}). Provide matching values or do not provide the device_name input argument."
            )
    elif device == EyeTracker.Generic:
        if rec_info is not None:
            device_name = rec_info.eye_tracker_name
        if not device_name:
            raise RuntimeError(
                f"For a {device.value} device, the device_name parameter should be set or the eye tracker name should be set in recording info, but it was not"
            )

    return device, rec_info, device_name


def _store_data(
    output_dir: pathlib.Path,
    gaze: pd.DataFrame | None,
    frame_ts: pd.DataFrame | None,
    rec_info: Recording,
    gaze_fname: str = naming.gaze_data_fname,
    frame_ts_fname: str = naming.frame_timestamps_fname,
    rec_info_fname: str = Recording.default_json_file_name,
    source_dir_as_relative_path: bool = False,
) -> None:
    """Write imported data (gaze, frame timestamps, recording info) to disk.

    Saves DataFrames as tab-separated CSV files via polars for speed, and
    serializes rec_info as JSON. If the recording duration is unknown, it is
    estimated from the available data.

    Args:
        output_dir: Directory where output files are written.
        gaze: Gaze data DataFrame with a timestamp index, or None.
        frame_ts: Frame timestamps DataFrame, or None.
        rec_info: Recording metadata to serialize.
        gaze_fname: Filename for the gaze data CSV.
        frame_ts_fname: Filename for the frame timestamps CSV.
        rec_info_fname: Filename for the recording info JSON.
        source_dir_as_relative_path: If True, convert the source directory
            in rec_info to a path relative to output_dir before saving.

    """
    # write the gaze data to a csv file (polars saves to file much faster than pandas)
    if gaze is not None:
        pl.from_pandas(gaze, include_index=True).write_csv(
            output_dir / gaze_fname, separator="\t", null_value="nan", float_precision=8
        )

    # store frame timestamps
    if frame_ts is not None:
        pl.from_pandas(frame_ts, include_index=True).write_csv(
            output_dir / frame_ts_fname, separator="\t", float_precision=8
        )

    # store rec info
    if source_dir_as_relative_path:
        rec_info.source_directory = pathlib.Path(os.path.relpath(rec_info.source_directory, output_dir))
    # if duration not known, fill it
    if not rec_info.duration and (gaze is not None or frame_ts is not None):
        # make a reasonable estimate of duration
        durations = []
        if gaze is not None:
            durations.append(gaze.index[-1] - gaze.index[0])
        if frame_ts is not None:
            durations.append(frame_ts.timestamp.iloc[-1])
        rec_info.duration = round(max(durations))
    rec_info.store_as_json(output_dir / rec_info_fname)


__all__ = [
    "SMI_ETG",
    "VPS_19",
    "SeeTrue_STONE",
    "VPS_Lite",
    "adhawk_mindlink",
    "argus_ETVision",
    "check_device",
    "check_folders",
    "check_output_dir",
    "check_source_dir",
    "do_import",
    "generic",
    "get_recording_info",
    "pupil_core",
    "pupil_invisible",
    "pupil_neon",
    "tobii_G2",
    "tobii_G3",
]
