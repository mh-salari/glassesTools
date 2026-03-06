"""Dynamic validation marker setup, config conversion, and analysis."""

import importlib.resources
import json
import math
import pathlib
import shutil
from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd

from ... import aruco, marker
from ..config import get_markers, get_targets

if TYPE_CHECKING:
    from .. import Plane


def deploy_setup_and_script(output_dir: str | pathlib.Path, overwrite: bool = False) -> list[str]:
    """Copy default dynamic validation setup and PsychoPy script to output_dir.

    Copies ``markerPositions.csv``, ``targetPositions.csv``,
    ``setup.json``, and ``stim.py`` from the bundled package.

    Args:
        output_dir: Directory to copy files into.
        overwrite: If True, overwrite existing files.

    Returns:
        List of filenames that were NOT copied (already existed and
        ``overwrite`` is False).

    Raises:
        RuntimeError: If ``output_dir`` does not exist.

    """
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError(f'The requested directory "{output_dir}" does not exist')

    # copy over all files
    not_copied: list[str] = []
    for r in ["markerPositions.csv", "targetPositions.csv", "setup.json", "stim.py"]:
        out_file = output_dir / r
        if out_file.exists() and not overwrite:
            not_copied.append(r)
            continue
        with importlib.resources.path(__package__, r) as p:
            shutil.copyfile(p, out_file)
    return not_copied


def _get_extent(extent: float, distance: float, psychopy_unit: str) -> float:
    """Convert a size value from PsychoPy units to cm."""
    match psychopy_unit:
        case "cm":
            return extent
        case "deg" | "degFlatPos":
            # sizes are not corrected for flat screen when unit is degFlatPos, so same as deg
            # for factor, see comment in _get_position
            return extent * distance * 0.017455
        case _:
            raise ValueError(f"PsychoPy unit {psychopy_unit} is not understood")


def _get_position(position: tuple[float, float], distance: float, psychopy_unit: str) -> tuple[float, float]:
    """Convert an (x, y) position from PsychoPy units to cm."""
    match psychopy_unit:
        case "cm":
            return position
        case "deg":
            # 1 deg (centered) at 1 cm is approximately pi/180 (2*tand(.5)), seems to be the logic for PsychoPy
            # this is then apparently hardcoded as 0.017455, which is rounded off a bit wrong...
            fac = distance * 0.017455
            return tuple(p * fac for p in position)
        case "degFlatPos":
            # positions corrected for flat screen, sizes not
            x, y = (math.radians(x) for x in position)
            return (
                math.hypot(distance, math.tan(y) * distance) * math.tan(x),
                math.hypot(distance, math.tan(x) * distance) * math.tan(y),
            )
        case _:
            raise ValueError(f"PsychoPy unit {psychopy_unit} is not understood")


def setup_to_automatic_coding(
    config_dir: str | pathlib.Path | None = None, file_name: str = "setup.json"
) -> dict[str, int | list[int]]:
    """Extract automatic coding parameters from the dynamic validation setup.

    Returns segment marker IDs (start/end) and ArUco border bits from
    the PsychoPy ``setup.json``.

    Args:
        config_dir: Path to directory containing the setup file, or
            None to use the bundled default.
        file_name: Name of the JSON setup file.

    Returns:
        A dict with ``start_IDs``, ``end_IDs``, and ``border_bits``
        keys.

    """
    if config_dir is not None:
        with pathlib.Path(pathlib.Path(config_dir) / file_name).open("r", encoding="utf-8") as f:
            setup = json.load(f)
    else:
        with importlib.resources.open_text(__package__, file_name) as f:
            setup = json.load(f)

    out = setup["validation"]["segment_marker"]
    out["border_bits"] = setup["aruco"]["border_bits"]
    return out


def setup_to_plane_config(
    output_dir: str | pathlib.Path, config_dir: str | pathlib.Path | None = None, file_name: str = "setup.json"
) -> dict[str, list[marker.MarkerID]]:
    """Convert PsychoPy ``setup.json`` to validation plane config files.

    Reads the PsychoPy dynamic validation setup, converts target and
    marker positions from PsychoPy units to cm, computes the plane
    extent, writes ``validationSetup.txt`` and converted CSV files, and
    returns the segmentation marker setup.

    Args:
        output_dir: Directory to write the converted config files into.
        config_dir: Path to directory containing the setup file, or
            None to use the bundled default.
        file_name: Name of the JSON setup file.

    Returns:
        A dict with ``start_markers``, ``end_markers``,
        ``marker_border_bits``, and ``shown_between_repetitions`` keys.

    Raises:
        RuntimeError: If ``output_dir`` does not exist.

    """
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError(f'The requested directory "{output_dir}" does not exist')

    if config_dir is not None:
        with pathlib.Path(pathlib.Path(config_dir) / file_name).open("r", encoding="utf-8") as f:
            setup = json.load(f)
    else:
        with importlib.resources.open_text(__package__, file_name) as f:
            setup = json.load(f)

    # get units in which positions are expressed, and viewing distance which may be needed for converting them
    marker_units = (
        setup["validation"]["markers"]["units"]
        if "units" in setup["validation"]["markers"]
        else setup["screen"]["units"]
    )
    target_units = (
        setup["validation"]["targets"]["units"]
        if "units" in setup["validation"]["targets"]
        else setup["screen"]["units"]
    )
    dist = setup["screen"]["viewing_distance"]
    # load target and marker positions (default ones from package if config dir is not specified)
    target_positions = get_targets(config_dir, setup["validation"]["targets"]["file"], __package__)
    marker_positions = get_markers(config_dir, setup["validation"]["markers"]["file"], __package__)
    # convert positions to cm
    target_positions[["x", "y"]] = target_positions.apply(
        lambda r: _get_position((r["x"], r["y"]), dist, target_units), axis=1, result_type="expand"
    )
    marker_positions[["x", "y"]] = marker_positions.apply(
        lambda r: _get_position((r["x"], r["y"]), dist, marker_units), axis=1, result_type="expand"
    )
    # negate y as positive y is upward for PsychoPy script but downward for our logic
    target_positions["y"] = -target_positions["y"]
    marker_positions["y"] = -marker_positions["y"]
    # Positions are centers relative to screen center; shift to (0,0) = top-left,
    # accounting for sizes so whole markers and targets fit on the reference image
    t_size = _get_extent(setup["validation"]["targets"]["look"]["diameter_max"], dist, target_units)
    m_size = _get_extent(setup["validation"]["markers"]["size"], dist, marker_units)
    min_t = target_positions[["x", "y"]].min() - t_size / 2
    min_m = marker_positions[["x", "y"]].min() - m_size / 2
    x_min = min(min_t["x"], min_m["x"])
    y_min = min(min_t["y"], min_m["y"])
    target_positions["x"] -= x_min
    marker_positions["x"] -= x_min
    target_positions["y"] -= y_min
    marker_positions["y"] -= y_min

    # get plane size
    max_t = target_positions[["x", "y"]].max() + t_size / 2
    max_m = marker_positions[["x", "y"]].max() + m_size / 2
    x_max = max(max_t["x"], max_m["x"])
    y_max = max(max_t["y"], max_m["y"])

    # write appropriate validationSetup.txt file
    val_setup = {
        "distance": dist,
        "mode": "cm",
        "arucoDictionary": setup["aruco"]["dict"],
        "markerBorderBits": setup["aruco"]["border_bits"],
        "markerSide": m_size,
        "markerPosFile": "markerPositions_converted.csv",
        "targetPosFile": "targetPositions_converted.csv",
        "targetType": "Thaler",
        "targetDiameter": t_size,
        "showGrid": 0,
        "gridCols": x_max,
        "gridRows": y_max,
        "minNumMarkers": 3,
        "centerTarget": setup["validation"]["targets"]["center_target"],
        "referencePosterSize": 1920,
    }

    # add marker indicator(s) as columns to target file
    for i, _ in enumerate(setup["validation"]["markers"]["replace_IDs"]):
        marker_ids = (
            target_positions.index.to_numpy()
            + setup["validation"]["markers"]["replace_ID_start"]
            + i * setup["validation"]["markers"]["replace_ID_offset"]
        )
        target_positions[f"marker_{i}"] = marker_ids

    # store everything to files
    with pathlib.Path(output_dir / "validationSetup.txt").open("w", encoding="utf-8") as f:
        f.writelines(f"{key} = {value}\n" for key, value in val_setup.items())
    target_positions.to_csv(output_dir / val_setup["targetPosFile"], float_format="%.8f")
    marker_positions.to_csv(output_dir / val_setup["markerPosFile"], float_format="%.8f")

    # last, get segmentation setup (markers used for start and end of validation interval) and info about repetitions
    segmentation_markers: dict[str, list[marker.MarkerID]] = {}
    aruco_dict_id = aruco.str_to_dict_id(setup["aruco"]["dict"])
    for s, o in zip(("start_IDs", "end_IDs"), ("start_markers", "end_markers"), strict=True):
        segmentation_markers[o] = [marker.MarkerID(m, aruco_dict_id) for m in setup["validation"]["segment_marker"][s]]
    segmentation_markers["marker_border_bits"] = setup["aruco"]["border_bits"]
    n_repetitions = setup["validation"]["n_repetitions"]
    segmentation_markers["shown_between_repetitions"] = (
        n_repetitions > 1 and setup["validation"]["show_segment_between_repetitions"]
    )
    return segmentation_markers


def get_marker_observations(
    validation_plane: "Plane", working_dir: pathlib.Path, name: str = "", missing_ok: bool = False
) -> tuple[dict[int, pd.DataFrame], dict[int, list[marker.MarkerID]]]:
    """Load and organize dynamic marker observations per target from detection files.

    For each target on the validation plane, reads the corresponding
    marker detection files, converts them to boolean presence signals,
    and merges multiple markers for the same target into a single
    DataFrame for robustness against spotty detection.

    Args:
        validation_plane: The dynamic validation plane containing
            marker-to-target mappings.
        working_dir: Directory containing marker detection TSV files.
        name: Optional plane name used in error messages.
        missing_ok: If True, silently skip targets whose marker files
            are all missing.

    Returns:
        A tuple of (marker observations dict mapping target ID to
        presence DataFrame, markers-per-target dict mapping target ID
        to list of ``MarkerID``).

    Raises:
        FileNotFoundError: If all marker files for a target are missing
            and ``missing_ok`` is False.

    """
    # organize markers
    markers_per_target: dict[int, list[marker.MarkerID]] = defaultdict(list)
    for m in validation_plane.dynamic_markers:
        t = validation_plane.dynamic_markers[m][0]
        markers_per_target[t].append(marker.MarkerID(m, validation_plane.aruco_dict_id))
    markers_per_target = dict(
        markers_per_target
    )  # get rid of defaultdict now its no longer needed so we get normal indexing

    # determine what marker files to read
    all_marker_ids = [m for ms in markers_per_target for m in markers_per_target[ms]]
    # for each target, check at least one of the marker files exists
    for t in markers_per_target:
        missing = [
            not marker.get_file_name(m.m_id, m.aruco_dict_id, working_dir).is_file() for m in markers_per_target[t]
        ]
        if all(missing) and not missing_ok:
            file_missing = [marker.get_file_name(m.m_id, m.aruco_dict_id, None) for m in markers_per_target[t]]
            missing_str = "\n- ".join(file_missing)
            extra = f"on plane {name} " if name else ""
            raise FileNotFoundError(f"None of the marker files for target {t} {extra}were found:\n- {missing_str}")
        # remove missing from list of markers to load
        if any(missing):
            for i, m in enumerate(missing):
                if not m:
                    continue
                all_marker_ids.remove(markers_per_target[t][i])

    # load all markers and recode so we just have a boolean indicating when markers are present
    marker_observations = {
        m: marker.read_dataframe_from_file(m.m_id, m.aruco_dict_id, working_dir).set_index("frame_idx")
        for m in all_marker_ids
    }
    marker_observations = {
        m: marker.code_for_presence(marker_observations[m], allow_failed=True)
        for m in marker_observations
        if not marker_observations[m].empty
    }

    # target presentations may be encoded by multiple markers simultaneously
    # merge all markers for the target to be more robust to choppy detection
    marker_observations_per_target: dict[int, pd.DataFrame] = {}
    for t in markers_per_target:
        for m in markers_per_target[t]:
            if m not in marker_observations or marker_observations[m].empty:
                continue
            if t not in marker_observations_per_target:
                marker_observations_per_target[t] = marker_observations[m]
            else:
                marker_observations_per_target[t] = marker_observations_per_target[t].combine_first(
                    marker_observations[m]
                )

    return marker_observations_per_target, markers_per_target
