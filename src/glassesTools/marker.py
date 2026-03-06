"""Marker data types, pose storage, detection analysis, and formatting utilities."""

import operator
import pathlib
import typing
from collections import defaultdict

import numpy as np
import pandas as pd

from . import data_files, drawing, json, naming, ocv


class MarkerID(typing.NamedTuple):
    """Identifier for an ArUco marker combining its ID and dictionary."""

    m_id: int
    aruco_dict_id: int


def marker_id_to_str(m: MarkerID) -> str:
    """Return a human-readable string for a MarkerID."""
    from . import aruco  # noqa: PLC0415

    return f"{m.m_id} ({aruco.dict_id_to_str[m.aruco_dict_id]})"


def _serialize_marker_id(m: MarkerID) -> dict[str, str | int]:
    from . import aruco  # noqa: PLC0415

    return {"m_id": m.m_id, "aruco_dict": aruco.dict_id_to_str[m.aruco_dict_id]}


def _deserialize_marker_id(m: dict[str, str | int]) -> MarkerID:
    from . import aruco  # noqa: PLC0415

    return MarkerID(
        m_id=m["m_id"],
        aruco_dict_id=aruco.str_to_dict_id(m["aruco_dict_id" if "aruco_dict_id" in m else "aruco_dict"]),
    )


json.register_type(json.TypeEntry(MarkerID, "__config.MarkerID__", _serialize_marker_id, _deserialize_marker_id))


class Marker:
    """A detected or defined marker with center, corners, color, and rotation."""

    def __init__(
        self,
        key: int,
        center: np.ndarray,
        corners: list[np.ndarray] | None = None,
        color: str | None = None,
        rot: float = 0.0,
    ) -> None:
        """Initialize a Marker with its key, center position, and optional properties."""
        self.key = key
        self.center = center
        self.corners = corners
        self.color = color
        self.rot = rot

    def __str__(self) -> str:
        """Return a human-readable string describing the marker."""
        return f"[{self.key}]: center @ ({self.center[0]:.2f}, {self.center[1]:.2f}), rot {self.rot:.0f} deg"

    def shift(self, offset: np.ndarray) -> None:
        """Shift marker center and corners by the given offset."""
        self.center += offset
        if self.corners:
            for c in self.corners:
                c += offset  # noqa: PLW2901


def corners_intersection(corners: np.ndarray) -> np.ndarray:
    """Compute the intersection point of the two diagonals of a quadrilateral."""
    line1 = (corners[0], corners[2])
    line2 = (corners[1], corners[3])
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a: tuple[float, float], b: tuple[float, float]) -> float:
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise ValueError("lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y]).astype("float32")


class Pose:
    """Marker pose (rotation and translation vectors) for a single frame."""

    _columns_compressed: typing.ClassVar[dict[str, int]] = {"frame_idx": 1, "R_vec": 3, "T_vec": 3}
    _non_float: typing.ClassVar[dict[str, type]] = {"frame_idx": int}

    def __init__(self, frame_idx: int, R_vec: np.ndarray | None = None, T_vec: np.ndarray | None = None) -> None:
        """Initialize a Pose with frame index and optional rotation/translation vectors."""
        self.frame_idx: int = frame_idx
        self.R_vec: np.ndarray | None = R_vec
        self.T_vec: np.ndarray | None = T_vec

    def pose_successful(self) -> bool:
        """Return True if both rotation and translation vectors are available."""
        return self.R_vec is not None and self.T_vec is not None

    def draw_frame_axis(
        self, frame: np.ndarray, camera_params: ocv.CameraParams, arm_length: float, sub_pixel_fac: int = 8
    ) -> None:
        """Draw the pose coordinate axes on a video frame."""
        if not camera_params.has_intrinsics():
            return
        drawing.opencv_frame_axis(frame, camera_params, self.R_vec, self.T_vec, arm_length, 3, sub_pixel_fac)


def read_dict_from_file(file_name: str | pathlib.Path, episodes: list[list[int]] | None = None) -> dict[int, Pose]:
    """Read marker poses from a TSV file into a dictionary keyed by frame index."""
    return data_files.read_file(file_name, Pose, True, True, False, False, episodes=episodes)[0]


def write_list_to_file(poses: list[Pose], file_name: str | pathlib.Path, skip_failed: bool = False) -> None:
    """Write a list of marker poses to a TSV file."""
    data_files.write_array_to_file(poses, file_name, Pose._columns_compressed, skip_all_nan=skip_failed)


def get_file_name(marker_id: int, aruco_dict_id: int, folder: str | pathlib.Path | None) -> pathlib.Path:
    """Build the TSV file path for a marker's pose data."""
    from . import aruco  # noqa: PLC0415

    file_name = f"{naming.marker_pose_prefix}{aruco.dict_id_to_str[aruco_dict_id]}_{marker_id}.tsv"
    if folder is None:
        return file_name
    folder = pathlib.Path(folder)
    return folder / file_name


def read_dataframe_from_file(marker_id: int, aruco_dict_id: int, folder: str | pathlib.Path) -> pd.DataFrame:
    """Read a marker pose TSV file into a pandas DataFrame."""
    file = get_file_name(marker_id, aruco_dict_id, folder)
    return pd.read_csv(file, sep="\t", dtype=defaultdict(lambda: float, **Pose._non_float))


@typing.overload
def code_for_presence(markers: pd.DataFrame, allow_failed: bool = False) -> pd.DataFrame: ...
@typing.overload
def code_for_presence(
    markers: dict[typing.Any, pd.DataFrame], allow_failed: bool = False
) -> dict[typing.Any, pd.DataFrame]: ...
def code_for_presence(
    markers: pd.DataFrame | dict[typing.Any, pd.DataFrame], allow_failed: bool = False
) -> pd.DataFrame | dict[typing.Any, pd.DataFrame]:
    """Add a boolean presence column to marker DataFrames."""
    if isinstance(markers, dict):
        for i in markers:
            markers[i] = _code_for_presence_impl(markers[i], f"{i}_", allow_failed)
    else:
        markers = _code_for_presence_impl(markers, "", allow_failed)
    return markers


def _code_for_presence_impl(markers: pd.DataFrame, lbl_extra: str, allow_failed: bool = False) -> pd.DataFrame:
    new_col_lbl = f"marker_{lbl_extra}presence"
    markers.insert(
        len(markers.columns),
        new_col_lbl,
        True
        if allow_failed
        else markers[[c for c in markers.columns if c != "frame_idx"]].notna().all(axis="columns"),
    )
    markers = markers[["frame_idx", new_col_lbl]] if "frame_idx" in markers else markers[[new_col_lbl]]
    markers = markers.astype({new_col_lbl: bool})  # ensure the new column is bool
    return markers


def expand_detection(markers: pd.DataFrame, fill_value: typing.Any) -> pd.DataFrame:  # noqa: ANN401
    """Expand a marker detection DataFrame to cover all frames in its range."""
    if "frame_idx" in markers.columns:
        min_fr_idx = markers["frame_idx"].min()
        max_fr_idx = markers["frame_idx"].max()
        new_index = pd.Index(range(min_fr_idx, max_fr_idx + 1), name="frame_idx")
        return markers.set_index("frame_idx").reindex(new_index, fill_value=fill_value).reset_index()
    if markers.index.name != "frame_idx":
        raise ValueError(
            f'It was expected that the name of the index is "frame_idx". It was "{markers.index.name}" instead. This may mean this dataframe does not contain the expected information. Cannot continue.'
        )
    min_fr_idx = markers.index.min()
    max_fr_idx = markers.index.max()
    new_index = pd.Index(range(min_fr_idx, max_fr_idx + 1), name="frame_idx")
    return markers.reindex(new_index, fill_value=fill_value)


def get_appearance_starts_ends(
    m: pd.DataFrame, max_gap_duration: int, min_duration: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find start and end frame indices of marker appearance intervals."""
    vals = np.pad(m["marker_presence"].to_numpy().astype(int), (1, 1), "constant", constant_values=(0, 0))
    d = np.diff(vals)
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    gaps = starts[1:] - ends[:-1]
    # fill gaps in marker detection
    gapi = np.nonzero(gaps <= max_gap_duration)[0]
    starts = np.delete(starts, gapi + 1)
    ends = np.delete(ends, gapi)
    # remove too short
    lengths = ends - starts
    shorti = np.nonzero(lengths < min_duration)[0]
    starts = np.delete(starts, shorti)
    ends = np.delete(ends, shorti)
    # turn first and last frames into frame_idx values
    if "frame_idx" in m.columns:
        return (
            m.loc[starts, "frame_idx"].to_numpy(copy=True),
            m.loc[ends - 1, "frame_idx"].to_numpy(copy=True),
        )  # NB: -1 so that ends point to last frame during which marker was last seen (and to not index out of the array)
    if m.index.name == "frame_idx":
        return m.index[starts].to_numpy(copy=True), m.index[ends - 1].to_numpy(copy=True)
    return None


def get_sequence_interval(
    starts: dict[MarkerID, list[int]],
    ends: dict[MarkerID, list[int]],
    pattern: list[MarkerID],
    max_intermarker_gap_duration: int,
    side: str = "start",
) -> np.ndarray:
    """Find intervals matching a marker sequence pattern with bounded inter-marker gaps."""
    pairs: list[tuple[int, int]] = []
    for i in range(len(ends[pattern[0]])):
        end_idx = i
        for j in range(len(pattern) - 1):
            if end_idx is None:
                break
            end = ends[pattern[j]][end_idx]
            gaps = starts[pattern[j + 1]] - end
            end_idx = get_smallest_gap_end(gaps, max_intermarker_gap_duration)
        if end_idx is not None:
            pairs.append((starts[pattern[0]][i], ends[pattern[-1]][end_idx]))

    idx = 0 if side == "start" else 1
    return np.array([p[idx] for p in pairs])


def get_smallest_gap_end(gaps: np.ndarray, max_intermarker_gap_duration: int) -> int | None:
    """Return the index of the smallest valid gap, or None if no gap qualifies."""
    gapi = np.nonzero(np.logical_and(gaps >= 0, gaps <= max_intermarker_gap_duration))[0]
    if gapi.size:
        # if there are multiple that qualify, take the smallest gap
        mini = np.argmin(gaps[gapi])
        return gapi[mini]
    return None


def format_duplicate_markers_msg(markers: set[tuple[int, int]]) -> str:
    """Format a human-readable message listing duplicate markers grouped by dictionary."""
    from . import aruco  # noqa: PLC0415

    # NB: input should be dictionary families, not dicts themselves
    # organize per dictionary family
    dict_markers: dict[int, list[int]] = defaultdict(list)
    for m, d in markers:
        dict_markers[d].append(m)
    dict_markers = {d: sorted(dict_markers[d]) for d in dict_markers}
    out = ""
    for i, d in enumerate(dict_markers):
        s = "s" if len(dict_markers[d]) > 1 else ""
        ids = ", ".join(str(x) for x in dict_markers[d])
        d_str, is_family = aruco.family_to_str[d]
        f_str = " family" if is_family else ""
        msg = f"marker{s} {ids} for the {d_str} dictionary{f_str}"
        if i == 0:
            out = msg
        elif i == len(dict_markers) - 1:
            out += f" and {msg}"
        else:
            out += f", {msg}"
    return out


def format_marker_sequence_msg(marker_set: list[tuple[int, int]]) -> str:
    """Format a human-readable message for a sequence of markers."""
    from . import aruco  # noqa: PLC0415

    # NB: input should be dictionary families, not dicts themselves
    # turn each dict into a string/family
    marker_set_str: list[tuple[str, bool, int]] = []
    all_same_family_or_dict = len({x[0] for x in marker_set}) == 1
    marker_set.sort(key=operator.itemgetter(0))
    if not all_same_family_or_dict:
        marker_set.sort(key=operator.itemgetter(1))
    for m in marker_set:
        d_str, is_family = aruco.family_to_str[m[1]]
        marker_set_str.append((d_str, is_family, m[0]))
    if all_same_family_or_dict:
        m_str = ", ".join(str(m[2]) for m in marker_set_str)
        m_str += " from the " + (
            f"{marker_set_str[0][0]} family" if marker_set_str[0][1] else f"{marker_set_str[0][0]} dict"
        )
    else:
        m_str = ", ".join(f"{m[2]} ({m[0] + (' family' if m[1] else '')})" for m in marker_set_str)
    return m_str
