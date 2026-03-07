"""Head-referenced gaze data: storage, I/O, and drawing.

"Head-referenced" means gaze is expressed in the scene camera's coordinate
frame — a 2D pixel position on the scene video (``gaze_pos_vid``) and
optionally a 3D point in the camera's world space (``gaze_pos_3d``).
Per-eye gaze direction and origin vectors are also stored when available.

Multiple timestamp variants support multi-recording synchronization:
``_VOR`` (vestibulo-ocular reflex clock), ``_ref`` (reference recording
clock), and ``_ori`` (original value before clock substitution).
"""

import pathlib
from typing import ClassVar

import numpy as np

from . import data_files, drawing, ocv, transforms


class Gaze:
    """A single head-referenced gaze sample.

    Each instance holds one row from a ``gazeData.tsv`` file: a 2D gaze
    position on the scene video, optional 3D gaze point, and optional
    per-eye direction/origin vectors. Class attributes (``_columns_compressed``,
    ``_non_float``, ``_columns_optional``) define the TSV schema used by
    ``data_files.read_file`` and ``data_files.write_array_to_file``.

    """

    _columns_compressed: ClassVar[dict[str, int]] = {
        "timestamp": 1,
        "timestamp_VOR": 1,
        "timestamp_ref": 1,
        "frame_idx": 1,
        "frame_idx_VOR": 1,
        "frame_idx_ref": 1,
        "gaze_pos_vid": 2,
        "gaze_pos_3d": 3,
        "gaze_dir_l": 3,
        "gaze_ori_l": 3,
        "gaze_dir_r": 3,
        "gaze_ori_r": 3,
    }
    _non_float: ClassVar[dict[str, type]] = {"frame_idx": int, "frame_idx_VOR": int, "frame_idx_ref": int}
    _columns_optional: ClassVar[list[str]] = ["timestamp_VOR", "frame_idx_VOR", "timestamp_ref", "frame_idx_ref"]

    def __init__(
        self,
        timestamp: float,
        frame_idx: int,
        gaze_pos_vid: np.ndarray,
        timestamp_ori: float | None = None,
        frame_idx_ori: int | None = None,
        timestamp_VOR: float | None = None,
        frame_idx_VOR: int | None = None,
        timestamp_ref: float | None = None,
        frame_idx_ref: int | None = None,
        gaze_pos_3d: np.ndarray | None = None,
        gaze_dir_l: np.ndarray | None = None,
        gaze_ori_l: np.ndarray | None = None,
        gaze_dir_r: np.ndarray | None = None,
        gaze_ori_r: np.ndarray | None = None,
    ) -> None:
        """Create a gaze sample.

        Args:
            timestamp: Primary timestamp (may be overwritten with a
                preferred clock variant by ``data_files.read_file``).
            frame_idx: Primary frame index in the scene video.
            gaze_pos_vid: 2D gaze position in scene video pixels ``[x, y]``.
            timestamp_ori: Original timestamp before clock substitution.
            frame_idx_ori: Original frame index before clock substitution.
            timestamp_VOR: Timestamp on the VOR (vestibulo-ocular reflex) clock.
            frame_idx_VOR: Frame index on the VOR clock.
            timestamp_ref: Timestamp on the reference recording's clock.
            frame_idx_ref: Frame index in the reference recording's video.
            gaze_pos_3d: 3D gaze point in camera world space ``[x, y, z]``.
            gaze_dir_l: Left eye gaze direction unit vector.
            gaze_ori_l: Left eye gaze origin (eye position in 3D).
            gaze_dir_r: Right eye gaze direction unit vector.
            gaze_ori_r: Right eye gaze origin (eye position in 3D).

        """
        self.timestamp: float = timestamp
        self.frame_idx: int = frame_idx

        # Original values preserved before data_files.read_file overwrites
        # the main timestamp/frame_idx with a preferred clock variant
        self.timestamp_ori: float = timestamp_ori
        self.frame_idx_ori: int = frame_idx_ori
        # Alternative clock sources for multi-recording synchronization
        self.timestamp_VOR: float = timestamp_VOR
        self.frame_idx_VOR: int = frame_idx_VOR
        self.timestamp_ref: float = timestamp_ref
        self.frame_idx_ref: int = frame_idx_ref

        self.gaze_pos_vid: np.ndarray = gaze_pos_vid
        self.gaze_pos_3d: np.ndarray = gaze_pos_3d
        self.gaze_dir_l: np.ndarray = gaze_dir_l
        self.gaze_ori_l: np.ndarray = gaze_ori_l
        self.gaze_dir_r: np.ndarray = gaze_dir_r
        self.gaze_ori_r: np.ndarray = gaze_ori_r

    def draw(
        self,
        img: np.ndarray,
        camera_params: ocv.CameraParams | None = None,
        sub_pixel_fac: int = 1,
        clr: tuple[int, int, int] = (0, 255, 0),
        draw_3d_gaze_point: bool = True,
        world_clr: tuple[int, int, int] = (0, 255, 255),
        radius: int = 8,
        world_radius: int = 5,
        thickness: int = 2,
        world_thickness: int = -1,
    ) -> None:
        """Draw the gaze point on an image, optionally including the 3D projection.

        The 2D gaze point (``gaze_pos_vid``) is always drawn. If a 3D gaze
        point and camera intrinsics are available, the 3D point is also
        projected onto the image. These two may differ — e.g. the AdHawk
        MindLink applies a parallax correction using vergence that shifts the
        2D point relative to the naively projected 3D point.

        Args:
            img: Image to draw on (modified in place).
            camera_params: Camera intrinsics/extrinsics for 3D projection.
            sub_pixel_fac: Sub-pixel rendering factor.
            clr: BGR color for the 2D gaze circle.
            draw_3d_gaze_point: Whether to also draw the projected 3D point.
            world_clr: BGR color for the 3D gaze circle.
            radius: Radius of the 2D gaze circle.
            world_radius: Radius of the 3D gaze circle.
            thickness: Line thickness for the 2D circle (-1 = filled).
            world_thickness: Line thickness for the 3D circle (-1 = filled).

        """
        drawing.opencv_circle(img, self.gaze_pos_vid, radius, clr, thickness, sub_pixel_fac)
        if (
            draw_3d_gaze_point
            and self.gaze_pos_3d is not None
            and camera_params is not None
            and camera_params.has_intrinsics()
        ):
            # Project the 3D world-space gaze point onto the 2D image plane
            a = transforms.project_points(
                np.array(self.gaze_pos_3d).reshape(1, 3),
                camera_params,
                rot_vec=camera_params.rotation_vec,
                trans_vec=camera_params.position,
            ).flatten()
            drawing.opencv_circle(img, a, world_radius, world_clr, world_thickness, sub_pixel_fac)


def read_dict_from_file(
    file_name: str | pathlib.Path, episodes: list[list[int]] | None = None, ts_column_suffixes: list[str] | None = None
) -> tuple[dict[int, list[Gaze]], int]:
    """Read head-referenced gaze data from a TSV file.

    Args:
        file_name: Path to the ``gazeData.tsv`` file.
        episodes: Optional ``[[start, end], ...]`` frame ranges to keep.
        ts_column_suffixes: Preferred timestamp column suffixes, in order.

    Returns:
        Tuple of (dict mapping frame index to list of ``Gaze`` samples,
        max frame index).

    """
    return data_files.read_file(
        file_name, Gaze, False, False, True, True, episodes=episodes, ts_fridx_field_suffixes=ts_column_suffixes
    )


def write_dict_to_file(
    gazes: list[Gaze] | dict[int, list[Gaze]], file_name: str | pathlib.Path, skip_missing: bool = False
) -> None:
    """Write head-referenced gaze data to a TSV file.

    Args:
        gazes: Gaze samples as a flat list or dict keyed by frame index.
        file_name: Output TSV file path.
        skip_missing: If True, drop rows where all gaze vectors are NaN.

    """
    data_files.write_array_to_file(
        gazes, file_name, Gaze._columns_compressed, Gaze._columns_optional, skip_all_nan=skip_missing
    )
