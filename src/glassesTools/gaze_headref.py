"""Head-referenced gaze data: storage, I/O, and drawing."""

import pathlib
from typing import ClassVar

import numpy as np

from . import data_files, drawing, ocv, transforms


class Gaze:
    """A single head-referenced gaze sample with video and 3-D coordinates.

    Class attributes describe the TSV columns used for storage.
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
        """Initialize a gaze sample from timestamps, frame indices, and gaze vectors."""
        self.timestamp: float = timestamp
        self.frame_idx: int = frame_idx

        # optional timestamps and frame_idxs
        self.timestamp_ori: float = timestamp_ori  # timestamp field can be set to any of these three. Keep here a copy of the original timestamp
        self.frame_idx_ori: int = frame_idx_ori
        self.timestamp_VOR: float = timestamp_VOR
        self.frame_idx_VOR: int = frame_idx_VOR
        self.timestamp_ref: float = timestamp_ref
        self.frame_idx_ref: int = frame_idx_ref  # frameidx _in reference video_ not in this eye tracker's video (unless this is the reference)

        self.gaze_pos_vid: np.ndarray = gaze_pos_vid  # gaze point on the scene video
        self.gaze_pos_3d: np.ndarray = gaze_pos_3d  # gaze point in the world (often binocular gaze point)
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
        """Draw the gaze point on an image, optionally including the 3-D projection."""
        drawing.openCVCircle(img, self.gaze_pos_vid, radius, clr, thickness, sub_pixel_fac)
        # draw 3D gaze point as well, usually coincides with 2D gaze point, but not always. E.g. the Adhawk MindLink may
        # apply a correction for parallax error to the projected gaze point using the vergence signal.
        if (
            draw_3d_gaze_point
            and self.gaze_pos_3d is not None
            and camera_params is not None
            and camera_params.has_intrinsics()
        ):
            a = transforms.project_points(
                np.array(self.gaze_pos_3d).reshape(1, 3),
                camera_params,
                rot_vec=camera_params.rotation_vec,
                trans_vec=camera_params.position,
            ).flatten()
            drawing.openCVCircle(img, a, world_radius, world_clr, world_thickness, sub_pixel_fac)


def read_dict_from_file(
    file_name: str | pathlib.Path, episodes: list[list[int]] | None = None, ts_column_suffixes: list[str] | None = None
) -> tuple[dict[int, list[Gaze]], int]:
    """Read head-referenced gaze data from a TSV file into a dict keyed by frame index."""
    return data_files.read_file(
        file_name, Gaze, False, False, True, True, episodes=episodes, ts_fridx_field_suffixes=ts_column_suffixes
    )


def write_dict_to_file(
    gazes: list[Gaze] | dict[int, list[Gaze]], file_name: str | pathlib.Path, skip_missing: bool = False
) -> None:
    """Write head-referenced gaze data to a TSV file."""
    data_files.write_array_to_file(
        gazes, file_name, Gaze._columns_compressed, Gaze._columns_optional, skip_all_nan=skip_missing
    )
