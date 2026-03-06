"""Plane and target-plane definitions for ArUco marker-based gaze mapping."""

from __future__ import annotations

import math
import pathlib
import typing

import cv2
import numpy as np
import pandas as pd
from matplotlib import colors

from . import data_files, drawing, marker, transforms

if typing.TYPE_CHECKING:
    from . import aruco


class Coordinate(typing.NamedTuple):
    """2D coordinate with x and y components."""

    x: float = 0.0
    y: float = 0.0


class Plane:
    """A planar surface defined by ArUco markers, used for gaze projection."""

    default_ref_image_name = "reference_image.png"

    def __init__(
        self,
        markers: str | pathlib.Path | pd.DataFrame,
        marker_size: float,
        plane_size: Coordinate,
        aruco_dict_id: int = cv2.aruco.DICT_4X4_250,
        marker_border_bits: int = 1,
        pos_size_scale_fac: float = 1.0,
        unit: str | None = None,
        package_to_read_from: str | None = None,
        ref_image_store_path: str | pathlib.Path | None = None,
        ref_image_size: int = 1920,
        min_num_markers: int = 3,
    ) -> None:
        """Initialize a plane from marker positions, size, and ArUco dictionary settings."""
        self.marker_size = marker_size * pos_size_scale_fac
        # marker positions
        self.markers: dict[int, marker.Marker] = {}
        self._all_marker_ids: list[int] = []
        self.plane_size = plane_size
        self.bbox: list[float] = [0.0, 0.0, self.plane_size.x, self.plane_size.y]
        self._origin: Coordinate = Coordinate(0.0, 0.0)

        # marker specs
        self.aruco_dict_id = aruco_dict_id
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_id)
        self.marker_border_bits = marker_border_bits
        self.unit = unit

        # processing specs
        self.min_num_markers = min_num_markers

        # prep markers
        self._load_markers(markers, pos_size_scale_fac, package_to_read_from)

        # get reference image of plane
        if ref_image_store_path:
            ref_image_store_path = pathlib.Path(ref_image_store_path)

        # get image (always create reference image, to be safe (settings may have changed))
        img = self._store_reference_image(ref_image_store_path, ref_image_size)

        self._ref_image_size = ref_image_size
        self._ref_image_cache: dict[int, np.ndarray] = {ref_image_size: img}

    def set_origin(self, origin: Coordinate) -> None:
        """Set the plane origin, shifting all markers and the bounding box accordingly."""
        offset = np.array(origin) - self._origin

        for i in self.markers:
            self.markers[i].shift(-offset)

        self.bbox[0] -= offset[0]
        self.bbox[2] -= offset[0]
        self.bbox[1] -= offset[1]
        self.bbox[3] -= offset[1]

        self._origin = origin

    def get_ref_image(self, im_size: int | None = None, as_rgb: bool = False) -> np.ndarray:
        """Return a copy of the reference image, optionally resized or in RGB."""
        if im_size is None:
            im_size = self._ref_image_size
        if not isinstance(im_size, int):
            raise TypeError(f"width input should be an int, not {type(im_size)}")
        # check we have the image, if not, add to cache
        if im_size not in self._ref_image_cache:
            scale = float(im_size) / self._ref_image_size
            self._ref_image_cache[im_size] = cv2.resize(
                self._ref_image_cache[self._ref_image_size], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        # return
        if as_rgb:
            return self._ref_image_cache[im_size][:, :, [2, 1, 0]].copy()
        # OpenCV's BGR
        return self._ref_image_cache[im_size].copy()

    def draw(
        self,
        img: np.ndarray,
        x: float,
        y: float,
        sub_pixel_fac: int = 1,
        color: tuple[int, ...] | None = None,
        size: int = 6,
    ) -> None:
        """Draw a gaze point on the plane reference image."""
        if not math.isnan(x):
            xy = transforms.to_image_pos(x, y, self.bbox, [img.shape[1], img.shape[0]])
            if color is None:
                drawing.opencv_circle(img, xy, 8, (0, 255, 0), -1, sub_pixel_fac)
                color = (0, 0, 0)
            drawing.opencv_circle(img, xy, size, color, -1, sub_pixel_fac)

    def _load_markers(
        self, markers: str | pathlib.Path | pd.DataFrame, pos_size_scale_fac: float, package_to_read_from: str | None
    ) -> None:
        from . import aruco  # noqa: PLC0415

        # read in aruco marker positions
        if isinstance(markers, pd.DataFrame):
            marker_pos = markers
        else:
            marker_pos = data_files.read_coord_file(markers, package_to_read_from)
        if marker_pos is None:
            raise RuntimeError(
                f"No markers could be read from the file {markers}, check it exists and contains markers"
            )

        # keep track of all IDs so we can check for duplicates
        self._all_marker_ids = marker_pos.index.to_list()
        marker_dict_size = aruco.get_dict_size(self.aruco_dict_id)
        for m_id in self._all_marker_ids:
            if m_id >= marker_dict_size:
                raise ValueError(
                    f"This plane is set up using the dictionary {aruco.dict_id_to_str[self.aruco_dict_id]} which only has {marker_dict_size} markers, which means that valid IDs are 0-{marker_dict_size - 1}. However, this plane is configured to contain a marker number {m_id} that is not a valid marker for this dictionary."
                )

        # turn into marker objects
        marker_pos.x *= pos_size_scale_fac
        marker_pos.y *= pos_size_scale_fac

        # determine default marker size
        half_size_default = self.marker_size / 2.0

        for idx, row in marker_pos.iterrows():
            c = row[["x", "y"]].to_numpy(copy=True)
            # rotate markers (negative because plane coordinate system)
            rot = row[["rotation_angle"]].to_numpy()[0] if "rotation_angle" in row else 0.0
            rotr = -math.radians(rot)
            rot_mat = np.array([[math.cos(rotr), math.sin(rotr)], [-math.sin(rotr), math.cos(rotr)]])
            # get marker size
            half_size = row[["size"]].to_numpy()[0] / 2.0 * pos_size_scale_fac if "size" in row else half_size_default
            # top left first, and clockwise: same order as detected ArUco marker corners
            tl = c + np.matmul(rot_mat, np.array([-half_size, -half_size]))
            tr = c + np.matmul(rot_mat, np.array([half_size, -half_size]))
            br = c + np.matmul(rot_mat, np.array([half_size, half_size]))
            bl = c + np.matmul(rot_mat, np.array([-half_size, half_size]))

            self.markers[idx] = marker.Marker(idx, c, corners=[tl, tr, br, bl], rot=rot)

    def get_marker_ids(self) -> dict[str | int, list[marker.MarkerID]]:
        """Return marker IDs for this plane grouped by plane name."""
        return {"plane": [marker.MarkerID(m_id, self.aruco_dict_id) for m_id in self._all_marker_ids]}

    def get_aruco_board(self) -> cv2.aruco.Board:
        """Build an OpenCV ArUco Board from this plane's markers."""
        from . import aruco  # noqa: PLC0415

        board_corner_points = []
        ids = []
        for key in self.markers:
            ids.append(key)
            marker_corner_points = np.vstack(self.markers[key].corners).astype("float32")
            board_corner_points.append(marker_corner_points)
        return aruco.create_board(board_corner_points, ids, self.aruco_dict)

    def get_plane_setup(self) -> aruco.PlaneSetup:
        """Return an ArUco PlaneSetup for this plane."""
        from . import aruco  # noqa: PLC0415

        return aruco.PlaneSetup(
            plane=self,
            aruco_detector_params={"markerBorderBits": self.marker_border_bits},
            min_num_markers=self.min_num_markers,
        )

    def _store_reference_image(self, path: pathlib.Path, im_size: int) -> np.ndarray:
        # get image with markers
        bbox_extents = [
            self.bbox[2] - self.bbox[0],
            math.fabs(self.bbox[3] - self.bbox[1]),
        ]  # math.fabs to deal with bboxes where (-,-) is bottom left
        aspect_ratio = bbox_extents[0] / bbox_extents[1]
        if aspect_ratio > 1:
            width = im_size
            height = math.ceil(im_size / aspect_ratio)
        else:
            width = math.ceil(im_size * aspect_ratio)
            height = im_size

        img = np.zeros((height, width), np.uint8)
        img[:] = 255
        # collect all markers
        corner_points = []
        ids = []
        # for checking if markers fit
        x_margin = bbox_extents[0] / width / 5  # ignore .2 pixel or less
        y_margin = bbox_extents[1] / height / 5  # ignore .2 pixel or less
        for key in self.markers:
            ids.append(key)
            corners = np.vstack(self.markers[key].corners).astype("float32")
            # check we're on the plane
            if (
                np.any(corners[:, 0] < -x_margin)
                or np.any(corners[:, 0] > self.plane_size.x + x_margin)
                or np.any(corners[:, 1] < -y_margin)
                or np.any(corners[:, 1] > self.plane_size.y + y_margin)
            ):
                center = ", ".join(f"{v:.4f}" for v in self.markers[key].center)
                corners = [", ".join(f"{v:.4f}" for v in c) for c in corners]
                plane_corners = [", ".join(f"{v:.4f}" for v in c) for c in (self.bbox[:2], self.bbox[2:])]
                raise ValueError(
                    f"Marker {key} with center positioned at ({center}), size {self.marker_size:.4f} and rotation {self.markers[key].rot:.1f} deg would\nhave its corners at ({corners[0]}), ({corners[1]}), ({corners[2]}), and ({corners[3]}),\nwhich is outside the defined plane which ranges from ({plane_corners[0]}) to ({plane_corners[1]}). Ensure all\nsizes and positions are in the same unit (e.g. mm) and check the marker position csv file, marker size and plane size."
                )
            corner_points.append(corners)

        # get info about marker positions on the board
        corner_points = np.dstack(corner_points)
        corner_points -= np.expand_dims(np.array([self.bbox[:2]]), 2).astype("float32")

        # get position and size of marker in the generated image
        corner_points[:, 0, :] = corner_points[:, 0, :] / bbox_extents[0] * float(img.shape[1])
        corner_points[:, 1, :] = corner_points[:, 1, :] / bbox_extents[1] * float(img.shape[0])

        # get marker size
        pix_sz = np.vstack((
            np.hypot(corner_points[0, 0, :] - corner_points[1, 0, :], corner_points[0, 1, :] - corner_points[1, 1, :]),
            np.hypot(corner_points[1, 0, :] - corner_points[2, 0, :], corner_points[1, 1, :] - corner_points[2, 1, :]),
        )).T
        # marker should be square
        pix_sz = np.round(np.min(pix_sz, 1)).astype("int")

        # place markers
        for i, sz, pos in zip(ids, pix_sz, np.moveaxis(corner_points, -1, 0), strict=True):
            # make marker
            marker_image = np.zeros((sz, sz), dtype=np.uint8)
            marker_image = self.aruco_dict.generateImageMarker(i, sz, marker_image, self.marker_border_bits)

            # put in image
            if pos[0, 1] == pos[1, 1] and pos[1, 0] == pos[2, 0] and pos[0, 0] < pos[1, 0]:
                # marker is aligned to image axes and not rotated, just blit
                ori = np.round(pos[0, :]).astype("int")
                img[ori[1] : ori[1] + sz, ori[0] : ori[0] + sz] = marker_image
                continue

            # set up affine transformation for placing marker in image
            in_corners = np.array([
                [-0.5, -0.5],
                [marker_image.shape[1] - 0.5, -0.5],
                [marker_image.shape[1] - 0.5, marker_image.shape[0] - 0.5],
            ]).astype("float32")
            transformation = cv2.getAffineTransform(in_corners, pos[:3, :])

            # perform affine transformation (i.e. rotate marker)
            img = cv2.warpAffine(
                marker_image,
                transformation,
                (img.shape[1], img.shape[0]),
                img,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT,
            )

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if path:
            cv2.imwrite(path, img)

        return img


class TargetPlane(Plane):
    """A plane with target points for gaze validation or calibration."""

    def __init__(
        self,
        markers: str | pathlib.Path | pd.DataFrame,
        targets: str | pathlib.Path | pd.DataFrame,
        marker_size: float,
        plane_size: Coordinate,
        aruco_dict_id: int = cv2.aruco.DICT_4X4_250,
        marker_border_bits: int = 1,
        pos_size_scale_fac: float = 1.0,
        unit: str | None = None,
        package_to_read_from: str | None = None,
        ref_image_store_path: str | pathlib.Path | None = None,
        ref_image_size: int = 1920,
        min_num_markers: int = 3,
    ) -> None:
        """Initialize a target plane with marker and target positions."""
        # get targets first, so that they can be drawn on the reference image
        self.targets: dict[int, marker.Marker] = {}
        self._load_targets(targets, pos_size_scale_fac, package_to_read_from)

        # call base class
        super().__init__(
            markers,
            marker_size,
            plane_size,
            aruco_dict_id,
            marker_border_bits,
            pos_size_scale_fac,
            unit,
            package_to_read_from,
            ref_image_store_path,
            ref_image_size,
            min_num_markers,
        )

    def set_origin(self, origin: Coordinate) -> None:
        """Set the plane origin, also shifting all targets."""
        for i in self.targets:
            self.targets[i].shift(-np.array(origin))
        super().set_origin(origin)

    def _load_targets(
        self, targets: str | pathlib.Path | pd.DataFrame, pos_size_scale_fac: float, package_to_read_from: str | None
    ) -> None:
        # read in target positions
        if isinstance(targets, pd.DataFrame):
            target_pos = targets
        else:
            target_pos = data_files.read_coord_file(targets, package_to_read_from)
        if target_pos is None:
            raise RuntimeError(
                f"No targets could be read from the file {targets}, check it exists and contains targets"
            )

        target_pos["center"] = list(target_pos[["x", "y"]].to_numpy() * pos_size_scale_fac)
        target_pos = target_pos.drop([x for x in target_pos.columns if x not in {"center", "color"}], axis=1)
        self.targets = {
            idx: marker.Marker(idx, **kwargs)
            for idx, kwargs in zip(target_pos.index, target_pos.to_dict(orient="records"), strict=False)
        }

    def get_target_ids(self) -> list[int]:
        """Return the list of target IDs."""
        return list(self.targets.keys())

    def _store_reference_image(self, path: pathlib.Path, im_size: int) -> np.ndarray:
        # first call superclass method to generate image without targets
        img = super()._store_reference_image(path, im_size)
        height, width = img.shape[:2]

        # add targets
        sub_pixel_fac = 8  # for sub-pixel positioning
        for key in self.targets:
            # check we're on the plane
            if (
                np.any(self.targets[key].center[0] < 0)
                or np.any(self.targets[key].center[0] > self.plane_size.x)
                or np.any(self.targets[key].center[1] < 0)
                or np.any(self.targets[key].center[1] > self.plane_size.y)
            ):
                center = ", ".join(f"{v:.4f}" for v in self.targets[key].center)
                plane_corners = [", ".join(f"{v:.4f}" for v in c) for c in (self.bbox[:2], self.bbox[2:])]
                raise ValueError(
                    f"Target {key} positioned at ({center}) is outside the defined\nplane which ranges from ({plane_corners[0]}) to ({plane_corners[1]}). Ensure all\nsizes and positions are in the same unit (e.g. mm) and check the target position csv file and plane size."
                )

            # 1. determine position on image
            circle_pos = transforms.to_image_pos(*self.targets[key].center, self.bbox, [width, height])

            # 2. draw
            clr = tuple(
                int(i * 255)
                for i in (colors.to_rgb(self.targets[key].color)[::-1] if self.targets[key].color else (0.0, 0.0, 1.0))
            )  # need BGR color ordering
            drawing.opencv_circle(img, circle_pos, 15, clr, -1, sub_pixel_fac)

        if path:
            cv2.imwrite(path, img)

        return img
