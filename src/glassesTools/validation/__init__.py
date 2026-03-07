"""Validation plane setup and dynamic marker handling."""

import math
import pathlib
import typing
from collections import defaultdict

import pandas as pd

from .. import aruco
from .. import marker as _marker
from .. import plane as _plane
from . import config
from . import default_poster as default_poster
from . import dynamic as dynamic


class Plane(_plane.TargetPlane):
    """A validation plane with targets and optional dynamic markers."""

    def __init__(
        self,
        config_dir: str | pathlib.Path | None,
        validation_config: dict[str, typing.Any] | None = None,
        is_dynamic: bool = False,
        **kwarg: typing.Any,
    ) -> None:
        """Initialize validation plane from config directory.

        If config_dir is None, the default config will be used.

        Args:
            config_dir: Path to directory containing validation config files,
                or None to use the default config.
            validation_config: Pre-loaded validation config dict. If None,
                loaded from config_dir.
            is_dynamic: Whether to load dynamic marker definitions from the
                target positions file.
            **kwarg: Additional keyword arguments passed to ``TargetPlane``.

        """
        if config_dir is not None:
            config_dir = pathlib.Path(config_dir)

        # get validation config, if needed
        if validation_config is None:
            validation_config = config.get_validation_setup(config_dir)
        self.config = validation_config

        # get marker width
        if self.config["mode"] == "deg":
            # 1° visual angle at viewing distance (cm) converted to mm
            self.cell_size_mm = 2.0 * math.tan(math.radians(0.5)) * self.config["distance"] * 10
        else:
            self.cell_size_mm = 10  # 1cm

        # get board size
        plane_size = _plane.Coordinate(
            self.config["gridCols"] * self.cell_size_mm, self.config["gridRows"] * self.cell_size_mm
        )

        # get targets first, so that any dynamic markers can be split off and then the rest passed to base class
        self.targets: dict[int, _marker.Marker] = {}
        self.dynamic_markers: dict[
            int, tuple[int, int]
        ] = {}  # {marker ID: (target ID, marker_N column in target file)} (keep latter around for good error reporting)
        self._dynamic_markers_cache: dict[int, _marker.MarkerID] | None = (
            None  # different format, for efficient return from get_marker_ids()
        )
        targets, origin = self._get_targets(config_dir, self.config, is_dynamic)

        # call base class
        markers = config.get_markers(config_dir, self.config["markerPosFile"])
        if "ref_image_store_path" not in kwarg:
            kwarg["ref_image_store_path"] = None
        super().__init__(
            markers,
            targets,
            self.config["markerSide"],
            plane_size,
            self.config["arucoDictionary"],
            self.config["markerBorderBits"],
            self.cell_size_mm,
            "mm",
            ref_image_size=self.config["referencePosterSize"],
            min_num_markers=self.config["minNumMarkers"],
            **kwarg,
        )

        # set center
        self.set_origin(origin)

    def _get_targets(
        self,
        config_dir: str | pathlib.Path | None,
        validation_setup: dict[str, typing.Any],
        is_dynamic: bool,
    ) -> tuple[pd.DataFrame, _plane.Coordinate]:
        """Load target positions and optional dynamic marker mappings.

        Poster space: (0,0) is origin (might be center target), (-,-)
        bottom left.

        Args:
            config_dir: Path to validation config directory.
            validation_setup: The validation config dict.
            is_dynamic: Whether to extract dynamic marker columns.

        Returns:
            A tuple of (targets DataFrame scaled to mm, origin coordinate).

        """
        # read in target positions
        targets = config.get_targets(config_dir, validation_setup["targetPosFile"])
        if targets is not None:
            targets_center = targets[["x", "y"]] * self.cell_size_mm
            if is_dynamic:
                # split of columns indicating markers that signal appearance of a target
                markers = pd.concat([targets.pop(c) for c in targets.columns if c.startswith("marker_")], axis=1)
            origin = _plane.Coordinate(
                *targets_center.loc[validation_setup["centerTarget"]].to_numpy()
            )  # NB: need origin in scaled space
            # load with dynamic markers, if any
            if is_dynamic:
                marker_columns = {c: int(c.removeprefix("marker_")) for c in markers}

                def _store_markers(r: pd.Series) -> None:
                    for c, col_idx in marker_columns.items():
                        self.dynamic_markers[int(r[c])] = (int(r.name), col_idx)

                markers.apply(_store_markers, axis=1)
        else:
            self.dynamic_markers.clear()
            origin = _plane.Coordinate(0.0, 0.0)
        self._dynamic_markers_cache = None
        return targets, origin

    def get_marker_ids(self) -> dict[str | int, list[_marker.MarkerID]]:
        """Return all marker IDs including dynamic markers.

        Returns:
            A dict mapping marker group keys to lists of MarkerID objects,
            combining the base plane markers with any dynamic markers.

        """
        if self._dynamic_markers_cache is None:
            self._dynamic_markers_cache = defaultdict(list)
            # {marker ID: (target ID, marker_N column in target file)} -> {marker_N column in target file: [(marker_id, aruco_dict)]}
            for m in self.dynamic_markers:
                self._dynamic_markers_cache[self.dynamic_markers[m][1]].append(_marker.MarkerID(m, self.aruco_dict_id))
        return super().get_marker_ids() | self._dynamic_markers_cache

    def is_dynamic(self) -> bool:
        """Return whether this plane uses dynamic markers."""
        return bool(self.dynamic_markers)

    def get_dynamic_marker_setup(self) -> aruco.MarkerSetup:
        """Return the ArUco marker setup for detecting dynamic markers."""
        return aruco.MarkerSetup(aruco_detector_params={"markerBorderBits": self.marker_border_bits}, detect_only=True)
