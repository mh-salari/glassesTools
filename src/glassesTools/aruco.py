"""ArUco marker detection, board management, and multi-detector coordination.

Provides three main abstractions:

- ``Detector``: wraps a single OpenCV ``ArucoDetector`` for one ArUco dictionary
  (or family of compatible dictionaries). Handles detection, board refinement,
  duplicate resolution, and visualization.
- ``Manager``: accepts planes and individual markers from the caller, consolidates
  them into a minimal set of ``Detector`` instances (one per dictionary family),
  and ensures each detector runs at most once per frame via caching.
- Module-level helpers for dictionary/family lookups, board creation, detection
  refinement, and duplicate-ID filtering.

ArUco dictionary families (e.g. 4x4_50, 4x4_100, 4x4_250) share the same
marker patterns but with different dictionary sizes. Within a family, a single
detector using the largest dictionary can detect all markers.
"""

import itertools
import pathlib
import typing

import cv2
import numpy as np

from . import annotation, drawing, marker, ocv, plane, pose, transforms

default_dict = cv2.aruco.DICT_4X4_250

dict_id_to_str: dict[int, str] = {
    getattr(cv2.aruco, k): k
    for k in [
        "DICT_4X4_50",
        "DICT_4X4_100",
        "DICT_4X4_250",
        "DICT_4X4_1000",
        "DICT_5X5_50",
        "DICT_5X5_100",
        "DICT_5X5_250",
        "DICT_5X5_1000",
        "DICT_6X6_50",
        "DICT_6X6_100",
        "DICT_6X6_250",
        "DICT_6X6_1000",
        "DICT_7X7_50",
        "DICT_7X7_100",
        "DICT_7X7_250",
        "DICT_7X7_1000",
        "DICT_ARUCO_ORIGINAL",
        "DICT_APRILTAG_16H5",
        "DICT_APRILTAG_25H9",
        "DICT_APRILTAG_36H10",
        "DICT_APRILTAG_36H11",
        "DICT_ARUCO_MIP_36H12",
    ]
}


def str_to_dict_id(aruco_dict_name: str) -> int:
    """Convert an ArUco dictionary name string to its OpenCV integer ID.

    Args:
        aruco_dict_name: Name string like ``"DICT_4X4_250"``.

    Returns:
        The corresponding ``cv2.aruco`` integer constant.

    Raises:
        ValueError: If the name is not a recognized ArUco dictionary.

    """
    if not hasattr(cv2.aruco, aruco_dict_name):
        raise ValueError(f'ArUco dictionary with name "{aruco_dict_name}" is not known.')
    return getattr(cv2.aruco, aruco_dict_name)


# Family grouping: dictionaries in the same family share marker patterns but
# have different dictionary sizes. A larger dictionary can detect all markers
# of smaller ones in the same family.
dict_id_to_family = {
    cv2.aruco.DICT_4X4_50: 0,
    cv2.aruco.DICT_4X4_100: 0,
    cv2.aruco.DICT_4X4_250: 0,
    cv2.aruco.DICT_4X4_1000: 0,
    cv2.aruco.DICT_5X5_50: 1,
    cv2.aruco.DICT_5X5_100: 1,
    cv2.aruco.DICT_5X5_250: 1,
    cv2.aruco.DICT_5X5_1000: 1,
    cv2.aruco.DICT_6X6_50: 2,
    cv2.aruco.DICT_6X6_100: 2,
    cv2.aruco.DICT_6X6_250: 2,
    cv2.aruco.DICT_6X6_1000: 2,
    cv2.aruco.DICT_7X7_50: 3,
    cv2.aruco.DICT_7X7_100: 3,
    cv2.aruco.DICT_7X7_250: 3,
    cv2.aruco.DICT_7X7_1000: 3,
    cv2.aruco.DICT_ARUCO_ORIGINAL: 4,
    cv2.aruco.DICT_APRILTAG_16H5: 5,
    cv2.aruco.DICT_APRILTAG_25H9: 6,
    cv2.aruco.DICT_APRILTAG_36H10: 7,
    cv2.aruco.DICT_APRILTAG_36H11: 8,
    cv2.aruco.DICT_ARUCO_MIP_36H12: 9,
}
# Maps family integer to (display_name, is_family): is_family=True for NxN
# families that group multiple dict sizes, False for standalone dictionaries.
family_to_str = {
    0: ("DICT_4X4", True),
    1: ("DICT_5X5", True),
    2: ("DICT_6X6", True),
    3: ("DICT_7X7", True),
    4: ("DICT_ARUCO_ORIGINAL", False),
    5: ("DICT_APRILTAG_16H5", False),
    6: ("DICT_APRILTAG_25H9", False),
    7: ("DICT_APRILTAG_36H10", False),
    8: ("DICT_APRILTAG_36H11", False),
    9: ("DICT_ARUCO_MIP_36H12", False),
}


class PlaneSetup(typing.TypedDict):
    """Configuration for a plane's ArUco detection setup.

    Attributes:
        plane: The plane defining the marker layout and board geometry.
        aruco_detector_params: Custom OpenCV detector parameter overrides.
        aruco_refine_params: Custom OpenCV refine parameter overrides.
        min_num_markers: Minimum detected markers for a successful plane detection.

    """

    plane: plane.Plane
    aruco_detector_params: dict[str, typing.Any]
    aruco_refine_params: dict[str, typing.Any]
    min_num_markers: int


class MarkerSetup(typing.TypedDict):
    """Configuration for an individual ArUco marker's detection setup.

    Attributes:
        aruco_detector_params: Custom OpenCV detector parameter overrides.
        detect_only: If True, detect without pose estimation.
        size: Physical marker size in world units (used to build 3D object points).

    """

    aruco_detector_params: dict[str, typing.Any]
    detect_only: bool
    size: float


def reduce_to_families(dictionary_ids: list[int]) -> tuple[list[int], dict[int, int]]:
    """Reduce a list of ArUco dictionary IDs to the minimal set needed.

    Groups dictionaries by family and selects the largest from each family.
    A single large dictionary can detect all markers from smaller dictionaries
    in the same family.

    Args:
        dictionary_ids: ArUco dictionary IDs to consolidate.

    Returns:
        Tuple of (needed dictionary IDs, mapping from requested to used ID).

    """
    # deduplicate while preserving input order (set.add returns None → falsy)
    seen: set[int] = set()
    aruco_dicts = [x for x in dictionary_ids if x not in seen and not seen.add(x)]
    # first organize by family
    by_family: dict[int, list[int]] = {}
    for d in aruco_dicts:
        f = dict_id_to_family[d]
        if f not in by_family:
            by_family[f] = []
        by_family[f].append(d)
    # for each family, if there are multiple dicts, get the largest
    needed_dicts = [max(by_family[f], key=get_dict_size) for f in by_family]
    # make a mapping of dictionary (requested) to dictionary (used)
    aruco_dict_mapping = {d: d2 for f, d2 in zip(by_family, needed_dicts, strict=True) for d in by_family[f]}
    return needed_dicts, aruco_dict_mapping


def get_dict_size(dictionary_id: int) -> int:
    """Return the number of markers in the given ArUco dictionary.

    Args:
        dictionary_id: OpenCV ArUco dictionary integer constant.

    Returns:
        Number of marker IDs defined in the dictionary.

    """
    return cv2.aruco.getPredefinedDictionary(dictionary_id).bytesList.shape[0]


def get_marker_image(size: int, m_id: int, ArUco_dict_id: int, marker_border_bits: int) -> np.ndarray | None:
    """Generate and return an ArUco marker image.

    Args:
        size: Output image size in pixels (square).
        m_id: Marker ID to generate.
        ArUco_dict_id: OpenCV ArUco dictionary integer constant.
        marker_border_bits: Width of the marker border in bits.

    Returns:
        Grayscale marker image, or ``None`` if ``m_id`` exceeds the dictionary size.

    """
    if m_id >= get_dict_size(ArUco_dict_id):
        return None
    marker_image = np.zeros((size, size), dtype=np.uint8)
    return cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(ArUco_dict_id), m_id, size, marker_image, marker_border_bits
    )


def deploy_marker_images(
    output_dir: str | pathlib.Path, size: int, ArUco_dict_id: int, marker_border_bits: int = 1
) -> None:
    """Generate and save all marker images for the given ArUco dictionary.

    Writes one PNG per marker ID to ``output_dir``.

    Args:
        output_dir: Directory to write the PNG files into.
        size: Output image size in pixels (square).
        ArUco_dict_id: OpenCV ArUco dictionary integer constant.
        marker_border_bits: Width of the marker border in bits.

    """
    for m_id in range(get_dict_size(ArUco_dict_id)):
        marker_image = get_marker_image(size, m_id, ArUco_dict_id, marker_border_bits)
        if marker_image is not None:
            cv2.imwrite(output_dir / f"{m_id}.png", marker_image)


class Detector:
    """ArUco marker detector for a single dictionary or family of dictionaries.

    Build-up pattern: call ``add_plane`` / ``add_individual_marker`` to register
    targets, then ``create_detector`` to finalize. After that, call
    ``detect_markers`` per frame. Detection parameters from all registered
    planes/markers are merged; conflicting parameters raise an error.

    """

    def __init__(self, dictionary_id: int) -> None:
        """Initialize the detector with the given ArUco dictionary ID.

        Args:
            dictionary_id: OpenCV ArUco dictionary integer constant.

        """
        self.dictionary_id = dictionary_id
        self._family = dict_id_to_family[self.dictionary_id]
        self._is_family = family_to_str[self._family][1]

        # registered targets and their OpenCV representations
        self.planes: dict[str, PlaneSetup] = {}
        self._boards: dict[str, cv2.aruco.Board] = {}
        self.individual_markers: dict[int, MarkerSetup] = {}
        self._indiv_marker_points: dict[int, np.ndarray] = {}

        # marker ID bookkeeping for filtering detected vs expected markers
        self._plane_marker_ids: dict[str, set[int]] = {}
        self._individual_marker_ids: set[int] = set()
        self._all_markers: set[int] = set()

        # accumulated user-specified OpenCV parameter overrides (merged across all targets)
        self._user_detector_params: dict[str, typing.Any] = {}
        self._user_refine_params: dict[str, typing.Any] = {}

        self._det: cv2.aruco.ArucoDetector | None = None

        self._last_detect_output: tuple[dict[str, dict[str]], dict[str], dict[str], list[np.ndarray]] = {}

    def add_plane(self, name: str, setup: PlaneSetup) -> None:
        """Register a plane with this detector.

        Validates dictionary compatibility, merges any custom OpenCV parameters,
        builds the board, and records the plane's marker IDs for later filtering.

        Args:
            name: Unique name for this plane.
            setup: Plane configuration including the ``Plane`` object and parameters.

        """
        self._check_dict(setup["plane"].aruco_dict_id, "plane")
        self.planes[name] = setup
        if "aruco_detector_params" in self.planes[name] and self.planes[name]["aruco_detector_params"]:
            self._update_parameters("detector", self.planes[name]["aruco_detector_params"])
        if "aruco_refine_params" in self.planes[name] and self.planes[name]["aruco_refine_params"]:
            self._update_parameters("refine", self.planes[name]["aruco_refine_params"])
        self._boards[name] = self.planes[name]["plane"].get_aruco_board()

        # only track the "plane" marker set; other groups (e.g. individual
        # markers defined on the plane) are registered separately by the caller
        markers = self.planes[name]["plane"].get_marker_ids()
        for ms in markers:
            if ms != "plane":
                continue
            m_ids = {m.m_id for m in markers[ms]}
            self._all_markers.update(m_ids)
            self._plane_marker_ids[name] = m_ids

    def add_individual_marker(self, mark: marker.MarkerID, setup: MarkerSetup) -> None:
        """Register an individual marker with this detector.

        Validates dictionary compatibility, merges any custom OpenCV parameters,
        and builds 3D object points for pose estimation (unless ``detect_only``).

        Args:
            mark: Marker identifier (ID + dictionary).
            setup: Marker configuration including size and detection parameters.

        """
        self._check_dict(mark.aruco_dict_id, "individual marker")
        self.individual_markers[mark.m_id] = setup
        if (
            "aruco_detector_params" in self.individual_markers[mark.m_id]
            and self.individual_markers[mark.m_id]["aruco_detector_params"]
        ):
            # individual markers don't use board refinement, so only detector params
            self._update_parameters("detector", self.individual_markers[mark.m_id]["aruco_detector_params"])
        self._all_markers.add(mark.m_id)
        self._individual_marker_ids.add(mark.m_id)
        # build 3D object points: square centered at origin on Z=0 plane
        marker_size = (
            self.individual_markers[mark.m_id].get("size", None)
            if not self.individual_markers[mark.m_id].get("detect_only", False)
            else None
        )
        if not marker_size or marker_size < 0.0:
            marker_points = None
        else:
            # corners in OpenCV order: top-left, top-right, bottom-right, bottom-left
            marker_points = np.array([
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ])
        self._indiv_marker_points[mark.m_id] = marker_points

    def _check_dict(self, dict_id: int, what: str) -> None:
        """Validate that ``dict_id`` is compatible with this detector's dictionary.

        For family detectors, the new dict must be in the same family and not
        larger than the detector's dictionary. For standalone dictionaries, it
        must match exactly.

        Args:
            dict_id: OpenCV dictionary ID of the target being registered.
            what: Human-readable label for error messages (e.g. ``"plane"``).

        Raises:
            ValueError: If the dictionary is incompatible.

        """
        if self._is_family:
            family = dict_id_to_family[dict_id]
            if family != self._family:
                raise ValueError(
                    f"The dictionary for this new {what}, {dict_id_to_str[dict_id]}, is not part of the family ({family_to_str[family][0]}) used for this detector. Use dictionary {dict_id_to_str[self.dictionary_id]} or smaller."
                )
            # within a family, OpenCV IDs are sequential by size, so a higher ID
            # means a larger dictionary that this detector can't fully cover
            if dict_id > self.dictionary_id:
                raise ValueError(
                    f"The dictionary for this new {what}, {dict_id_to_str[dict_id]}, contains more markers than the dictionary used for this detector ({dict_id_to_str[self.dictionary_id]}). Use a dictionary with more markers when creating this detector."
                )
        elif dict_id != self.dictionary_id:
            raise ValueError(
                f"The dictionary for this new {what}, {dict_id_to_str[dict_id]}, does not match the dictionary used for this detector ({dict_id_to_str[self.dictionary_id]})."
            )

    def _update_parameters(self, which: str, new_params: dict) -> None:
        """Merge user-supplied OpenCV parameters into the accumulated set.

        Each parameter is validated against the OpenCV class and checked for
        conflicts with previously registered values. Identical values are
        accepted (idempotent), but different values for the same parameter
        raise an error.

        Args:
            which: ``"detector"`` or ``"refine"``.
            new_params: Parameter name-value pairs to merge.

        Raises:
            ValueError: If ``which`` is unknown or a parameter conflicts.
            AttributeError: If a parameter name is not recognized by OpenCV.

        """
        if which == "detector":
            param_dict = self._user_detector_params
            cls = cv2.aruco.DetectorParameters
        elif which == "refine":
            param_dict = self._user_refine_params
            cls = cv2.aruco.RefineParameters
        else:
            raise ValueError(f'parameter type "{which}" not understood')
        for p, val in new_params.items():
            if not hasattr(cls, p):
                raise AttributeError(f"{p} is not a valid parameter for cv2.aruco.{cls.__name__}")
            # same value is fine (idempotent), different value is a conflict
            if p in param_dict and val != param_dict[p]:
                fam_str, is_family = family_to_str[dict_id_to_family[self.dictionary_id]]
                dict_str = f"{fam_str} family" if is_family else f"{dict_id_to_str[self.dictionary_id]} dictionary"
                raise ValueError(
                    f"You have already set the parameter {p} to {param_dict[p]} and are now trying to set it to {val}, in the detector for the {dict_str}. Resolve this conflict by checking this setting for all planes and individual markers using the {dict_str}."
                )
            param_dict[p] = val

    def create_detector(self) -> None:
        """Build and store the OpenCV ``ArucoDetector`` from accumulated parameters.

        Must be called after all planes and individual markers have been
        registered and before the first call to ``detect_markers``.
        """
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # good default, user can override
        refine_params = cv2.aruco.RefineParameters()
        for p in self._user_detector_params:
            setattr(detector_params, p, self._user_detector_params[p])
        for p in self._user_refine_params:
            setattr(refine_params, p, self._user_refine_params[p])

        self._det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(self.dictionary_id), detector_params, refine_params
        )

    def detect_markers(
        self, image: cv2.UMat, camera_params: ocv.CameraParams
    ) -> tuple[dict[str, dict[str]], dict[str], dict[str], list[np.ndarray]]:
        """Run the full detection pipeline on a single frame.

        Pipeline per plane: raw detection → filter to plane's markers →
        resolve duplicate IDs via reprojection → refine (recover missed
        markers from rejected pile). Individual and unexpected markers are
        filtered from the raw detection.

        Args:
            image: Input image (BGR or grayscale).
            camera_params: Camera intrinsics for board refinement and
                duplicate resolution.

        Returns:
            A 4-tuple of (plane detections dict, individual markers dict,
            unexpected markers dict, rejected corner list).

        """
        img_points, ids, rejected_img_points = self._detect_markers(image, self._det)

        # --- per-plane processing ---
        out_planes: dict[str] = {}
        for p in self.planes:
            if ids is not None:
                # keep only markers belonging to this plane
                pl_img_points, pl_ids = filter_detections(img_points, ids, self._plane_marker_ids[p])
                # resolve duplicate IDs (same marker detected multiple times)
                ok, corners_consistent, ids_consistent, rejected_indices = filter_board_duplicates(
                    self._boards[p], pl_img_points, pl_ids, camera_params
                )
                if ok:
                    if rejected_indices:
                        rejected_img_points += tuple(img_points[i] for i in rejected_indices)
                    pl_img_points = corners_consistent
                    pl_ids = ids_consistent

                # try to recover missed markers from the rejected pile
                recovered_ids = None
                if len(pl_ids) > self.planes[p]["min_num_markers"]:
                    pl_img_points, pl_ids, rejected_img_points, recovered_ids = self._refine_detection(
                        image, pl_img_points, pl_ids, rejected_img_points, self._det, self._boards[p], camera_params
                    )

                out_planes[p] = dict(
                    zip(["img_points", "ids", "recovered_ids"], (pl_img_points, pl_ids, recovered_ids), strict=True)
                )
            else:
                out_planes[p] = None

        # --- individual markers: keep only registered ones ---
        out_individual: dict[str] = {}
        out_individual["img_points"], out_individual["ids"] = self._filter_detections(
            img_points, ids, self._individual_marker_ids
        )

        # --- unexpected markers: everything NOT in any registered set ---
        unexpected_markers: dict[str] = {}
        unexpected_markers["img_points"], unexpected_markers["ids"] = self._filter_detections(
            img_points, ids, self._all_markers, keep_expected=False
        )

        self._last_detect_output = (out_planes, out_individual, unexpected_markers, rejected_img_points)
        return self._last_detect_output

    @staticmethod
    def _detect_markers(
        image: cv2.UMat, det: cv2.aruco.ArucoDetector
    ) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
        """Run raw ArUco detection and normalize ``ids`` to ``None`` when empty."""
        img_points, ids, rejected_img_points = det.detectMarkers(image)
        if np.any(ids is None):
            ids = None
        return img_points, ids, rejected_img_points

    @staticmethod
    def _refine_detection(
        image: cv2.UMat,
        detected_corners: list[np.ndarray],
        detected_ids: np.ndarray,
        rejected_corners: list[np.ndarray],
        det: cv2.aruco.ArucoDetector,
        board: cv2.aruco.Board,
        camera_params: ocv.CameraParams,
    ) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray], np.ndarray | None]:
        """Delegate to module-level :func:`refine_detection`."""
        return refine_detection(image, detected_corners, detected_ids, rejected_corners, det, board, camera_params)

    @staticmethod
    def _filter_detections(
        img_points: list[np.ndarray],
        ids: np.ndarray,
        expected_ids: list[np.ndarray],
        keep_expected: bool = True,
    ) -> tuple[tuple[np.ndarray, ...] | list[np.ndarray], np.ndarray | None]:
        """Delegate to module-level :func:`filter_detections`."""
        return filter_detections(img_points, ids, expected_ids, keep_expected)

    def get_matching_image_board_points(
        self, plane_name: str, detect_tuple: tuple | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return matched 3D-2D point pairs for pose estimation of a plane.

        Uses the board's ``matchImagePoints`` to pair detected image corners
        with their known 3D positions on the board.

        Args:
            plane_name: Name of the registered plane.
            detect_tuple: Detection output to use; defaults to the last
                ``detect_markers`` result.

        Returns:
            ``(object_points, image_points)`` arrays, or ``(None, None)`` if
            the plane was not detected or has too few markers.

        """
        if detect_tuple is None:
            detect_tuple = self._last_detect_output
        if (
            plane_name not in detect_tuple[0]
            or detect_tuple[0][plane_name]["ids"] is None
            or not detect_tuple[0][plane_name]["img_points"]
        ):
            return None, None
        obj_p, img_p = self._boards[plane_name].matchImagePoints(
            detect_tuple[0][plane_name]["img_points"], detect_tuple[0][plane_name]["ids"]
        )
        # each marker contributes 4 corner points
        if img_p is None or int(img_p.shape[0] / 4) < self.planes[plane_name]["min_num_markers"]:
            return None, None
        return obj_p, img_p

    def get_individual_marker_points(
        self, marker_id: int, detect_tuple: tuple | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return 3D-2D point pairs for pose estimation of an individual marker.

        Args:
            marker_id: Integer marker ID.
            detect_tuple: Detection output to use; defaults to the last
                ``detect_markers`` result.

        Returns:
            ``(object_points, image_points)`` arrays, or ``(None, None)`` if
            the marker was not detected.

        """
        if detect_tuple is None:
            detect_tuple = self._last_detect_output
        if (
            detect_tuple[1]["ids"] is None
            or not detect_tuple[1]["img_points"]
            or marker_id not in detect_tuple[1]["ids"]
        ):
            return None, None
        img_points = detect_tuple[1]["img_points"][detect_tuple[1]["ids"].flatten().tolist().index(marker_id)]
        return self._indiv_marker_points[marker_id], img_points

    def visualize(
        self,
        frame: np.ndarray,
        detect_tuple: tuple | None = None,
        sub_pixel_fac: int = 8,
        plane_marker_color: tuple[int, int, int] | None = (0, 255, 0),
        recovered_plane_marker_color: tuple[int, int, int] | None = (255, 255, 0),
        individual_marker_color: tuple[int, int, int] | None = (255, 0, 255),
        unexpected_marker_color: tuple[int, int, int] | None = (150, 253, 253),
        rejected_marker_color: tuple[int, int, int] | None = None,
    ) -> None:
        """Draw detected markers on the frame using color-coded borders.

        Each category (plane, recovered, individual, unexpected, rejected) is
        drawn with its own color.  Set a color to ``None`` to skip that
        category.

        Args:
            frame: BGR image to draw on (modified in-place).
            detect_tuple: Detection output to visualize; defaults to the last
                ``detect_markers`` result.
            sub_pixel_fac: Sub-pixel drawing precision factor.
            plane_marker_color: BGR color for plane markers.
            recovered_plane_marker_color: BGR color for board-refinement
                recovered markers.
            individual_marker_color: BGR color for individual markers.
            unexpected_marker_color: BGR color for unexpected markers.
            rejected_marker_color: BGR color for rejected candidates
                (``None`` to hide).

        """
        if detect_tuple is None:
            detect_tuple = self._last_detect_output
        special_highlight = []

        # rejected markers are hidden by default, useful for debugging
        if rejected_marker_color is not None:
            cv2.aruco.drawDetectedMarkers(frame, detect_tuple[3], None, borderColor=rejected_marker_color)

        # plane markers, with recovered markers highlighted in a different color
        if plane_marker_color is not None:
            for p in detect_tuple[0]:
                if not detect_tuple[0][p] or "ids" not in detect_tuple[0][p] or len(detect_tuple[0][p]["ids"]) == 0:
                    continue
                if (
                    recovered_plane_marker_color is not None
                    and detect_tuple[0][p]["recovered_ids"] is not None
                    and len(detect_tuple[0][p]["recovered_ids"]) > 0
                ):
                    special_highlight = [detect_tuple[0][p]["recovered_ids"], recovered_plane_marker_color]
                drawing.aruco_detected_markers(
                    frame,
                    detect_tuple[0][p]["img_points"],
                    detect_tuple[0][p]["ids"],
                    border_color=plane_marker_color,
                    sub_pixel_fac=sub_pixel_fac,
                    special_highlight=special_highlight,
                )
        if (
            individual_marker_color is not None
            and detect_tuple[1]["ids"] is not None
            and len(detect_tuple[1]["ids"]) > 0
        ):
            drawing.aruco_detected_markers(
                frame,
                detect_tuple[1]["img_points"],
                detect_tuple[1]["ids"],
                border_color=individual_marker_color,
                sub_pixel_fac=sub_pixel_fac,
            )
        if (
            unexpected_marker_color is not None
            and detect_tuple[2]["ids"] is not None
            and len(detect_tuple[2]["ids"]) > 0
        ):
            drawing.aruco_detected_markers(
                frame,
                detect_tuple[2]["img_points"],
                detect_tuple[2]["ids"],
                border_color=unexpected_marker_color,
                sub_pixel_fac=sub_pixel_fac,
            )


class Manager:
    """Consolidate planes and individual markers into a minimal set of ArUco detectors.

    Workflow: register planes/markers → ``consolidate_setup()`` →
    ``register_with_estimator()``. The manager ensures each underlying
    ``Detector`` runs at most once per frame (via ``_det_cache``), even if
    multiple planes share the same ArUco family.

    """

    def __init__(self) -> None:
        """Initialize with empty plane and marker registries."""
        # --- registration phase (populated by add_plane / add_individual_marker) ---
        self.planes: dict[str, PlaneSetup] = {}
        self.plane_proc_intervals: dict[str, tuple[annotation.EventType, list[int] | list[list[int]]] | None] = {}
        self._plane_to_detector: dict[str, int] = {}
        self.individual_markers: dict[marker.MarkerID, MarkerSetup] = {}
        self.individual_markers_proc_intervals: dict[
            str, tuple[annotation.EventType, list[int] | list[list[int]]] | None
        ] = {}

        # --- consolidation phase (populated by consolidate_setup) ---
        self._detectors: dict[int, Detector] = {}
        # per-detector cache: dict_id → (frame_idx, detection_result)
        # ensures each detector runs at most once per frame
        self._det_cache: dict[int, tuple[int, tuple]] = {}
        # prevents double-drawing when multiple planes share a detector
        self._last_viz_frame_idx: dict[int, int] = {}

        # visualization colors stored in BGR (OpenCV convention)
        self._plane_marker_color = (0, 255, 0)
        self._recovered_plane_marker_color = (255, 255, 0)
        self._individual_marker_color = (255, 0, 255)
        self._unexpected_marker_color = (128, 255, 255)
        self._rejected_marker_color = None

    def add_plane(
        self,
        plane: str,
        planes_setup: PlaneSetup,
        processing_intervals: tuple[annotation.EventType, list[int] | list[list[int]]] | None = None,
    ) -> None:
        """Register a plane for ArUco detection.

        The plane is stored for later consolidation; no detector is created
        yet.

        Args:
            plane: Unique name for this plane.
            planes_setup: Plane configuration (``Plane`` object + parameters).
            processing_intervals: Optional frame intervals during which this
                plane should be detected.

        Raises:
            ValueError: If the plane name is already registered.

        """
        if plane in self.planes:
            raise ValueError(f'Cannot register the plane "{plane}", it is already registered')
        self.planes[plane] = planes_setup
        self.plane_proc_intervals[plane] = processing_intervals

    def add_individual_marker(
        self,
        mark: marker.MarkerID,
        marker_setup: MarkerSetup,
        processing_intervals: tuple[annotation.EventType, list[int] | list[list[int]]] | None = None,
    ) -> None:
        """Register an individual marker for ArUco detection.

        Args:
            mark: Marker identifier (ID + dictionary).
            marker_setup: Marker configuration.
            processing_intervals: Optional frame intervals during which this
                marker should be detected.

        Raises:
            ValueError: If this marker is already registered.

        """
        if mark in self.individual_markers:
            raise ValueError(
                f"Cannot register the individual marker {marker.marker_id_to_str(mark)}, it is already registered"
            )
        self.individual_markers[mark] = marker_setup
        self.individual_markers_proc_intervals[mark] = processing_intervals

    def set_visualization_colors(
        self,
        plane_marker_color: tuple[int, int, int] | None = (0, 255, 0),
        recovered_plane_marker_color: tuple[int, int, int] | None = (0, 255, 255),
        individual_marker_color: tuple[int, int, int] | None = (255, 0, 255),
        unexpected_marker_color: tuple[int, int, int] | None = (255, 255, 128),
        rejected_marker_color: tuple[int, int, int] | None = None,
    ) -> None:
        """Set visualization colors for marker drawing.

        Colors are provided in RGB order by the caller but stored internally
        in BGR order (OpenCV convention). Pass ``None`` for any category to
        hide it.

        Args:
            plane_marker_color: RGB color for plane markers.
            recovered_plane_marker_color: RGB color for recovered markers.
            individual_marker_color: RGB color for individual markers.
            unexpected_marker_color: RGB color for unexpected markers.
            rejected_marker_color: RGB color for rejected candidates.

        """
        if plane_marker_color is not None:
            plane_marker_color = plane_marker_color[::-1]
        self._plane_marker_color = plane_marker_color
        if recovered_plane_marker_color is not None:
            recovered_plane_marker_color = recovered_plane_marker_color[::-1]
        self._recovered_plane_marker_color = recovered_plane_marker_color
        if individual_marker_color is not None:
            individual_marker_color = individual_marker_color[::-1]
        self._individual_marker_color = individual_marker_color
        if unexpected_marker_color is not None:
            unexpected_marker_color = unexpected_marker_color[::-1]
        self._unexpected_marker_color = unexpected_marker_color
        if rejected_marker_color is not None:
            rejected_marker_color = rejected_marker_color[::-1]
        self._rejected_marker_color = rejected_marker_color

    def consolidate_setup(self, allow_duplicated_markers: bool = False) -> None:
        """Build the minimal set of detectors from all registered planes and markers.

        Groups planes and individual markers by ArUco dictionary family,
        creates one ``Detector`` per family, and validates marker uniqueness.

        Args:
            allow_duplicated_markers: If True, warn instead of raising on
                duplicate marker IDs across planes/markers.

        """
        # validate marker uniqueness: each marker ID must belong to exactly one
        # plane or individual marker registration to avoid ambiguous detections
        all_markers: set[marker.MarkerID] = set()
        err_msg = "Markers are not unique across planes and individual markers"
        for p in self.planes:
            markers = self.planes[p]["plane"].get_marker_ids()
            for ms in markers:
                if ms != "plane":
                    # N.B.: other markers should be registered by caller as individual markers
                    continue
                if overlap := all_markers.intersection(markers[ms]):
                    t_err_msg = f'{err_msg} for plane "{p}", duplicated markers: {marker.format_duplicate_markers_msg({(m.m_id, dict_id_to_family[m.aruco_dict_id]) for m in overlap})}'
                    if allow_duplicated_markers:
                        print(f"Warning: {t_err_msg}")
                    else:
                        raise RuntimeError(t_err_msg)
                all_markers.update(markers[ms])
        for m in self.individual_markers:
            if m in all_markers:
                t_err_msg = f'{err_msg}: individual marker "{m}" is also used elsewhere'
                if allow_duplicated_markers:
                    print(f"Warning: {t_err_msg}")
                else:
                    raise RuntimeError(t_err_msg)
            all_markers.add(m)

        # see for which marker dicts we need detectors to service all these
        # also determine mapping of requested ArUco dicts to these detectors
        needed_dicts, dict_mapping = reduce_to_families({m.aruco_dict_id for m in all_markers})

        # organize planes and individual markers into the dict that will be used for their detection
        planes_organized: dict[int, list[str]] = {d: [] for d in needed_dicts}
        indiv_markers_organized: dict[int, list[marker.MarkerID]] = {d: [] for d in needed_dicts}
        for p in self.planes:
            det_dict = dict_mapping[self.planes[p]["plane"].aruco_dict_id]
            planes_organized[det_dict].append(p)
        for m in self.individual_markers:
            det_dict = dict_mapping[m.aruco_dict_id]
            indiv_markers_organized[det_dict].append(m)

        # make the needed detectors
        self._detectors.clear()
        self._plane_to_detector.clear()
        for d in needed_dicts:
            self._detectors[d] = Detector(d)
            for p in planes_organized[d]:
                self._detectors[d].add_plane(p, self.planes[p])
                self._plane_to_detector[p] = d
            for m in indiv_markers_organized[d]:
                self._detectors[d].add_individual_marker(m, self.individual_markers[m])
            self._detectors[d].create_detector()

    def register_with_estimator(self, estimator: pose.Estimator) -> None:
        """Register all planes and individual markers with a pose estimator.

        Hooks the manager's detection and visualization methods into the
        estimator's per-frame callbacks.

        Args:
            estimator: The pose estimator to register with.

        """
        for p in self.planes:
            estimator.add_plane(
                p,
                self._detect_plane,
                self.plane_proc_intervals[p],
                # visualization callbacks drop the camera_params arg (4th param)
                lambda pn, fi, fr, _: self._visualize_plane(pn, fi, fr),
            )
        for m in self.individual_markers:
            estimator.add_individual_marker(
                m,
                self._detect_individual_marker,
                self.individual_markers_proc_intervals[m],
                lambda k, fi, fr, _: self._visualize_individual_marker(k, fi, fr),
            )

    def _detect_plane(
        self, plane_name: str, frame_idx: int, frame: np.ndarray, camera_parameters: ocv.CameraParams
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Detection callback for a plane — returns 3D-2D point pairs for pose."""
        if plane_name not in self._plane_to_detector:
            raise ValueError(f"The plane {plane_name} is not known")
        aruco_dict_id = self._plane_to_detector[plane_name]
        detect_tuple = self._get_detector_cache(aruco_dict_id, frame_idx, frame, camera_parameters)
        if not detect_tuple[0] or plane_name not in detect_tuple[0] or not detect_tuple[0][plane_name]:
            return None, None
        return self._detectors[aruco_dict_id].get_matching_image_board_points(plane_name, detect_tuple)

    def _detect_individual_marker(
        self, mark: marker.MarkerID, frame_idx: int, frame: np.ndarray, camera_parameters: ocv.CameraParams
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Detection callback for an individual marker — returns 3D-2D point pairs for pose."""
        if mark not in self.individual_markers:
            raise ValueError(f"The individual marker {marker.marker_id_to_str(mark)} is not known")
        detect_tuple = self._get_detector_cache(mark.aruco_dict_id, frame_idx, frame, camera_parameters)
        if not detect_tuple[1] or detect_tuple[1]["ids"] is None or mark.m_id not in detect_tuple[1]["ids"]:
            return None, None
        return self._detectors[mark.aruco_dict_id].get_individual_marker_points(mark.m_id, detect_tuple)

    def _get_detector_cache(
        self, aruco_dict_id: int, frame_idx: int, frame: np.ndarray | None, camera_parameters: ocv.CameraParams | None
    ) -> tuple | None:
        """Return cached detection results, running the detector if needed.

        Each detector runs at most once per frame: if the cached frame_idx
        matches, the stored result is reused. This is key when multiple
        planes share the same detector.
        """
        if aruco_dict_id not in self._det_cache or self._det_cache[aruco_dict_id][0] != frame_idx:
            if frame is None:
                return None
            detect_tuple = self._detectors[aruco_dict_id].detect_markers(frame, camera_parameters)
            self._det_cache[aruco_dict_id] = (frame_idx, detect_tuple)
        return self._det_cache[aruco_dict_id][1]

    def _visualize_plane(self, plane_name: str, frame_idx: int, frame: np.ndarray) -> None:
        """Visualization callback for a plane.

        Draws all marker categories for the plane's detector. Skips if this
        detector was already drawn on the current frame (multiple planes may
        share a detector, but visualization only needs to happen once).
        """
        if plane_name not in self._plane_to_detector:
            raise ValueError(f"The plane {plane_name} is not known")
        aruco_dict_id = self._plane_to_detector[plane_name]
        if aruco_dict_id in self._last_viz_frame_idx and self._last_viz_frame_idx[aruco_dict_id] == frame_idx:
            return
        # frame=None: don't re-run detection, only use cached results
        detect_tuple = self._get_detector_cache(aruco_dict_id, frame_idx, None, None)
        if detect_tuple is not None:
            frame = self._detectors[aruco_dict_id].visualize(
                frame,
                detect_tuple,
                plane_marker_color=self._plane_marker_color,
                recovered_plane_marker_color=self._recovered_plane_marker_color,
                individual_marker_color=self._individual_marker_color,
                unexpected_marker_color=self._unexpected_marker_color,
                rejected_marker_color=self._rejected_marker_color,
            )
            self._last_viz_frame_idx[aruco_dict_id] = frame_idx

    def _visualize_individual_marker(self, mark: marker.MarkerID, frame_idx: int, frame: np.ndarray) -> None:
        """Visualization callback for an individual marker.

        Same deduplication logic as ``_visualize_plane``.
        """
        if mark not in self.individual_markers:
            raise ValueError(f"The individual marker {marker.marker_id_to_str(mark)} is not known")
        if (
            mark.aruco_dict_id in self._last_viz_frame_idx
            and self._last_viz_frame_idx[mark.aruco_dict_id] == frame_idx
        ):
            return
        detect_tuple = self._get_detector_cache(mark.aruco_dict_id, frame_idx, None, None)
        if detect_tuple is not None:
            frame = self._detectors[mark.aruco_dict_id].visualize(
                frame,
                detect_tuple,
                plane_marker_color=self._plane_marker_color,
                recovered_plane_marker_color=self._recovered_plane_marker_color,
                individual_marker_color=self._individual_marker_color,
                unexpected_marker_color=self._unexpected_marker_color,
                rejected_marker_color=self._rejected_marker_color,
            )
            self._last_viz_frame_idx[mark.aruco_dict_id] = frame_idx


def create_board(
    board_corner_points: list[np.ndarray], ids: list[int], ArUco_dict: cv2.aruco.Dictionary
) -> cv2.aruco.Board:
    """Create an ArUco board from corner points, marker IDs, and a dictionary.

    Args:
        board_corner_points: List of (4,2) arrays with 2D corner positions.
        ids: Marker IDs corresponding to each corner set.
        ArUco_dict: OpenCV ArUco dictionary for the board.

    Returns:
        OpenCV ArUco board object.

    """
    # Reshape from list of 2D corner arrays to a single 3D array at Z=0
    board_corner_points = np.dstack(board_corner_points)  # list of 2D arrays -> 3D array
    board_corner_points = np.rollaxis(board_corner_points, -1)  # 4x2xN -> Nx4x2
    board_corner_points = np.pad(
        board_corner_points, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=(0.0, 0.0)
    )  # Nx4x2 -> Nx4x3 (Z=0 for all points)
    return cv2.aruco.Board(board_corner_points, ArUco_dict, np.array(ids))


def refine_detection(
    image: cv2.UMat,
    detected_corners: list[np.ndarray],
    detected_ids: np.ndarray,
    rejected_corners: list[np.ndarray],
    det: cv2.aruco.ArucoDetector,
    board: cv2.aruco.Board,
    camera_parameters: ocv.CameraParams,
) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray], np.ndarray | None]:
    """Refine detected markers using the board layout and camera parameters.

    Uses OpenCV's ``refineDetectedMarkers`` to recover markers from the
    rejected pool if they match expected board positions, and works around
    corner shape bugs in some OpenCV versions.

    Args:
        image: Input image for re-detection.
        detected_corners: Initially detected corner arrays.
        detected_ids: Initially detected marker IDs.
        rejected_corners: Rejected candidate corner arrays.
        det: OpenCV ``ArucoDetector`` to use.
        board: Board defining expected marker positions.
        camera_parameters: Camera intrinsics for refinement.

    Returns:
        Tuple of (refined corners, refined IDs, remaining rejected corners,
        recovered IDs that were newly matched).

    """
    img_points, ids, rejected_img_points, _ = det.refineDetectedMarkers(
        image=image,
        board=board,
        detectedCorners=detected_corners,
        detectedIds=detected_ids,
        rejectedCorners=rejected_corners,
        cameraMatrix=camera_parameters.camera_mtx,
        distCoeffs=camera_parameters.distort_coeffs,
    )
    # workaround: some OpenCV versions return corners as (4,2) instead of (1,4,2)
    if img_points and img_points[0].shape[0] == 4:
        img_points = [np.reshape(c, (1, 4, 2)) for c in img_points]
    if rejected_img_points and rejected_img_points[0].shape[0] == 4:
        rejected_img_points = [np.reshape(c, (1, 4, 2)) for c in rejected_img_points]
    # recovered = IDs in output but not in input (newly matched from rejected pile)
    recovered_ids = None
    if detected_ids is not None and ids is not None:
        recovered_ids = np.array(list(set(ids.flatten()) - set(detected_ids.flatten()))).reshape((-1, 1))

    return img_points, ids, rejected_img_points, recovered_ids


def filter_detections(
    img_points: list[np.ndarray],
    ids: np.ndarray,
    expected_ids: list[np.ndarray],
    keep_expected: bool = True,
) -> tuple[tuple[np.ndarray, ...] | list[np.ndarray], np.ndarray | None]:
    """Filter marker detections to keep or exclude expected IDs.

    Args:
        img_points: Detected corner arrays.
        ids: Corresponding marker IDs.
        expected_ids: Set of IDs to keep (or exclude if *keep_expected* is False).
        keep_expected: If True, keep only expected IDs; if False, keep
            everything except expected IDs.

    Returns:
        Filtered (img_points, ids) tuple.

    """
    if ids is None or not img_points:
        return img_points, ids
    if not keep_expected:
        expected_ids = set(ids.flatten()) - set(expected_ids)
    if not expected_ids:
        # optimization. If output will definitely be empty, return directly
        return (), None
    to_remove = np.where([x not in expected_ids for x in ids.flatten()])[0]
    ids = np.delete(ids, to_remove, axis=0)
    img_points = tuple(v for i, v in enumerate(img_points) if i not in to_remove)
    return img_points, ids


def has_duplicates(ids: np.ndarray) -> bool:
    """Return True if any marker ID appears more than once.

    Args:
        ids: Marker ID array from detection output.

    """
    if ids is None or len(ids) == 0:
        return False
    flat = ids.flatten().astype(int)
    return len(flat) != len(np.unique(flat))


def group_indices_by_id(ids: np.ndarray) -> dict[int, list[int]]:
    """Group detection indices by marker ID.

    Args:
        ids: Marker ID array from detection output.

    Returns:
        Dict mapping each marker ID to the list of indices where it appears.

    """
    id_to_indices: dict[int, list[int]] = {}
    if ids is None or len(ids) == 0:
        return id_to_indices
    for i, mid in enumerate(ids.flatten()):
        id_to_indices.setdefault(int(mid), []).append(i)
    return id_to_indices


def _build_board_objpoints_map(board: cv2.aruco.Board) -> dict[int, np.ndarray]:
    """Map each board marker ID to its known (4,3) 3D corner positions."""
    return {
        int(mid): np.asarray(obj4x3, dtype=np.float32)
        for obj4x3, mid in zip(board.getObjPoints(), board.getIds().flatten(), strict=True)
    }


def _corners_4x2(c: np.ndarray) -> np.ndarray:
    """Normalize a corner array to (4,2) for reprojection error computation.

    OpenCV returns corners in different shapes across versions and API calls:
    ``(4,1,2)``, ``(1,4,2)``, or ``(4,2)``. This function handles all of them.

    Args:
        c: Corner array in any of the above shapes.

    Returns:
        Float32 array of shape ``(4, 2)``.

    Raises:
        ValueError: If the shape cannot be normalized to ``(4, 2)``.

    """
    c = np.asarray(c)
    if c.shape == (4, 2):
        return c.astype(np.float32)
    if c.shape == (4, 1, 2):
        return c.reshape(4, 2).astype(np.float32)
    if c.shape == (1, 4, 2):
        return c.reshape(4, 2).astype(np.float32)
    # Fallback: flatten last two dims to 2 cols if possible
    c2 = c.reshape(-1, 2)
    if c2.shape[0] == 4:
        return c2.astype(np.float32)
    raise ValueError(f"Unexpected corner shape: {c.shape}")


def _mean_corner_error_projected(
    observed_4x2: np.ndarray, projected_4x2: np.ndarray, test_rotations: bool = True
) -> float:
    """Mean L2 error between observed and projected corners.

    When ``test_rotations`` is True, tries all 4 cyclic rotations of the
    observed corners and returns the best (lowest) error.  This guards
    against corner ordering mismatches between detection and projection.

    Args:
        observed_4x2: Detected corners, shape ``(4, 2)``.
        projected_4x2: Reprojected corners, shape ``(4, 2)``.
        test_rotations: Whether to try all 4 corner rotations.

    Returns:
        Mean per-corner L2 distance (pixels).

    """
    if not test_rotations:
        return np.linalg.norm(observed_4x2 - projected_4x2, axis=1).mean()

    best = float("inf")
    for rot in range(4):
        obs_rot = np.roll(observed_4x2, -rot, axis=0)
        err = np.linalg.norm(obs_rot - projected_4x2, axis=1).mean()
        best = min(best, err)
    return best


def _estimate_board_pose(
    board: cv2.aruco.Board, corners_list: list[np.ndarray], ids: np.ndarray, camera_params: ocv.CameraParams
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """Estimate board pose from a set of detected corners via solvePnP.

    Args:
        board: Board defining expected 3D marker positions.
        corners_list: Detected corner arrays.
        ids: Detected marker IDs.
        camera_params: Camera intrinsics.

    Returns:
        ``(ok, rvec, tvec)`` where ``ok`` is False if pose estimation fails.

    """
    obj_p, img_p = board.matchImagePoints(corners_list, ids)
    if len(obj_p) == 0:
        return False, None, None
    retval, rvec, tvec, _ = pose.estimate_pose(obj_p, img_p, camera_params)
    if retval <= 0:
        return False, None, None
    return True, rvec, tvec


def filter_board_duplicates(
    board: cv2.aruco.Board,
    corners: list[np.ndarray],
    ids: np.ndarray,
    camera_params: ocv.CameraParams,
    *,
    min_markers: int = 2,
    max_combinations: int | None = 5000,
    test_corner_rotations: bool = True,
) -> tuple[bool, list[np.ndarray], np.ndarray, list[int]]:
    """Resolve duplicate marker IDs by exhaustive reprojection-error search.

    When the same marker ID is detected multiple times in one frame (e.g.
    a real marker and a false positive), this function tries every combination
    of choosing one detection per duplicated ID, estimates the board pose for
    each, and keeps the combination with the lowest mean reprojection error.
    Single-ID detections are always included.

    Args:
        board: Board defining expected 3D marker positions.
        corners: Detected corner arrays.
        ids: Detected marker IDs.
        camera_params: Camera intrinsics for pose estimation and reprojection.
        min_markers: Minimum detections for a valid pose estimate.
        max_combinations: Safety limit on the Cartesian product of duplicate
            choices.  ``None`` disables the limit.
        test_corner_rotations: Whether to try corner rotations when computing
            reprojection error (guards against corner order mismatches).

    Returns:
        ``(ok, corners_consistent, ids_consistent, rejected_indices)`` where
        ``rejected_indices`` lists the input indices that were NOT selected.

    """
    # --- Early outs ---
    if ids is None or len(ids) == 0:
        return False, [], np.empty((0, 1), dtype=np.int32), []

    # If all IDs are unique, keep all (no pose computation)
    if not has_duplicates(ids):
        kept_indices = list(range(len(ids)))
        corners_consistent = [corners[i] for i in kept_indices]
        ids_consistent = ids.reshape(-1, 1).copy()
        return True, corners_consistent, ids_consistent, []

    if len(corners) != len(ids):
        raise ValueError(f"corners (len={len(corners)}) and ids (len={len(ids)}) mismatch.")

    # Group detections by ID
    id2idx = group_indices_by_id(ids)
    singles: list[int] = []
    dup_groups: list[list[int]] = []
    for idxs in id2idx.values():
        if len(idxs) == 1:
            singles.append(idxs[0])
        else:
            dup_groups.append(idxs)

    # Compute total combinations (product of lengths of duplicate groups)
    num_combinations = 1
    for g in dup_groups:
        num_combinations *= len(g)

    if max_combinations is not None and num_combinations > max_combinations:
        # Too many; caller assumed "few duplicates", so bail rather than blow up.
        return False, [], np.empty((0, 1), dtype=ids.dtype), []

    # Precompute object points per ID
    board_map = _build_board_objpoints_map(board)
    # Prepare an array form of ids for fast slicing
    ids_arr = ids.reshape(-1, 1)

    best_err = float("inf")
    best_indices: list[int] = []

    # Iterate all choices, one index from each duplicate group
    for choice in itertools.product(*dup_groups):
        # Selected detections = singles + one per duplicated ID group
        selected = singles + list(choice)

        if len(selected) < min_markers:
            # Insufficient constraints to estimate pose robustly
            continue

        # Estimate pose for this combination
        sel_corners = [corners[i] for i in selected]
        sel_ids = ids_arr[selected]  # shape (K,1)

        retval, rvec, tvec = _estimate_board_pose(board, sel_corners, sel_ids, camera_params)
        if retval <= 0:
            # Pose failed; skip this combination
            continue

        # Compute mean reprojection error for the selected markers
        # Project each selected marker's 3D corners with the pose and compare to observed
        total_err = 0.0
        total_corners = 0

        for i in selected:
            mid = int(ids_arr[i, 0])
            if mid not in board_map:
                # Board doesn't define this ID; skip (shouldn't happen if detections are on the same board)
                continue
            obj4x3 = board_map[mid]  # (4,3)
            proj4x2 = transforms.project_points(obj4x3, camera_params, rot_vec=rvec, trans_vec=tvec)

            obs4x2 = _corners_4x2(corners[i])
            err = _mean_corner_error_projected(obs4x2, proj4x2, test_rotations=test_corner_rotations)
            total_err += err * 4  # 4 corners
            total_corners += 4

        if total_corners == 0:
            # No valid error; skip
            continue

        mean_err = total_err / total_corners

        # Keep the lowest-error combination (tie-breaker: keep the one with more markers)
        if (mean_err < best_err) or (np.isclose(mean_err, best_err) and len(selected) > len(best_indices)):
            best_err = mean_err
            best_indices = selected

    if not best_indices:
        return False, [], np.empty((0, 1), dtype=ids.dtype), []

    # Prepare outputs (preserve corner shapes and return OpenCV-style ids)
    best_indices = sorted(best_indices)
    corners_consistent = [corners[i] for i in best_indices]
    ids_consistent = ids_arr[best_indices].reshape(-1, 1).astype(ids.dtype, copy=False)

    return True, corners_consistent, ids_consistent, list(set(range(len(ids))) - set(best_indices))
