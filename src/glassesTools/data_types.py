"""Methods for computing angular gaze offsets (how far gaze is from a target point).

Multiple methods are available, ranging from simple (assumed viewing distance +
homography) to precise (per-eye 3D gaze rays with camera pose). The ``DataType``
enum defines these methods, while selection and validation functions handle choosing
the best available method for a given recording.

NB: all ``pose_*`` methods require a calibrated scene camera.
"""

import enum
import math
import typing

import numpy as np

from . import gaze_worldref, json, pose, transforms
from . import utils as _utils


class DataType(_utils.AutoName):
    """Type of data to use for computing angular measures."""

    viewpos_vidpos_homography = enum.auto()  # use homography to map gaze from video to plane, and viewing distance defined in config (combined with the assumptions that the viewing position (eye) is located directly in front of the plane's center and that the plane is oriented perpendicularly to the line of sight) to compute angular measures
    pose_vidpos_homography = (
        enum.auto()
    )  # use homography to map gaze from video to plane, and pose information w.r.t. plane to compute angular measures
    pose_vidpos_ray = enum.auto()  # use camera calibration to map gaze position on scene video to cyclopean gaze vector, and pose information w.r.t. plane to compute angular measures
    pose_world_eye = enum.auto()  # use provided gaze position in world (often a binocular gaze point), and pose information w.r.t. plane to compute angular measures
    pose_left_eye = (
        enum.auto()
    )  # use provided left eye gaze vector, and pose information w.r.t. plane to compute angular measures
    pose_right_eye = (
        enum.auto()
    )  # use provided right eye gaze vector, and pose information w.r.t. plane to compute angular measures
    pose_left_right_avg = (
        enum.auto()
    )  # report average of left (pose_left_eye) and right (pose_right_eye) eye angular values

    def __str__(self) -> str:
        """Return the name so pandas serializes this in a user-friendly way."""
        return self.name


def data_type_val_to_enum_val(x: int | str) -> DataType:
    """Convert an integer or string to the corresponding DataType enum value.

    The integer mapping provides backward compatibility with older config files
    that used numeric identifiers.

    Args:
        x: Integer ID or string name of the data type.

    Returns:
        The matching DataType enum member.

    """
    return _utils.str_int_2_enum_val(
        x,
        DataType,
        {
            1: "viewpos_vidpos_homography",
            2: "pose_vidpos_homography",
            3: "pose_vidpos_ray",
            4: "pose_world_eye",
            5: "pose_left_eye",
            6: "pose_right_eye",
            7: "pose_left_right_avg",
        },
    )


# compatible_reg_names: older JSON files used "glassesValidator.DataQualityType" as the tag
json.register_type(
    json.TypeEntry(
        DataType,
        "__enum.DataType__",
        _utils.enum_val_2_str,
        data_type_val_to_enum_val,
        compatible_reg_names=["glassesValidator.DataQualityType"],
    )
)


def get_explanation(dq: DataType) -> tuple[str, str]:
    """Return a short name and a detailed description for the given data type.

    Used for GUI display and user-facing documentation of each method.

    Args:
        dq: The data type to describe.

    Returns:
        Tuple of (short display name, detailed description string).

    """
    ler_name = "Left eye ray + pose"
    rer_name = "Right eye ray + pose"
    match dq:
        case DataType.viewpos_vidpos_homography:
            return (
                "Homography + view distance",
                (
                    "Use homography to map gaze position from the scene video to "
                    "the validation plane, and use an assumed viewing distance (see "
                    "the project's configuration) to compute angular measures "
                    "in degrees with respect to the scene camera. In this mode, it is "
                    "assumed that the eye is located exactly in front of the center of "
                    "the plane and that the plane is oriented perpendicularly to the "
                    "line of sight from this assumed viewing position."
                ),
            )
        case DataType.pose_vidpos_homography:
            return (
                "Homography + pose",
                (
                    "Use homography to map gaze position from the scene video to "
                    "the validation plane, and use the determined pose of the scene "
                    "camera (requires a calibrated camera) to compute angular "
                    "measures in degrees with respect to the scene camera."
                ),
            )
        case DataType.pose_vidpos_ray:
            return (
                "Video ray + pose",
                (
                    "Use camera calibration to turn gaze position from the scene "
                    "video into a direction vector, and determine gaze position on "
                    "the validation plane by intersecting this vector with the "
                    "plane. Then, use the determined pose of the scene camera "
                    "(requires a calibrated camera) to compute angular "
                    "measures in degrees with respect to the scene camera."
                ),
            )
        case DataType.pose_world_eye:
            return (
                "World gaze position + pose",
                (
                    "Use the gaze position in the world provided by the eye tracker "
                    "(often a binocular gaze point) to determine gaze position on the "
                    "validation plane by turning it into a direction vector with respect "
                    "to the scene camera and intersecting this vector with the plane. "
                    "Then, use the determined pose of the scene camera "
                    "(requires a calibrated camera) to compute angular "
                    "measures in degrees with respect to the scene camera."
                ),
            )
        case DataType.pose_left_eye:
            return (
                ler_name,
                (
                    "Use the gaze direction vector for the left eye provided by "
                    "the eye tracker to determine gaze position on the validation "
                    "plane by intersecting this vector with the plane. "
                    "Then, use the determined pose of the scene camera "
                    "(requires a camera calibration) to compute angular "
                    "measures in degrees with respect to the left eye."
                ),
            )
        case DataType.pose_right_eye:
            return (
                rer_name,
                (
                    "Use the gaze direction vector for the right eye provided by "
                    "the eye tracker to determine gaze position on the validation "
                    "plane by intersecting this vector with the plane. "
                    "Then, use the determined pose of the scene camera "
                    "(requires a camera calibration) to compute angular "
                    "measures in degrees with respect to the right eye."
                ),
            )
        case DataType.pose_left_right_avg:
            return (
                "Average eye rays + pose",
                (
                    "For each time point, take angular offset between the left and "
                    "right gaze positions and the fixation target and average them "
                    "to compute angular measures in degrees. Requires "
                    f"'{ler_name}' and '{rer_name}' to be enabled."
                ),
            )


def get_world_gaze_fields_for_data_type(angle_type: DataType) -> list[str | None]:
    """Return the ``gaze_worldref.Gaze`` field names for a data type.

    Returns a 3-element list: [origin, 3D gaze point, 2D plane point].
    Elements are None when the data type doesn't use that field (e.g. origin
    is None for camera-centric methods where the camera is the implicit origin).

    Args:
        angle_type: The data type to look up.

    Returns:
        List of field name strings (or None) for [origin, 3D point, 2D plane point].

    Raises:
        NotImplementedError: If the data type has no field mapping.

    """
    # field 1: origin of gaze vector (None if scene camera)
    # field 2: 3D gaze point in camera space (None if not available)
    # field 3: 2D gaze point on plane in plane space
    match angle_type:
        case DataType.viewpos_vidpos_homography:
            # from camera perspective, using homography
            # viewpos_vidpos_homography: using assumed viewing distance
            fields = [None, None, "gazePosPlane2D_vidPos_homography"]
        case DataType.pose_vidpos_homography:
            # from camera perspective, using homography
            # pose_vidpos_homography   : using pose info
            fields = [None, "gazePosCam_vidPos_homography", "gazePosPlane2D_vidPos_homography"]
        case DataType.pose_vidpos_ray:
            # from camera perspective, using 3D gaze point ray
            fields = [None, "gazePosCam_vidPos_ray", "gazePosPlane2D_vidPos_ray"]
        case DataType.pose_world_eye:
            # using 3D world gaze position, with respect to eye tracker reference frame's origin
            fields = [None, "gazePosCamWorld", "gazePosPlane2DWorld"]
        case DataType.pose_left_eye:
            fields = ["gazeOriCamLeft", "gazePosCamLeft", "gazePosPlane2DLeft"]
        case DataType.pose_right_eye:
            fields = ["gazeOriCamRight", "gazePosCamRight", "gazePosPlane2DRight"]
        case _:
            raise NotImplementedError(f"Logic for gaze angle type {angle_type} not implemented. Contact developer.")
    return fields


def get_available_data_types(plane_gazes: dict[int, list[gaze_worldref.Gaze]]) -> list[DataType]:
    """Determine which data types have sufficient non-NaN data in the plane gazes.

    Checks each data type's required fields across all gaze samples. A data type
    is available if at least one sample has all required fields present (non-NaN).

    Args:
        plane_gazes: Dict mapping frame indices to lists of gaze records.

    Returns:
        List of data types that have usable data.

    """
    dq_have: list[DataType] = []
    for dq in DataType:
        if dq == DataType.pose_left_right_avg:
            continue  # special case handled below (needs both left and right eye data)
        fields = get_world_gaze_fields_for_data_type(dq)
        have_data = np.vstack(
            tuple(
                [(a := getattr(s, f)) is not None and not np.any(np.isnan(a)) for v in plane_gazes.values() for s in v]
                for f in fields
                if f is not None
            )
        )
        if np.any(np.all(have_data, axis=0)):
            dq_have.append(dq)

    if (DataType.pose_left_eye in dq_have) and (DataType.pose_right_eye in dq_have):
        dq_have.append(DataType.pose_left_right_avg)
    return dq_have


def select_data_types_to_use(
    dq_types: typing.Iterable[DataType] | DataType | str | None,
    dq_have: list[DataType],
    allow_dq_fallback: bool = True,
) -> list[DataType]:
    """Select which data types to use, validating against available types.

    Validates user-requested data types against what's available, converting
    strings to enum values and removing unavailable types (or raising if
    fallback is disabled). If no types remain, falls back with priority:
    ``pose_vidpos_ray`` > ``pose_vidpos_homography`` > ``viewpos_vidpos_homography``.

    Args:
        dq_types: Requested data types (or None for automatic selection).
        dq_have: Available data types from ``get_available_data_types``.
        allow_dq_fallback: If True, silently remove unavailable types.
            If False, raise on unavailable types.

    Returns:
        List of validated data types to use.

    Raises:
        ValueError: If a string doesn't match any known data type.
        TypeError: If an element is neither a DataType nor a string.
        RuntimeError: If a required data type is unavailable and fallback is disabled.

    """
    if dq_types is not None:
        dq_types = [dq_types] if isinstance(dq_types, (DataType, str)) else list(dq_types)

    if dq_types:
        # Iterate in reverse so we can safely delete elements by index
        for i, dq in reversed(list(enumerate(dq_types))):
            if not isinstance(dq, DataType):
                if isinstance(dq, str):
                    if hasattr(DataType, dq):
                        dq_types[i] = getattr(DataType, dq)
                    else:
                        raise ValueError(
                            f"The string '{dq}' is not a known data type. Known types: {[e.name for e in DataType]}"
                        )
                else:
                    raise TypeError(
                        f"The variable 'dq' should be a string with one of the following values: {[e.name for e in DataType]}"
                    )
            if dq_types[i] not in dq_have:
                if allow_dq_fallback:
                    del dq_types[i]
                else:
                    raise RuntimeError(
                        f"Data type {dq} could not be used as its not available for this recording. Available data types: {[e.name for e in dq_have]}"
                    )

        if DataType.pose_left_right_avg in dq_types and (
            (DataType.pose_left_eye not in dq_have) or (DataType.pose_right_eye not in dq_have)
        ):
            if allow_dq_fallback:
                dq_types.remove(DataType.pose_left_right_avg)
            else:
                raise RuntimeError(
                    f"Cannot use the data type {DataType.pose_left_right_avg} because it requires having data types {DataType.pose_left_eye} and {DataType.pose_right_eye} available, but one or both are not available. Available data types: {[e.name for e in dq_have]}"
                )

    if not dq_types:
        if DataType.pose_vidpos_ray in dq_have:
            # highest priority is DataType.pose_vidpos_ray
            dq_types.append(DataType.pose_vidpos_ray)
        elif DataType.pose_vidpos_homography in dq_have:
            # else at least try to use pose (shouldn't occur, if we have pose we have a calibrated camera, which means we should have the above)
            dq_types.append(DataType.pose_vidpos_homography)
        else:
            # else we're down to falling back on an assumed viewing distance
            if DataType.viewpos_vidpos_homography not in dq_have:
                raise RuntimeError(
                    f"Even data type {DataType.viewpos_vidpos_homography} could not be used, bare minimum failed for some weird reason. Contact developer."
                )
            dq_types.append(DataType.viewpos_vidpos_homography)

    return dq_types


def calculate_gaze_angles_to_point(
    plane_gazes: dict[int, list[gaze_worldref.Gaze]],
    poses: dict[int, pose.Pose],
    points: dict[int, np.ndarray],
    d_types: list[DataType],
    points_for_homography: dict[int, np.ndarray] | None = None,
    viewing_distance: float | None = None,
) -> tuple[list[int], np.ndarray, dict[int, dict[DataType, np.ndarray]]]:
    """Compute angular offsets between gaze and target points for each data type.

    For each target point and data type, computes the angular offset decomposed
    into [total, horizontal, vertical] components in degrees.

    Args:
        plane_gazes: Dict mapping frame indices to lists of gaze records.
        poses: Dict mapping frame indices to camera pose objects.
        points: Dict mapping target IDs to 3D world coordinates.
        d_types: Data types to compute angles for.
        points_for_homography: Target positions for homography-based methods
            (2D plane coords, different representation than *points*).
        viewing_distance: Assumed viewing distance in mm (for
            ``viewpos_vidpos_homography`` only).

    Returns:
        Tuple of (frame indices, timestamps, nested dict of angular offsets
        per target per data type).

    """
    out: dict[int, dict[DataType, np.ndarray]] = {}
    frame_idxs = None
    timestamps = None
    for t, point_val in points.items():
        points_cam_space: dict[int, np.ndarray] = {}
        out[t] = {}
        for d_type in d_types:
            if d_type == DataType.pose_left_right_avg:
                continue  # special case handled below
            # get data
            fr_idxs, ts, ori, gaze, gaze_plane = collect_gaze_data(plane_gazes, d_type, viewing_distance)
            if frame_idxs is None:
                frame_idxs = fr_idxs
            if timestamps is None:
                timestamps = ts
            out[t][d_type] = np.full((gaze.shape[0], 3), np.nan)

            # compute
            for i, f_idx in enumerate(fr_idxs):
                if d_type == DataType.viewpos_vidpos_homography:
                    # get vectors based on assumed viewing distance (from config), without using pose info
                    v_gaze = gaze[i, :]
                    v_target = points_for_homography[t]
                else:
                    # use 3D vectors known given pose information
                    if f_idx not in poses:
                        continue
                    if f_idx not in points_cam_space:
                        points_cam_space[f_idx] = poses[f_idx].world_frame_to_cam(point_val)

                    # get vectors from origin to target and to gaze point
                    v_gaze = gaze[i, :] - ori[i, :]
                    v_target = points_cam_space[f_idx] - ori[i, :]

                # Total angular offset between gaze and target vectors
                ang_2d = transforms.angle_between(v_target, v_gaze)
                # Decompose into horizontal/vertical using the angle between
                # gaze and target projected onto the plane surface.
                # Result: [total_angle, horizontal_component, vertical_component]
                on_plane_angle = math.atan2(gaze_plane[i, 1] - point_val[1], gaze_plane[i, 0] - point_val[0])
                out[t][d_type][i, :] = ang_2d * np.array([1.0, math.cos(on_plane_angle), math.sin(on_plane_angle)])

        # special case for average of left and right eye
        if DataType.pose_left_right_avg in d_types:
            out[t][DataType.pose_left_right_avg] = np.dstack((
                out[t][DataType.pose_left_eye],
                out[t][DataType.pose_right_eye],
            )).mean(axis=2)

    return frame_idxs, timestamps, out


def collect_gaze_data(
    plane_gazes: dict[int, list[gaze_worldref.Gaze]], d_type: DataType, viewing_distance: float | None = None
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect and stack gaze data from all frames into aligned numpy arrays.

    Flattens the per-frame gaze records into contiguous arrays for efficient
    vectorized computation in ``calculate_gaze_angles_to_point``.

    Args:
        plane_gazes: Dict mapping frame indices to lists of gaze records.
        d_type: The data type determining which gaze fields to extract.
        viewing_distance: Assumed viewing distance in mm (required for
            ``viewpos_vidpos_homography``, ignored otherwise).

    Returns:
        Tuple of (frame_indices, timestamps, origins, gaze_points, plane_points)
        as aligned arrays.

    Raises:
        ValueError: If *viewing_distance* is None for ``viewpos_vidpos_homography``.

    """
    if d_type == DataType.viewpos_vidpos_homography and viewing_distance is None:
        raise ValueError(
            f"When using data type {DataType.viewpos_vidpos_homography}, a viewing distance (in mm) should be provided."
        )

    frame_idxs = [k for k, v in plane_gazes.items() for _ in v]
    ts = [s.timestamp for v in plane_gazes.values() for s in v]

    fields = get_world_gaze_fields_for_data_type(d_type)
    # Origin is at the camera (zeros) for camera-centric methods
    if fields[0] is None:
        ori = np.zeros((len(ts), 3))
    else:
        ori = np.vstack([getattr(s, fields[0]) for v in plane_gazes.values() for s in v])
    gaze_plane = np.vstack([getattr(s, fields[2]) for v in plane_gazes.values() for s in v])
    if fields[1] is None:
        if not d_type == DataType.viewpos_vidpos_homography:
            raise NotImplementedError("This field should be set, is a special case not implemented? Contact developer")
        # No 3D gaze point available — synthesize one from 2D plane position
        # plus assumed viewing distance as the Z component
        gaze = np.hstack((gaze_plane[:, 0:2], np.full((gaze_plane.shape[0], 1), viewing_distance)))
    else:
        gaze = np.vstack([getattr(s, fields[1]) for v in plane_gazes.values() for s in v])

    return frame_idxs, ts, ori, gaze, gaze_plane
