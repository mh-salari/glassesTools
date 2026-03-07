"""Coordinate transforms, homography estimation, and camera projection utilities."""

import typing

import cv2
import numpy as np

from . import marker, ocv

M = typing.TypeVar("M", bound=int)
N = typing.TypeVar("N", bound=int)


def to_norm_pos(x: float, y: float, bbox: list[float]) -> list[float]:
    """Transform world-unit plane coordinates to a normalized ``[0, 1]`` position.

    Input ``(0, 0)`` is bottom-left in world space; output ``(0, 0)`` is
    top-left in image space (y is flipped).

    Args:
        x: X coordinate in world units.
        y: Y coordinate in world units.
        bbox: Bounding box ``[left, top, right, bottom]`` in world units.

    Returns:
        Normalized ``[x, y]`` in ``[0, 1]`` range.

    """
    extents = [bbox[2] - bbox[0], bbox[1] - bbox[3]]
    # bbox[1]-y instead of y-bbox[3] to flip y axis
    pos = [(x - bbox[0]) / extents[0], (bbox[1] - y) / extents[1]]
    return pos


def to_image_pos(
    x: float, y: float, bbox: list[float], img_size: list[float], margin: list[float] | None = None
) -> list[float]:
    """Transform world-unit plane coordinates to pixel position in an image.

    Args:
        x: X coordinate in world units.
        y: Y coordinate in world units.
        bbox: Bounding box ``[left, top, right, bottom]`` in world units.
        img_size: Active image area ``[width, height]`` in pixels
            (excluding margin).
        margin: Pixel offset ``[x_margin, y_margin]`` added to the result.

    Returns:
        Pixel position ``[px, py]``.

    """
    if margin is None:
        margin = [0, 0]
    # fractional position between bounding box edges, (0,0) in bottom left
    pos = to_norm_pos(x, y, bbox)
    # turn into int, add margin
    pos = [p * s + m for p, s, m in zip(pos, img_size, margin, strict=True)]
    return pos


def in_bbox(x: float, y: float, bbox: list[float], margin: float = 0.0) -> bool:
    """Return ``True`` if the point is within the bounding box.

    Args:
        x: X coordinate in world units.
        y: Y coordinate in world units.
        bbox: Bounding box ``[left, top, right, bottom]`` in world units.
        margin: Fractional tolerance added to each edge (e.g. 0.1 allows
            10% overshoot).

    Returns:
        Whether the point is inside the (optionally expanded) box.

    """
    pos = to_norm_pos(x, y, bbox)
    return (pos[0] >= -margin and pos[0] <= 1 + margin) and (pos[1] >= -margin and pos[1] <= 1 + margin)


def dist_from_bbox(x: float, y: float, bbox: list[float]) -> float:
    """Return the normalized distance from a point to the nearest bbox edge.

    Returns ``0.0`` if the point is inside the bounding box.

    Args:
        x: X coordinate in world units.
        y: Y coordinate in world units.
        bbox: Bounding box ``[left, top, right, bottom]`` in world units.

    Returns:
        Distance in normalized coordinates (where the bbox spans ``[0, 1]``).

    """
    pos = to_norm_pos(x, y, bbox)
    if (pos[0] >= 0 and pos[0] <= 1) and (pos[1] >= 0 and pos[1] <= 1):
        return 0.0  # inside bbox
    # compute max distance from edge of bbox
    dx = pos[0] if pos[0] < 0.0 else pos[0] - 1
    dy = pos[1] if pos[1] < 0.0 else pos[1] - 1
    return abs(max(dx, dy))


def estimate_homography_known_marker(
    known: list[marker.Marker], detected_corners: list[np.ndarray], detected_ids: np.ndarray
) -> np.ndarray | None:
    """Estimate homography from detected ArUco markers matched against known positions.

    Matches detected marker IDs to the ``known`` dict, collects matching
    corner pairs, and computes a homography.

    Args:
        known: Dict of known ``Marker`` objects keyed by marker ID.
        detected_corners: Per-marker corner arrays from ArUco detection.
        detected_ids: Detected marker ID array.

    Returns:
        3x3 homography matrix, or ``None`` if fewer than 4 points match.

    Raises:
        ValueError: If *detected_ids* and *detected_corners* have
            different lengths.

    """
    img_points = []
    obj_points = []
    detected_ids = detected_ids.flatten()
    if len(detected_ids) != len(detected_corners):
        raise ValueError(
            f"unexpected number of IDs ({len(detected_ids)}) given number of corner arrays ({len(detected_corners)})"
        )
    for i in range(len(detected_ids)):
        if detected_ids[i] in known:
            dc = detected_corners[i]
            if dc.shape[0] == 1 and dc.shape[1] == 4:
                dc = np.reshape(dc, (4, 1, 2))
            img_points.extend([x.flatten() for x in dc])
            obj_points.extend(known[detected_ids[i]].corners)

    if len(img_points) < 4:
        return None

    # compute Homography
    return estimate_homography(obj_points, img_points)


def estimate_homography(
    obj_points: list[np.ndarray] | np.ndarray, img_points: list[np.ndarray] | np.ndarray
) -> np.ndarray:
    """Estimate a homography mapping image points to object points.

    Args:
        obj_points: 2D object (world) point coordinates.
        img_points: 2D image point coordinates.

    Returns:
        3x3 homography matrix, or ``None`` if ``findHomography`` fails.

    """
    img_points = np.float32(img_points)
    obj_points = np.float32(obj_points)
    h, _ = cv2.findHomography(img_points, obj_points)
    return h


def apply_homography(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply a homography matrix to a set of 2D points.

    Args:
        points: Nx2 array of 2D points.
        h: 3x3 homography matrix.

    Returns:
        Transformed Nx2 array, or NaN array if input is all NaN.

    """
    if np.all(np.isnan(points)):
        return np.full_like(points, np.nan)

    return cv2.perspectiveTransform(points.astype("float").reshape((-1, 1, 2)), h).reshape((-1, 2))


def distort_points(
    points_cam: np.ndarray[tuple[M, typing.Literal[2]], np.dtype[np.float64]], cam_params: ocv.CameraParams
) -> np.ndarray[tuple[M, typing.Literal[2]], np.dtype[np.float64]]:
    """Apply lens distortion to undistorted camera-space 2D points.

    Unprojects through a distortion-free model, then reprojects through
    the full distortion model.

    Args:
        points_cam: Mx2 array of undistorted 2D points.
        cam_params: Camera intrinsic parameters.

    Returns:
        Mx2 array of distorted 2D points, or NaN if input is all NaN.

    """
    if np.all(np.isnan(points_cam)):
        return np.full_like(points_cam, np.nan)

    if cam_params.has_opencv_camera():
        # unproject, ignoring distortion as this is an undistorted point
        points_w = cv2.undistortPoints(points_cam.astype("float"), cam_params.camera_mtx, np.zeros((1, 5)))
        # reproject, applying distortion
        return cv2.projectPoints(
            cv2.convertPointsToHomogeneous(points_w),
            np.zeros((1, 1, 3)),
            np.zeros((1, 1, 3)),
            cam_params.camera_mtx,
            cam_params.distort_coeffs,
        )[0].reshape((-1, 2))
    if cam_params.has_colmap_camera():
        # unproject, ignoring distortion as this is an undistorted point
        points_w = cam_params.colmap_camera_no_distortion.cam_from_img(points_cam.reshape((-1, 2)))
        # reproject, applying distortion
        return cam_params.colmap_camera.img_from_cam(points_w)
    return np.full_like(points_cam, np.nan)


def undistort_points(
    points_cam: np.ndarray[tuple[M, typing.Literal[2]], np.dtype[np.float64]], cam_params: ocv.CameraParams
) -> np.ndarray[tuple[M, typing.Literal[2]], np.dtype[np.float64]]:
    """Remove lens distortion from distorted camera-space 2D points.

    For OpenCV cameras, uses ``undistortPoints`` with re-projection back
    to pixel coordinates.  For COLMAP cameras, unprojects through the
    distortion model and reprojects through a distortion-free model.

    Args:
        points_cam: Mx2 array of distorted 2D points.
        cam_params: Camera intrinsic parameters.

    Returns:
        Mx2 array of undistorted 2D points, or NaN if input is all NaN.

    """
    if np.all(np.isnan(points_cam)):
        return np.full_like(points_cam, np.nan)

    if cam_params.has_opencv_camera():
        return cv2.undistortPoints(
            points_cam.astype("float"), cam_params.camera_mtx, cam_params.distort_coeffs, P=cam_params.camera_mtx
        ).reshape((-1, 2))  # P=cameraMatrix to reproject to camera
    if cam_params.has_colmap_camera():
        # unproject, removing distortion
        points_w = cam_params.colmap_camera.cam_from_img(points_cam.reshape((-1, 2)))
        # reproject, without applying distortion
        return cam_params.colmap_camera_no_distortion.img_from_cam(points_w)
    return np.full_like(points_cam, np.nan)


def unproject_points(
    points_cam: np.ndarray[tuple[M, typing.Literal[2]], np.dtype[np.float64]], cam_params: ocv.CameraParams
) -> np.ndarray[tuple[M, typing.Literal[3]], np.dtype[np.float64]]:
    """Unproject 2D camera points to 3D rays, removing distortion.

    Returns homogeneous 3D coordinates on the z=1 plane.

    Args:
        points_cam: Mx2 array of 2D image points.
        cam_params: Camera intrinsic parameters.

    Returns:
        Mx3 array of 3D points, or NaN if input is all NaN.

    """
    if np.all(np.isnan(points_cam)):
        return np.full((points_cam.shape[0], 3), np.nan)

    if cam_params.has_opencv_camera():
        points_w = cv2.undistortPoints(
            points_cam.reshape((-1, 2)).astype("float"), cam_params.camera_mtx, cam_params.distort_coeffs
        ).reshape((-1, 2))
        return cv2.convertPointsToHomogeneous(points_w).reshape((-1, 3))
    if cam_params.has_colmap_camera():
        return cv2.convertPointsToHomogeneous(
            cam_params.colmap_camera.cam_from_img(points_cam.reshape((-1, 2)))
        ).reshape((-1, 3))
    return np.full((points_cam.shape[0], 3), np.nan)


def project_points(
    points_world: np.ndarray[tuple[M, typing.Literal[3]], np.dtype[np.float64]],
    cam_params: ocv.CameraParams,
    ignore_distortion: bool = False,
    rot_vec: np.ndarray[tuple[typing.Literal[3]], np.dtype[np.float64]] | None = None,
    trans_vec: np.ndarray[tuple[typing.Literal[3]], np.dtype[np.float64]] | None = None,
) -> np.ndarray[tuple[M, typing.Literal[2]], np.dtype[np.float64]]:
    """Project 3D world points to 2D camera image coordinates.

    Supports optional extrinsic rotation and translation vectors for
    transforming from world to camera frame before projection.

    Args:
        points_world: Mx3 array of 3D points.
        cam_params: Camera intrinsic parameters.
        ignore_distortion: If ``True``, project without applying distortion.
        rot_vec: Rodrigues rotation vector for world-to-camera transform.
        trans_vec: Translation vector for world-to-camera transform.

    Returns:
        Mx2 array of 2D image coordinates, or NaN if input is all NaN.

    """
    if np.all(np.isnan(points_world)):
        return np.full((points_world.shape[0], 2), np.nan)

    if cam_params.has_opencv_camera():
        return cv2.projectPoints(
            points_world.astype("float"),
            np.zeros((1, 1, 3)) if rot_vec is None else rot_vec,
            np.zeros((1, 1, 3)) if trans_vec is None else trans_vec,
            cam_params.camera_mtx,
            np.zeros((1, 5)) if ignore_distortion else cam_params.distort_coeffs,
        )[0].reshape((-1, 2))
    if cam_params.has_colmap_camera():
        if rot_vec is not None and trans_vec is not None:
            r_mat = cv2.Rodrigues(rot_vec)[0]
            rt_mat = np.hstack((r_mat, trans_vec.reshape(3, 1)))
            points_world = np.matmul(
                rt_mat, cv2.convertPointsToHomogeneous(points_world.reshape((-1, 3))).reshape((-1, 4)).T
            ).T
        if ignore_distortion:
            return cam_params.colmap_camera_no_distortion.img_from_cam(points_world.reshape((-1, 3)))
        return cam_params.colmap_camera.img_from_cam(points_world.reshape((-1, 3))).reshape((-1, 2))
    return np.full((points_world.shape[0], 2), np.nan)


def intersect_plane_ray(
    plane_normal: np.ndarray,
    plane_point: np.ndarray,
    ray_direction: np.ndarray,
    ray_point: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Find the intersection point of a ray with a plane.

    Based on the Rosetta Code ray-plane intersection algorithm.

    Args:
        plane_normal: Unit normal vector of the plane.
        plane_point: Any point on the plane.
        ray_direction: Direction vector of the ray.
        ray_point: Origin point of the ray.
        epsilon: Tolerance for parallelism check.

    Returns:
        3D intersection point, or NaN array if the ray is parallel
        to the plane.

    """
    ndotu = plane_normal.dot(ray_direction)
    if abs(ndotu) < epsilon:
        # raise RuntimeError("no intersection or line is within plane")
        return np.full((3,), np.nan)

    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    return w + si * ray_direction + plane_point


def _vecdot(x1: np.ndarray, x2: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute vector dot product along an axis (numpy<2 compatibility shim)."""
    # np.vecdot was added in numpy 2.0; this provides equivalent functionality
    ndim = max(x1.ndim, x2.ndim)
    x1_shape = (1,) * (ndim - x1.ndim) + tuple(x1.shape)
    x2_shape = (1,) * (ndim - x2.ndim) + tuple(x2.shape)
    if x1_shape[axis] != x2_shape[axis]:
        raise ValueError("x1 and x2 must have the same size along the given axis")

    x1_, x2_ = np.broadcast_arrays(x1, x2)
    x1_ = np.moveaxis(x1_, axis, -1)
    x2_ = np.moveaxis(x2_, axis, -1)

    res = x1_[..., None, :] @ x2_[..., None]
    return res[..., 0, 0].copy()


def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute the angle in degrees between vectors *v1* and *v2*.

    Uses the numerically stable ``arctan2(||cross||, dot)`` formulation.

    Args:
        v1: First vector or array of vectors.
        v2: Second vector or array of vectors.

    Returns:
        Angle(s) in degrees.

    """
    return (180.0 / np.pi) * np.arctan2(
        np.linalg.norm(np.cross(v1, v2), axis=min((1, v1.ndim - 1, v2.ndim - 1))), _vecdot(v1, v2)
    )
