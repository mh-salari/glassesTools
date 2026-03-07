"""OpenCV drawing primitives with sub-pixel anti-aliasing and marker visualization."""

import math

import cv2
import numpy as np

from . import marker, ocv, transforms


def opencv_circle(
    img: np.ndarray, center_coordinates: np.ndarray, radius: float, color: tuple, thickness: int, sub_pixel_fac: int
) -> None:
    """Draw an anti-aliased circle with sub-pixel precision.

    Coordinates are scaled by *sub_pixel_fac* (must be a power of 2) and
    passed to OpenCV with a fixed-point ``shift``, giving sub-pixel
    rendering accuracy.

    Args:
        img: Image to draw on (modified in-place).
        center_coordinates: ``(x, y)`` center of the circle.
        radius: Circle radius in pixels.
        color: BGR color tuple.
        thickness: Line thickness (negative for filled).
        sub_pixel_fac: Sub-pixel precision factor (power of 2).

    """
    # Scale to fixed-point, guard against NaN / int overflow
    p = [np.round(x * sub_pixel_fac) for x in center_coordinates]
    if np.all([not math.isnan(x) and abs(x) < np.iinfo(np.intc).max for x in p]):
        p = tuple(int(x) for x in p)
        cv2.circle(
            img,
            p,
            int(np.round(radius * sub_pixel_fac)),
            color,
            thickness,
            lineType=cv2.LINE_AA,
            shift=int(math.log2(sub_pixel_fac)),
        )


def opencv_line(
    img: np.ndarray, start_point: np.ndarray, end_point: np.ndarray, color: tuple, thickness: int, sub_pixel_fac: int
) -> None:
    """Draw an anti-aliased line with sub-pixel precision.

    Args:
        img: Image to draw on (modified in-place).
        start_point: ``(x, y)`` start of the line.
        end_point: ``(x, y)`` end of the line.
        color: BGR color tuple.
        thickness: Line thickness in pixels.
        sub_pixel_fac: Sub-pixel precision factor (power of 2).

    """
    sp = [np.round(x * sub_pixel_fac) for x in start_point]
    ep = [np.round(x * sub_pixel_fac) for x in end_point]
    if np.all([not math.isnan(x) and abs(x) < np.iinfo(np.intc).max for x in sp]) and np.all([
        not math.isnan(x) and abs(x) < np.iinfo(np.intc).max for x in ep
    ]):
        sp = tuple(int(x) for x in sp)
        ep = tuple(int(x) for x in ep)
        cv2.line(img, sp, ep, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(sub_pixel_fac)))


def opencv_rectangle(
    img: np.ndarray, p1: np.ndarray, p2: np.ndarray, color: tuple, thickness: int, sub_pixel_fac: int
) -> None:
    """Draw an anti-aliased rectangle with sub-pixel precision.

    Args:
        img: Image to draw on (modified in-place).
        p1: ``(x, y)`` top-left corner.
        p2: ``(x, y)`` bottom-right corner.
        color: BGR color tuple.
        thickness: Line thickness (negative for filled).
        sub_pixel_fac: Sub-pixel precision factor (power of 2).

    """
    p1 = [np.round(x * sub_pixel_fac) for x in p1]
    p2 = [np.round(x * sub_pixel_fac) for x in p2]
    if np.all([not math.isnan(x) and abs(x) < np.iinfo(np.intc).max for x in p1]) and np.all([
        not math.isnan(x) and abs(x) < np.iinfo(np.intc).max for x in p2
    ]):
        p1 = tuple(int(x) for x in p1)
        p2 = tuple(int(x) for x in p2)
        cv2.rectangle(img, p1, p2, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(sub_pixel_fac)))


def opencv_polylines(
    img: np.ndarray, pts: np.ndarray, is_closed: bool, color: tuple, thickness: int, sub_pixel_fac: int
) -> None:
    """Draw anti-aliased polylines with sub-pixel precision.

    Args:
        img: Image to draw on (modified in-place).
        pts: Nx2 array of ``(x, y)`` vertices.
        is_closed: If ``True``, connect the last point to the first.
        color: BGR color tuple.
        thickness: Line thickness in pixels.
        sub_pixel_fac: Sub-pixel precision factor (power of 2).

    """
    pts = np.round(pts * sub_pixel_fac).astype(np.int32)
    if np.all([not math.isnan(x) and abs(x) < np.iinfo(np.intc).max for x in pts.flatten()]):
        cv2.polylines(
            img, [pts], is_closed, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(sub_pixel_fac))
        )


def opencv_frame_axis(
    img: np.ndarray,
    cam_params: ocv.CameraParams,
    rvec: np.ndarray,
    tvec: np.ndarray,
    arm_length: float,
    thickness: int,
    sub_pixel_fac: int,
    position: list[float] | None = None,
    offset_x: float = 0,
    offset_y: float = 0,
) -> None:
    """Draw 3D coordinate axes on an image using camera pose, with anti-aliasing.

    Draws X (red), Y (green), Z (blue) axes from an origin point,
    z-sorted so that nearer axes are drawn on top.

    Args:
        img: Image to draw on (modified in-place).
        cam_params: Camera intrinsic parameters for projection.
        rvec: Rodrigues rotation vector (world-to-camera).
        tvec: Translation vector (world-to-camera).
        arm_length: Length of each axis arm in world units.
        thickness: Line thickness in pixels.
        sub_pixel_fac: Sub-pixel precision factor (power of 2).
        position: 3D origin offset ``[x, y, z]`` in world coordinates.
        offset_x: Horizontal pixel offset applied after projection.
        offset_y: Vertical pixel offset applied after projection.

    """
    if position is None:
        position = [0.0, 0.0, 0.0]
    # Build 4 points: origin + one endpoint per axis, shifted by position
    points = np.vstack((np.zeros((1, 3)), arm_length * np.eye(3))) + np.vstack(4 * [np.asarray(position)])
    cam_points = transforms.project_points(points, cam_params, rot_vec=rvec, trans_vec=tvec)
    offset = np.array([offset_x, offset_y])
    # Z-sort axes so farther ones are drawn first (painter's algorithm)
    r_mat = cv2.Rodrigues(rvec)[0]
    rt_mat = np.hstack((r_mat, tvec.reshape(3, 1)))
    world_points = np.matmul(rt_mat, cv2.convertPointsToHomogeneous(points[1:, :]).reshape((-1, 4)).T)
    order = np.argsort(world_points[-1, :])[::-1]
    # draw
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    for i in order:
        opencv_line(
            img,
            cam_points[0].flatten() - offset,
            cam_points[i + 1].flatten() - offset,
            colors[i],
            thickness,
            sub_pixel_fac,
        )


def aruco_detected_markers(
    img: np.ndarray,
    corners: list,
    ids: np.ndarray | None,
    border_color: tuple = (0, 255, 0),
    draw_ids: bool = True,
    sub_pixel_fac: int = 1,
    special_highlight: list | None = None,
) -> None:
    """Draw detected ArUco markers with anti-aliased borders and optional ID labels.

    Similar to OpenCV's built-in marker drawing, but with anti-aliased
    rendering and the ability to highlight specific markers with custom
    colors.

    Args:
        img: Image to draw on (modified in-place).
        corners: Per-marker corner arrays from ArUco detection.
        ids: Detected marker ID array, or ``None``.
        border_color: Default BGR border color for marker edges.
        draw_ids: If ``True``, draw marker ID labels at each center.
        sub_pixel_fac: Sub-pixel precision factor (power of 2).
        special_highlight: Flat list of ``[id_set, color, id_set, color, ...]``
            pairs.  Markers whose ID is in a given *id_set* are drawn
            with the corresponding *color* instead of *border_color*.

    """
    if special_highlight is None:
        special_highlight = []
    # Derive text and corner colors by channel-swapping the border color
    text_color = list(border_color)
    corner_color = list(border_color)
    text_color[0], text_color[1] = text_color[1], text_color[0]  # swap B and R
    corner_color[1], corner_color[2] = corner_color[2], corner_color[1]  # swap G and R

    draw_ids = draw_ids and (ids is not None) and len(ids) > 0

    for i in range(len(corners)):
        corner = corners[i][0]
        # draw marker sides
        side_color = border_color
        for s, c in zip(special_highlight[::2], special_highlight[1::2], strict=True):
            if s is not None and ids[i][0] in s:
                side_color = c
        for j in range(4):
            p0 = corner[j, :]
            p1 = corner[(j + 1) % 4, :]
            opencv_line(img, p0, p1, side_color, 1, sub_pixel_fac)

        # draw first corner mark
        p1 = corner[0]
        opencv_rectangle(img, corner[0] - 3, corner[0] + 3, corner_color, 1, sub_pixel_fac)

        # draw IDs if wanted
        if draw_ids:
            c = marker.corners_intersection(corner)
            cv2.putText(
                img,
                str(ids[i][0]),
                tuple(c.astype(np.intc)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
                lineType=cv2.LINE_AA,
            )
