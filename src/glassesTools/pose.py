"""Pose estimation, homography computation, and video-based pose processing."""

import enum
import pathlib
import typing

import cv2
import numpy as np

from . import annotation, data_files, drawing, intervals, marker, ocv, timestamps, transforms


class Pose:
    """Camera pose and homography for a single frame relative to a plane.

    Stores the results of PnP pose estimation (rotation vector, translation
    vector, reprojection error) and/or a homography matrix.  Provides methods
    for converting between camera frame, world (plane) frame, and image
    coordinates.  Internally caches the rotation matrix and its inverse for
    efficiency.

    Attributes:
        frame_idx: Video frame index.
        pose_N_points: Number of image points used for the pose estimate;
            ``0`` if no valid pose.
        pose_reprojection_error: RMS reprojection error, or ``-1.0``.
        pose_R_vec: Rodrigues rotation vector (3,), or ``None``.
        pose_T_vec: Translation vector (3,), or ``None``.
        homography_N_points: Number of image points used for the homography;
            ``0`` if no valid homography.
        homography_mat: 3x3 homography matrix, or ``None``.

    """

    _columns_compressed: typing.ClassVar[dict[str, int]] = {
        "frame_idx": 1,
        "pose_N_points": 1,
        "pose_reprojection_error": 1,
        "pose_R_vec": 3,
        "pose_T_vec": 3,
        "homography_N_points": 1,
        "homography_mat": 9,
    }
    _non_float: typing.ClassVar[dict[str, type]] = {
        "frame_idx": int,
        "pose_ok": bool,
        "pose_N_points": int,
        "homography_N_points": int,
    }
    # backwards compatibility
    _column_patches: typing.ClassVar[dict[str, tuple[str, typing.Callable[[int], int]]]] = {
        "pose_N_markers": ("pose_N_points", lambda x: x * 4),
        "homography_N_markers": ("homography_N_points", lambda x: x * 4),
    }

    def __init__(
        self,
        frame_idx: int,
        pose_N_points: int = 0,
        pose_reprojection_error: float = -1.0,
        pose_R_vec: np.ndarray | None = None,
        pose_T_vec: np.ndarray | None = None,
        homography_N_points: int = 0,
        homography_mat: np.ndarray | None = None,
    ) -> None:
        """Initialize pose with frame index and optional pose/homography data.

        Args:
            frame_idx: Video frame index.
            pose_N_points: Number of image points used for the pose estimate
                (4 per marker when using ArUco).  ``0`` means unavailable.
            pose_reprojection_error: RMS reprojection error, or ``-1.0``.
            pose_R_vec: Rodrigues rotation vector (3,).
            pose_T_vec: Translation vector (3,).
            homography_N_points: Number of image points used for the
                homography.  ``0`` means unavailable.
            homography_mat: 3x3 homography matrix (may be passed flat).

        """
        self.frame_idx: int = frame_idx
        self.pose_N_points: int = pose_N_points
        self.pose_reprojection_error: float = pose_reprojection_error
        self.pose_R_vec: np.ndarray | None = pose_R_vec
        self.pose_T_vec: np.ndarray | None = pose_T_vec
        self.homography_N_points: int = homography_N_points
        self.homography_mat: np.ndarray | None = (
            homography_mat.reshape(3, 3) if homography_mat is not None else homography_mat
        )

        # lazily computed caches for rotation matrices and plane geometry
        self._RMat = None
        self._RtMat = None
        self._plane_normal = None
        self._plane_point = None
        self._RMatInv = None
        self._RtMatInv = None
        self._i_homography_mat = None

    def pose_successful(self) -> bool:
        """Return whether a valid camera pose estimate is available."""
        return self.pose_N_points > 0

    def homography_successful(self) -> bool:
        """Return whether a valid homography estimate is available."""
        return self.homography_N_points > 0

    def draw_frame_axis(
        self,
        img: np.ndarray,
        camera_params: ocv.CameraParams,
        arm_length: float,
        thickness: int,
        sub_pixel_fac: int,
        position: list[float] | None = None,
    ) -> None:
        """Draw 3D coordinate axes on the image at the estimated pose location.

        Does nothing if pose vectors or camera intrinsics are unavailable.

        Args:
            img: Image to draw on (modified in-place).
            camera_params: Camera intrinsic parameters.
            arm_length: Length of each axis arm in world units.
            thickness: Line thickness in pixels.
            sub_pixel_fac: Sub-pixel rendering factor.
            position: 3D origin point in world coordinates for the axes.
                Defaults to ``[0, 0, 0]``.

        """
        if position is None:
            position = [0.0, 0.0, 0.0]
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or not camera_params.has_intrinsics():
            return
        drawing.opencv_frame_axis(
            img, camera_params, self.pose_R_vec, self.pose_T_vec, arm_length, thickness, sub_pixel_fac, position
        )

    def cam_frame_to_world(self, point: np.ndarray) -> np.ndarray:
        """Transform a 3D point from camera frame to the plane's world frame.

        Args:
            point: 3D point in camera coordinates.

        Returns:
            3D point in world (plane) coordinates, or NaN array if pose
            is unavailable.

        """
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(point)):
            return np.full((3,), np.nan)

        if self._RtMatInv is None:
            if self._RMatInv is None:
                if self._RMat is None:
                    self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
                self._RMatInv = self._RMat.T
            self._RtMatInv = np.hstack((self._RMatInv, np.matmul(-self._RMatInv, self.pose_T_vec.reshape(3, 1))))

        return np.matmul(self._RtMatInv, np.append(np.array(point), 1.0).reshape((4, 1))).flatten()

    def world_frame_to_cam(self, point: np.ndarray) -> np.ndarray:
        """Transform a 3D point from the plane's world frame to camera frame.

        Args:
            point: 3D point in world (plane) coordinates.

        Returns:
            3D point in camera coordinates, or NaN array if pose
            is unavailable.

        """
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(point)):
            return np.full((3,), np.nan)

        if self._RtMat is None:
            if self._RMat is None:
                self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
            self._RtMat = np.hstack((self._RMat, self.pose_T_vec.reshape(3, 1)))

        return np.matmul(self._RtMat, np.append(np.array(point), 1.0).reshape((4, 1))).flatten()

    def plane_to_cam_pose(self, point_plane: np.ndarray, camera_params: ocv.CameraParams) -> np.ndarray:
        """Project a 3D point on the plane to a 2D camera image location using pose.

        Args:
            point_plane: 3D point in plane coordinates (z should be 0).
            camera_params: Camera intrinsic parameters.

        Returns:
            2D image coordinates, or NaN array if pose is unavailable.

        """
        if (self.pose_R_vec is None) or (self.pose_T_vec is None):
            return np.full((2,), np.nan)
        return transforms.project_points(
            point_plane, camera_params, rot_vec=self.pose_R_vec, trans_vec=self.pose_T_vec
        ).flatten()

    def cam_to_plane_pose(self, point: np.ndarray, camera_params: ocv.CameraParams) -> tuple[np.ndarray, np.ndarray]:
        """Project a 2D camera image point onto the plane using pose.

        Unprojects the image point to a 3D ray, intersects it with the plane
        in camera space, then transforms the intersection to world coordinates.

        Args:
            point: 2D image coordinates.
            camera_params: Camera intrinsic parameters.

        Returns:
            Tuple of ``(plane_xy, cam_intersection)`` where *plane_xy* is the
            2D position on the plane and *cam_intersection* is the 3D
            intersection point in camera space.  Both are NaN arrays if
            pose or intrinsics are unavailable.

        """
        if (
            (self.pose_R_vec is None)
            or (self.pose_T_vec is None)
            or np.any(np.isnan(point))
            or not camera_params.has_intrinsics()
        ):
            return np.full((2,), np.nan), np.full((3,), np.nan)

        g_3d = transforms.unproject_points(point, camera_params)

        # intersect the ray from the camera origin with the plane (in camera space)
        pos_cam = self.vector_intersect(g_3d)

        # convert camera-space intersection to plane coordinates; z should be ~0
        (x, y, _z) = self.cam_frame_to_world(pos_cam)
        return np.asarray([x, y]), pos_cam

    def plane_to_cam_homography(self, point: np.ndarray, camera_params: ocv.CameraParams) -> np.ndarray:
        """Project a 2D plane point to a 2D camera image location using homography.

        Uses the inverse homography, then re-applies distortion if intrinsics
        are available.

        Args:
            point: 2D point in plane coordinates.
            camera_params: Camera intrinsic parameters.

        Returns:
            2D image coordinates, or NaN array if homography is unavailable.

        """
        if self.homography_mat is None:
            return np.full((2,), np.nan)

        if self._i_homography_mat is None:
            self._i_homography_mat = np.linalg.inv(self.homography_mat)
        out = transforms.apply_homography(point, self._i_homography_mat).flatten()
        if camera_params.has_intrinsics():
            out = transforms.distort_points(out, camera_params).flatten()
        return out

    def cam_to_plane_homography(self, point_cam: np.ndarray, camera_params: ocv.CameraParams) -> np.ndarray:
        """Project a 2D camera image point onto the plane using homography.

        Undistorts the image point if intrinsics are available, then applies
        the homography.

        Args:
            point_cam: 2D image coordinates.
            camera_params: Camera intrinsic parameters.

        Returns:
            2D plane coordinates, or NaN array if homography is unavailable.

        """
        if self.homography_mat is None:
            return np.full((2,), np.nan)

        if camera_params.has_intrinsics():
            point_cam = transforms.undistort_points(point_cam, camera_params).flatten()
        return transforms.apply_homography(point_cam, self.homography_mat).flatten()

    def get_origin_on_image(self, camera_params: ocv.CameraParams) -> np.ndarray:
        """Return the plane origin ``(0, 0)`` projected onto the camera image.

        Args:
            camera_params: Camera intrinsic parameters.

        Returns:
            2D image coordinates.

        """
        return self.get_plane_point_on_image(np.zeros((1, 2)), camera_params)

    def get_plane_point_on_image(self, point: np.ndarray, camera_params: ocv.CameraParams) -> np.ndarray:
        """Project a plane point onto the camera image, preferring pose over homography.

        Args:
            point: 2D point in plane coordinates.
            camera_params: Camera intrinsic parameters.

        Returns:
            2D image coordinates, or NaN array if neither pose nor
            homography is available.

        """
        if self.pose_successful() and camera_params.has_intrinsics():
            a = self.plane_to_cam_pose(np.append(np.array(point), 0.0), camera_params)
        elif self.homography_successful():
            a = self.plane_to_cam_homography(point, camera_params)
        else:
            a = np.full((2,), np.nan)
        return a

    def vector_intersect(self, vector: np.ndarray, origin: np.ndarray | None = None) -> np.ndarray:
        """Find the intersection of a ray with the plane in camera space.

        Lazily computes and caches the plane normal and a point on the plane
        from the rotation and translation vectors.

        Args:
            vector: 3D ray direction (will be normalized internally).
            origin: 3D ray origin.  Defaults to ``[0, 0, 0]`` (camera
                origin).

        Returns:
            3D intersection point in camera space, or NaN array if pose
            is unavailable.

        """
        if origin is None:
            origin = np.array([0.0, 0.0, 0.0])
        if (self.pose_R_vec is None) or (self.pose_T_vec is None) or np.any(np.isnan(vector)):
            return np.full((3,), np.nan)

        if self._plane_normal is None:
            if self._RtMat is None:
                if self._RMat is None:
                    self._RMat = cv2.Rodrigues(self.pose_R_vec)[0]
                self._RtMat = np.hstack((self._RMat, self.pose_T_vec.reshape(3, 1)))

            # get plane normal
            self._plane_normal = self._RMat[:, 2]  # equivalent to: np.matmul(self._RMat, np.array([0., 0., 1.]))
            # get point on plane (just use plane's origin, doesn't matter which point)
            self._plane_point = self._RtMat[:, 3]  # equivalent to: np.matmul(self._RtMat, np.array([0., 0., 0., 1.]))

        # normalize vector
        vector /= np.linalg.norm(vector)

        # find intersection of vector (e.g. 3D gaze) with plane
        return transforms.intersect_plane_ray(
            self._plane_normal, self._plane_point, vector.flatten(), origin.flatten()
        )


def read_dict_from_file(file_name: str | pathlib.Path, episodes: list[list[int]] | None = None) -> dict[int, Pose]:
    """Read pose data from a TSV file into a dict keyed by frame index.

    Args:
        file_name: Path to the TSV file.
        episodes: Optional frame-index intervals to restrict reading.

    Returns:
        Dict mapping frame index to ``Pose`` objects.

    """
    return data_files.read_file(file_name, Pose, True, True, False, False, episodes=episodes)[0]


def write_list_to_file(poses: list[Pose], file_name: str | pathlib.Path, skip_failed: bool = False) -> None:
    """Write a list of ``Pose`` objects to a TSV file.

    Args:
        poses: Pose objects to write.
        file_name: Output file path.
        skip_failed: If ``True``, omit rows where all values are NaN.

    """
    data_files.write_array_to_file(poses, file_name, Pose._columns_compressed, skip_all_nan=skip_failed)


class Status(enum.Enum):
    """Processing status for a single video frame."""

    Ok = enum.auto()
    Skip = enum.auto()
    Finished = enum.auto()


_T = typing.TypeVar("_T")


class Estimator:
    """Video-based pose estimator that processes frames to detect planes and markers.

    Plane detection functions, individual marker detection functions, and
    extra processing functions are registered before processing begins.
    Each registration can include processing intervals and an optional
    visualizer callback.  The estimator then processes the video frame by
    frame, invoking the registered functions for each frame that falls
    within their defined intervals.

    """

    def __init__(
        self,
        video_file: str | pathlib.Path,
        frame_timestamp_file: str | pathlib.Path | timestamps.VideoTimestamps,
        camera_calibration_file: str | pathlib.Path | ocv.CameraParams,
    ) -> None:
        """Initialize estimator with video, timestamps, and camera calibration.

        Args:
            video_file: Path to the video file.
            frame_timestamp_file: Per-frame timestamps, as a path or
                pre-loaded ``VideoTimestamps``.
            camera_calibration_file: Camera calibration, as a path or
                pre-loaded ``CameraParams``.

        """
        self.video_ts = (
            frame_timestamp_file
            if isinstance(frame_timestamp_file, timestamps.VideoTimestamps)
            else timestamps.VideoTimestamps(frame_timestamp_file)
        )
        self.video = ocv.CV2VideoReader(video_file, self.video_ts.timestamps)
        self.cam_params = (
            camera_calibration_file
            if isinstance(camera_calibration_file, ocv.CameraParams)
            else ocv.CameraParams.read_from_file(camera_calibration_file)
        )

        self.plane_functions: dict[
            str, typing.Callable[[str, int, np.ndarray, ocv.CameraParams], tuple[np.ndarray, np.ndarray]]
        ] = {}
        self.plane_intervals: dict[str, tuple[annotation.EventType, list[int] | list[list[int]]]] = {}
        self.plane_visualizers: dict[str, typing.Callable[[str, int, np.ndarray, np.ndarray], None] | None] = {}

        self.individual_marker_functions: dict[
            _T, typing.Callable[[_T, int, np.ndarray, ocv.CameraParams], tuple[np.ndarray, np.ndarray | None]]
        ] = {}
        self.individual_marker_intervals: dict[_T, tuple[annotation.EventType, list[int] | list[list[int]]]] = {}
        self.individual_marker_visualizers: dict[
            _T, typing.Callable[[_T, int, np.ndarray, np.ndarray], None] | None
        ] = {}

        self.extra_proc_functions: dict[
            str, typing.Callable[[str, int, np.ndarray, ocv.CameraParams, typing.Any], tuple]
        ] = {}
        self.extra_proc_intervals: dict[str, tuple[annotation.EventType, list[int] | list[list[int]]] | None] = {}
        self.extra_proc_parameters: dict[str, dict[str, typing.Any]] = {}
        self.extra_proc_visualizers: dict[str, typing.Callable[[str, np.ndarray, int, typing.Any], None] | None] = {}

        self._cache: (
            tuple[
                Status,
                dict[str, Pose],
                dict[_T, marker.Pose],
                dict[str, tuple[int, typing.Any]],
                tuple[np.ndarray, int, float],
            ]
            | None
        ) = None  # self._cache[4][1] is frame number

        self.allow_early_exit = True
        self.progress_updater: typing.Callable[[], None] | None = None

        self.do_visualize = False
        self.sub_pixel_fac = 8
        self.plane_axis_arm_length = 25
        self.individual_marker_axis_arm_length = 25
        self.show_extra_processing_output = True

        self._first_frame = True

    def add_plane(
        self,
        plane: str,
        plane_function: typing.Callable[
            [str, int, np.ndarray, ocv.CameraParams], tuple[np.ndarray, np.ndarray] | None
        ],
        processing_intervals: tuple[annotation.EventType, list[int] | list[list[int]]] | None = None,
        plane_visualizer: typing.Callable[[str, int, np.ndarray, np.ndarray], None] | None = None,
    ) -> None:
        """Register a plane detection function for video processing.

        Args:
            plane: Unique name for this plane.
            plane_function: Callable that receives ``(plane_name, frame_idx,
                frame, camera_params)`` and returns ``(object_points,
                img_points)`` or ``None``.
            processing_intervals: Frame intervals during which this plane
                should be processed, or ``None`` for all frames.
            plane_visualizer: Optional callback for drawing detection results.

        Raises:
            RuntimeError: If called after processing has started.
            ValueError: If *plane* is already registered.

        """
        if not self._first_frame:
            raise RuntimeError("You cannot register planes once video processing has started")
        if plane in self.plane_functions:
            raise ValueError(f'Cannot register the plane "{plane}", it is already registered')
        self.plane_functions[plane] = plane_function
        self.plane_intervals[plane] = processing_intervals
        self.plane_visualizers[plane] = plane_visualizer

    def add_individual_marker(
        self,
        key: _T,
        individual_marker_function: typing.Callable[
            [_T, int, np.ndarray, ocv.CameraParams], tuple[np.ndarray, np.ndarray | None]
        ],
        processing_intervals: tuple[annotation.EventType, list[int] | list[list[int]]] | None = None,
        individual_marker_visualizer: typing.Callable[[str, int, np.ndarray, np.ndarray], None] | None = None,
    ) -> None:
        """Register an individual marker detection function for video processing.

        Args:
            key: Unique identifier for this marker.
            individual_marker_function: Callable that receives ``(key,
                frame_idx, frame, camera_params)`` and returns
                ``(object_points, img_points)``.
            processing_intervals: Frame intervals during which this marker
                should be processed, or ``None`` for all frames.
            individual_marker_visualizer: Optional callback for drawing
                detection results.

        Raises:
            RuntimeError: If called after processing has started.
            ValueError: If *key* is already registered.

        """
        if not self._first_frame:
            raise RuntimeError("You cannot register individual markers once video processing has started")
        if key in self.individual_marker_functions:
            raise ValueError(f"Cannot register the individual marker {key}, it is already registered")
        self.individual_marker_functions[key] = individual_marker_function
        self.individual_marker_intervals[key] = processing_intervals
        self.individual_marker_visualizers[key] = individual_marker_visualizer

    def register_extra_processing_fun(
        self,
        name: str,
        processing_intervals: tuple[annotation.EventType, list[int] | list[list[int]]] | None,
        func: typing.Callable[[str, int, np.ndarray, ocv.CameraParams, typing.Any], tuple],
        func_parameters: dict[str, typing.Any],
        visualizer: typing.Callable[[str, np.ndarray, int, typing.Any], None],
    ) -> None:
        """Register an extra processing function for video processing.

        Args:
            name: Unique name for this processing function.
            processing_intervals: Frame intervals, or ``None`` for all frames.
            func: Callable that receives ``(name, frame_idx, frame,
                camera_params, **func_parameters)``.
            func_parameters: Keyword arguments passed to *func*.
            visualizer: Optional callback for drawing processing results.

        Raises:
            RuntimeError: If called after processing has started.
            ValueError: If *name* is already registered.

        """
        if not self._first_frame:
            raise RuntimeError("You cannot register extra processing functions once video processing has started")
        if name in self.extra_proc_functions:
            raise ValueError(f'Cannot register the extra processing function "{name}", it is already registered')
        self.extra_proc_intervals[name] = processing_intervals
        self.extra_proc_functions[name] = func
        self.extra_proc_parameters[name] = func_parameters
        self.extra_proc_visualizers[name] = visualizer

    def set_allow_early_exit(self, allow_early_exit: bool) -> None:
        """Set whether processing stops after the last defined interval.

        Args:
            allow_early_exit: If ``False``, process all frames regardless
                of interval boundaries.

        """
        self.allow_early_exit = allow_early_exit

    def set_progress_updater(self, progress_updater: typing.Callable[[], None]) -> None:
        """Set the callback invoked after each frame is processed.

        Args:
            progress_updater: Zero-argument callback.

        """
        self.progress_updater = progress_updater

    def set_visualize_on_frame(self, do_visualize: bool) -> None:
        """Enable or disable drawing detections on frames during processing.

        Args:
            do_visualize: Whether to draw on frames.

        """
        self.do_visualize = do_visualize

    def get_video_info(self) -> tuple[int, int, float]:
        """Return video ``(width, height, fps)``."""
        return (
            int(self.video.get_prop(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.video.get_prop(cv2.CAP_PROP_FRAME_HEIGHT)),
            self.video.get_prop(cv2.CAP_PROP_FPS),
        )

    def estimate_pose(
        self, object_points: np.ndarray, img_points: np.ndarray, flags: int = cv2.SOLVEPNP_ITERATIVE
    ) -> tuple[int, np.ndarray | None, np.ndarray | None, float]:
        """Estimate camera pose using this estimator's camera parameters.

        Delegates to the module-level :func:`estimate_pose`.

        Args:
            object_points: Nx3 array of 3D object points.
            img_points: Nx1x2 array of 2D image points.
            flags: OpenCV PnP solver flag.

        Returns:
            Tuple of ``(n_points, r_vec, t_vec, reprojection_error)``.

        """
        return estimate_pose(object_points, img_points, self.cam_params, flags)

    def estimate_homography(self, object_points: np.ndarray, img_points: np.ndarray) -> tuple[int, np.ndarray | None]:
        """Estimate homography using this estimator's camera parameters.

        Delegates to the module-level :func:`estimate_homography`.

        Args:
            object_points: Nx3 array of 3D object points.
            img_points: Nx1x2 array of 2D image points.

        Returns:
            Tuple of ``(n_points, homography_matrix)``.

        """
        return estimate_homography(object_points, img_points, self.cam_params)

    def estimate_pose_and_homography(self, frame_idx: int, object_points: np.ndarray, img_points: np.ndarray) -> Pose:
        """Estimate both camera pose and homography for a single frame.

        Args:
            frame_idx: Video frame index.
            object_points: Nx3 array of 3D object points.
            img_points: Nx1x2 array of 2D image points.

        Returns:
            ``Pose`` populated with both pose and homography results.

        """
        pose = Pose(frame_idx)
        if (
            object_points is not None and img_points is not None and img_points.shape[0] >= 4
        ):  # at least four image points needed
            # get camera pose
            pose.pose_N_points, pose.pose_R_vec, pose.pose_T_vec, pose.pose_reprojection_error = self.estimate_pose(
                object_points, img_points
            )

            # also get homography (direct image plane to plane in world transform)
            pose.homography_N_points, pose.homography_mat = self.estimate_homography(object_points, img_points)
        return pose

    def process_one_frame(
        self, wanted_frame_idx: int | None = None
    ) -> tuple[
        Status,
        dict[str, Pose],
        dict[str, marker.Pose],
        dict[str, tuple[int, typing.Any]],
        tuple[np.ndarray, int, float],
    ]:
        """Process a single video frame.

        Reads the next (or a specific) frame, determines which registered
        planes, individual markers, and extra processing functions should
        run based on their intervals, executes detection and pose estimation,
        and optionally visualizes the results on the frame.

        Args:
            wanted_frame_idx: Frame index to process.  ``None`` reads the
                next sequential frame.

        Returns:
            A 5-tuple ``(status, poses, markers, extras, frame_data)``
            where *frame_data* is ``(frame, frame_idx, frame_ts)``.

        """
        if wanted_frame_idx is not None and self._cache is not None and self._cache[4][1] == wanted_frame_idx:
            return self._cache

        should_exit, frame, frame_idx, frame_ts = self.video.read_frame(
            report_gap=True, wanted_frame_idx=wanted_frame_idx
        )

        if should_exit or (
            self.allow_early_exit
            and (
                (not self.plane_intervals or intervals.beyond_last_interval(frame_idx, self.plane_intervals))
                and (
                    not self.individual_marker_intervals
                    or intervals.beyond_last_interval(frame_idx, self.individual_marker_intervals)
                )
                and (
                    not self.extra_proc_intervals
                    or intervals.beyond_last_interval(frame_idx, self.extra_proc_intervals)
                )
            )
        ):
            self._cache = Status.Finished, None, None, None, (None, None, None)
            return self._cache
        if self.progress_updater:
            self.progress_updater()

        if self._first_frame:
            self._first_frame = False

        # check we're in a current interval, else skip processing
        # NB: have to spool through like this, setting specific frame to read
        # with cap.set(cv2.CAP_PROP_POS_FRAMES) doesn't seem to work reliably
        # for VFR video files
        planes_for_this_frame = [
            p for p in self.plane_functions if intervals.is_in_interval(frame_idx, self.plane_intervals[p])
        ]
        indiv_markers_for_this_frame = [
            i
            for i in self.individual_marker_functions
            if intervals.is_in_interval(frame_idx, self.individual_marker_intervals[i])
        ]
        extra_processing_for_this_frame = [
            e for e in self.extra_proc_functions if intervals.is_in_interval(frame_idx, self.extra_proc_intervals[e])
        ]
        if frame is None or (
            not planes_for_this_frame and not indiv_markers_for_this_frame and not extra_processing_for_this_frame
        ):
            # we don't have a valid frame or nothing to do, continue to next
            self._cache = Status.Skip, None, None, None, (frame, frame_idx, frame_ts)
            return self._cache

        pose_out: dict[str, Pose] = {}
        individual_marker_out: dict[_T, marker.Pose] = {}
        extra_processing_out: dict[str, tuple[int, typing.Any]] = {}
        if planes_for_this_frame:
            # detect fiducials
            plane_points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for p in planes_for_this_frame:
                det_output = self.plane_functions[p](p, frame_idx, frame, self.cam_params)
                if det_output[0] is not None:
                    plane_points[p] = det_output
            # determine pose
            for p, points in plane_points.items():
                pose_out[p] = self.estimate_pose_and_homography(frame_idx, *points)

        if indiv_markers_for_this_frame:
            # detect fiducials
            indiv_marker_points: dict[_T, tuple[np.ndarray, np.ndarray]] = {}
            for i in indiv_markers_for_this_frame:
                det_output = self.individual_marker_functions[i](i, frame_idx, frame, self.cam_params)
                if (
                    det_output[1] is not None
                ):  # object points may not be available (e.g. when marker size is not set), so check for image points
                    indiv_marker_points[i] = det_output
            # determine pose, if wanted
            for i, i_points in indiv_marker_points.items():
                mpose = marker.Pose(frame_idx)
                if (
                    i_points[0] is not None
                ):  # object points may not be available (e.g. when marker size is not set). If so, skip pose estimation
                    _, mpose.R_vec, mpose.T_vec, _ = self.estimate_pose(*i_points, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                individual_marker_out[i] = mpose

        for e in extra_processing_for_this_frame:
            eproc = self.extra_proc_functions[e](e, frame_idx, frame, self.cam_params, **self.extra_proc_parameters[e])
            if eproc is not None:
                extra_processing_out[e] = (frame_idx, eproc)

        # now that all processing is done, handle visualization, if any
        if self.do_visualize:
            # first draw all detection output
            if planes_for_this_frame:
                for p, p_pts in plane_points.items():
                    if self.plane_visualizers[p] is None:
                        continue
                    self.plane_visualizers[p](p, frame_idx, frame, p_pts[0])
            if indiv_markers_for_this_frame:
                for i, i_pts in indiv_marker_points.items():
                    if self.individual_marker_visualizers[i] is None:
                        continue
                    self.individual_marker_visualizers[i](i, frame_idx, frame, i_pts[0])
            for e in extra_processing_for_this_frame:
                if self.show_extra_processing_output and self.extra_proc_visualizers[e] and e in extra_processing_out:
                    self.extra_proc_visualizers[e](e, frame, *extra_processing_out[e])

            # now also draw pose, if wanted
            if self.plane_axis_arm_length:
                for p_pose in pose_out.values():
                    if p_pose.pose_successful():
                        p_pose.draw_frame_axis(
                            frame, self.cam_params, self.plane_axis_arm_length, 3, sub_pixel_fac=self.sub_pixel_fac
                        )
            if self.individual_marker_axis_arm_length:
                for i_mpose in individual_marker_out.values():
                    if i_mpose.pose_successful():
                        i_mpose.draw_frame_axis(
                            frame, self.cam_params, self.individual_marker_axis_arm_length, self.sub_pixel_fac
                        )

        self._cache = Status.Ok, pose_out, individual_marker_out, extra_processing_out, (frame, frame_idx, frame_ts)
        return self._cache

    def process_video(
        self,
    ) -> tuple[dict[str, list[Pose]], dict[_T, list[marker.Pose]], dict[str, list[tuple[int, typing.Any]]]]:
        """Process the entire video from start to finish.

        Calls :meth:`process_one_frame` in a loop until finished, collecting
        all successful results.

        Returns:
            Tuple of ``(poses, individual_markers, extra_processing)`` where
            each is a dict keyed by registered name/key with lists of per-frame
            results.

        """
        poses_out: dict[str, list[Pose]] = {p: [] for p in self.plane_functions}
        individual_markers_out: dict[_T, list[marker.Pose]] = {i: [] for i in self.individual_marker_functions}
        extra_processing_out: dict[str, list[tuple[int, typing.Any]]] = {e: [] for e in self.extra_proc_functions}
        while True:
            status, plane, individual_marker, extra_proc, _ = self.process_one_frame()
            if status == Status.Finished:
                break
            if status == Status.Skip:
                continue
            # store outputs
            for p in plane:
                poses_out[p].append(plane[p])
            for i in individual_marker:
                individual_markers_out[i].append(individual_marker[i])
            for e in extra_proc:
                extra_processing_out[e].append(extra_proc[e])

        return poses_out, individual_markers_out, extra_processing_out


def estimate_pose(
    object_points: np.ndarray,
    img_points: np.ndarray,
    cam_params: ocv.CameraParams,
    flags: int = cv2.SOLVEPNP_ITERATIVE,
) -> tuple[int, np.ndarray | None, np.ndarray | None, float]:
    """Estimate camera pose via PnP from object-image point correspondences.

    For cameras with OpenCV intrinsics, uses ``solvePnPGeneric`` directly.
    For non-OpenCV cameras (e.g. COLMAP-only), unprojects to an identity
    camera space first, then solves PnP and computes reprojection error
    manually.

    Args:
        object_points: Nx3 array of 3D object points.
        img_points: Nx1x2 array of corresponding 2D image points.
        cam_params: Camera intrinsic parameters.
        flags: OpenCV PnP solver flag.

    Returns:
        Tuple of ``(n_points, r_vec, t_vec, reprojection_error)`` where
        *n_points* is ``0`` if estimation failed or was not possible.

    """
    n_points, r_vec, t_vec, reprojection_error = 0, None, None, -1.0
    if object_points is None or not cam_params.has_intrinsics() or object_points.shape[0] < 4:
        return n_points, r_vec, t_vec, reprojection_error

    if cam_params.has_opencv_camera():
        n_solutions, r_vec, t_vec, reprojection_error = cv2.solvePnPGeneric(
            object_points,
            img_points,
            cam_params.camera_mtx,
            cam_params.distort_coeffs,
            np.empty(1),
            np.empty(1),
            flags=flags,
        )
        n_points = object_points.shape[0] if n_solutions else 0
        reprojection_error = reprojection_error[0][0]
    else:
        # non-OpenCV camera (e.g. COLMAP-only): unproject to 3D, then re-project
        # through an identity camera so solvePnPGeneric can be used
        points_w = transforms.unproject_points(img_points, cam_params)
        points_cam = transforms.project_points(
            points_w, ocv.CameraParams(cam_params.resolution, np.identity(3), np.zeros((5, 1)))
        )
        n_points, r_vec, t_vec, _ = cv2.solvePnPGeneric(
            object_points,
            points_cam.reshape((-1, 1, 2)),
            np.identity(3),
            np.zeros((5, 1)),
            np.empty(1),
            np.empty(1),
            flags=flags,
        )
        # solvePnPGeneric's error is meaningless here (identity camera units);
        # recompute against original image points using the real camera model
        if n_points:
            proj_points = transforms.project_points(object_points, cam_params, rot_vec=r_vec[0], trans_vec=t_vec[0])
            reprojection_error = cv2.norm(
                proj_points.astype("float32").reshape((-1, 1, 2)), img_points, cv2.NORM_L2
            ) / np.sqrt(2 * proj_points.shape[0])
        else:
            reprojection_error = np.nan
    return n_points, r_vec[0], t_vec[0], reprojection_error


def estimate_homography(
    object_points: np.ndarray, img_points: np.ndarray, cam_params: ocv.CameraParams
) -> tuple[int, np.ndarray | None]:
    """Estimate homography from object-image point correspondences.

    Undistorts image points if camera intrinsics are available before
    computing the homography.

    Args:
        object_points: Nx3 array of 3D object points.
        img_points: Nx1x2 array of corresponding 2D image points.
        cam_params: Camera intrinsic parameters.

    Returns:
        Tuple of ``(n_points, homography_matrix)`` where *n_points* is
        ``0`` if estimation failed.

    """
    n_points, h = 0, None
    if object_points is None or object_points.shape[0] < 4:
        return n_points, h

    # use undistorted marker corners if possible
    if cam_params is not None and cam_params.has_intrinsics():
        img_points = transforms.undistort_points(img_points.reshape((-1, 2)), cam_params).reshape((-1, 1, 2))

    h = transforms.estimate_homography(object_points, img_points)
    if h is not None:
        n_points = object_points.shape[0]
    return n_points, h
