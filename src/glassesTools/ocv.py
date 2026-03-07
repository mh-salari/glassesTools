"""OpenCV camera parameter handling and video reading utilities.

Provides ``CameraParams`` for storing intrinsic/extrinsic camera parameters
with automatic conversion to COLMAP camera models, and ``CV2VideoReader``
for sequential frame reading with timestamp-based tracking.
"""

import bisect
import copy
import pathlib
import warnings
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pycolmap


class CameraParams:
    """Camera intrinsic and extrinsic parameters with optional COLMAP camera model.

    When OpenCV parameters are provided without a COLMAP dict, they are
    automatically converted to a ``FULL_OPENCV`` COLMAP camera model for use
    with COLMAP's undistortion functions. A distortion-free variant is also
    created for operations that need undistorted coordinates.

    Attributes:
        resolution: Image size as ``(width, height)`` array, or ``None``.
        camera_mtx: OpenCV 3x3 camera matrix, or ``None``.
        distort_coeffs: OpenCV distortion coefficients (1D), or ``None``.
        rotation_vec: Rodrigues rotation vector (3,), or ``None``.
        position: Camera position in world coordinates (3,), or ``None``.
        colmap_camera: COLMAP camera model, or ``None``.
        colmap_camera_no_distortion: Same as *colmap_camera* but with
            all distortion parameters zeroed out, or ``None``.

    """

    def __init__(
        self,
        resolution: np.ndarray | None,
        camera_mtx: np.ndarray | None,
        distort_coeffs: np.ndarray | None = None,
        rotation_vec: np.ndarray | None = None,
        position: np.ndarray | None = None,
        colmap_camera_dict: dict[str, Any] | None = None,
    ) -> None:
        """Initialize camera parameters.

        Args:
            resolution: Image ``(width, height)`` array.
            camera_mtx: OpenCV 3x3 camera intrinsic matrix.
            distort_coeffs: OpenCV distortion coefficients.
            rotation_vec: Rodrigues rotation vector for extrinsics.
            position: Camera position in world coordinates.
            colmap_camera_dict: Pre-built COLMAP camera dict.  If provided,
                used directly instead of converting from OpenCV parameters.

        """
        self.resolution: np.ndarray | None = resolution.flatten() if resolution is not None else None
        self.camera_mtx: np.ndarray | None = camera_mtx
        self.distort_coeffs: np.ndarray | None = distort_coeffs.flatten() if distort_coeffs is not None else None
        self.rotation_vec: np.ndarray | None = rotation_vec.flatten() if rotation_vec is not None else None
        self.position: np.ndarray | None = position.flatten() if position is not None else None

        self.colmap_camera: pycolmap.Camera | None = None
        self.colmap_camera_no_distortion: pycolmap.Camera | None = None

        # build COLMAP camera: prefer explicit dict, otherwise convert from OpenCV
        if colmap_camera_dict:
            self.colmap_camera = pycolmap.Camera(colmap_camera_dict)
        elif self.has_opencv_camera():
            self.colmap_camera = pycolmap.Camera.create(
                0, pycolmap.CameraModelId.FULL_OPENCV, self.camera_mtx[0, 0], *self.resolution
            )

            # map OpenCV intrinsics into COLMAP's parameter array:
            # [fx, fy, cx, cy, distortion_coeffs...]
            cal_params = np.zeros((self.colmap_camera.extra_params_idxs()[-1] + 1,))
            cal_params[self.colmap_camera.focal_length_idxs()] = [self.camera_mtx[0, 0], self.camera_mtx[1, 1]]
            cal_params[self.colmap_camera.principal_point_idxs()] = self.camera_mtx[0:2, 2]
            if len(self.distort_coeffs) > len(self.colmap_camera.extra_params_idxs()):
                self.colmap_camera = None
                print(
                    f"Warning: could not make colmap FULL_OPENCV camera as there are too many distortion parameters {len(self.distort_coeffs)}"
                )
            else:
                cal_params[self.colmap_camera.extra_params_idxs()[0 : len(self.distort_coeffs)]] = (
                    self.distort_coeffs.flatten()
                )
                self.colmap_camera.params = cal_params
        if self.colmap_camera is not None:
            # distortion-free variant: zero out params[4:] (everything after fx, fy, cx, cy)
            cam_dict = copy.deepcopy(self.colmap_camera.todict())
            cam_dict["params"][4:] = 0
            self.colmap_camera_no_distortion = pycolmap.Camera(cam_dict)

    @staticmethod
    def read_from_file(file_name: str | pathlib.Path) -> "CameraParams":
        """Read camera parameters from an OpenCV FileStorage XML/YAML file.

        Returns an empty ``CameraParams(None, None)`` if the file does not
        exist.  The file stores a rotation matrix which is converted to a
        Rodrigues vector.

        Args:
            file_name: Path to the XML/YAML file.

        Returns:
            Populated ``CameraParams`` instance.

        """
        file_name = pathlib.Path(file_name)
        if not file_name.is_file():
            return CameraParams(None, None)

        fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
        resolution = fs.getNode("resolution").mat()
        # intrinsics
        camera_matrix = fs.getNode("cameraMatrix").mat()
        dist_coeff = fs.getNode("distCoeff").mat()
        # extrinsics
        camera_rotation = fs.getNode("rotation").mat()
        if camera_rotation is not None:
            camera_rotation = cv2.Rodrigues(camera_rotation)[0]  # need rotation vector, not rotation matrix
        camera_position = fs.getNode("position").mat()
        # colmap camera dict, if available
        colmap_camera = {}
        colmap_camera_n = fs.getNode("colmap_camera")
        if not colmap_camera_n.empty():
            colmap_camera["camera_id"] = int(colmap_camera_n.getNode("camera_id").real())
            colmap_camera["model"] = getattr(pycolmap.CameraModelId, colmap_camera_n.getNode("model").string())
            colmap_camera["width"] = int(colmap_camera_n.getNode("width").real())
            colmap_camera["height"] = int(colmap_camera_n.getNode("height").real())
            colmap_camera["params"] = colmap_camera_n.getNode("params").mat()
            colmap_camera["has_prior_focal_length"] = bool(colmap_camera_n.getNode("width").real())
        fs.release()

        return CameraParams(resolution, camera_matrix, dist_coeff, camera_rotation, camera_position, colmap_camera)

    def has_opencv_camera(self) -> bool:
        """Return True if OpenCV camera matrix and distortion coefficients are available."""
        return (self.camera_mtx is not None) and (self.distort_coeffs is not None)

    def has_colmap_camera(self) -> bool:
        """Return True if a COLMAP camera model is available."""
        return self.colmap_camera is not None

    def has_intrinsics(self) -> bool:
        """Return True if any intrinsic camera model (OpenCV or COLMAP) is available."""
        return self.has_opencv_camera() or self.has_colmap_camera()

    def has_extrinsics(self) -> bool:
        """Return True if rotation vector and position are available."""
        return (self.rotation_vec is not None) and (self.position is not None)


class CV2VideoReader:
    """Sequential forward-only video reader with timestamp-based frame tracking.

    Uses spooling (sequential reads) instead of seeking because OpenCV's
    ``CAP_PROP_POS_MSEC`` seeking was found to be unreliable.  Includes gap
    detection logic that corrects the frame index when the video contains
    corrupt or missing frames.

    Attributes:
        file: Path to the video file.
        nframes: Total number of frames (derived from the timestamp array).
        frame_idx: Index of the last frame read, or ``-1`` before any reads.

    """

    def __init__(self, file: str | pathlib.Path, timestamps: list[float] | np.ndarray | pd.DataFrame) -> None:
        """Initialize video reader.

        Args:
            file: Path to the video file.
            timestamps: Per-frame timestamps in milliseconds.  Accepts a list,
                a 1-D numpy array, or a DataFrame with a ``"timestamp"`` column.

        Raises:
            RuntimeError: If OpenCV cannot open the video file.

        """
        self.file = pathlib.Path(file)
        if isinstance(timestamps, list):
            self._ts = np.array(timestamps)
        elif isinstance(timestamps, pd.DataFrame):
            self._ts = timestamps["timestamp"].to_numpy()
        else:
            self._ts = timestamps

        self._cap = cv2.VideoCapture(self.file)
        if not self._cap.isOpened():
            raise RuntimeError(f'the file "{self.file!s}" could not be opened')
        self.nframes = len(self._ts)
        self.frame_idx = -1
        self._last_good_ts = (-1, -1.0, -1.0)  # (frame_idx, opencv_ts, file_ts)
        self._cache: tuple[bool, np.ndarray, int, float] | None = None

    def __del__(self) -> None:
        """Release the underlying OpenCV video capture."""
        self._cap.release()

    def get_prop(self, cv2_prop: int) -> float:
        """Get a property value from the underlying ``cv2.VideoCapture``.

        Args:
            cv2_prop: OpenCV property identifier (e.g. ``cv2.CAP_PROP_FPS``).

        Returns:
            The property value, or ``0.0`` if unsupported.

        """
        return self._cap.get(cv2_prop)

    def set_prop(self, cv2_prop: int, val: float) -> bool:
        """Set a property value on the underlying ``cv2.VideoCapture``.

        Args:
            cv2_prop: OpenCV property identifier.
            val: Value to set.

        Returns:
            Whether the property was set successfully.

        """
        return self._cap.set(cv2_prop, val)

    def read_frame(
        self, report_gap: bool = False, wanted_frame_idx: int | None = None
    ) -> tuple[bool, np.ndarray | None, int | None, float | None]:
        """Read the next frame, or a specific frame by spooling forward.

        Seeking via ``CAP_PROP_POS_MSEC`` is unreliable, so this method reads
        frames sequentially until the wanted index is reached.  If the video
        contains corrupt or missing frames (gaps), the method detects the
        discontinuity by comparing OpenCV's internal timestamp against the
        externally-supplied timestamps and corrects ``frame_idx`` accordingly.

        Only forward access is supported.  Requesting a frame behind the
        current position returns the last cached frame with a warning.

        Args:
            report_gap: If ``True``, print a message when a frame
                discontinuity (gap) is detected.
            wanted_frame_idx: Frame index to read.  ``None`` reads the next
                sequential frame.

        Returns:
            A 4-tuple ``(done, frame, frame_idx, timestamp)``:

            - *done*: ``True`` when the end of the video is reached.
            - *frame*: BGR image array, or ``None`` if the frame was corrupt
              or the video ended.
            - *frame_idx*: Index of the returned frame, or ``None`` at end.
            - *timestamp*: Timestamp from the externally-supplied list, or
              ``None`` at end.

        Raises:
            ValueError: If *wanted_frame_idx* is out of bounds.

        """
        if wanted_frame_idx is not None:
            if wanted_frame_idx < 0 or wanted_frame_idx >= self.nframes:
                raise ValueError(f"wanted_frame_idx ({wanted_frame_idx}) out of bounds ([0-{self.nframes - 1}])")
        else:
            wanted_frame_idx = self.frame_idx + 1

        if self.frame_idx > wanted_frame_idx:
            warnings.warn(
                f"Requested frame ({wanted_frame_idx}) was earlier than current position of reader (frame {self.frame_idx}). Impossible to deliver because this video reader strictly advances forward. Returning last read frame",
                RuntimeWarning,
                stacklevel=2,
            )
            # this condition can only occur if we've already read something and thus have a cache, so this check should never trigger
            if self._cache is None:
                raise RuntimeError("No cache, unexpected failure mode, contact developer")
            return self._cache
        if self._cache is not None and self._cache[2] == wanted_frame_idx:
            return self._cache

        while True:
            ret, frame = self._cap.read()
            # OpenCV's timestamp ignores mp4 edit lists, so it may differ from
            # the true presentation timestamps — we only use it for gap detection
            ocv_ts = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            self.frame_idx += 1

            # ret==False can also mean a single corrupt frame, not necessarily end-of-video;
            # only treat it as finished if we're at the very start or past 99% of expected frames
            if not ret and (self.frame_idx == 0 or self.frame_idx / self.nframes > 0.99):
                self._cache = True, None, None, None
                return self._cache

            ts_from_list = self._ts[self.frame_idx]
            if self.frame_idx == 1 or ocv_ts > 0.0:
                # Gap detection: if the elapsed time according to OpenCV is larger than
                # what our timestamp list expects, frames were skipped (e.g. corruption)
                if (
                    self._last_good_ts[0] != -1
                    and ts_from_list - self._last_good_ts[2] < ocv_ts - self._last_good_ts[1] - 1
                ):  # 1 ms leeway for precision or mismatched timestamps
                    # Correct frame_idx by measuring the jump in OpenCV time,
                    # then finding the closest match in our own timestamp list
                    # (offset relative to the last good point, so we're robust
                    # to OpenCV ignoring edit lists)
                    t_jump = ocv_ts - self._last_good_ts[1]
                    tss = self._ts - self._last_good_ts[2]
                    self.frame_idx = self._find_closest_idx(t_jump, tss)
                    ts_from_list = self._ts[self.frame_idx]
                    if report_gap and self.frame_idx - self._last_good_ts[0] > 1:
                        print(
                            f"Frame discontinuity detected (jumped from {self._last_good_ts[0]} to {self.frame_idx}), there are probably corrupt frames in your video"
                        )
                self._last_good_ts = (self.frame_idx, ocv_ts, ts_from_list)

            # keep spooling until we arrive at the wanted frame
            if self.frame_idx == wanted_frame_idx:
                if not ret or frame is None:
                    # we might not have a valid frame, but we're not done yet
                    self._cache = False, None, self.frame_idx, ts_from_list
                else:
                    self._cache = False, frame, self.frame_idx, ts_from_list
                return self._cache

    @staticmethod
    def _find_closest_idx(time: float, times: np.ndarray) -> int:
        """Find the index in *times* whose value is closest to *time*.

        Uses binary search for efficiency.

        Args:
            time: Target timestamp to match.
            times: Sorted array of timestamps.

        Returns:
            Index of the nearest timestamp.

        """
        idx = bisect.bisect(times, time)
        if abs(times[idx - 1] - time) < abs(times[idx] - time):
            idx -= 1
        return idx

    def report_frame(self, interval: int = 100) -> None:
        """Print the current frame index at regular intervals.

        Args:
            interval: Print every *interval*-th frame.

        """
        if self.frame_idx % interval == 0:
            print(f"  frame {self.frame_idx}")
