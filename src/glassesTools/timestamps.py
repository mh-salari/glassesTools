"""Timestamp handling and video frame timestamp lookup."""

import bisect
import datetime
import enum
import pathlib

import numpy as np
import pandas as pd

from . import json


class Timestamp:
    """A Unix timestamp with a human-readable display string."""

    def __init__(self, unix_time: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> None:
        """Initialize with a Unix timestamp and optional strftime format."""
        self.fmt = fmt
        self.display = ""
        self.value = 0
        self.update(unix_time)

    def update(self, unix_time: float) -> None:
        """Update the stored timestamp and regenerate the display string."""
        self.value = int(unix_time)
        if self.value == 0:
            self.display = ""
        else:
            self.display = datetime.datetime.fromtimestamp(unix_time, tz=datetime.UTC).strftime(self.fmt)


json.register_type(json.TypeEntry(Timestamp, "__Timestamp__", lambda x: x.value, Timestamp))


class Type(enum.Enum):
    """Type of video timestamps to use (normal or stretched)."""

    Normal = enum.auto()
    Stretched = enum.auto()


class VideoTimestamps:
    """Lookup table for video frame timestamps, loaded from a TSV file."""

    def __init__(self, file_name: str | pathlib.Path) -> None:
        """Load frame timestamps from a tab-separated file."""
        self.timestamp_dict: dict[int, float] = {}
        self.indices: list[int] = []
        self.timestamps: list[float] = []
        self._ifi: float = None

        df = pd.read_csv(file_name, delimiter="\t", index_col="frame_idx")
        self.timestamp_dict = df.to_dict()["timestamp"]

        df = df.reset_index()
        df = df[df["frame_idx"] != -1]
        self.indices = df["frame_idx"].to_list()
        self.timestamps = df["timestamp"].to_list()

        self.timestamp_stretched_dict: dict[int, float] = None
        self.timestamps_stretched: list[float] = None
        self._ifi_stretched: float = None
        self.has_stretched = "timestamp_stretched" in df.columns
        if self.has_stretched:
            self.timestamp_stretched_dict = df.to_dict()["timestamp_stretched"]
            self.timestamps_stretched = df["timestamp_stretched"].to_list()

    def get_timestamp(self, idx: int, which: Type = Type.Normal) -> float:
        """Return the timestamp for a given frame index, or -1.0 if not found."""
        idx = int(idx)
        match which:
            case Type.Normal:
                d = self.timestamp_dict
            case Type.Stretched:
                if not self.has_stretched:
                    raise RuntimeError("stretched timestamps are not available for this video")
                d = self.timestamp_stretched_dict
        if idx in d:
            return d[idx]
        return -1.0

    def find_frame(self, ts: float, which: Type = Type.Normal) -> int:
        """Return the frame index nearest to the given timestamp."""
        match which:
            case Type.Normal:
                timestamps = self.timestamps
            case Type.Stretched:
                if not self.has_stretched:
                    raise RuntimeError("stretched timestamps are not available for this video")
                timestamps = self.timestamps_stretched

        idx = bisect.bisect(timestamps, ts)
        # return nearest
        if idx >= len(timestamps):
            return self.indices[-1]
        if idx > 0 and abs(timestamps[idx - 1] - ts) < abs(timestamps[idx] - ts):
            return self.indices[idx - 1]
        return self.indices[idx]

    def get_last(self, which: Type = Type.Normal) -> tuple[int, float]:
        """Return the last frame index and its timestamp."""
        match which:
            case Type.Normal:
                timestamps = self.timestamps
            case Type.Stretched:
                if not self.has_stretched:
                    raise RuntimeError("stretched timestamps are not available for this video")
                timestamps = self.timestamps_stretched
        return self.indices[-1], timestamps[-1]

    def get_ifi(self, which: Type = Type.Normal) -> float:
        """Return the mean inter-frame interval (IFI) in milliseconds."""
        match which:
            case Type.Normal:
                if self._ifi is None:
                    self._ifi = np.mean(np.diff(self.timestamps))
                return self._ifi
            case Type.Stretched:
                if not self.has_stretched:
                    raise RuntimeError("stretched timestamps are not available for this video")
                if self._ifi_stretched is None:
                    self._ifi_stretched = np.mean(np.diff(self.timestamps_stretched))
                return self._ifi_stretched
