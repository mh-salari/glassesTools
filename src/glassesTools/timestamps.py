"""Timestamp handling and video frame timestamp lookup.

Provides ``Timestamp`` for JSON-serializable Unix epoch values and
``VideoTimestamps`` for loading per-frame timestamp tables from TSV files.
``VideoTimestamps`` supports both normal and *stretched* timestamps — the
latter are linearly remapped to a reference recording's timeline during
multi-recording synchronization.
"""

import bisect
import datetime
import enum
import pathlib

import numpy as np
import pandas as pd

from . import json


class Timestamp:
    """A Unix timestamp with a human-readable display string.

    Stores the epoch value as an integer and generates a formatted display
    string on update. Registered with the custom JSON type system for
    serialization in recording metadata.

    Attributes:
        fmt: strftime format string for display.
        display: Human-readable representation, or empty string if value is 0.
        value: Unix epoch timestamp (integer seconds).

    """

    def __init__(self, unix_time: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> None:
        """Create a Timestamp from a Unix epoch value.

        Args:
            unix_time: Unix epoch timestamp (seconds).
            fmt: strftime format for the display string.

        """
        self.fmt = fmt
        self.display = ""
        self.value = 0
        self.update(unix_time)

    def update(self, unix_time: float) -> None:
        """Replace the stored timestamp and regenerate the display string.

        Args:
            unix_time: New Unix epoch timestamp (seconds).

        """
        self.value = int(unix_time)
        if self.value == 0:
            self.display = ""
        else:
            self.display = datetime.datetime.fromtimestamp(unix_time, tz=datetime.UTC).strftime(self.fmt)


# Serializes as bare epoch integer; Timestamp constructor serves as deserializer
json.register_type(json.TypeEntry(Timestamp, "__Timestamp__", lambda x: x.value, Timestamp))


class Type(enum.Enum):
    """Selector for which timestamp column to use in ``VideoTimestamps``.

    ``Stretched`` timestamps are linearly remapped to a reference recording's
    clock during multi-recording synchronization and may not be available for
    all videos.

    """

    Normal = enum.auto()
    Stretched = enum.auto()


class VideoTimestamps:
    """Lookup table for video frame timestamps, loaded from a TSV file.

    Maintains both a dict (for O(1) lookup by frame index) and a sorted
    list (for bisect-based nearest-frame search by timestamp). Optionally
    loads a ``timestamp_stretched`` column if present.

    Attributes:
        timestamp_dict: Mapping from frame index to normal timestamp.
        indices: Sorted list of valid frame indices (excludes -1 sentinel).
        timestamps: Corresponding normal timestamps, same order as *indices*.
        has_stretched: Whether stretched timestamps are available.

    """

    def __init__(self, file_name: str | pathlib.Path) -> None:
        """Load frame timestamps from a tab-separated file.

        Args:
            file_name: Path to a TSV file with ``frame_idx`` and ``timestamp``
                columns, and optionally ``timestamp_stretched``.

        """
        self.timestamp_dict: dict[int, float] = {}
        self.indices: list[int] = []
        self.timestamps: list[float] = []
        self._ifi: float | None = None

        df = pd.read_csv(file_name, delimiter="\t", index_col="frame_idx")
        # Dict for O(1) frame→timestamp lookup (includes -1 sentinel rows)
        self.timestamp_dict = df.to_dict()["timestamp"]

        # Sorted lists for bisect search — exclude -1 sentinel frames
        df = df.reset_index()
        df = df[df["frame_idx"] != -1]
        self.indices = df["frame_idx"].to_list()
        self.timestamps = df["timestamp"].to_list()

        self.timestamp_stretched_dict: dict[int, float] | None = None
        self.timestamps_stretched: list[float] | None = None
        self._ifi_stretched: float | None = None
        self.has_stretched = "timestamp_stretched" in df.columns
        if self.has_stretched:
            self.timestamp_stretched_dict = df.to_dict()["timestamp_stretched"]
            self.timestamps_stretched = df["timestamp_stretched"].to_list()

    def get_timestamp(self, idx: int, which: Type = Type.Normal) -> float:
        """Look up the timestamp for a specific frame index.

        Args:
            idx: Frame index to look up.
            which: Which timestamp column to use.

        Returns:
            The timestamp, or -1.0 if *idx* is not in the table.

        Raises:
            RuntimeError: If stretched timestamps are requested but unavailable.

        """
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
        """Find the frame index whose timestamp is nearest to *ts*.

        Uses binary search (``bisect``) on the sorted timestamp list, then
        compares the two candidates straddling the insertion point.

        Args:
            ts: Target timestamp to search for.
            which: Which timestamp column to search in.

        Returns:
            The frame index of the nearest timestamp.

        Raises:
            RuntimeError: If stretched timestamps are requested but unavailable.

        """
        match which:
            case Type.Normal:
                timestamps = self.timestamps
            case Type.Stretched:
                if not self.has_stretched:
                    raise RuntimeError("stretched timestamps are not available for this video")
                timestamps = self.timestamps_stretched

        # bisect gives the insertion point; compare neighbors for nearest
        idx = bisect.bisect(timestamps, ts)
        if idx >= len(timestamps):
            return self.indices[-1]
        if idx > 0 and abs(timestamps[idx - 1] - ts) < abs(timestamps[idx] - ts):
            return self.indices[idx - 1]
        return self.indices[idx]

    def get_last(self, which: Type = Type.Normal) -> tuple[int, float]:
        """Return the last frame index and its timestamp.

        Args:
            which: Which timestamp column to use.

        Returns:
            A ``(frame_index, timestamp)`` tuple for the final frame.

        Raises:
            RuntimeError: If stretched timestamps are requested but unavailable.

        """
        match which:
            case Type.Normal:
                timestamps = self.timestamps
            case Type.Stretched:
                if not self.has_stretched:
                    raise RuntimeError("stretched timestamps are not available for this video")
                timestamps = self.timestamps_stretched
        return self.indices[-1], timestamps[-1]

    def get_ifi(self, which: Type = Type.Normal) -> float:
        """Return the mean inter-frame interval (IFI).

        The result is in the same units as the stored timestamps
        (typically milliseconds). Computed lazily and cached.

        Args:
            which: Which timestamp column to compute IFI from.

        Returns:
            Mean time between consecutive frames.

        Raises:
            RuntimeError: If stretched timestamps are requested but unavailable.

        """
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
