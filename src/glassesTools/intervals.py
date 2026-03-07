"""Utilities for checking frame membership in annotation intervals.

Intervals are ``(EventType, bounds)`` tuples. *bounds* is either a flat list
``[start, end, start, end, ...]`` (as stored on disk) or a nested list of pairs
``[[start, end], [start, end], ...]`` (as used at runtime). Point events have
single-element bounds ``[frame]``. Functions here accept both formats and
unflatten automatically when needed.
"""

from . import annotation


def is_in_interval(frame_idx: int, intervals: tuple[annotation.EventType, list[int] | list[list[int]]]) -> bool:
    """Check whether *frame_idx* falls within any of the given intervals.

    Args:
        frame_idx: Frame index to test.
        intervals: An ``(EventType, bounds)`` tuple, or None to indicate
            all frames are valid.

    Returns:
        True if *frame_idx* is inside any interval (or if *intervals* is None).

    """
    if intervals is None:
        return True  # no interval defined, all frames should be processed

    # intervals[1] is the bounds list from the (EventType, bounds) tuple
    for iv in intervals[1]:
        if len(iv) == 1:
            # point event — match only the exact frame
            if frame_idx == iv[0]:
                return True
        elif frame_idx >= iv[0] and frame_idx <= iv[1]:
            return True
    return False


def which_interval(
    frame_idx: int, intervals: dict[str, tuple[annotation.EventType, list[int] | list[list[int]]]]
) -> tuple[list[str] | None, list[list[int]] | None]:
    """Find all named annotation intervals that contain *frame_idx*.

    Args:
        frame_idx: Frame index to test.
        intervals: Dict mapping annotation names to ``(EventType, bounds)``
            tuples. Bounds may be flat or nested — flat bounds are
            unflattened automatically.

    Returns:
        A ``(keys, ivals)`` tuple where *keys* lists the matching annotation
        names and *ivals* lists the corresponding ``[start, end]`` bounds,
        or ``(None, None)`` if *intervals* is empty or not a dict.

    """
    if not isinstance(intervals, dict) or not intervals:
        return None, None
    # Auto-unflatten: if any category's first bound element is a scalar
    # rather than a list, the bounds are still in flat on-disk format
    if any(not isinstance(intervals[k][1][0], list) for k in intervals if intervals[k][1]):
        intervals = annotation.unflatten_annotation_dict(intervals, add_incomplete_intervals=True)

    keys = []
    ivals = []
    for k in intervals:
        for iv in intervals[k][1]:
            if len(iv) == 1:
                if frame_idx == iv[0]:
                    keys.append(k)
                    ivals.append(iv)
            elif frame_idx >= iv[0] and frame_idx <= iv[1]:
                keys.append(k)
                ivals.append(iv)

    return keys, ivals


def beyond_last_interval(
    frame_idx: int, intervals: dict[str, tuple[annotation.EventType, list[int] | list[list[int]]]]
) -> bool:
    """Check whether *frame_idx* is past every annotation category's last interval.

    Used as an early-exit condition in frame processing loops: once the
    current frame is beyond all defined intervals, no further frames can
    match either.

    Args:
        frame_idx: Frame index to test.
        intervals: Dict mapping annotation names to ``(EventType, bounds)``
            tuples.

    Returns:
        True only if *frame_idx* exceeds the end of every category's last
        interval. Returns False if any category is unbounded (None) or empty.

    """
    if not intervals:
        return False
    if isinstance(intervals, dict):
        # If ANY category still has frames ahead, we can't exit early
        for iv_val in intervals.values():
            if iv_val is None:
                # None means "process all frames" — never exhausted
                return False
            if not iv_val[1]:
                return False
            # Check the last element of the last interval bound
            if isinstance(iv_val[1][-1], list):
                # nested format: [-1] is last interval, [-1][-1] is its end
                if frame_idx <= iv_val[1][-1][-1]:
                    return False
            elif frame_idx <= iv_val[1][-1]:
                # flat format: [-1] is the last value in the flat list
                return False
        # All categories exhausted
        return True
    return False
