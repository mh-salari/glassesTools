"""Assign fixation intervals to validation targets."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import gaze_worldref, marker, naming


def distance(
    targets: dict[int, np.ndarray],
    fixations: str | pathlib.Path | pd.DataFrame,
    do_global_shift: bool = True,
    max_dist_fac: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Assign fixation intervals to targets based on spatial distance.

    For each target, finds the nearest unused fixation that exceeds a
    minimum duration. A global shift can be applied to center fixations
    and targets before matching, improving robustness when gaze data has
    an overall offset. Fixations farther than ``max_dist_fac`` times the
    nearest inter-target distance are rejected.

    Args:
        targets: Mapping of target ID to position array ``[x, y]``.
        fixations: Either a DataFrame or path to a TSV file with columns
            ``xpos``, ``ypos``, ``dur``, ``start``, ``startT``, ``endT``.
        do_global_shift: If True, center fixations and targets by
            removing the median fixation offset and mean target offset
            before distance computation.
        max_dist_fac: Maximum allowed distance to a target, expressed as
            a fraction of the smallest inter-target distance.

    Returns:
        A tuple of (selected intervals DataFrame indexed by target ID,
        unselected intervals DataFrame or None if all fixations were
        assigned).

    """
    # read input if needed
    if not isinstance(fixations, pd.DataFrame):
        fixations = pd.read_csv(fixations, sep="\t", index_col=False)

    # for each target, find closest fixation
    min_dur = 100  # ms
    used = np.zeros((fixations["start"].size), dtype="bool")
    selected = np.empty((len(targets),), dtype="int")
    selected[:] = -999

    t_x = np.array([targets[t][0] for t in targets])
    t_y = np.array([targets[t][1] for t in targets])
    off_f_x = off_f_y = off_t_x = off_t_y = 0.0
    if do_global_shift:
        # first, center the problem. That means determine and remove any overall shift from the
        # data and the targets, to improve robustness of assigning fixations points to targets.
        # Else, if all data is e.g. shifted up by more than half the distance between
        # validation targets, target assignment would fail
        off_f_x = fixations["xpos"].median()
        off_f_y = fixations["ypos"].median()
        off_t_x = t_x.mean()
        off_t_y = t_y.mean()

    # reject fixations farther than max_dist_fac * intertarget distance
    dist_lim = np.inf
    if len(t_x) > 1:
        # arbitrarily take first target and find closest target to it
        dist = np.hypot(t_x[0] - t_x[1:], t_y[0] - t_y[1:])
        min_dist = dist.min()
        if min_dist > 0:
            dist_lim = min_dist * max_dist_fac

    for i, t in zip(range(len(targets)), targets, strict=True):
        if np.all(used):
            # all fixations used up, can't assign anything to remaining targets
            continue
        # select fixation
        dist = np.hypot(
            fixations["xpos"] - off_f_x - (targets[t][0] - off_t_x),
            fixations["ypos"] - off_f_y - (targets[t][1] - off_t_y),
        )
        dist[used] = np.inf  # make sure fixations already bound to a target are not used again
        dist[fixations["dur"] < min_dur] = np.inf  # make sure that fixations that are too short are not selected
        i_fix = np.argmin(dist)
        if dist[i_fix] <= dist_lim:
            selected[i] = i_fix
            used[i_fix] = True

    # prep return values
    selected_intervals = pd.DataFrame(columns=["xpos", "ypos", "startT", "endT"])  # sets which columns to copy
    selected_intervals.index.name = "target"
    for i, t in zip(range(len(targets)), targets, strict=True):
        if selected[i] == -999:
            continue
        selected_intervals.loc[t] = fixations.iloc[selected[i]]
    other_intervals = fixations.loc[np.logical_not(used), ["xpos", "ypos", "startT"]]
    return selected_intervals, None if other_intervals.empty else other_intervals


def dynamic_markers(
    marker_observations_per_target: dict[int, pd.DataFrame],  # {target: dataframe}
    markers_per_target: dict[int, list[marker.MarkerID]],  # {target: [marker,...]}, one or multiple markers per target
    timestamps_file: str | pathlib.Path,
    episode: list[int],
    skip_first_duration: float,
    max_gap_duration: int,
    min_duration: int,
    name: str = "",
    allow_missing: bool = False,
) -> tuple[pd.DataFrame, None]:
    """Assign intervals to targets using dynamic marker appearance signals.

    Each target is associated with one or more ArUco markers. This
    function examines marker detection data within the given episode,
    fills gaps in the detection signal, finds contiguous appearance
    windows, and selects the longest window for each target. The
    resulting intervals are expressed in recording timestamps (not
    frame indices).

    Args:
        marker_observations_per_target: Mapping of target ID to a
            DataFrame of marker detection observations.
        markers_per_target: Mapping of target ID to a list of
            ``MarkerID`` objects associated with that target.
        timestamps_file: Path to the frame timestamps TSV file used to
            convert frame indices to recording timestamps.
        episode: Two-element list ``[start_frame, end_frame]`` defining
            the episode to process.
        skip_first_duration: Time in ms to skip at the start of each
            detected appearance window (allowing fixation to settle).
        max_gap_duration: Maximum gap in frames between detections that
            is still treated as a single contiguous appearance.
        min_duration: Minimum duration in frames for an appearance
            window to be accepted.
        name: Optional episode name used in error messages.
        allow_missing: If True, silently skip targets with no marker
            observations instead of raising an error.

    Returns:
        A tuple of (selected intervals DataFrame indexed by target ID
        with ``startT`` and ``endT`` columns, None).

    Raises:
        RuntimeError: If no markers for a target were observed during
            the episode and ``allow_missing`` is False.

    """
    # frame timestamps are needed because the returned intervals should be in recording time, not frame indices
    timestamps = pd.read_csv(timestamps_file, delimiter="\t", index_col="frame_idx")
    ts_col = "timestamp_stretched" if "timestamp_stretched" in timestamps else "timestamp"

    # make local copy of marker_observations, containing only the current episode
    marker_observations_per_target = {
        t: mo.loc[episode[0] : episode[1], :] for t, mo in marker_observations_per_target.items()
    }
    # check we have data for at least one of the markers for a given target
    for t in marker_observations_per_target:
        if marker_observations_per_target[t].empty and not allow_missing:
            missing_str = "\n- ".join([marker.marker_id_to_str(m) for m in markers_per_target[t]])
            extra = (
                f"from frame {episode[0]} to frame {episode[1]}"
                if not name
                else f'"{name}" from frame {episode[0]} to frame {episode[1]}'
            )
            raise RuntimeError(
                f"None of the markers for target {t} were observed during the episode {extra}:\n- {missing_str}"
            )

    # fill gaps between detections with False to get a continuous presence signal
    marker_observations_per_target = {
        t: marker.expand_detection(marker_observations_per_target[t], fill_value=False)
        for t in marker_observations_per_target
        if not marker_observations_per_target[t].empty
    }

    # for each target, see when it is presented using the marker presence signal
    selected_intervals = pd.DataFrame(columns=["startT", "endT"])
    selected_intervals.index.name = "target"
    for t in marker_observations_per_target:
        start, end = marker.get_appearance_starts_ends(
            marker_observations_per_target[t], max_gap_duration, min_duration
        )
        if start.size == 0:
            continue
        # in case there are multiple (e.g. spotty detection), choose longest
        durs = np.array(end) - np.array(start) + 1
        maxi = np.argmax(durs)
        ts = timestamps.loc[[start[maxi], end[maxi]], ts_col].to_numpy(copy=True)
        ts[0] += skip_first_duration
        if ts[0] >= ts[1]:
            continue
        selected_intervals.loc[t] = ts
    return selected_intervals, None


def plot(
    selected_intervals: pd.DataFrame,
    other_intervals: pd.DataFrame | None,
    targets: dict[int, np.ndarray],
    gazes: str | pathlib.Path | dict[int, list[gaze_worldref.Gaze]],
    episode: list[int],
    output_directory: str | pathlib.Path,
    filename_stem: str = naming.fixation_assignment_prefix,
    iteration: int = 0,
    background_image: tuple[np.ndarray, list[float]] | None = None,  # (image, extent in mm [l r t b])
    plot_limits: list[list[float]] | None = None,
) -> None:
    """Plot fixation-to-target assignment overlaid on the poster image.

    Draws all fixation intervals as a connected path, then overlays red
    lines from each assigned fixation to its matched target. If the
    intervals lack ``xpos``/``ypos`` columns, gaze positions are computed
    from the raw gaze data within each interval's time range.

    Args:
        selected_intervals: DataFrame of assigned intervals, indexed by
            target ID with ``startT``, ``endT``, and optionally ``xpos``
            and ``ypos`` columns.
        other_intervals: DataFrame of unassigned intervals (same
            columns), or None.
        targets: Mapping of target ID to position array ``[x, y]``.
        gazes: Either a dict of frame-indexed gaze samples or a path to
            a gaze data file.
        episode: Two-element list ``[start_frame, end_frame]``.
        output_directory: Directory where the plot PNG will be saved.
        filename_stem: Prefix for the output filename.
        iteration: Zero-based iteration index, used in the filename.
        background_image: Tuple of (image array, extent ``[l, r, t, b]``
            in mm) for the poster background, or None.
        plot_limits: Optional ``[[x_min, x_max], [y_min, y_max]]`` axis
            limits.

    """
    output_directory = pathlib.Path(output_directory)
    # if we do not have x and y positions for the gaze intervals, make them
    if "xpos" not in selected_intervals.columns or (
        other_intervals is not None and "xpos" not in other_intervals.columns
    ):
        # read input if needed
        if not isinstance(gazes, dict):
            gazes = gaze_worldref.read_dict_from_file(gazes)
        if "xpos" not in selected_intervals.columns:
            samples_per_frame = {k: v for (k, v) in gazes.items() if k >= episode[0] and k <= episode[1]}
            has_ray = np.any(
                np.logical_not(np.isnan([s.gazePosPlane2D_vidPos_ray for v in gazes.values() for s in v]))
            )
            field = "gazePosPlane2D_vidPos_ray" if has_ray else "gazePosPlane2D_vidPos_homography"
            for t in selected_intervals.index:
                st, et = selected_intervals.loc[t, ["startT", "endT"]].to_numpy()
                data = [
                    getattr(s, field)
                    for v in samples_per_frame.values()
                    for s in v
                    if s.timestamp >= st and s.timestamp <= et
                ]
                if data:
                    gaze = np.vstack(data)
                    selected_intervals.loc[t, ["xpos", "ypos"]] = np.nanmedian(gaze, axis=0)
        if other_intervals is not None and "xpos" not in other_intervals.columns:
            for t in other_intervals.index:
                st, et = other_intervals.loc[t, ["startT", "endT"]].to_numpy()
                data = [
                    getattr(s, field)
                    for v in samples_per_frame.values()
                    for s in v
                    if s.timestamp >= st and s.timestamp <= et
                ]
                if data:
                    gaze = np.vstack(data)
                    other_intervals.loc[t, ["xpos", "ypos"]] = np.nanmedian(gaze, axis=0)
    # combine intervals for a single time-ordered path
    if other_intervals is not None:
        all_intervals = pd.concat(
            (selected_intervals.set_index("startT"), other_intervals.set_index("startT")), join="inner"
        ).sort_index()
    else:
        all_intervals = selected_intervals.set_index("startT").sort_index()

    f = plt.figure(dpi=300)
    plt.imshow(background_image[0], extent=background_image[1], alpha=0.5)
    # draw all intervals
    plt.plot(all_intervals["xpos"], all_intervals["ypos"], "b-")
    plt.plot(all_intervals["xpos"], all_intervals["ypos"], "go")
    # draw target matching
    for t, row in selected_intervals.iterrows():
        plt.plot([row["xpos"], targets[t][0]], [row["ypos"], targets[t][1]], "r-")

    # cosmetics
    plt.xlabel("mm")
    plt.ylabel("mm")
    if plot_limits is not None:
        plt.xlim(plot_limits[0])
        plt.ylim(plot_limits[1])
    plt.gca().invert_yaxis()

    f.savefig(output_directory / f"{filename_stem}_interval_{iteration + 1:02d}.png")
    plt.close(f)


def to_tsv(
    selected_intervals: pd.DataFrame,
    output_directory: str | pathlib.Path,
    filename_stem: str = naming.fixation_assignment_prefix,
    iteration: int = 0,
) -> None:
    """Write selected intervals to a TSV file.

    Drops position columns (``xpos``, ``ypos``) and renames time
    columns to ``start_timestamp`` and ``end_timestamp``. Appends to the
    file for iterations after the first.

    Args:
        selected_intervals: DataFrame of assigned intervals indexed by
            target ID.
        output_directory: Directory where the TSV file will be written.
        filename_stem: Prefix for the output filename.
        iteration: Zero-based iteration index. The first iteration
            writes a header; subsequent iterations append.

    Raises:
        ValueError: If the ``marker_interval`` column exists but its
            values don't match the expected iteration number.

    """
    output_directory = pathlib.Path(output_directory)
    selected_intervals = selected_intervals.drop(
        columns=[c for c in ("xpos", "ypos") if c in selected_intervals.columns]
    ).rename(columns={"startT": "start_timestamp", "endT": "end_timestamp"})
    if "marker_interval" in selected_intervals.columns:
        if not all(selected_intervals["marker_interval"] == iteration + 1):
            raise ValueError(f"marker_interval column values do not match expected iteration {iteration + 1}")
    else:
        selected_intervals.insert(0, "marker_interval", iteration + 1)

    selected_intervals.to_csv(
        output_directory / f"{filename_stem}.tsv",
        mode="w" if iteration == 0 else "a",
        header=iteration == 0,
        sep="\t",
        na_rep="nan",
        float_format="%.3f",
    )
