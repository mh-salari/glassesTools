"""Fixation classification using I2MC on plane-projected gaze data.

Runs the I2MC (Identification by Two-Means Clustering) algorithm on
world-referenced gaze that has been projected onto a reference plane.
When per-eye signals are available, I2MC uses those for better robustness,
but fixation positions are recalculated using the ray or homography gaze
signal — which matches the visualization and is more reliable for some
devices.
"""

import math
import pathlib
import typing

import I2MC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import gaze_worldref, naming


def from_plane_gaze(
    gazes: str | pathlib.Path | dict[int, list[gaze_worldref.Gaze]],
    classification_intervals: list[list[int]],
    output_directory: str | pathlib.Path,
    I2MC_settings_override: dict[str, typing.Any] | None = None,
    filename_stem: str = naming.fixation_classification_prefix,
    do_plot: bool = True,
    plot_limits: list[list[float]] | None = None,
) -> None:
    """Run I2MC fixation classification on gaze data projected to a plane.

    Processes each classification interval independently, writing per-interval
    TSV results and optional diagnostic plots.

    Args:
        gazes: World-referenced gaze samples, or path to a TSV file.
        classification_intervals: ``[[start, end], ...]`` frame ranges.
            Use ``end=-1`` for "until end of recording".
        output_directory: Directory for output TSV and plot files.
        I2MC_settings_override: Optional dict to override I2MC parameters.
        filename_stem: Prefix for output filenames.
        do_plot: Whether to generate diagnostic plots.
        plot_limits: Axis limits for plots (``[[xmin, xmax], [ymin, ymax]]``).

    Raises:
        RuntimeError: If no gaze data channels are available.

    """
    output_directory = pathlib.Path(output_directory)

    # read input if needed
    if not isinstance(gazes, dict):
        gazes = gaze_worldref.read_dict_from_file(gazes)

    # set I2MC options
    opt = {"xres": None, "yres": None}  # dummy values for required options
    opt["missingx"] = math.nan
    opt["missingy"] = math.nan
    opt["maxdisp"] = 50  # mm
    opt["windowtimeInterp"] = 0.25  # s
    opt["maxMergeDist"] = 20  # mm
    opt["maxMergeTime"] = 81  # ms
    opt["minFixDur"] = 50  # ms

    # I2MC requires a fixed sampling frequency, but eye trackers have varying
    # rates. The exact value doesn't matter much (I2MC mainly uses it to convert
    # time-based parameters to samples). Snap to the nearest known frequency
    # for which we have tested I2MC filter settings.
    ts = np.array([s.timestamp for v in gazes.values() for s in v])
    ts_diff = np.diff(ts)
    ts_diff = ts_diff[ts_diff > 0]  # drop zero/negative gaps (duplicate timestamps)
    rec_freq = np.round(np.mean(1000.0 / ts_diff))  # empirical Hz
    known_freqs = [30.0, 50.0, 60.0, 90.0, 120.0, 200.0]
    opt["freq"] = known_freqs[np.abs(known_freqs - rec_freq).argmin()]
    if opt["freq"] == 200.0:
        pass  # defaults are good
    elif opt["freq"] == 120.0:
        opt["downsamples"] = [2, 3, 5]
        opt["chebyOrder"] = 7
    elif opt["freq"] in {50.0, 60.0}:
        opt["downsamples"] = [2, 5]
        opt["downsampFilter"] = False
    else:
        # 90 Hz, 30 Hz
        opt["downsamples"] = [2, 3]
        opt["downsampFilter"] = False

    # apply setting overrides from caller, if any
    if I2MC_settings_override:
        for k in I2MC_settings_override:
            if I2MC_settings_override[k] is not None:
                opt[k] = I2MC_settings_override[k]

    # Probe which gaze channels have any non-NaN data across the whole recording
    has_left = np.any(np.logical_not(np.isnan([s.gazePosPlane2DLeft for v in gazes.values() for s in v])))
    has_right = np.any(np.logical_not(np.isnan([s.gazePosPlane2DRight for v in gazes.values() for s in v])))
    has_ray = np.any(np.logical_not(np.isnan([s.gazePosPlane2D_vidPos_ray for v in gazes.values() for s in v])))
    has_homography = np.any(
        np.logical_not(np.isnan([s.gazePosPlane2D_vidPos_homography for v in gazes.values() for s in v]))
    )
    for idx, iv in enumerate(classification_intervals):
        gazes_to_classify = {k: v for (k, v) in gazes.items() if k >= iv[0] and (iv[1] == -1 or k <= iv[1])}
        # Doing detection on the world data if available is good, but we should plot using the ray (if
        # available) or homography data, as that corresponds to the gaze visualization provided in the
        # software, and for some recordings/devices the world-based coordinates can be far off.
        if has_ray:
            ray_x = np.array([s.gazePosPlane2D_vidPos_ray[0] for v in gazes_to_classify.values() for s in v])
            ray_y = np.array([s.gazePosPlane2D_vidPos_ray[1] for v in gazes_to_classify.values() for s in v])
        elif has_homography:
            homography_x = np.array([
                s.gazePosPlane2D_vidPos_homography[0] for v in gazes_to_classify.values() for s in v
            ])
            homography_y = np.array([
                s.gazePosPlane2D_vidPos_homography[1] for v in gazes_to_classify.values() for s in v
            ])

        data = {}
        data["time"] = np.array([s.timestamp for v in gazes_to_classify.values() for s in v])
        need_recalc_fix = False
        if has_left and has_right:
            # Per-eye signals give I2MC better robustness for classification,
            # but fixation positions need recalculating with ray/homography data
            data["L_X"] = np.array([s.gazePosPlane2DLeft[0] for v in gazes_to_classify.values() for s in v])
            data["L_Y"] = np.array([s.gazePosPlane2DLeft[1] for v in gazes_to_classify.values() for s in v])
            data["R_X"] = np.array([s.gazePosPlane2DRight[0] for v in gazes_to_classify.values() for s in v])
            data["R_Y"] = np.array([s.gazePosPlane2DRight[1] for v in gazes_to_classify.values() for s in v])
            need_recalc_fix = True
        elif has_ray:
            data["average_X"] = ray_x
            data["average_Y"] = ray_y
        elif has_homography:
            data["average_X"] = homography_x
            data["average_Y"] = homography_y
        else:
            raise RuntimeError("No data available to process")

        # run event classification to find fixations
        fixations, data_i2mc, par_i2mc = I2MC.I2MC(data, opt, False)

        # When per-eye data was used for classification, replace it with the
        # ray/homography signal and recalculate fixation positions — per-eye
        # world coordinates can be inaccurate for some devices
        if need_recalc_fix:
            data_i2mc = data_i2mc.drop(columns=["L_X", "L_Y", "R_X", "R_Y"], errors="ignore")
            data_i2mc["average_X"] = ray_x if has_ray else homography_x
            data_i2mc["average_Y"] = ray_y if has_ray else homography_y
            # recalculate fixation positions based on gaze position on video data
            fixations = I2MC.get_fixations(
                data_i2mc["finalweights"].array,
                data_i2mc["time"].array,
                data_i2mc["average_X"],
                data_i2mc["average_Y"],
                data_i2mc["average_missing"],
                par_i2mc,
            )

        # store to file
        fix_df = pd.DataFrame(fixations)
        fix_df.to_csv(
            output_directory / f"{filename_stem}_interval_{idx + 1:02d}.tsv",
            mode="w",
            na_rep="nan",
            sep="\t",
            index=False,
            float_format="%.3f",
        )

        # make timeseries plot of gaze data with fixations
        if do_plot:
            f = I2MC.plot.data_and_fixations(data_i2mc, fixations, fix_as_line=True, unit="mm", res=plot_limits)
            plt.gca().invert_yaxis()
            f.savefig(str(output_directory / f"{filename_stem}_interval_{idx + 1:02d}.png"))
            plt.close(f)
