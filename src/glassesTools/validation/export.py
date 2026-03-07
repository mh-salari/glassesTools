"""Export and summarize validation data quality metrics."""

import pathlib

import pandas as pd

from .. import data_types


def collect_data_quality(
    rec_dirs: list[str | pathlib.Path],
    file_name: str | dict[str, str] = "dataQuality.tsv",
    col_for_parent: str | None = None,
) -> tuple[pd.DataFrame | None, data_types.DataType | None, list[int] | None]:
    """Collect data quality metrics from recording directories into a single DataFrame.

    Reads ``dataQuality.tsv`` (or a dict of per-plane filenames) from
    each recording directory, concatenates them, converts the ``type``
    index level to :class:`~glassesTools.data_types.DataType` enum
    values, and selects a default data quality type for export.

    Args:
        rec_dirs: List of recording directory paths to scan.
        file_name: Either a single TSV filename or a dict mapping plane
            names to filenames.
        col_for_parent: If set, adds a column with this name containing
            the parent directory name of each recording.

    Returns:
        A tuple of (concatenated DataFrame, default DataType for export,
        list of target IDs), or ``(None, None, None)`` if no files were
        found or the result is empty.

    """
    rec_files: list = []
    idx_vals = ["recording"]
    if isinstance(file_name, dict):
        for f in file_name:
            for d in rec_dirs:
                f_path = pathlib.Path(d) / file_name[f]
                if not f_path.is_file():
                    continue
                kwargs = {"recording": f_path.parent.name, "plane": f}
                if col_for_parent:
                    kwargs[col_for_parent] = f_path.parent.parent.name
                rec_files.append((f_path, kwargs))
        idx_vals.append("plane")
        if col_for_parent:
            idx_vals.insert(0, col_for_parent)
    else:
        rec_files = [(pathlib.Path(rec) / file_name, {"recording": rec.name}) for rec in rec_dirs]
        rec_files = [f for f in rec_files if f[0].is_file()]
    if not rec_files:
        return None, None, None
    df = pd.concat((pd.read_csv(rec[0], delimiter="\t").assign(**rec[1]) for rec in rec_files), ignore_index=True)
    if df.empty:
        return None, None, None
    # set indices
    df = df.set_index([*idx_vals, "marker_interval", "type", "target"])
    # change type index into enum
    type_idx = df.index.names.index("type")
    df.index = df.index.set_levels(
        pd.CategoricalIndex([getattr(data_types.DataType, x) for x in df.index.levels[type_idx]]), level="type"
    )

    # see what we have
    dq_types = sorted(df.index.levels[type_idx], key=lambda dq: dq.value)
    targets = list(df.index.levels[df.index.names.index("target")])

    # good default selection of dq type to export
    if data_types.DataType.pose_vidpos_ray in dq_types:
        default_dq_type = data_types.DataType.pose_vidpos_ray
    elif data_types.DataType.pose_vidpos_homography in dq_types:
        default_dq_type = data_types.DataType.pose_vidpos_homography
    else:
        # ultimate fallback, just set first available as the one to export
        default_dq_type = dq_types[0]

    return df, default_dq_type, targets


def summarize_and_store_data_quality(
    df: pd.DataFrame,
    output_file_or_dir: str | pathlib.Path,
    dq_types: list[data_types.DataType],
    targets: list[int],
    average_over_targets: bool = False,
    include_data_loss: bool = False,
) -> None:
    """Filter, optionally average, and write data quality metrics to TSV.

    Removes unwanted data types and targets from the DataFrame,
    optionally averages metrics across targets (adding a
    ``num_targets`` count column), and writes the result to a TSV file.

    Args:
        df: Multi-indexed DataFrame from :func:`collect_data_quality`.
        output_file_or_dir: Output TSV path or directory (defaults to
            ``dataQuality.tsv`` if a directory).
        dq_types: Data quality types to keep.
        targets: Target IDs to keep.
        average_over_targets: If True, average metrics across targets.
        include_data_loss: If True, keep the ``data_loss`` column.

    """
    dq_types_have = sorted(df.index.levels[df.index.names.index("type")], key=lambda dq: dq.value)
    targets_have = list(df.index.levels[df.index.names.index("target")])

    # remove unwanted types of data quality
    dq_types_sel = [dq in dq_types for dq in dq_types_have]
    if not all(dq_types_sel):
        df = df.drop(index=[dq for i, dq in enumerate(dq_types_have) if not dq_types_sel[i]], level="type")
    # remove unwanted targets
    targets_sel = [t in targets for t in targets_have]
    if not all(targets_sel):
        df = df.drop(index=[t for i, t in enumerate(targets_have) if not targets_sel[i]], level="target")
    # remove unwanted data loss
    if not include_data_loss and "data_loss" in df.columns:
        df = df.drop(columns="data_loss")
    # average data if wanted
    if average_over_targets:
        gb = df.drop(columns="order").groupby([n for n in df.index.names if n != "target"], observed=True)
        count = gb.count()
        df = gb.mean()
        # add number of targets count (there may be some missing data)
        df.insert(0, "num_targets", count["acc"])

    # store
    output_file_or_dir = pathlib.Path(output_file_or_dir)
    if output_file_or_dir.is_dir():
        output_file_or_dir /= "dataQuality.tsv"
    df.to_csv(output_file_or_dir, mode="w", header=True, sep="\t", na_rep="nan", float_format="%.6f")


def export_data_quality(
    rec_dirs: list[str | pathlib.Path],
    output_file_or_dir: str | pathlib.Path,
    dq_types: list[data_types.DataType] | None = None,
    targets: list[int] | None = None,
    average_over_targets: bool = False,
    include_data_loss: bool = False,
) -> None:
    """Collect, summarize, and export data quality metrics from recording directories.

    Convenience wrapper that calls :func:`collect_data_quality` and
    :func:`summarize_and_store_data_quality`.

    Args:
        rec_dirs: List of recording directory paths.
        output_file_or_dir: Output TSV path or directory.
        dq_types: Data quality types to export. If None, uses the
            default from :func:`collect_data_quality`.
        targets: Target IDs to include. If None, includes all.
        average_over_targets: If True, average metrics across targets.
        include_data_loss: If True, include ``data_loss`` in output.

    """
    df, default_dq_type, targets_have = collect_data_quality(rec_dirs)
    if not dq_types:
        dq_types = [default_dq_type]
    if not targets:
        targets = targets_have
    summarize_and_store_data_quality(
        df, output_file_or_dir, dq_types, targets, average_over_targets, include_data_loss
    )


def export_et_sync(
    rec_dirs: list[str | pathlib.Path], in_file_name: str, output_file_or_dir: str | pathlib.Path
) -> None:
    """Collect and export eye tracker synchronization data from recording directories.

    Reads per-recording sync TSV files, concatenates them with
    ``session`` and ``recording`` columns, and writes the result.

    Args:
        rec_dirs: List of recording directory paths.
        in_file_name: Name of the sync TSV file within each recording
            directory.
        output_file_or_dir: Output TSV path or directory (defaults to
            ``et_sync.tsv`` if a directory).

    """
    sync_files = [
        (pathlib.Path(rec) / in_file_name, {"recording": rec.name, "session": rec.parent.name}) for rec in rec_dirs
    ]
    sync_files = [f for f in sync_files if f[0].is_file()]
    # get all sync files
    df = pd.concat((pd.read_csv(sync[0], delimiter="\t").assign(**sync[1]) for sync in sync_files), ignore_index=True)
    if df.empty:
        return
    df = df.set_index(["session", "recording", "interval"])
    # store
    output_file_or_dir = pathlib.Path(output_file_or_dir)
    if output_file_or_dir.is_dir():
        output_file_or_dir /= "et_sync.tsv"
    df.to_csv(output_file_or_dir, mode="w", header=True, sep="\t", na_rep="nan", float_format="%.6f")
