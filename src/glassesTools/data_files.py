"""Reading and writing tabular data files for eye tracker recordings."""

import importlib
import pathlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pandas as pd
import polars as pl


def get_column_labels(lbl: str, n: int = 3) -> list[str]:
    """Generate column label suffixes for multi-component data fields."""
    if n <= 3:
        return [f"{lbl}_{chr(c)}" for c in range(ord("x"), ord("x") + n)]
    if n == 9:
        return [f"{lbl}[{r},{c}]" for r in range(3) for c in range(3)]
    raise ValueError(f"n input should be <=3 or 9, was {n}")


def none_if_any_nan(vals: np.ndarray) -> np.ndarray | None:
    """Return the array as-is if it contains no NaN values, otherwise return None."""
    if not np.any(np.isnan(vals)):
        return vals
    return None


def all_nan_if_none(vals: np.ndarray | None, numel: int) -> np.ndarray:
    """Return the array, or a NaN-filled array of length *numel* if *vals* is None."""
    if vals is None:
        return np.full((numel,), np.nan)
    return vals


# read coordinate files (e.g. marker files, which have the colums ID, x, y, rotation_angle)
def _read_coord_file_impl(file: str | pathlib.Path) -> pd.DataFrame:
    return (
        pd
        .read_csv(file, dtype=defaultdict(lambda: np.float32, ID="int32", color="str"))
        .dropna(axis=0, how="all")
        .set_index("ID")
    )


def read_coord_file(file: str | pathlib.Path, package_to_read_from: str | None = None) -> pd.DataFrame | None:
    """Read a coordinate file (e.g. marker file with columns ID, x, y, rotation_angle).

    If *package_to_read_from* is given, read from package resources instead of the filesystem.
    """
    if package_to_read_from:
        with importlib.resources.path(package_to_read_from, file) as p:
            return _read_coord_file_impl(p)
    if file.is_file():
        return _read_coord_file_impl(file)
    return None


def uncompress_columns(cols_compressed: dict[str, int]) -> list[list[str]]:
    """Expand compressed column definitions into lists of individual column names."""
    return [get_column_labels(c, N) if (N := cols_compressed[c]) > 1 else [c] for c in cols_compressed]


def _get_col_name_with_suffix(base: str, suf: str) -> str:
    if not suf:
        return base
    return base + "_" + suf


def read_file(
    file_name: str | pathlib.Path,
    record_class: Any,
    drop_if_all_nan: bool,
    put_none_if_any_nan: bool,
    as_list_dict: bool,
    make_ori_ts_fridx: bool,
    episodes: list[list[int]] | None = None,
    ts_fridx_field_suffixes: list[str] | None = None,
    subset_var: str = "frame_idx",
) -> tuple[dict, Any]:
    """Read a tab-separated data file and return records grouped by *subset_var*."""
    # interrogate destination record_class
    cols_compressed: dict[str, int] = record_class._columns_compressed
    dtypes: dict[str, Any] = record_class._non_float
    column_patches: dict[str, tuple[str, Callable]] = (
        record_class._column_patches if hasattr(record_class, "_column_patches") else None
    )
    # add patches, if any, to dtypes so these column are read correctly to
    if column_patches is not None:
        dtypes |= {on: dtypes[nn] for on, (nn, _) in column_patches.items()}

    # read file and select, if wanted
    df = pd.read_csv(file_name, delimiter="\t", index_col=False, dtype=defaultdict(lambda: float, **dtypes))
    if episodes:
        sel = (df[subset_var] >= episodes[0][0]) & (df[subset_var] <= episodes[0][1])
        for e in episodes[1:]:
            sel |= (df[subset_var] >= e[0]) & (df[subset_var] <= e[1])
        df = df[sel]

    # if we have column renaming to do, do it now
    if column_patches is not None:
        # apply operations, if any
        for on, (_, op) in column_patches.items():
            if on not in df.columns or op is None:
                continue
            df[on] = op(df[on])
        # rename columns
        df = df.rename(columns={on: nn for on, (nn, _) in column_patches.items()})

    # figure out what the data columns are
    cols_uncompressed = uncompress_columns(cols_compressed)

    # drop rows where are all data columns are nan
    if drop_if_all_nan:
        df = df.dropna(how="all", subset=[c for cs in cols_uncompressed if len(cs) > 1 for c in cs])

    # group columns into numpy arrays, optionally insert None if missing
    for c, ac in zip(cols_compressed, cols_uncompressed, strict=True):
        if len(ac) == 1:
            continue  # nothing to do, would just copy column to itself
        if ac:
            if not any(a in df.columns for a in ac):
                continue
            if put_none_if_any_nan:
                df[c] = [
                    none_if_any_nan(x) for x in df[ac].to_numpy()
                ]  # make list of numpy arrays, or None if there are any NaNs in the array
            else:
                df[c] = list(df[ac].to_numpy())  # make list of numpy arrays
        else:
            df[c] = None

    # keep only the columns we want (this also puts them in the right order even if that doesn't matter since we use kwargs to construct objects)
    df = df[[c for c in cols_compressed if c in df.columns]]

    # if we have multiple timestamps of frame_idxs, make sure we keep a copy of the original one
    if make_ori_ts_fridx:
        df = df.copy()  # after the above operation to keep only the columns we want, we're looking at a slice of the larger df. To be able to make changes as we will here, use copy to get a new df with just the expected columns
        # make copies of original
        df["frame_idx_ori"] = df["frame_idx"]
        if "timestamp" in df.columns:
            df["timestamp_ori"] = df["timestamp"]
    # now put requested into normal timestamp column, if wanted
    if make_ori_ts_fridx and ts_fridx_field_suffixes:
        copied = False
        for suf in ts_fridx_field_suffixes:  # these are in order of preference
            field = _get_col_name_with_suffix("frame_idx", suf)
            if field not in df.columns:
                continue
            df["frame_idx"] = df[field]
            if "timestamp" in df.columns:
                df["timestamp"] = df[_get_col_name_with_suffix("timestamp", suf)]
            copied = True
            break
        if not copied:
            raise ValueError("None of the specified suffixes were found, can't continue")

    if as_list_dict:
        obj_list = [record_class(**kwargs) for kwargs in df.to_dict(orient="records")]

        # organize into dict by frame index
        objs = {}
        for k, v in zip(df[subset_var], obj_list, strict=True):
            objs.setdefault(k, []).append(v)
    else:
        objs = {
            idx: record_class(**kwargs)
            for idx, kwargs in zip(df[subset_var], df.to_dict(orient="records"), strict=True)
        }
    return objs, df[subset_var].max()


def write_array_to_file(
    objects: list[Any] | dict[int, list[Any]],
    file_name: str | pathlib.Path,
    cols_compressed: dict[str, int],
    drop_all_nan_cols: list[str] | None = None,
    skip_all_nan: bool = False,
) -> None:
    """Write a collection of data objects to a tab-separated file."""
    if not objects:
        return

    if drop_all_nan_cols is None:
        drop_all_nan_cols = []

    if isinstance(objects, dict):
        # flatten
        objects = [o for olist in objects.values() for o in olist]

    records = [{k: getattr(p, k) for k in vars(p) if not k.startswith("_")} for p in objects]
    df = pd.DataFrame.from_records(records)

    # unpack array columns
    cols_uncompressed = uncompress_columns(cols_compressed)
    for c, ac in zip(cols_compressed, cols_uncompressed, strict=True):
        if len(ac) > 1:
            df[ac] = np.vstack([all_nan_if_none(v, len(ac)).flatten() for v in df[c].to_numpy()])

    # if wanted, drop specific columns for which all rows are nan
    if drop_all_nan_cols:
        df = df.drop([c for c in drop_all_nan_cols if c in df and df[c].isna().all()], axis="columns")

    # if we have filled _ori timestamp and frame_idx columns, copy them back into plain timestamp
    # and frame_idx columns. NB: the _ori columns will be removed because they're not listed in
    # the possible file columns
    if "timestamp_ori" in df.columns and not df["timestamp_ori"].isna().all():
        df["timestamp"] = df["timestamp_ori"]
    if "frame_idx_ori" in df.columns and not df["frame_idx_ori"].isna().all():
        df["frame_idx"] = df["frame_idx_ori"]

    # keep only columns to be written out and order them correctly
    df = df[[c for cs in cols_uncompressed for c in cs if c in df.columns]]

    # drop rows where are all data columns are nan
    if skip_all_nan:
        df = df.dropna(how="all", subset=[c for cs in cols_uncompressed if len(cs) > 1 for c in cs])

    # convert to polars as that library saves to file waaay faster
    df = pl.from_pandas(df)
    df.write_csv(file_name, separator="\t", null_value="nan", float_precision=8)
