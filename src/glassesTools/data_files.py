"""Reading and writing tab-separated data files for eye tracker recordings.

Multi-component fields (e.g. 3D coordinates, 3x3 rotation matrices) are stored in
TSV files as separate columns (``pos_x``, ``pos_y``, ``pos_z``) but represented in
memory as single numpy arrays. This mapping is defined by "compressed column"
dicts (``{field_name: num_components}``) that record classes provide via their
``_columns_compressed`` attribute.
"""

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
    """Generate column names for a multi-component field.

    For n<=3, produces spatial suffixes: ``lbl_x``, ``lbl_y``, ``lbl_z``.
    For n==9, produces 3x3 matrix indices: ``lbl[0,0]`` through ``lbl[2,2]``.

    Args:
        lbl: Base field name.
        n: Number of components (1-3 for spatial, 9 for 3x3 matrix).

    Returns:
        List of individual column names.

    Raises:
        ValueError: If *n* is not in the supported range.

    """
    if n <= 3:
        # Generates _x, _y, _z suffixes via ASCII codes starting from 'x'
        return [f"{lbl}_{chr(c)}" for c in range(ord("x"), ord("x") + n)]
    if n == 9:
        return [f"{lbl}[{r},{c}]" for r in range(3) for c in range(3)]
    raise ValueError(f"n input should be <=3 or 9, was {n}")


def none_if_any_nan(vals: np.ndarray) -> np.ndarray | None:
    """Return the array if it has no NaN values, otherwise None.

    Used when reading multi-component fields: if any component is missing,
    the entire field is treated as absent rather than partially filled.

    Args:
        vals: Array to check.

    Returns:
        The input array, or None if any element is NaN.

    """
    if not np.any(np.isnan(vals)):
        return vals
    return None


def all_nan_if_none(vals: np.ndarray | None, numel: int) -> np.ndarray:
    """Return the array, or a NaN-filled array of length *numel* if *vals* is None.

    Inverse of ``none_if_any_nan`` — used when writing multi-component fields back
    to TSV, where absent fields must be represented as NaN columns.

    Args:
        vals: Array, or None if the field is absent.
        numel: Expected number of elements (used to size the NaN placeholder).

    Returns:
        The input array, or a NaN-filled array if *vals* is None.

    """
    if vals is None:
        return np.full((numel,), np.nan)
    return vals


def _read_coord_file_impl(file: str | pathlib.Path) -> pd.DataFrame:
    """Read a coordinate CSV file (e.g. marker definitions: ID, x, y, rotation_angle).

    Uses a defaultdict dtype so all columns default to float32 except ID (int32)
    and color (str).
    """
    return (
        pd
        .read_csv(file, dtype=defaultdict(lambda: np.float32, ID="int32", color="str"))
        .dropna(axis=0, how="all")
        .set_index("ID")
    )


def read_coord_file(file: str | pathlib.Path, package_to_read_from: str | None = None) -> pd.DataFrame | None:
    """Read a coordinate file (e.g. marker definitions with columns ID, x, y, rotation_angle).

    Args:
        file: Filename or path to the coordinate file.
        package_to_read_from: If given, read from this package's resources
            instead of the filesystem.

    Returns:
        DataFrame indexed by ID, or None if the file doesn't exist.

    """
    if package_to_read_from:
        with importlib.resources.path(package_to_read_from, file) as p:
            return _read_coord_file_impl(p)
    if file.is_file():
        return _read_coord_file_impl(file)
    return None


def uncompress_columns(cols_compressed: dict[str, int]) -> list[list[str]]:
    """Expand compressed column definitions into lists of individual column names.

    E.g. ``{"pos": 3, "frame_idx": 1}`` becomes ``[["pos_x", "pos_y", "pos_z"], ["frame_idx"]]``.

    Args:
        cols_compressed: Dict mapping field names to their component counts.

    Returns:
        List of column name lists, one per field.

    """
    return [get_column_labels(c, N) if (N := cols_compressed[c]) > 1 else [c] for c in cols_compressed]


def _get_col_name_with_suffix(base: str, suf: str) -> str:
    """Join a base column name with a suffix, or return base if suffix is empty."""
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
    """Read a tab-separated data file and return records grouped by *subset_var*.

    The *record_class* must provide class attributes describing the file schema:
    ``_columns_compressed`` (field-to-component-count mapping), ``_non_float``
    (non-float dtype overrides), and optionally ``_column_patches`` (column
    rename/transform rules as ``{old_name: (new_name, transform_callable)}``).

    Args:
        file_name: Path to the TSV file.
        record_class: Class with schema attributes and used as constructor for records.
        drop_if_all_nan: Drop rows where all multi-component data columns are NaN.
        put_none_if_any_nan: Replace multi-component arrays with None if any
            component is NaN.
        as_list_dict: If True, group multiple records per key into lists.
            If False, one record per key.
        make_ori_ts_fridx: If True, preserve original timestamp/frame_idx before
            overwriting with a preferred suffix variant.
        episodes: Optional list of ``[start, end]`` frame index ranges to keep.
        ts_fridx_field_suffixes: Suffixes to try (in preference order) for
            overwriting the main timestamp/frame_idx columns.
        subset_var: Column name used as the grouping key.

    Returns:
        Tuple of (grouped records dict, max value of *subset_var*).

    Raises:
        ValueError: If *ts_fridx_field_suffixes* is given but none are found.

    """
    # record_class defines the file schema via class attributes
    cols_compressed: dict[str, int] = record_class._columns_compressed
    dtypes: dict[str, Any] = record_class._non_float
    column_patches: dict[str, tuple[str, Callable]] = (
        record_class._column_patches if hasattr(record_class, "_column_patches") else None
    )
    # If a column will be renamed (old_name → new_name), the old name in the file
    # needs the same dtype as the new name so pandas reads it correctly.
    if column_patches is not None:
        dtypes |= {on: dtypes[nn] for on, (nn, _) in column_patches.items()}

    # read file and select, if wanted
    df = pd.read_csv(file_name, delimiter="\t", index_col=False, dtype=defaultdict(lambda: float, **dtypes))
    if episodes:
        sel = (df[subset_var] >= episodes[0][0]) & (df[subset_var] <= episodes[0][1])
        for e in episodes[1:]:
            sel |= (df[subset_var] >= e[0]) & (df[subset_var] <= e[1])
        df = df[sel]

    # Column patches: first apply transform functions (e.g. unit conversion),
    # then rename old column names to their canonical names.
    # Order matters — transforms run on the old name before renaming.
    if column_patches is not None:
        for on, (_, op) in column_patches.items():
            if on not in df.columns or op is None:
                continue
            df[on] = op(df[on])
        df = df.rename(columns={on: nn for on, (nn, _) in column_patches.items()})

    # figure out what the data columns are
    cols_uncompressed = uncompress_columns(cols_compressed)

    # Only check multi-component columns for all-NaN rows — a single scalar
    # column being NaN is normal (e.g. missing timestamp), but all components
    # of a vector field being NaN means no data for that sample.
    if drop_if_all_nan:
        df = df.dropna(how="all", subset=[c for cs in cols_uncompressed if len(cs) > 1 for c in cs])

    # Pack individual columns (e.g. pos_x, pos_y, pos_z) into single numpy
    # arrays per row, stored in the compressed column name (e.g. pos).
    for c, ac in zip(cols_compressed, cols_uncompressed, strict=True):
        if len(ac) == 1:
            continue  # scalar field, nothing to pack
        if ac:
            if not any(a in df.columns for a in ac):
                continue  # none of the expected columns exist in the file
            if put_none_if_any_nan:
                df[c] = [none_if_any_nan(x) for x in df[ac].to_numpy()]
            else:
                df[c] = list(df[ac].to_numpy())
        else:
            df[c] = None

    # Keep only expected columns in schema order
    df = df[[c for c in cols_compressed if c in df.columns]]

    # When multiple timestamp/frame_idx variants exist (e.g. from different
    # clocks), save the originals as _ori columns before overwriting the main
    # ones with the preferred variant.
    if make_ori_ts_fridx:
        # Column selection above produces a view; copy to avoid SettingWithCopyWarning
        df = df.copy()
        df["frame_idx_ori"] = df["frame_idx"]
        if "timestamp" in df.columns:
            df["timestamp_ori"] = df["timestamp"]
    # Overwrite main timestamp/frame_idx with the first matching suffix variant
    if make_ori_ts_fridx and ts_fridx_field_suffixes:
        copied = False
        for suf in ts_fridx_field_suffixes:  # tried in order of preference
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
    """Write a collection of data objects to a tab-separated file.

    Inverse of ``read_file``: unpacks numpy arrays back into individual TSV
    columns, restores original timestamps if ``_ori`` columns are present, and
    writes via polars for speed.

    Args:
        objects: Records as a flat list or a dict of ``{key: [records...]}``.
        file_name: Output TSV file path.
        cols_compressed: Field-to-component-count mapping (same as record class schema).
        drop_all_nan_cols: Column names to drop if all values are NaN.
        skip_all_nan: If True, drop rows where all multi-component columns are NaN.

    """
    if not objects:
        return

    if drop_all_nan_cols is None:
        drop_all_nan_cols = []

    if isinstance(objects, dict):
        # Flatten {key: [records...]} into a single list
        objects = [o for olist in objects.values() for o in olist]

    # Extract public attributes (skip _private) into dicts for DataFrame construction
    records = [{k: getattr(p, k) for k in vars(p) if not k.startswith("_")} for p in objects]
    df = pd.DataFrame.from_records(records)

    # Unpack numpy arrays back into individual columns (inverse of read_file's packing)
    cols_uncompressed = uncompress_columns(cols_compressed)
    for c, ac in zip(cols_compressed, cols_uncompressed, strict=True):
        if len(ac) > 1:
            df[ac] = np.vstack([all_nan_if_none(v, len(ac)).flatten() for v in df[c].to_numpy()])

    # if wanted, drop specific columns for which all rows are nan
    if drop_all_nan_cols:
        df = df.drop([c for c in drop_all_nan_cols if c in df and df[c].isna().all()], axis="columns")

    # Restore original timestamp/frame_idx from _ori columns if present.
    # The _ori columns themselves get dropped below since they're not in the schema.
    if "timestamp_ori" in df.columns and not df["timestamp_ori"].isna().all():
        df["timestamp"] = df["timestamp_ori"]
    if "frame_idx_ori" in df.columns and not df["frame_idx_ori"].isna().all():
        df["frame_idx"] = df["frame_idx_ori"]

    # keep only columns to be written out and order them correctly
    df = df[[c for cs in cols_uncompressed for c in cs if c in df.columns]]

    if skip_all_nan:
        df = df.dropna(how="all", subset=[c for cs in cols_uncompressed if len(cs) > 1 for c in cs])

    # Polars writes CSV significantly faster than pandas
    df = pl.from_pandas(df)
    df.write_csv(file_name, separator="\t", null_value="nan", float_precision=8)
