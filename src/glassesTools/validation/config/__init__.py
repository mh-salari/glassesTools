"""Validation configuration file reading and parsing."""

import importlib.resources
import pathlib
import typing
from shlex import shlex

import numpy as np
import pandas as pd

from ... import aruco as _aruco
from ... import data_files as _data_files


def _read_glasses_validator_config_file(file: typing.IO[str]) -> dict[str, str]:
    """Read key=value pairs from a glassesValidator config file into a dict.

    Uses :class:`shlex` to tokenize, treating ``=`` as whitespace and
    ``%`` as the comment character.

    Args:
        file: An open text file-like object to read from.

    Returns:
        A dict mapping config keys to their string values.

    """
    lexer = shlex(file)
    lexer.whitespace += "="
    lexer.wordchars += (
        ".[],"  # don't split extensions of filenames in the input file, and accept stuff from python list syntax
    )
    lexer.commenters = "%"
    return dict(zip(lexer, lexer, strict=False))


_default_poster_package = ".".join([*__package__.split(".")[:-1], "default_poster"])


def get_validation_setup(
    config_dir: str | pathlib.Path | None = None, config_file: str = "validationSetup.txt"
) -> dict[str, typing.Any]:
    """Read and parse a validation setup config file.

    Falls back to the default config bundled with the package if no
    ``config_dir`` is provided. Numeric values are parsed to int or
    float, and the ArUco dictionary name is converted to its integer ID.

    Args:
        config_dir: Path to directory containing the config file, or
            None to use the bundled default.
        config_file: Name of the config file.

    Returns:
        A dict of parsed configuration values.

    """
    if config_dir is not None:
        with (pathlib.Path(config_dir) / config_file).open() as f:
            validation_config = _read_glasses_validator_config_file(f)
    else:
        # fall back on default config included with package
        with importlib.resources.open_text(_default_poster_package, config_file) as f:
            validation_config = _read_glasses_validator_config_file(f)

    # parse numerics into int or float
    for key, val in validation_config.items():
        if np.all([c.isdigit() for c in val]):
            validation_config[key] = int(val)
        else:
            try:
                validation_config[key] = float(val)
            except ValueError:
                pass  # just keep value as a string
    # backwards compatibility
    if "arucoDictionary" not in validation_config:
        validation_config["arucoDictionary"] = "DICT_4X4_250"
    # check aruco dictionary name, and convert to ID
    validation_config["arucoDictionary"] = _aruco.str_to_dict_id(validation_config["arucoDictionary"])
    return validation_config


def _read_coord_file(config_dir: str | pathlib.Path | None, file: str, package: str) -> pd.DataFrame | None:
    """Read a coordinate CSV from config_dir or from a bundled package."""
    if config_dir is not None:
        return _data_files.read_coord_file(pathlib.Path(config_dir) / file)
    return _data_files.read_coord_file(file, package)


def get_targets(
    config_dir: str | pathlib.Path | None = None,
    file: str = "targetPositions.csv",
    package: str = _default_poster_package,
) -> pd.DataFrame | None:
    """Read target positions from config directory or default package.

    Args:
        config_dir: Path to config directory, or None for defaults.
        file: Target positions CSV filename.
        package: Fallback package containing the default file.

    Returns:
        A DataFrame of target positions, or None if the file is missing.

    """
    return _read_coord_file(config_dir, file, package)


def get_markers(
    config_dir: str | pathlib.Path | None = None,
    file: str = "markerPositions.csv",
    package: str = _default_poster_package,
) -> pd.DataFrame | None:
    """Read marker positions from config directory or default package.

    Args:
        config_dir: Path to config directory, or None for defaults.
        file: Marker positions CSV filename.
        package: Fallback package containing the default file.

    Returns:
        A DataFrame of marker positions, or None if the file is missing.

    """
    return _read_coord_file(config_dir, file, package)
