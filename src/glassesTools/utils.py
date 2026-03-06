"""General utility functions for color conversion, enum handling, and text formatting."""

import colorsys
import enum
import os
import pathlib
import types
import typing

import numpy as np


def hex_to_rgba_0_1(hex_str: str) -> tuple[float, float, float, float]:
    """Convert a hex color string to RGBA tuple with values in 0-1 range."""
    r = int(hex_str[1:3], base=16) / 255
    g = int(hex_str[3:5], base=16) / 255
    b = int(hex_str[5:7], base=16) / 255
    a = int(hex_str[7:9], base=16) / 255 if len(hex_str) > 7 else 1.0
    return (r, g, b, a)


def rgba_0_1_to_hex(rgba: tuple[float, ...]) -> str:
    """Convert an RGBA tuple with values in 0-1 range to a hex color string."""
    r = f"{int(rgba[0] * 255):02x}"
    g = f"{int(rgba[1] * 255):02x}"
    b = f"{int(rgba[2] * 255):02x}"
    a = f"{int(rgba[3] * 255):02x}" if len(rgba) > 3 else "FF"
    return f"#{r}{g}{b}{a}"


def get_colors(n_colors: int, saturation: float, value: float) -> list[tuple[float, float, float]]:
    """Generate evenly spaced colors in HSV space, returned as RGB tuples."""
    color_steps = 1 / (n_colors + 1)
    return [colorsys.hsv_to_rgb(i * color_steps, saturation, value) for i in range(n_colors)]


def get_hour_minutes_seconds_ms(dur_seconds: float) -> tuple[float, float, float, float]:
    """Split a duration in seconds into hours, minutes, seconds, and milliseconds."""
    hours, remainder = divmod(dur_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    seconds, ms = divmod(seconds, 1)
    return hours, minutes, seconds, ms


def format_duration(dur: float, show_ms: bool) -> str:
    """Format a duration in seconds as a human-readable time string."""
    hours, minutes, seconds, ms = get_hour_minutes_seconds_ms(dur)
    if round(ms, 3) == 1.0:
        # prevent getting timecode x:xx:xx.1000
        hours, minutes, seconds, ms = get_hour_minutes_seconds_ms(round(dur))
    dur_str = f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"
    if show_ms:
        dur_str += f".{ms * 1000:03.0f}"
    return dur_str


class AutoName(enum.Enum):
    """Enum that auto-generates values from member names, replacing underscores with spaces."""

    def _generate_next_value_(name: str, _start: int, _count: int, _last_values: list[str]) -> str:  # noqa: N805
        return name.strip("_").replace("__", "-").replace("_", " ")


def enum_val_2_str(x: enum.Enum) -> str:
    """Convert an enum value to a stable string representation."""
    # to ensure that string representation of enum is constant over Python versions (it isn't for enum.IntEnum at least)
    return f"{type(x).__name__}.{x.name}"


E = typing.TypeVar("E")


def str_int_2_enum_val(
    x: str | int | E, enum_cls: type[E], patches: typing.Mapping[str | int, str] | None = None
) -> E:
    """Convert a string or int to an enum value, with optional name patches."""
    if isinstance(x, enum_cls):
        return x
    str_val = patches[x] if isinstance(x, int) and patches and x in patches else x.rsplit(".", maxsplit=1)[-1]
    if patches is not None and str_val in patches:
        str_val = patches[str_val]
    # if its the name of an enum member, return by attribute
    if hasattr(enum_cls, str_val):
        return getattr(enum_cls, str_val)
    # else assume its the value, so try to construct with value
    return enum_cls(str_val)


def cartesian_product(*arrays: np.ndarray) -> np.ndarray:
    """Compute the cartesian product of input arrays."""
    ndim = len(arrays)
    return np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim)


def fast_scandir(dir_path: pathlib.Path) -> list[pathlib.Path]:
    """Recursively scan a directory and return all subdirectory paths."""
    if not dir_path.is_dir():
        return []
    subfolders = [pathlib.Path(f.path) for f in os.scandir(dir_path) if f.is_dir()]
    for subfolder in list(subfolders):
        subfolders.extend(fast_scandir(subfolder))
    return subfolders


def unpack_none_union(annotation: type) -> tuple[type, bool]:
    """Unpack a Union type that includes None, returning the inner type and whether None was present."""
    # below handles both types.Optional and direct unions with None
    if (
        typing.get_origin(annotation) in {typing.Union, types.UnionType}
        and (args := typing.get_args(annotation))[-1] == types.NoneType
    ):
        return typing.Union[args[:-1]], True  # noqa: UP007
    return annotation, False


def set_all(
    inp: dict[int, bool],
    value: bool,
    subset: list[int] | None = None,
    predicate: typing.Callable[[int], bool] | None = None,
) -> None:
    """Set all values in a dict to the given value, optionally filtered by subset and predicate."""
    if subset is None:
        subset = (r for r in inp)
    for r in subset:
        if r in inp and (not predicate or predicate(r)):
            inp[r] = value


def trim_str(text: str, length: int | None = None, till_newline: bool = True, newline_ellipsis: bool = False) -> str:
    """Trim a string to a maximum length and/or first line."""
    if text and till_newline:
        temp = text.splitlines()
        if temp:
            text = temp[0]
        if len(temp) > 1 and newline_ellipsis:
            text += ".."
    if length:
        text = (text[: length - 2] + "..") if len(text) > length else text
    return text
