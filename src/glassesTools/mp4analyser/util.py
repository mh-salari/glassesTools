"""Utility functions for reading binary data with struct.unpack."""

import struct
from typing import BinaryIO


def read_u8(fp: BinaryIO) -> int:
    """Read an unsigned 8-bit integer."""
    return struct.unpack(">B", fp.read(1))[0]


def read_u16(fp: BinaryIO) -> int:
    """Read an unsigned 16-bit big-endian integer."""
    return struct.unpack(">H", fp.read(2))[0]


def read_u32(fp: BinaryIO) -> int:
    """Read an unsigned 32-bit big-endian integer."""
    return struct.unpack(">I", fp.read(4))[0]


def read_u64(fp: BinaryIO) -> int:
    """Read an unsigned 64-bit big-endian integer."""
    return struct.unpack(">Q", fp.read(8))[0]


def read_i8(fp: BinaryIO) -> int:
    """Read a signed 8-bit integer."""
    return struct.unpack(">b", fp.read(1))[0]


def read_i16(fp: BinaryIO) -> int:
    """Read a signed 16-bit big-endian integer."""
    return struct.unpack(">h", fp.read(2))[0]


def read_i32(fp: BinaryIO) -> int:
    """Read a signed 32-bit big-endian integer."""
    return struct.unpack(">i", fp.read(4))[0]


def read_i64(fp: BinaryIO) -> int:
    """Read a signed 64-bit big-endian integer."""
    return struct.unpack(">q", fp.read(8))[0]


def read_u8_8(fp: BinaryIO) -> float:
    """Read an unsigned 8.8 fixed-point number."""
    ipart, fpart = struct.unpack(">2B", fp.read(2))
    return ipart + (fpart / 256)


def read_u16_16(fp: BinaryIO) -> float:
    """Read an unsigned 16.16 fixed-point number."""
    ipart, fpart = struct.unpack(">2H", fp.read(4))
    return ipart + (fpart / 65536)


def read_i8_8(fp: BinaryIO) -> float:
    """Read a signed 8.8 fixed-point number."""
    ipart = struct.unpack(">b", fp.read(1))[0]
    fpart = struct.unpack("B", fp.read(1))[0]
    return ipart + (fpart / 256)
