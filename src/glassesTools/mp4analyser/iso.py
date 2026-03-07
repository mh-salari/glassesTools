"""ISO/IEC 14496-12 MP4 box definitions and file parser.

Contains class definitions for all supported ISO base media file format boxes,
the Mp4File top-level container, and a box factory function.
"""

import binascii
import datetime
import logging
import pathlib
import struct
from typing import BinaryIO

from . import non_iso
from .core import Header, Mp4Box, Mp4FullBox
from .summary import Summary
from .util import (
    read_i8_8,
    read_i16,
    read_i32,
    read_i64,
    read_u8,
    read_u8_8,
    read_u16,
    read_u16_16,
    read_u32,
    read_u64,
)

logger = logging.getLogger(__name__)

# Supported box
# 'ftyp', 'pdin', 'moov', 'mvhd', 'meta', 'trak', 'tkhd', 'tref', 'trgr', 'edts', 'elst', 'mdia',
# 'mdhd', 'hdlr', 'elng', 'minf', 'vmhd', 'smhd', 'hmhd', 'nmhd', 'dinf', 'dref', 'url ', 'urn ',
# 'stbl', 'stsd', 'stts', 'ctts', 'cslg', 'stsc', 'stsz', 'stz2', 'stco', 'co64', 'stss', 'stsh',
# 'padb', 'stdp', 'sdtp', 'sbgp', 'sgpd', 'subs', 'saiz', 'saio', 'udta', 'mvex', 'mehd', 'trex',
# 'leva', 'moof', 'mfhd', 'traf', 'tfhd', 'trun', 'tfdt', 'mfra', 'tfra', 'mfro', 'mdat', 'free',
# 'skip', 'cprt', 'tsel', 'strk', 'stri', 'strd', 'iloc', 'ipro', 'rinf', 'sinf', 'frma', 'schm',
# 'xml ', 'pitm', 'iref', 'meco', 'mere', 'styp', 'sidx', 'ssix', 'prft', 'avc1', 'hvc1', 'avcC',
# 'hvcC', 'btrt', 'pasp', 'mp4a', 'ac-3', 'ec-3', 'esds', 'dac3', 'dec3', 'ilst', 'data', 'pssh',
# 'senc'
# Not supported
# 'sthd', 'iinf', 'bxml', 'fiin', 'paen', 'fire', 'fpar', 'fecr', 'segr', 'gitn', 'idat'


def box_factory(fp: BinaryIO, header: Header, parent: Mp4Box) -> Mp4Box:
    """Create a box instance by looking up the class from the header type."""
    # Normalise header type so it can be expressed in a Python Class name
    box_type = header.type.replace(" ", "_").replace("-", "_").lower()
    box_class = globals().get(box_type.capitalize() + "Box")
    if box_class:
        return box_class(fp, header, parent)
    return non_iso.box_factory_non_iso(fp, header, parent)


class Mp4File:
    """Mp4File Class, effectively the top-level container"""

    def __init__(self, filename: str) -> None:
        """Parse the MP4 file and build the box tree."""
        self.filename = filename
        self.type = "file"
        self.children = []
        self.summary = {}
        try:
            with pathlib.Path(filename).open("rb") as f:
                end_of_file = False
                while not end_of_file:
                    current_header = Header(f)
                    current_box = box_factory(f, current_header, self)
                    self.children.append(current_box)
                    if current_box.size == 0:
                        end_of_file = True
                    if len(f.read(4)) != 4:
                        end_of_file = True
                    else:
                        f.seek(-4, 1)
            f.close()
            self._generate_samples_from_moov()
            self._generate_samples_from_moofs()
        except Exception:
            # catch exception in case we can continue
            logger.exception("error in %s after child %d", filename, len(self.children))

    def _generate_samples_from_moov(self) -> None:
        """Identify media samples in mdat for full mp4 file."""
        mdats = [mbox for mbox in self.children if mbox.type == "mdat"]
        # generate a sample list if there is a moov that contains traks N.B only ever 0,1 moov boxes
        if moov := next((box for box in self.children if box.type == "moov"), None):
            traks = [tbox for tbox in moov.children if tbox.type == "trak"]
            sample_list = []
            for trak in traks:
                trak_id = next(box for box in trak.children if box.type == "tkhd").box_info["track_ID"]
                mdia = next(box for box in trak.children if box.type == "mdia")
                minf = next(box for box in mdia.children if box.type == "minf")
                samplebox = next(box for box in minf.children if box.type == "stbl")
                chunk_offsets = next(box for box in samplebox.children if box.type in {"stco", "co64"}).box_info[
                    "entry_list"
                ]
                sample_size_box = next(box for box in samplebox.children if box.type in {"stsz", "stz2"})
                if sample_size_box.box_info["sample_size"] > 0:
                    sample_sizes = [
                        {"entry_size": sample_size_box.box_info["sample_size"]}
                        for _ in range(sample_size_box.box_info["sample_count"])
                    ]
                else:
                    sample_sizes = sample_size_box.box_info["entry_list"]
                sample_to_chunks = next(box for box in samplebox.children if box.type == "stsc").box_info["entry_list"]
                s2c_index = 0
                next_run = 0
                sample_idx = 0
                for i, chunk in enumerate(chunk_offsets, 1):
                    if i >= next_run:
                        samples_per_chunk = sample_to_chunks[s2c_index]["samples_per_chunk"]
                        s2c_index += 1
                        next_run = (
                            sample_to_chunks[s2c_index]["first_chunk"]
                            if s2c_index < len(sample_to_chunks)
                            else len(chunk_offsets) + 1
                        )
                    chunk_dict = {
                        "track_ID": trak_id,
                        "chunk_ID": i,
                        "chunk_offset": chunk["chunk_offset"],
                        "samples_per_chunk": samples_per_chunk,
                        "chunk_samples": [],
                    }
                    sample_offset = chunk["chunk_offset"]
                    for j, sample in enumerate(
                        sample_sizes[sample_idx : sample_idx + samples_per_chunk], sample_idx + 1
                    ):
                        chunk_dict["chunk_samples"].append({
                            "sample_ID": j,
                            "size": sample["entry_size"],
                            "offset": sample_offset,
                        })
                        sample_offset += sample["entry_size"]
                    sample_list.append(chunk_dict)
                    sample_idx += samples_per_chunk
            # sample_list could be empty, say, for mpeg-dash initialization segment
            if sample_list:
                # sort by chunk offset to get interleaved list
                sample_list.sort(key=lambda k: k["chunk_offset"])
                for mdat in mdats:
                    mdat_sample_list = [
                        sample
                        for sample in sample_list
                        if mdat.start_of_box < sample["chunk_offset"] < (mdat.start_of_box + mdat.size)
                    ]
                    if mdat_sample_list:
                        mdat.box_info["message"] = "Has samples."
                    mdat.sample_list = mdat_sample_list

    def _generate_samples_from_moofs(self) -> None:
        """Generate samples within mdats of media segments for fragmented mp4 files.

        Media segments are 1 moof (optionally preceded by an styp) followed by 1 or more contiguous mdats.
        """
        i = 0
        while i < len(self.children) - 1:
            if self.children[i].type == "moof":
                moof = self.children[i]
                media_segment = {"moof_box": moof, "mdat_boxes": []}
                sequence_number = next(mfhd for mfhd in moof.children if mfhd.type == "mfhd").box_info[
                    "sequence_number"
                ]
                while i < len(self.children) - 1 and self.children[i + 1].type == "mdat":
                    media_segment["mdat_boxes"].append(self.children[i + 1])
                    i += 1
                # I've only ever seen 1 traf in a moof, but the standard says there could be more
                data_offset = 0
                for j, traf in enumerate([tbox for tbox in moof.children if tbox.type == "traf"]):
                    # read tfhd, there will be one
                    tfhd = next(hbox for hbox in traf.children if hbox.type == "tfhd")
                    trak_id = tfhd.box_info["track_id"]
                    if "base_data_offset" in tfhd.box_info:
                        data_offset = tfhd.box_info["base_data_offset"]
                    elif tfhd.box_info["default_base_is_moof"]:
                        data_offset = media_segment["moof_box"].start_of_box
                    elif j > 0:
                        # according to spec. should be set end of data for last fragment
                        pass
                    else:
                        data_offset = media_segment["moof_box"].start_of_box
                    for k, trun in enumerate([rbox for rbox in traf.children if rbox.type == "trun"], 1):
                        if "data_offset" in trun.box_info:
                            data_offset += trun.box_info["data_offset"]
                        run_dict = {
                            "sequence_number": sequence_number,
                            "track_ID": trak_id,
                            "run_ID": k,
                            "run_offset": data_offset,
                            "sample_count": trun.box_info["sample_count"],
                            "run_samples": [],
                        }
                        has_sample_size = trun.flags & 0x0200 == 0x0200
                        for l, sample in enumerate(trun.box_info["samples"], 1):
                            if not has_sample_size:
                                sample_size = tfhd.box_info["default_sample_size"]
                            else:
                                sample_size = sample["sample_size"]
                            run_dict["run_samples"].append({
                                "sample_ID": l,
                                "size": sample_size,
                                "offset": data_offset,
                            })
                            data_offset += sample_size
                        for mdat in media_segment["mdat_boxes"]:
                            if (
                                mdat.start_of_box < run_dict["run_offset"]
                                and (mdat.start_of_box + mdat.size) >= data_offset
                            ):
                                mdat.box_info["message"] = "Has samples."
                                mdat.sample_list.append(run_dict)
            i += 1

    def read_bytes(self, offset: int, num_bytes: int) -> bytes:
        """Read raw bytes from the file at the given offset."""
        with pathlib.Path(self.filename).open("rb") as f:
            f.seek(offset)
            bytes_read = f.read(num_bytes)
        f.close()
        return bytes_read

    def get_summary(self) -> dict:
        """Return a summary dictionary for this MP4 file."""
        if not self.summary:
            self.summary = Summary(self)
        return self.summary.data

    def search_boxes_for_type(self, box_type: str) -> list[Mp4Box]:
        """Search all child boxes recursively for the given type."""
        type_matches = []
        for box in self.children:
            if box.type == box_type:
                type_matches.append(box)
            if box.children:
                type_matches += box.search_child_boxes_for_type(box_type)
        return type_matches


# Box classes


class FreeBox(Mp4Box):
    """Free space box (free/skip) — padding or deleted data."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            pass
        finally:
            fp.seek(self.start_of_box + self.size)


SkipBox = FreeBox


class FtypBox(Mp4Box):
    """File type and compatibility box (ftyp)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info = {
                "major_brand": fp.read(4).decode("utf-8"),
                "minor_version": f"{read_u32(fp):#010x}",
                "compatible_brands": [],
            }
            bytes_left = self.size - (self.header.header_size + 8)
            while bytes_left > 0:
                self.box_info["compatible_brands"].append(fp.read(4).decode("utf-8"))
                bytes_left -= 4
        finally:
            fp.seek(self.start_of_box + self.size)


StypBox = FtypBox


class ColrBox(Mp4Box):
    """Colour information box (colr)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["color_type"] = fp.read(4).decode("utf-8")
            if self.box_info["color_type"] == "nclx":
                self.box_info["color_primaries"] = read_u16(fp)
                self.box_info["transfer_characteristics"] = read_u16(fp)
                self.box_info["matrix_coefficients"] = read_u16(fp)
                self.box_info["full_range_flag"] = read_u8(fp) >> 7
        finally:
            fp.seek(self.start_of_box + self.size)


class PdinBox(Mp4FullBox):
    """Progressive download information box (pdin)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        end_of_box = self.start_of_box + self.size
        try:
            self.box_info["rates"] = []
            while fp.tell() < end_of_box:
                self.box_info("rates").append({"rate": read_u32(fp), "initial_delay": read_u32(fp)})
        finally:
            fp.seek(end_of_box)


class ContainerBox(Mp4Box):
    """Generic container box that recursively parses child boxes."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            bytes_left = self.size - self.header.header_size
            while bytes_left > 7:
                current_header = Header(fp)
                current_box = box_factory(fp, current_header, self)
                self.children.append(current_box)
                bytes_left -= current_box.size
        finally:
            fp.seek(self.start_of_box + self.size)


# All these are pure container boxes
DinfBox = MinfBox = MdiaBox = TrefBox = EdtsBox = TrakBox = MoofBox = MoovBox = ContainerBox
UdtaBox = TrgrBox = MvexBox = MfraBox = StrkBox = StrdBox = RinfBox = SinfBox = MecoBox = ContainerBox
GmhdBox = SchiBox = ContainerBox


class MetaBox(Mp4Box):
    """Seems to be a discrepancy between Apple atom spec and ISO about whether this is a versioned box"""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            bytes_left = self.size - self.header.header_size
            first_four_bytes = read_u32(fp)
            second_four_bytes = fp.read(4)
            if second_four_bytes.decode("utf-8", errors="ignore") == "hdlr":
                # it's non-versioned
                fp.seek(-8, 1)
            else:
                # it's versioned
                self.version = first_four_bytes >> 24
                self.flags = first_four_bytes & 0xFFFFFF
                fp.seek(-4, 1)
                bytes_left -= 4
            while bytes_left > 7:
                current_header = Header(fp)
                current_box = box_factory(fp, current_header, self)
                self.children.append(current_box)
                bytes_left -= current_box.size
        finally:
            fp.seek(self.start_of_box + self.size)


class MdatBox(Mp4Box):
    """Media data box (mdat)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        self.sample_list = []
        try:
            self.box_info["message"] = "No samples found."
        finally:
            fp.seek(self.start_of_box + self.size)


class MvhdBox(Mp4FullBox):
    """Movie header box (mvhd)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            dt_base = datetime.datetime(1904, 1, 1, 0, 0, 0)
            if self.version == 1:
                self.box_info["creation_time"] = (dt_base + datetime.timedelta(seconds=(read_u64(fp)))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.box_info["modification_time"] = (dt_base + datetime.timedelta(seconds=(read_u64(fp)))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.box_info["timescale"] = read_u32(fp)
                self.box_info["duration"] = read_u64(fp)
            else:
                self.box_info["creation_time"] = (dt_base + datetime.timedelta(seconds=(read_u32(fp)))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.box_info["modification_time"] = (dt_base + datetime.timedelta(seconds=(read_u32(fp)))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.box_info["timescale"] = read_u32(fp)
                self.box_info["duration"] = read_u32(fp)
            self.box_info["rate"] = read_u16_16(fp)
            self.box_info["volume"] = read_u8_8(fp)
            fp.seek(10, 1)
            self.box_info["matrix"] = [f"{b:#010x}" for b in struct.unpack(">9I", fp.read(36))]
            fp.seek(24, 1)
            self.box_info["next_track_id"] = read_u32(fp)
        finally:
            fp.seek(self.start_of_box + self.size)


class MfhdBox(Mp4FullBox):
    """Movie fragment header box (mfhd)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["sequence_number"] = read_u32(fp)
        finally:
            fp.seek(self.start_of_box + self.size)


class MehdBox(Mp4FullBox):
    """Movie extends header box (mehd)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            if self.version == 1:
                self.box_info["fragment_duration"] = read_u64(fp)
            else:
                self.box_info["fragment_duration"] = read_u32(fp)
        finally:
            fp.seek(self.start_of_box + self.size)


class ElstBox(Mp4FullBox):
    """Edit list box (elst)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["entry_count"] = read_u32(fp)
            self.box_info["entry_list"] = []
            for _ in range(self.box_info["entry_count"]):
                if self.version == 1:
                    self.box_info["entry_list"].append({
                        "segment_duration": read_u64(fp),
                        "media_time": read_i64(fp),
                        "media_rate": read_u16_16(fp),
                    })
                else:
                    self.box_info["entry_list"].append({
                        "segment_duration": read_u32(fp),
                        "media_time": read_i32(fp),
                        "media_rate": read_u16_16(fp),
                    })
        finally:
            fp.seek(self.start_of_box + self.size)


class TkhdBox(Mp4FullBox):
    """Track header box (tkhd)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            dt_base = datetime.datetime(1904, 1, 1, 0, 0, 0)
            if self.version == 1:
                self.box_info["creation_time"] = (dt_base + datetime.timedelta(seconds=(read_u64(fp)))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.box_info["modification_time"] = (dt_base + datetime.timedelta(seconds=(read_u64(fp)))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.box_info["track_ID"] = read_u32(fp)
                fp.seek(4, 1)
                self.box_info["duration"] = read_u64(fp)
            else:
                self.box_info["creation_time"] = (dt_base + datetime.timedelta(seconds=(read_u32(fp)))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.box_info["modification_time"] = (dt_base + datetime.timedelta(seconds=(read_u32(fp)))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.box_info["track_ID"] = read_u32(fp)
                fp.seek(4, 1)
                self.box_info["duration"] = read_u32(fp)
            fp.seek(8, 1)
            self.box_info["layer"] = read_i16(fp)
            self.box_info["alternate_group"] = read_i16(fp)
            self.box_info["volume"] = read_u8_8(fp)
            fp.seek(2, 1)
            self.box_info["matrix"] = [f"{b:#010x}" for b in struct.unpack(">9I", fp.read(36))]
            self.box_info["width"] = read_u16_16(fp)
            self.box_info["height"] = read_u16_16(fp)
        finally:
            fp.seek(self.start_of_box + self.size)


class TfhdBox(Mp4FullBox):
    """Track fragment header box (tfhd)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["track_id"] = read_u32(fp)
            if self.flags & 0x000001 == 0x000001:
                self.box_info["base_data_offset"] = read_u64(fp)
            if self.flags & 0x000002 == 0x000002:
                self.box_info["sample_description_index"] = read_u32(fp)
            if self.flags & 0x000008 == 0x000008:
                self.box_info["default_sample_duration"] = read_u32(fp)
            if self.flags & 0x000010 == 0x000010:
                self.box_info["default_sample_size"] = read_u32(fp)
            if self.flags & 0x000020 == 0x000020:
                self.box_info["default_sample_flags"] = f"{read_u32(fp):#08x}"
            self.box_info["duration_is_empty"] = self.flags & 0x010000 == 0x010000
            # depending on value of base data offset this flag mght be ignored
            self.box_info["default_base_is_moof"] = self.flags & 0x020000 == 0x020000
        finally:
            fp.seek(self.start_of_box + self.size)


class TrexBox(Mp4FullBox):
    """Track extends defaults box (trex)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["track_ID"] = read_u32(fp)
            self.box_info["default_sample_description_index"] = read_u32(fp)
            self.box_info["default_sample_duration"] = read_u32(fp)
            self.box_info["default_sample_size"] = read_u32(fp)
            self.box_info["default_sample_flags"] = f"{read_u32(fp):#08x}"
        finally:
            fp.seek(self.start_of_box + self.size)


class LevaBox(Mp4FullBox):
    """Level assignment box (leva)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["level_count"] = read_u8(fp)
            self.box_info["level_list"] = []
            for _ in range(self.box_info["level_count"]):
                level_dict = {"track_ID": read_u32(fp)}
                pad_assign = read_u8(fp)
                level_dict["padding_flag"] = pad_assign // 128
                level_dict["assignment_type"] = pad_assign % 128
                if level_dict["assignment_type"] == 0:
                    level_dict["grouping_type"] = fp.read(4).decode("utf-8")
                elif level_dict["assignment_type"] == 1:
                    level_dict["grouping_type"] = fp.read(4).decode("utf-8")
                    level_dict["grouping_type_parameter"] = read_u32(fp)
                elif level_dict["assignment_type"] == 4:
                    level_dict["sub_track_id"] = read_u32(fp)
                self.box_info["level_list"].append(level_dict)
        finally:
            fp.seek(self.start_of_box + self.size)


class TfraBox(Mp4FullBox):
    """Track fragment random access box (tfra)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["track_ID"] = read_u32(fp)
            length_fields = read_u32(fp)
            self.box_info["length_size_of_traf_num"] = length_fields >> 4 & 3
            self.box_info["length_size_of_trun_num"] = length_fields >> 2 & 3
            self.box_info["length_size_of_sample_num"] = length_fields & 3
            self.box_info["number_of_entry"] = read_u32(fp)
            self.box_info["entry_list"] = []
            for _ in range(self.box_info["number_of_entry"]):
                entry_dict = {}
                if self.version == 1:
                    entry_dict["time"] = read_u64(fp)
                    entry_dict["moof_offset"] = read_u64(fp)
                else:
                    entry_dict["time"] = read_u32(fp)
                    entry_dict["moof_offset"] = read_u32(fp)
                if self.box_info["length_size_of_traf_num"] == 0:
                    entry_dict["traf_number"] = read_u8(fp)
                elif self.box_info["length_size_of_traf_num"] == 1:
                    entry_dict["traf_number"] = read_u16(fp)
                elif self.box_info["length_size_of_traf_num"] == 3:
                    entry_dict["traf_number"] = read_u32(fp)
                if self.box_info["length_size_of_trun_num"] == 0:
                    entry_dict["trun_number"] = read_u8(fp)
                elif self.box_info["length_size_of_trun_num"] == 1:
                    entry_dict["trun_number"] = read_u16(fp)
                elif self.box_info["length_size_of_trun_num"] == 3:
                    entry_dict["trun_number"] = read_u32(fp)
                if self.box_info["length_size_of_sample_num"] == 0:
                    entry_dict["sample_number"] = read_u8(fp)
                elif self.box_info["length_size_of_sample_num"] == 1:
                    entry_dict["sample_number"] = read_u16(fp)
                elif self.box_info["length_size_of_sample_num"] == 3:
                    entry_dict["sample_number"] = read_u32(fp)
                self.box_info["entry_list"].append(entry_dict)
        finally:
            fp.seek(self.start_of_box + self.size)


class MfroBox(Mp4FullBox):
    """Movie fragment random access offset box (mfro)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["size"] = read_u32(fp)
        finally:
            fp.seek(self.start_of_box + self.size)


class CprtBox(Mp4FullBox):
    """Copyright box (cprt)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            # I think this is right
            lang = read_u16(fp)
            if lang == 0:
                self.box_info["language"] = "0x00"
            else:
                ch1 = str(chr(96 + (lang >> 10 & 31)))
                ch2 = str(chr(96 + (lang >> 5 & 31)))
                ch3 = str(chr(96 + (lang & 31)))
                self.box_info["language"] = ch1 + ch2 + ch3
            bytes_left = self.size - (self.header.header_size + 2)
            self.box_info["name"] = fp.read(bytes_left).decode("utf-8", errors="ignore")
        finally:
            fp.seek(self.start_of_box + self.size)


class TselBox(Mp4FullBox):
    """Track selection box (tsel)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["switch_group"] = read_u32(fp)
            bytes_left = self.size - (self.header.header_size + 8)
            attr_list = []
            while bytes_left > 0:
                attr_list.append(fp.read(4).decode("utf-8"))
                bytes_left -= 4
            self.box_info["attributes"] = attr_list
        finally:
            fp.seek(self.start_of_box + self.size)


class StriBox(Mp4FullBox):
    """Sub-track information box (stri)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["switch_group"] = read_u16(fp)
            self.box_info["alternate_group"] = read_u16(fp)
            self.box_info["sub_track_ID"] = read_u32(fp)
            bytes_left = self.size - (self.header.header_size + 12)
            attr_list = []
            while bytes_left > 0:
                attr_list.append(fp.read(4).decode("utf-8"))
                bytes_left -= 4
            self.box_info["attributes"] = attr_list
        finally:
            fp.seek(self.start_of_box + self.size)


class IlocBox(Mp4FullBox):
    """Item location box (iloc)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["offset_size"] = read_u32(fp)
            self.box_info["length_size"] = read_u32(fp)
            self.box_info["base_offset_size"] = read_u32(fp)
            if self.version in {1, 2}:
                self.box_info["index_size"] = read_u32(fp)
            else:
                self.box_info["reserved"] = read_u32(fp)
            if self.version < 2:
                self.box_info["item_count"] = read_u16(fp)
            elif self.version == 2:
                self.box_info["item_count"] = read_u32(fp)
            self.box_info["item_list"] = []
            for _ in range(self.box_info["item_count"]):
                item = {}
                if self.version < 2:
                    item["item_ID"] = read_u16(fp)
                elif self.version == 2:
                    item["item_ID"] = read_u32(fp)
                if self.version in {1, 2}:
                    item["construction_method"] = read_u16(fp) % 16
                item["data_reference_index"] = read_u16(fp)
                if self.box_info["offset_size"] == 4:
                    item["base_offset"] = read_u32(fp)
                elif self.box_info["offset_size"] == 8:
                    item["base_offset"] = read_u64(fp)
                item["extent_count"] = read_u16(fp)
                item["extent_list"] = []
                for _ in range(item["extent_count"]):
                    extent = {}
                    if self.version in {1, 2}:
                        if self.box_info["index_size"] == 4:
                            extent["extent_index"] = read_u32(fp)
                        elif self.box_info["index_size"] == 8:
                            extent["extent_index"] = read_u64(fp)
                    if self.box_info["offset_size"] == 4:
                        extent["extent_offset"] = read_u32(fp)
                    elif self.box_info["offset_size"] == 8:
                        extent["extent_offset"] = read_u64(fp)
                    if self.box_info["length_size"] == 4:
                        extent["extent_length"] = read_u32(fp)
                    elif self.box_info["length_size"] == 8:
                        extent["extent_length"] = read_u64(fp)
                    item["extent_list"].append(extent)
                self.box_info["item_list"].append(item)
        finally:
            fp.seek(self.start_of_box + self.size)


class IproBox(Mp4FullBox):
    """Item protection box (ipro)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["protection_count"] = read_u16(fp)
            for _ in range(self.box_info["protection_count"]):
                current_header = Header(fp)
                current_box = box_factory(fp, current_header, self)
                self.children.append(current_box)
        finally:
            fp.seek(self.start_of_box + self.size)


class FrmaBox(Mp4Box):
    """Original format box (frma)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["data_format"] = fp.read(4).decode("utf-8")
        finally:
            fp.seek(self.start_of_box + self.size)


class SchmBox(Mp4FullBox):
    """Scheme type box (schm)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["scheme_type"] = fp.read(4).decode("utf-8")
            self.box_info["scheme_version"] = read_u32(fp)
            if self.flags & 0x000001 == 0x000001:
                self.box_info["data_offset"] = fp.read(self.size - (self.header.header_size + 12)).decode("utf-8")
        finally:
            fp.seek(self.start_of_box + self.size)


class Xml_Box(Mp4FullBox):
    """XML container box (xml)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["xml_data"] = fp.read(self.size - (self.header.header_size + 4)).decode(
                "utf-8", errors="ignore"
            )
        finally:
            fp.seek(self.start_of_box + self.size)


class PitmBox(Mp4FullBox):
    """Primary item reference box (pitm)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            if self.version == 0:
                self.box_info["item_ID"] = read_u16(fp)
            else:
                self.box_info["item_ID"] = read_u32(fp)
        finally:
            fp.seek(self.start_of_box + self.size)


# This is just a versioned container box
class IrefBox(Mp4FullBox):
    """Item reference box (iref)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            bytes_left = self.size - self.header.header_size
            while bytes_left > 7:
                current_header = Header(fp)
                current_box = box_factory(fp, current_header, self)
                self.children.append(current_box)
                bytes_left -= current_box.size
        finally:
            fp.seek(self.start_of_box + self.size)


class MereBox(Mp4FullBox):
    """Metabox relation box (mere)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["first_metabox_handler_type"] = fp.read(4).decode("utf-8")
            self.box_info["second_metabox_handler_type"] = fp.read(4).decode("utf-8")
            self.box_info["metabox_relation"] = read_u8(fp)
        finally:
            fp.seek(self.start_of_box + self.size)


class TrunBox(Mp4FullBox):
    """Track fragment run box (trun)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["sample_count"] = read_u32(fp)
            if self.flags & 0x000001 == 0x000001:
                self.box_info["data_offset"] = read_i32(fp)
            if self.flags & 0x000004 == 0x000004:
                self.box_info["first_sample_flags"] = f"{read_u32(fp):#08x}"
            has_sample_duration = self.flags & 0x000100 == 0x000100
            has_sample_size = self.flags & 0x000200 == 0x000200
            has_sample_flags = self.flags & 0x000400 == 0x000400
            has_scto = self.flags & 0x000800 == 0x000800
            sample_list = []
            for _ in range(self.box_info["sample_count"]):
                sample = {}
                if has_sample_duration:
                    sample["sample_duration"] = read_u32(fp)
                if has_sample_size:
                    sample["sample_size"] = read_u32(fp)
                if has_sample_flags:
                    sample["sample_flags"] = f"{read_u32(fp):#08x}"
                if has_scto:
                    if self.version == 1:
                        self.box_info["sample_composition_time_offset"] = read_i32(fp)
                    else:
                        self.box_info["sample_composition_time_offset"] = read_u32(fp)
                sample_list.append(sample)
            self.box_info["samples"] = sample_list
        finally:
            fp.seek(self.start_of_box + self.size)


class TfdtBox(Mp4FullBox):
    """Track fragment decode time box (tfdt)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            if self.version == 1:
                self.box_info["baseMediaDecode"] = read_u64(fp)
            else:
                self.box_info["baseMediaDecode"] = read_u32(fp)
        finally:
            fp.seek(self.start_of_box + self.size)


class MdhdBox(Mp4FullBox):
    """Media header box (mdhd)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            dt_base = datetime.datetime(1904, 1, 1, 0, 0, 0)
            if self.version == 1:
                self.box_info["creation_time"] = (dt_base + datetime.timedelta(seconds=(read_u64(fp)))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.box_info["modification_time"] = (dt_base + datetime.timedelta(seconds=(read_u64(fp)))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.box_info["timescale"] = read_u32(fp)
                self.box_info["duration"] = read_u64(fp)
            else:
                self.box_info["creation_time"] = (dt_base + datetime.timedelta(seconds=(read_u32(fp)))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.box_info["modification_time"] = (dt_base + datetime.timedelta(seconds=(read_u32(fp)))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.box_info["timescale"] = read_u32(fp)
                self.box_info["duration"] = read_u32(fp)
            # I think this is right
            lang = struct.unpack(">H", fp.read(2))[0]
            if lang == 0:
                self.box_info["language"] = "0x00"
            else:
                ch1 = str(chr(96 + (lang >> 10 & 31)))
                ch2 = str(chr(96 + (lang >> 5 & 31)))
                ch3 = str(chr(96 + (lang % 32)))
                self.box_info["language"] = ch1 + ch2 + ch3
        finally:
            fp.seek(self.start_of_box + self.size)


class ElngBox(Mp4FullBox):
    """Extended language tag box (elng)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["extended_language"] = fp.read(self.size - (self.header.header_size + 4)).split(b"\x00")[0]
        finally:
            fp.seek(self.start_of_box + self.size)


class DrefBox(Mp4FullBox):
    """Data reference box (dref)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["entry_count"] = read_u32(fp)
            for _ in range(self.box_info["entry_count"]):
                current_header = Header(fp)
                current_box = box_factory(fp, current_header, self)
                self.children.append(current_box)
        finally:
            fp.seek(self.start_of_box + self.size)


class Url_Box(Mp4FullBox):
    """Data entry URL box (url)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            if self.flags & 0x000001 != 1:
                data_entry = fp.read(self.size - (self.header.header_size + 4))
                self.box_info["location"] = data_entry.decode("utf-8", errors="ignore")
        finally:
            fp.seek(self.start_of_box + self.size)


class Urn_Box(Mp4FullBox):
    """Data entry URN box (urn)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            if self.flags & 0x000001 != 1:
                _name, _sep, location = fp.read(self.size - (self.header.header_size + 4)).partition(b"\x00")
                self.box_info["name"] = location.decode("utf-8", errors="ignore")
                self.box_info["location"] = location.decode("utf-8", errors="ignore")
        finally:
            fp.seek(self.start_of_box + self.size)


class HdlrBox(Mp4FullBox):
    """Handler reference box (hdlr)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            fp.seek(4, 1)
            self.box_info["handler_type"] = fp.read(4).decode("utf-8")
            fp.seek(12, 1)
            bytes_left = self.size - (self.header.header_size + 25)  # string is null terminated
            self.box_info["name"] = fp.read(bytes_left).decode("utf-8", errors="ignore")
        finally:
            fp.seek(self.start_of_box + self.size)


class StblBox(ContainerBox):
    """Sample table box (stbl) — resolves cross-box dependencies after parsing children."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            # Some sample table boxes have dependencies on other sample table table boxes in order to read correctly
            # Fill stdp, sdtp lists using sample count in stsz
            sz = next((box for box in self.children if box.type in {"stsz", "stz2"}), None)
            sc = sz.box_info["sample_count"] if sz else False
            sdtp = next((box for box in self.children if box.type == "sdtp"), None)
            stdp = next((box for box in self.children if box.type == "stdp"), None)
            if sc and sdtp:
                sdtp.update_table(fp, sc)
            if sc and stdp:
                stdp.update_table(fp, sc)
        finally:
            fp.seek(self.start_of_box + self.size)


class TrafBox(ContainerBox):
    """Track fragment box (traf) — resolves SENC encryption info after parsing children."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            # if we have a senc box
            senc = next((box for box in self.children if box.type == "senc"), None)
            # check if it has sub-sampling before getting IV_size from sgpd or saiz
            if senc and senc.flags & 0x000002 == 0x000002:
                per_sample_iv_size = 0
                sgpd = next((box for box in self.children if box.type == "sgpd"), None)
                saiz = next((box for box in self.children if box.type == "saiz"), None)
                if sgpd and sgpd.box_info["grouping_type"] == "seig":
                    # I've only ever seen a single entry in seig so get IV size from this
                    per_sample_iv_size = sgpd.box_info["entry_list"][0]["per_sample_iv_size"]
                # try saiz
                elif saiz:
                    # does it have non-zero default size?
                    if saiz.box_info["default_sample_info_size"] > 0:
                        sample_size = saiz.box_info["default_sample_info_size"]
                    else:
                        sample_size = min(
                            sample["sample_info_size"] for sample in saiz.box_info["sample_info_size_list"]
                        )
                    # deduct 10 or 18 from sample size (8 or 16 byte IV + 2 byte sub-sample count) and divide by six
                    # (2 bytes clear data size + 4 bytes enc data size)
                    per_sample_iv_size = 8 if (sample_size - 10) % 6 == 0 else 16 if (sample_size - 18) % 6 == 0 else 0
                senc.populate_sample_table(fp, per_sample_iv_size)
        finally:
            fp.seek(self.start_of_box + self.size)


class VmhdBox(Mp4FullBox):
    """Video media header box (vmhd)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["graphicsmode"] = read_u16(fp)
            self.box_info["opcolor"] = struct.unpack(">3H", fp.read(6))
        finally:
            fp.seek(self.start_of_box + self.size)


class SmhdBox(Mp4FullBox):
    """Sound media header box (smhd)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["balance"] = read_i8_8(fp)
        finally:
            fp.seek(self.start_of_box + self.size)


class HmhdBox(Mp4FullBox):
    """Hint media header box (hmhd)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["maxPDUsize"] = read_u16(fp)
            self.box_info["avgPDUsize"] = read_u16(fp)
            self.box_info["maxbitrate"] = read_u32(fp)
            self.box_info["avgbitrate"] = read_u32(fp)
        finally:
            fp.seek(self.start_of_box + self.size)


class NmhdBox(Mp4FullBox):
    """Null media header box (nmhd)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)

        try:
            pass
        finally:
            fp.seek(self.start_of_box + self.size)


class StsdBox(Mp4FullBox):
    """Sample description box (stsd)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["entry_count"] = read_u32(fp)
            for _ in range(self.box_info["entry_count"]):
                current_header = Header(fp)
                current_box = box_factory(fp, current_header, self)
                self.children.append(current_box)
        finally:
            fp.seek(self.start_of_box + self.size)


class SttsBox(Mp4FullBox):
    """Decoding time to sample box (stts)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["entry_count"] = read_u32(fp)
            self.box_info["entry_list"] = []
            for _ in range(self.box_info["entry_count"]):
                self.box_info["entry_list"].append({"sample_count": read_u32(fp), "sample_delta": read_u32(fp)})
        finally:
            fp.seek(self.start_of_box + self.size)


class CttsBox(Mp4FullBox):
    """Composition time to sample box (ctts)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["entry_count"] = read_u32(fp)
            self.box_info["entry_list"] = []
            for _ in range(self.box_info["entry_count"]):
                if self.version == 1:
                    self.box_info["entry_list"].append({"sample_count": read_u32(fp), "sample_offset": read_i32(fp)})
                else:
                    self.box_info["entry_list"].append({"sample_count": read_u32(fp), "sample_offset": read_i32(fp)})
        finally:
            fp.seek(self.start_of_box + self.size)


class CslgBox(Mp4FullBox):
    """Composition to decode timeline mapping box (cslg)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            if self.version == 1:
                self.box_info["compositionToDTSShift"] = read_i64(fp)
                self.box_info["leastDecodeToDisplayDelta"] = read_i64(fp)
                self.box_info["greatestDecodeToDisplayDelta"] = read_i64(fp)
                self.box_info["compositionStartTime"] = read_i64(fp)
                self.box_info["compositionEndTime"] = read_i64(fp)
            else:
                self.box_info["compositionToDTSShift"] = read_i32(fp)
                self.box_info["leastDecodeToDisplayDelta"] = read_i32(fp)
                self.box_info["greatestDecodeToDisplayDelta"] = read_i32(fp)
                self.box_info["compositionStartTime"] = read_i32(fp)
                self.box_info["compositionEndTime"] = read_i32(fp)
        finally:
            fp.seek(self.start_of_box + self.size)


class StssBox(Mp4FullBox):
    """Sync sample box (stss)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["entry_count"] = read_u32(fp)
            self.box_info["entry_list"] = []
            for _ in range(self.box_info["entry_count"]):
                self.box_info["entry_list"].append({"sample_number": read_u32(fp)})
        finally:
            fp.seek(self.start_of_box + self.size)


class StshBox(Mp4FullBox):
    """Shadow sync sample box (stsh)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["entry_count"] = read_u32(fp)
            self.box_info["entry_list"] = []
            for _ in range(self.box_info["entry_count"]):
                self.box_info["entry_list"].append({
                    "shadowed_sample_number": read_u32(fp),
                    "sync_sample_number": read_u32(fp),
                })
        finally:
            fp.seek(self.start_of_box + self.size)


class StscBox(Mp4FullBox):
    """Sample to chunk box (stsc)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["entry_count"] = read_u32(fp)
            self.box_info["entry_list"] = []
            for _ in range(self.box_info["entry_count"]):
                first_chunk = read_u32(fp)
                samples_per_chunk = read_u32(fp)
                samples_description_index = read_u32(fp)
                self.box_info["entry_list"].append({
                    "first_chunk": first_chunk,
                    "samples_per_chunk": samples_per_chunk,
                    "samples_description_index": samples_description_index,
                })
        finally:
            fp.seek(self.start_of_box + self.size)


class StcoBox(Mp4FullBox):
    """Chunk offset box (stco)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["entry_count"] = read_u32(fp)
            self.box_info["entry_list"] = []
            for _ in range(self.box_info["entry_count"]):
                self.box_info["entry_list"].append({"chunk_offset": read_u32(fp)})
        finally:
            fp.seek(self.start_of_box + self.size)


class Co64Box(Mp4FullBox):
    """64-bit chunk offset box (co64)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["entry_count"] = read_u32(fp)
            self.box_info["entry_list"] = []
            for _ in range(self.box_info["entry_count"]):
                self.box_info["entry_list"].append({"chunk_offset": read_u64(fp)})
        finally:
            fp.seek(self.start_of_box + self.size)


class PadbBox(Mp4FullBox):
    """Padding bits box (padb)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["sample_count"] = read_u32(fp)
            self.box_info["sample_list"] = []
            for _ in range(self.box_info["sample_count"]):
                pads = read_u8(fp)
                self.box_info["entry_list"].append({"pad1": pads // 16, "pad2": pads % 16})
        finally:
            fp.seek(self.start_of_box + self.size)


class SubsBox(Mp4FullBox):
    """Sub-sample information box (subs)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["entry_count"] = read_u32(fp)
            self.box_info["entry_list"] = []
            for _ in range(self.box_info["entry_count"]):
                sample_delta = read_u32(fp)
                subsample_count = read_u16(fp)
                if subsample_count > 0:
                    subsample_list = []
                    for _ in range(subsample_count):
                        subsample_size = read_u32(fp) if self.version == 1 else read_u16(fp)
                        subsample_priority = read_u8(fp)
                        discardable = read_u8(fp)
                        codec_specific_parameters = read_u32(fp)
                        subsample_list.append({
                            "subsample_size": subsample_size,
                            "subsample_priority": subsample_priority,
                            "discardable": discardable,
                            "codec_specific_parameters": codec_specific_parameters,
                        })
                    self.box_info["entry_list"].append({
                        "sample_delta": sample_delta,
                        "subsample_count": subsample_count,
                        "subsample_list": subsample_list,
                    })
        finally:
            fp.seek(self.start_of_box + self.size)


class SbgpBox(Mp4FullBox):
    """Sample to group box (sbgp)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["grouping_type"] = fp.read(4).decode("utf-8")
            if self.version == 1:
                self.box_info["grouping_type_parameter"] = read_u32(fp)
            self.box_info["entry_count"] = read_u32(fp)
            self.box_info["entry_list"] = []
            for _ in range(self.box_info["entry_count"]):
                self.box_info["entry_list"].append({
                    "sample_count": read_u32(fp),
                    "group_description_index": read_u32(fp),
                })
        finally:
            fp.seek(self.start_of_box + self.size)


class SgpdBox(Mp4FullBox):
    """Sample group description box (sgpd)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["grouping_type"] = fp.read(4).decode("utf-8")
            if self.version == 1:
                self.box_info["default_length"] = read_u32(fp)
            elif self.version >= 2:
                self.box_info["default_sample_description_index"] = read_u32(fp)
            self.box_info["entry_count"] = read_u32(fp)
            self.box_info["entry_list"] = []
            for _ in range(self.box_info["entry_count"]):
                if self.box_info["default_length"] == 0 and self.version == 1:
                    description_length = read_u32(fp)
                else:
                    description_length = self.box_info["default_length"]
                sample_group_entry = binascii.b2a_hex(fp.read(description_length)).decode("utf-8")
                if self.box_info["grouping_type"] == "seig":
                    seig_dict = {
                        "crypto_byte_block": int(sample_group_entry[2], 16),
                        "skip_byte_block": int(sample_group_entry[3], 16),
                        "is_encrypted": int(sample_group_entry[5], 16),
                        "per_sample_iv_size": int(sample_group_entry[6:8], 16),
                        "kid": sample_group_entry[8:40],
                    }
                    if seig_dict["is_encrypted"] == 1 and seig_dict["per_sample_iv_size"] == 0:
                        seig_dict["constant_IV_size"] = int(sample_group_entry[40:42], 16)
                        seig_dict["constant_IV"] = sample_group_entry[42:]
                    self.box_info["entry_list"].append(seig_dict)
                else:
                    self.box_info["entry_list"].append({"sample_group_entry": sample_group_entry})
        finally:
            fp.seek(self.start_of_box + self.size)


class SaizBox(Mp4FullBox):
    """Sample auxiliary information sizes box (saiz)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            if self.flags & 0x000001 == 0x000001:
                self.box_info["aux_info_type"] = fp.read(4).decode("utf-8")
                self.box_info["aux_info_type_parameter"] = read_u32(fp)
            self.box_info["default_sample_info_size"] = read_u8(fp)
            self.box_info["sample_count"] = read_u32(fp)
            if self.box_info["default_sample_info_size"] == 0:
                self.box_info["sample_info_size_list"] = []
                for _ in range(self.box_info["sample_count"]):
                    self.box_info["sample_info_size_list"].append({"sample_info_size": read_u8(fp)})
        finally:
            fp.seek(self.start_of_box + self.size)


class SaioBox(Mp4FullBox):
    """Sample auxiliary information offsets box (saio)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            if self.flags & 0x000001 == 0x000001:
                self.box_info["aux_info_type"] = fp.read(4).decode("utf-8")
                self.box_info["aux_info_type_parameter"] = read_u32(fp)
            self.box_info["entry_count"] = read_u32(fp)
            self.box_info["offset_list"] = []
            for _ in range(self.box_info["entry_count"]):
                if self.version == 0:
                    self.box_info["offset_list"].append({"offset": read_u32(fp)})
                else:
                    self.box_info["offset_list"].append({"offset": read_u64(fp)})
        finally:
            fp.seek(self.start_of_box + self.size)


class StszBox(Mp4FullBox):
    """Sample size box (stsz)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["sample_size"] = read_u32(fp)
            self.box_info["sample_count"] = read_u32(fp)
            if self.box_info["sample_size"] == 0:
                self.box_info["entry_list"] = []
                for _ in range(self.box_info["sample_count"]):
                    self.box_info["entry_list"].append({"entry_size": read_u32(fp)})
        finally:
            fp.seek(self.start_of_box + self.size)


class Stz2Box(Mp4FullBox):
    """Compact sample size box (stz2)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["field_size"] = read_u32(fp)
            self.box_info["sample_count"] = read_u32(fp)
            self.box_info["entry_list"] = []
            for _ in range(self.box_info["sample_count"]):
                if self.box_info["field_size"] == 4:
                    mybyte = read_u8(fp)
                    self.box_info["entry_list"].append({"entry_size": mybyte // 16}, {"entry_size+": mybyte % 16})
                if self.box_info["field_size"] == 8:
                    self.box_info["entry_list"].append({"entry_size": read_u8(fp)})
                if self.box_info["field_size"] == 16:
                    self.box_info["entry_list"].append({"entry_size": read_u16(fp)})
        finally:
            fp.seek(self.start_of_box + self.size)


class StdpBox(Mp4FullBox):
    """Sample degradation priority box (stdp)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            pass
        finally:
            fp.seek(self.start_of_box + self.size)

    def update_table(self, fp: BinaryIO, sc: int) -> None:
        """Populate sample list by re-reading box data with known sample count."""
        fp_orig = fp.tell()
        fp.seek(self.start_of_box + self.header.header_size + 4)
        self.box_info["sample_list"] = []
        for _ in range(sc):
            self.box_info["sample_list"].append({"priority": read_u16(fp)})
        fp.seek(fp_orig)


class SdtpBox(Mp4FullBox):
    """Sample dependency type box (sdtp)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            pass
        finally:
            fp.seek(self.start_of_box + self.size)

    def update_table(self, fp: BinaryIO, sc: int) -> None:
        """Populate sample list by re-reading box data with known sample count."""
        fp_orig = fp.tell()
        fp.seek(self.start_of_box + self.header.header_size + 4)
        self.box_info["sample_list"] = []
        for _ in range(sc):
            the_byte = read_u8(fp)
            is_leading = the_byte >> 6
            depends_on = the_byte >> 4 & 3
            is_depended_on = the_byte >> 2 & 3
            has_redundancy = the_byte & 3
            self.box_info["sample_list"].append({
                "is_leading": is_leading,
                "sample_depends_on": depends_on,
                "sample_is_depended_on": is_depended_on,
                "sample_has_redundancy": has_redundancy,
            })
        fp.seek(fp_orig)


class SidxBox(Mp4FullBox):
    """Segment index box (sidx)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["reference_ID"] = read_u32(fp)
            self.box_info["timescale"] = read_u32(fp)
            if self.version == 0:
                self.box_info["earliest_presentation_time"] = read_u32(fp)
                self.box_info["first_offset"] = read_u32(fp)
            else:
                self.box_info["earliest_presentation_time"] = read_u64(fp)
                self.box_info["first_offset"] = read_u64(fp)
            fp.seek(2, 1)
            self.box_info["reference_count"] = read_u16(fp)
            self.box_info["reference_list"] = []
            for _ in range(self.box_info["reference_count"]):
                rt_sz = read_u32(fp)
                subsegment_dur = read_u32(fp)
                st_sz = read_u32(fp)
                self.box_info["reference_list"].append({
                    "reference_type": rt_sz >> 31,
                    "reference_size": rt_sz % 2147483648,
                    "subsegment_duration": subsegment_dur,
                    "starts_with_sap": st_sz >> 31,
                    "SAP_type": st_sz >> 28 & 7,
                    "SAP_delta_time": st_sz % 268435456,
                })
        finally:
            fp.seek(self.start_of_box + self.size)


class SsixBox(Mp4FullBox):
    """Subsegment index box (ssix)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["subsegment_count"] = read_u32(fp)
            self.box_info["subsegment_list"] = []
            for _ in range(self.box_info["subsegment_count"]):
                subsegment_dict = {"range_count": read_u32(fp)}
                range_list = []
                for _ in range(self.box_info["range_count"]):
                    l_r = read_u32(fp)
                    range_list.append({"level": l_r // 16777216, "range_size": l_r % 16777216})
                subsegment_dict["range_list"] = range_list
                self.box_info["subsegment_list"].append(subsegment_dict)
        finally:
            fp.seek(self.start_of_box + self.size)


class PrftBox(Mp4FullBox):
    """Producer reference time box (prft)."""

    def __init__(self, fp: BinaryIO, header: Header, parent: Mp4Box) -> None:
        """Parse box data from the file pointer."""
        super().__init__(fp, header, parent)
        try:
            self.box_info["reference_track_id"] = read_u32(fp)
            self.box_info["ntp_timestamp"] = read_u64(fp)
            if self.version == 0:
                self.box_info["media_time"] = read_u32(fp)
            else:
                self.box_info["media_time"] = read_u64(fp)
        finally:
            fp.seek(self.start_of_box + self.size)
