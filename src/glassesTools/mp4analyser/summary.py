"""Summary information extracted from an MP4 file."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .iso import Mp4File


class Summary:
    """Summary class for an Mp4File instance."""

    def __init__(self, mp4file: Mp4File) -> None:
        """Extract summary data from the given MP4 file."""
        boxes = mp4file.children
        self.data = {}
        self.data["filename"] = mp4file.filename
        self.data["filesize (bytes)"] = pathlib.Path(mp4file.filename).stat().st_size
        if fstyp := next((box for box in boxes if box.type in {"ftyp", "styp"}), None):
            self.data["brand"] = fstyp.box_info["major_brand"]
        # check if there is a moov and if there is a moov that contains traks N.B only ever 0,1 moov boxes
        if moov := next((box for box in boxes if box.type == "moov"), None):
            mvhd = next(box for box in moov.children if box.type == "mvhd")
            self.data["creation_time"] = mvhd.box_info["creation_time"]
            self.data["modification_time"] = mvhd.box_info["modification_time"]
            self.data["duration (secs)"] = round(mvhd.box_info["duration"] / mvhd.box_info["timescale"])
            if self.data["duration (secs)"] > 0:
                self.data["bitrate (bps)"] = round(8 * self.data["filesize (bytes)"] / self.data["duration (secs)"])
            traks = [tbox for tbox in moov.children if tbox.type == "trak"]
            self.data["track_list"] = []
            for trak in traks:
                this_trak = {}
                this_trak["track_id"] = next(box for box in trak.children if box.type == "tkhd").box_info["track_ID"]
                mdia = next(box for box in trak.children if box.type == "mdia")
                mdhd = next(box for box in mdia.children if box.type == "mdhd")
                hdlr = next(box for box in mdia.children if box.type == "hdlr")
                minf = next(box for box in mdia.children if box.type == "minf")
                stbl = next(box for box in minf.children if box.type == "stbl")
                t = mdhd.box_info["timescale"]
                d = mdhd.box_info["duration"]
                v = mdhd.version

                sz = next(box for box in stbl.children if box.type in {"stsz", "stz2"})
                sc = sz.box_info["sample_count"]
                if sz.box_info["sample_size"] > 0:
                    # uniform sample size
                    trak_size = sz.box_info["sample_size"] * sc
                else:
                    trak_size = sum(entry["entry_size"] for entry in sz.box_info["entry_list"])

                sample_rate = None
                if (d < 0xFFFFFFFF and v == 0) or (d < 0xFFFFFFFFFFFFFFFF and v == 1):
                    this_trak["track_duration (secs)"] = round(d / t)
                    if trak_size > 0 and this_trak["track_duration (secs)"] > 0:
                        this_trak["track_bitrate (calculated bps)"] = round(
                            8 * trak_size / this_trak["track_duration (secs)"]
                        )
                        sample_rate = round((sc * t) / d, 2)

                codec_info = next(box for box in stbl.children if box.type == "stsd").children[0]
                media = hdlr.box_info["handler_type"]
                if media == "vide":
                    this_trak["media_type"] = "video"
                    this_trak["codec_type"] = codec_info.type
                    this_trak["width"] = codec_info.box_info.get("width", "unknown")
                    this_trak["height"] = codec_info.box_info.get("height", "unknown")
                    if sample_rate is not None:
                        this_trak["frame_rate"] = sample_rate
                elif media == "soun":
                    this_trak["media_type"] = "audio"
                    this_trak["codec_type"] = codec_info.type
                    this_trak["channel_count"] = codec_info.box_info.get("audio_channel_count", "unknown")
                    this_trak["sample_rate"] = codec_info.box_info.get("audio_sample_rate", "unknown")
                else:
                    this_trak["media_type"] = media
                    this_trak["codec_type"] = codec_info.type

                self.data["track_list"].append(this_trak)
        else:
            # no moov found
            self.data["contains_moov"] = False

        if [box for box in boxes if box.type == "moof"]:
            self.data["contains_fragments"] = True
