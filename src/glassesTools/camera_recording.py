"""Camera recording data model and import utilities."""

import dataclasses
import os
import pathlib
import shutil
import typing
from enum import auto

from . import json, utils, video_utils


class Type(utils.AutoName):
    """Camera recording type classification."""

    External = auto()
    Head_attached = auto()


json.register_type(
    json.TypeEntry(
        Type, "__enum.camera_recording.Type__", utils.enum_val_2_str, lambda x: getattr(Type, x.split(".")[1])
    )
)


@dataclasses.dataclass
class Recording:
    """Data model for a camera recording with JSON serialization."""

    default_json_file_name: typing.ClassVar[str] = "recording_info.json"

    name: str
    type: Type
    video_file: str
    source_directory: pathlib.Path
    working_directory: pathlib.Path = ""
    duration: float = None

    def store_as_json(self, path: str | pathlib.Path) -> None:
        """Serialize recording metadata to a JSON file.

        Args:
            path: File or directory path. If a directory, uses
                ``default_json_file_name``.

        """
        path = pathlib.Path(path)
        if path.is_dir():
            path /= self.default_json_file_name
        # Strip subclass-added fields; keep only Recording's own fields
        # (excluding working_directory, which is inferred from the file path on load)
        to_dump = dataclasses.asdict(self)
        to_dump = {k: to_dump[k] for k in to_dump if k in Recording.__annotations__ and k != "working_directory"}
        # dump to file
        json.dump(to_dump, path)

    @staticmethod
    def load_from_json(path: str | pathlib.Path) -> "Recording":
        """Deserialize recording metadata from a JSON file.

        Args:
            path: File or directory path. If a directory, uses
                ``default_json_file_name``.

        Returns:
            A ``Recording`` with ``working_directory`` set to the
            parent of the JSON file.

        """
        path = pathlib.Path(path)
        if path.is_dir():
            path /= Recording.default_json_file_name
        kwds = json.load(path)
        if "type" in kwds:
            kwds["type"] = Type(kwds["type"])
        # Backwards compat: older files may lack a type field
        if "type" not in kwds:
            kwds["type"] = Type.External
        return Recording(**kwds, working_directory=path.parent)

    def get_video_path(self) -> pathlib.Path:
        """Resolve the full path to the video file.

        Falls back to the source directory if the file is not found in
        the working directory.

        Returns:
            Path to the video file.

        """
        vid = self.working_directory / self.video_file
        if not vid.is_file():
            if not self.source_directory.is_absolute():
                vid = (self.working_directory / self.source_directory / self.video_file).resolve()
            else:
                vid = self.source_directory / self.video_file
        return vid

    def get_source_directory(self) -> pathlib.Path | None:
        """Resolve the full path to the source directory.

        Returns:
            Absolute path to the source directory, or ``None`` if unset.

        """
        if not self.source_directory:
            return None
        if not self.source_directory.is_absolute():
            return (self.working_directory / self.source_directory).resolve()
        return self.source_directory


def do_import(
    rec_info: Recording,
    cam_cal_file: str | pathlib.Path | None = None,
    copy_video: bool = True,
    source_dir_as_relative_path: bool = False,
) -> Recording:
    """Import a camera recording into the working directory.

    Copies the video file and camera calibration, extracts frame
    timestamps, and writes recording metadata as JSON.

    Args:
        rec_info: Recording descriptor with ``source_directory`` and
            ``working_directory`` set.
        cam_cal_file: Path to camera calibration XML file to copy.
        copy_video: If ``True``, copy the video file into the working
            directory.
        source_dir_as_relative_path: If ``True``, store the source
            directory as a relative path in the JSON metadata.

    Returns:
        The updated ``Recording`` with duration filled in.

    Raises:
        ValueError: If ``working_directory`` is not set.
        FileNotFoundError: If the source video file does not exist.

    """
    if not rec_info.working_directory:
        raise ValueError("working_directory must be set on the rec_info object")
    rec_info.working_directory = pathlib.Path(rec_info.working_directory)
    ifile = rec_info.source_directory / rec_info.video_file
    if not ifile.is_file():
        raise FileNotFoundError(f"The camera recording file {ifile} was not found")
    print(f"processing: {rec_info.video_file} -> {rec_info.working_directory}")

    if not rec_info.working_directory.is_dir():
        rec_info.working_directory.mkdir()

    if copy_video:
        ofile = rec_info.working_directory / rec_info.video_file
        print("  Copy video file...")
        shutil.copy2(ifile, ofile)

    # also get its calibration
    print("  Getting camera calibration...")
    if cam_cal_file is not None:
        shutil.copy2(str(cam_cal_file), str(rec_info.working_directory / "calibration.xml"))
    else:
        print("  !! No camera calibration provided! Defaulting to hardcoded")

    # and frame timestamps
    print("  Getting frame timestamps...")
    ts = video_utils.get_frame_timestamps_from_video(rec_info.get_video_path())
    ts.to_csv(str(rec_info.working_directory / "frameTimestamps.tsv"), sep="\t")
    rec_info.duration = float(ts.timestamp.iat[-1] - ts.timestamp.iat[0])

    # store recording info to folder
    if source_dir_as_relative_path:
        rec_info.source_directory = pathlib.Path(
            os.path.relpath(rec_info.source_directory, rec_info.working_directory)
        )
    rec_info.store_as_json(rec_info.working_directory)

    return rec_info
