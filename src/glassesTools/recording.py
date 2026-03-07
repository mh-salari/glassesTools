"""Eye-tracker recording data model and discovery utilities."""

import dataclasses
import pathlib
import typing

import natsort

from . import json, utils
from .eyetracker import EyeTracker
from .timestamps import Timestamp


@dataclasses.dataclass
class Recording:
    """Data model for an eye-tracker recording with JSON serialization."""

    default_json_file_name: typing.ClassVar[str] = "recording_info.json"

    name: str = ""
    source_directory: pathlib.Path = ""
    working_directory: pathlib.Path = ""
    start_time: Timestamp = dataclasses.field(default_factory=lambda: Timestamp(0))
    duration: float | None = None
    eye_tracker: EyeTracker = EyeTracker.Unknown
    eye_tracker_name: str = ""  # name to show if eye_tracker is EyeTracker.Generic
    project: str = ""
    participant: str = ""
    firmware_version: str = ""
    glasses_serial: str = ""
    recording_unit_serial: str = ""
    recording_software_version: str = ""
    scene_camera_serial: str = ""
    scene_video_file: str = ""

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
        kwds["eye_tracker"] = EyeTracker(kwds["eye_tracker"])
        return Recording(**kwds, working_directory=path.parent)

    def get_scene_video_path(self) -> pathlib.Path:
        """Resolve the full path to the scene video file.

        Falls back to the source directory if the file is not found in
        the working directory.

        Returns:
            Path to the scene video file.

        """
        vid = self.working_directory / self.scene_video_file
        if not vid.is_file():
            if not self.source_directory.is_absolute():
                vid = (self.working_directory / self.source_directory / self.scene_video_file).resolve()
            else:
                vid = self.source_directory / self.scene_video_file
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


def find_recordings(
    paths: list[pathlib.Path], eye_tracker: EyeTracker, device_name: str | None = None
) -> list[Recording]:
    """Scan directories for valid eye-tracker recordings.

    Recursively walks each path, checks every subdirectory for a valid
    recording, and returns them sorted in natural OS order.

    Args:
        paths: Root directories to scan.
        eye_tracker: Eye tracker type to look for.
        device_name: Optional device name filter.

    Returns:
        Discovered recordings, naturally sorted by source directory.

    """
    from . import importing  # noqa: PLC0415

    all_recs = []
    for p in paths:
        all_dirs = utils.fast_scandir(p)
        all_dirs.append(p)
        for d in all_dirs:
            # check if dir is a valid recording
            if (recs := importing.get_recording_info(d, eye_tracker, device_name)) is not None:
                all_recs.extend(recs)

    # sort in order natural for OS
    return natsort.os_sorted(all_recs, lambda rec: rec.source_directory)
