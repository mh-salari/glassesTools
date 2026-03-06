"""Annotation types and event registry for marking temporal points and intervals in recordings."""

import dataclasses
from enum import Enum, auto

from . import json, utils


class Type(Enum):
    """Classification of annotation markers as either single points or intervals."""

    Point = auto()
    Interval = auto()


class EventType(utils.AutoName):
    """Types of annotation events used for synchronization, validation, and analysis."""

    Validate = auto()  # interval to be used for running glassesValidator
    Sync_Camera = auto()  # point to be used for synchronizing different cameras
    Sync_ET_Data = (
        auto()
    )  # episode to be used for synchronization of eye tracker data to scene camera (e.g. using VOR)
    Trial = auto()  # episode for which to map gaze to plane(s): output for files to be provided to user
    Target = auto()  # episode indicating when a specific target is being looked at
    Fixation = auto()  # episode indicating a fixation (e.g. detected by I2MC)


event_types = list(EventType)
json.register_type(
    json.TypeEntry(EventType, "__enum.Event__", utils.enum_val_2_str, lambda x: getattr(EventType, x.split(".")[1]))
)

type_map = {
    EventType.Validate: Type.Interval,
    EventType.Sync_Camera: Type.Point,
    EventType.Sync_ET_Data: Type.Interval,
    EventType.Trial: Type.Interval,
    EventType.Target: Type.Interval,
    EventType.Fixation: Type.Interval,
}

internal_types = {EventType.Target, EventType.Fixation}

tooltip_map = {
    EventType.Validate: "Validation episode",
    EventType.Sync_Camera: "Camera sync point",
    EventType.Sync_ET_Data: "Eye tracker synchronization episode",
    EventType.Trial: "Trial episode",
    EventType.Target: "Target episode",
    EventType.Fixation: "Fixation episode",
}

default_hotkeys = {
    EventType.Validate: "v",
    EventType.Sync_Camera: "c",
    EventType.Sync_ET_Data: "e",
    EventType.Trial: "t",
}


@dataclasses.dataclass
class Event:
    """A registered annotation event with its type, name, description, and optional hotkey."""

    event_type: EventType
    name: str
    description: str = ""
    hotkey: str = ""


EVENT_REGISTRY: list[Event] = []


def register_event(entry: Event) -> None:
    """Add an annotation event to the global registry."""
    EVENT_REGISTRY.append(entry)


def unregister_all_annotation_types() -> None:
    """Remove all annotation events from the global registry."""
    EVENT_REGISTRY.clear()


def get_events_by_type(event_type: EventType) -> list[Event]:
    """Return all registered events matching the given event type."""
    return [e for e in EVENT_REGISTRY if e.event_type == event_type]


def flatten_annotation_dict(
    annotations: dict[str, tuple[EventType, list[list[int]]]],
) -> dict[str, tuple[EventType, list[int]]]:
    """Flatten nested interval lists into a single flat list per annotation.

    Converts ``{name: (type, [[start, end], ...])}`` to ``{name: (type, [start, end, ...])}``.
    """
    annotations_flat: dict[str, tuple[EventType, list[int]]] = {}

    def _copy_flat_annotation(
        annotations: tuple[EventType, list[list[int]]],
    ) -> tuple[EventType, list[int]]:
        if annotations[1] and isinstance(annotations[1][0], list):
            return (annotations[0], [i for iv in annotations[1] for i in iv])
        return (annotations[0], annotations[1].copy())

    for e in EVENT_REGISTRY:  # iterate over this for consistent ordering
        if e.name not in annotations:
            continue
        annotations_flat[e.name] = _copy_flat_annotation(annotations[e.name])
    # add anything still missing
    for e_name, e_val in annotations.items():
        if e_name not in annotations_flat:
            annotations_flat[e_name] = _copy_flat_annotation(e_val)
    return annotations_flat


def unflatten_annotation_dict(
    annotations: dict[str, tuple[EventType, list[int]]], add_incomplete_intervals: bool = False
) -> dict[str, tuple[EventType, list[list[int]]]]:
    """Unflatten a flat annotation list back into nested interval pairs.

    Converts ``{name: (type, [start, end, ...])}`` to ``{name: (type, [[start, end], ...])}``.
    If *add_incomplete_intervals* is True, a trailing unpaired value is kept as ``[start]``.
    """
    annotations_unflat: dict[str, tuple[EventType, list[list[int]]]] = {}

    def _copy_unflat_annotation(
        annotations: tuple[EventType, list[int]],
    ) -> tuple[EventType, list[list[int]]]:
        if type_map[annotations[0]] == Type.Interval:
            result = (annotations[0], [annotations[1][m : m + 2] for m in range(0, len(annotations[1]) - 1, 2)])
            if add_incomplete_intervals and len(annotations[1]) % 2 == 1:
                result[1].append([annotations[1][-1]])
            return result
        return (annotations[0], [[tp] for tp in annotations[1]])

    for e in EVENT_REGISTRY:  # iterate over this for consistent ordering
        if e.name not in annotations:
            continue
        annotations_unflat[e.name] = _copy_unflat_annotation(annotations[e.name])
    # add anything still missing
    for e_name, e_val in annotations.items():
        if e_name not in annotations_unflat:
            annotations_unflat[e_name] = _copy_unflat_annotation(e_val)
    return annotations_unflat
