"""Annotation types and event registry for marking temporal points and intervals in recordings.

Each annotation event has an ``EventType`` (e.g. Validate, Trial, Sync_Camera) and is classified
as either a ``Point`` (single timestamp) or ``Interval`` (start/end pair). Some event types are
internal — generated programmatically (e.g. Target from marker detection, Fixation from fixation
classification) — while others are user-annotated via the GUI.

Downstream code registers concrete ``Event`` instances into a global ``EVENT_REGISTRY``, which
controls ordering and lookup. Annotations are stored as nested interval lists
(``[[start, end], ...]``) but can be flattened to ``[start, end, ...]`` for processing and
timeline display, with ``flatten_annotation_dict`` / ``unflatten_annotation_dict`` handling
the conversion.
"""

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
# JSON round-trip: serializes as "EventType.Validate", so the deserializer
# splits on "." to extract the member name for getattr lookup.
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

# Internal types are not user-annotated — they are generated programmatically
# (e.g. by target detection or fixation classification algorithms).
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
    """Add an annotation event to the global registry.

    Args:
        entry: The event to register.

    """
    EVENT_REGISTRY.append(entry)


def unregister_all_annotation_types() -> None:
    """Remove all annotation events from the global registry."""
    EVENT_REGISTRY.clear()


def get_events_by_type(event_type: EventType) -> list[Event]:
    """Return all registered events matching the given event type.

    Args:
        event_type: The event type to filter by.

    Returns:
        List of matching registered events.

    """
    return [e for e in EVENT_REGISTRY if e.event_type == event_type]


def flatten_annotation_dict(
    annotations: dict[str, tuple[EventType, list[list[int]]]],
) -> dict[str, tuple[EventType, list[int]]]:
    """Flatten nested interval lists into a single flat list per annotation.

    Converts ``{name: (type, [[start, end], ...])}`` to ``{name: (type, [start, end, ...])}``.

    Args:
        annotations: Dict mapping event names to ``(EventType, nested_list)`` tuples.

    Returns:
        Dict with the same keys but flat timestamp lists.

    """
    annotations_flat: dict[str, tuple[EventType, list[int]]] = {}

    def _copy_flat_annotation(
        annotations: tuple[EventType, list[list[int]]],
    ) -> tuple[EventType, list[int]]:
        # Guard: data may already be flat (list of ints) if it was never nested
        if annotations[1] and isinstance(annotations[1][0], list):
            return (annotations[0], [i for iv in annotations[1] for i in iv])
        return (annotations[0], annotations[1].copy())

    # Two-loop pattern: first pass uses EVENT_REGISTRY order for deterministic
    # output ordering, second pass catches any annotations not in the registry
    # (e.g. from plugins or newer event types).
    for e in EVENT_REGISTRY:
        if e.name not in annotations:
            continue
        annotations_flat[e.name] = _copy_flat_annotation(annotations[e.name])
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

    Args:
        annotations: Dict mapping event names to ``(EventType, flat_list)`` tuples.
        add_incomplete_intervals: Whether to keep a trailing unpaired start timestamp.

    Returns:
        Dict with the same keys but nested interval pair lists.

    """
    annotations_unflat: dict[str, tuple[EventType, list[list[int]]]] = {}

    def _copy_unflat_annotation(
        annotations: tuple[EventType, list[int]],
    ) -> tuple[EventType, list[list[int]]]:
        if type_map[annotations[0]] == Type.Interval:
            # Pair consecutive timestamps into [start, end] intervals
            result = (annotations[0], [annotations[1][m : m + 2] for m in range(0, len(annotations[1]) - 1, 2)])
            # An odd number of timestamps means a trailing start without an end
            # (e.g. user started an interval but hasn't closed it yet)
            if add_incomplete_intervals and len(annotations[1]) % 2 == 1:
                result[1].append([annotations[1][-1]])
            return result
        # Point-type annotations: wrap each individual timestamp in a list
        return (annotations[0], [[tp] for tp in annotations[1]])

    # Same two-loop pattern as flatten: registry first for ordering, then remainder
    for e in EVENT_REGISTRY:
        if e.name not in annotations:
            continue
        annotations_unflat[e.name] = _copy_unflat_annotation(annotations[e.name])
    for e_name, e_val in annotations.items():
        if e_name not in annotations_unflat:
            annotations_unflat[e_name] = _copy_unflat_annotation(e_val)
    return annotations_unflat
