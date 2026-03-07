"""Custom JSON serialization with type registry for round-tripping complex types.

Uses orjson for writing (fast, supports a custom ``default`` callback) and stdlib
``json`` for reading (orjson lacks ``object_hook`` support needed for decoding
tagged dicts back into Python types).

Types that aren't natively JSON-serializable (e.g. ``set``, ``tuple``, ``pathlib.Path``)
are wrapped in tagged dicts like ``{"builtin.set": [1, 2, 3]}`` so they survive a
serialization round-trip. Other modules register their own types via ``register_type``.
"""

import dataclasses
import json
import pathlib
import typing

import orjson


def json_encoder(obj: typing.Any) -> dict[str, typing.Any]:
    """Encode registered types to JSON-serializable dicts.

    Finds the most specific registered type matching ``obj`` and uses its
    ``to_json`` callable to produce a tagged dict ``{reg_name: value}``.

    Args:
        obj: The object to serialize.

    Returns:
        A single-key dict mapping the registered type name to the serialized value.

    Raises:
        TypeError: If no registered type matches ``obj``.

    """
    matches: dict[type, TypeEntry] = {}
    for t in TYPE_REGISTRY:
        if isinstance(obj, t.type):
            matches[t.type] = t
    if matches:
        # Multiple types may match via isinstance (e.g. a subclass matches both
        # itself and its parent). Pick the most derived type so we use the
        # correct serializer.
        best = None
        for t2 in matches:
            if best is None or issubclass(t2, best):
                best = t2
        return {matches[best].reg_name: matches[best].to_json(obj)}
    raise TypeError(f"type {type(obj)} cannot be serialized")


def dump(obj: typing.Any, file: pathlib.Path) -> None:
    """Serialize an object to a JSON file using orjson.

    Args:
        obj: The object to serialize.
        file: Path to the output JSON file.

    """
    # OPT_PASSTHROUGH_SUBCLASS ensures subclasses of dict/list/etc. are routed to
    # json_encoder instead of being silently serialized as their base type.
    data = orjson.dumps(obj, json_encoder, orjson.OPT_INDENT_2 | orjson.OPT_PASSTHROUGH_SUBCLASS)
    pathlib.Path(file).write_bytes(data)


def json_decoder(d: dict[str, typing.Any]) -> typing.Any:
    """Decode JSON dicts back to registered types.

    Checks each registered type's ``reg_name`` (and ``compatible_reg_names``)
    against the dict keys. Returns the first match reconstructed via ``from_json``,
    or the original dict if no type matches.

    Args:
        d: A decoded JSON dict (from ``json.load``'s ``object_hook``).

    Returns:
        The reconstructed Python object, or the original dict if unrecognized.

    """
    for t in TYPE_REGISTRY:
        if t.reg_name in d:
            return t.from_json(d[t.reg_name])
        # Also check legacy tag names so older JSON files still deserialize correctly
        if t.compatible_reg_names is not None:
            for r_name in t.compatible_reg_names:
                if r_name in d:
                    return t.from_json(d[r_name])
    # No registered type matched — return the plain dict
    return d


def load(file: pathlib.Path) -> typing.Any:
    """Load and deserialize a JSON file, decoding registered types.

    Args:
        file: Path to the JSON file.

    Returns:
        The deserialized Python object with registered types reconstructed.

    """
    with pathlib.Path(file).open(encoding="utf-8") as f:
        return json.load(f, object_hook=json_decoder)


def loads(payload: str) -> typing.Any:
    """Deserialize a JSON string, decoding registered types.

    Args:
        payload: The JSON string to deserialize.

    Returns:
        The deserialized Python object with registered types reconstructed.

    """
    return json.loads(payload, object_hook=json_decoder)


@dataclasses.dataclass
class TypeEntry:
    """Registry entry mapping a Python type to its JSON serialization format.

    Attributes:
        type: The Python type to match via ``isinstance``.
        reg_name: Tag name used as the JSON dict key for this type.
        to_json: Callable converting an instance to a JSON-serializable value.
        from_json: Callable reconstructing an instance from the JSON value.
        compatible_reg_names: Alternative tag names accepted during deserialization,
            for backward compatibility with older JSON files.

    """

    type: type
    reg_name: str
    to_json: typing.Callable
    from_json: typing.Callable
    compatible_reg_names: list[str] | None = None


TYPE_REGISTRY: list[TypeEntry] = []


def register_type(entry: TypeEntry) -> None:
    """Register a type for custom JSON serialization and deserialization.

    Args:
        entry: The type entry to add to the global registry.

    """
    TYPE_REGISTRY.append(entry)


# Built-in types that aren't natively JSON-serializable but are commonly used
# throughout the codebase. Other modules register domain-specific types at import time.
register_type(TypeEntry(pathlib.Path, "pathlib.Path", str, pathlib.Path))
register_type(TypeEntry(set, "builtin.set", list, set))
register_type(TypeEntry(tuple, "builtin.tuple", list, tuple))
