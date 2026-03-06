"""Custom JSON serialization with type registry for round-tripping complex types."""

import dataclasses
import json
import pathlib
import typing

import orjson


def json_encoder(obj: typing.Any) -> dict[str, typing.Any]:
    """Encode registered types to JSON-serializable dicts."""
    matches: dict[type, TypeEntry] = {}
    for t in TYPE_REGISTRY:
        if isinstance(obj, t.type):
            matches[t.type] = t
    if matches:
        # find most precise match and use that
        best = None
        for t2 in matches:
            if best is None or issubclass(t2, best):
                best = t2
        return {matches[best].reg_name: matches[best].to_json(obj)}
    raise TypeError(f"type {type(obj)} cannot be serialized")


def dump(obj: typing.Any, file: pathlib.Path) -> None:
    """Serialize an object to a JSON file using orjson."""
    data = orjson.dumps(obj, json_encoder, orjson.OPT_INDENT_2 | orjson.OPT_PASSTHROUGH_SUBCLASS)
    pathlib.Path(file).write_bytes(data)


def json_decoder(d: dict[str, typing.Any]) -> typing.Any:
    """Decode JSON dicts back to registered types."""
    for t in TYPE_REGISTRY:
        if t.reg_name in d:
            return t.from_json(d[t.reg_name])
        if t.compatible_reg_names is not None:
            for r_name in t.compatible_reg_names:
                if r_name in d:
                    return t.from_json(d[r_name])
    return d


def load(file: pathlib.Path) -> typing.Any:
    """Load and deserialize a JSON file, decoding registered types."""
    with pathlib.Path(file).open(encoding="utf-8") as f:
        return json.load(f, object_hook=json_decoder)


def loads(payload: str) -> typing.Any:
    """Deserialize a JSON string, decoding registered types."""
    return json.loads(payload, object_hook=json_decoder)


@dataclasses.dataclass
class TypeEntry:
    """Registry entry mapping a Python type to its JSON serialization format."""

    type: type
    reg_name: str
    to_json: typing.Callable
    from_json: typing.Callable
    compatible_reg_names: list[str] | None = None


TYPE_REGISTRY: list[TypeEntry] = []


def register_type(entry: TypeEntry) -> None:
    """Register a type for custom JSON serialization and deserialization."""
    TYPE_REGISTRY.append(entry)


register_type(TypeEntry(pathlib.Path, "pathlib.Path", str, pathlib.Path))
register_type(TypeEntry(set, "builtin.set", list, set))
register_type(TypeEntry(tuple, "builtin.tuple", list, tuple))
