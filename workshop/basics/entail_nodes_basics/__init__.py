"""entail-nodes-basics: a workshop package - reusable validators for custom nodes (entail product track P5;
LIBRARY_DESIGN.md 13.7). entail finds it through the entry point group `entail.nodes` and registers its validators by
name, so a program checks a value with them without importing this package:

    from entail import nodes
    nodes.check("app.answer", answer, "json_object", keys=("title",))

Each validator is a plain function of the value and keyword parameters that returns nodes.ok(), nodes.broken(why) or
nodes.unknown(why); the core runs it, records what it found and keeps it from breaking the program.
"""
import json

from entail import nodes

name = "basics"
version = "0.1.0"
requires = ">=2.0,<3"


def json_object(value, keys=()):
    """The value is a JSON object text that has these keys."""
    if not isinstance(value, str):
        return nodes.unknown(f"expected text, got {type(value).__name__}")
    try:
        obj = json.loads(value)
    except ValueError as e:
        return nodes.broken(f"not JSON ({e})")
    if not isinstance(obj, dict):
        return nodes.broken(f"JSON, but a {type(obj).__name__}, not an object")
    missing = [k for k in keys if k not in obj]
    return nodes.broken(f"missing keys {missing}") if missing else nodes.ok()


def max_chars(value, limit=None):
    """The text is at most `limit` characters."""
    if limit is None:
        return nodes.unknown("no limit given")
    n = len(value)
    return nodes.ok() if n <= limit else nodes.broken(f"{n} characters, the limit is {limit}")


def within_context(value, context=None, reserve=0):
    """A prompt's token count (the value: an int, or a list of token ids) leaves `reserve` tokens of a `context`-token
    window for the answer."""
    if context is None:
        return nodes.unknown("no context window given")
    n = value if isinstance(value, int) else len(value)
    return nodes.ok() if n + reserve <= context else \
        nodes.broken(f"{n} prompt tokens + {reserve} for the answer > the context of {context}")


def same_size(value, expected=None):
    """An image's (width, height) - or anything with .size - is the size that was asked for."""
    if expected is None:
        return nodes.unknown("no expected size given")
    got = tuple(getattr(value, "size", value))
    return nodes.ok() if got == tuple(expected) else nodes.broken(f"{got[0]}x{got[1]}, asked for "
                                                                 f"{expected[0]}x{expected[1]}")


validators = {"json_object": json_object, "max_chars": max_chars, "within_context": within_context,
              "same_size": same_size}
