"""A small LLM app with two custom nodes (entail product track P5): the answer must be a JSON object with a title and a
body, and short enough for the page it goes on. No model, no GPU: `generate` stands in for one.

    ENTAIL=load python examples/custom_nodes/llm_app.py            # every answer holds: nothing broken
    ENTAIL=load python examples/custom_nodes/llm_app.py --fault    # one answer is cut off: broken at app.answer
    entail serve                                                    # the nodes app.answer and app.summary

Without ENTAIL=load the checks do not run and the app is unchanged.
"""
import json
import sys

from entail import nodes

FAULT = "--fault" in sys.argv


@nodes.validator("json_object")
def json_object(value, keys=()):
    try:
        obj = json.loads(value)
    except ValueError as e:
        return nodes.broken(f"not JSON ({e})")
    missing = [k for k in keys if k not in obj]
    return nodes.broken(f"missing keys {missing}") if missing else nodes.ok()


@nodes.validator("max_chars")
def max_chars(value, limit):
    return nodes.ok() if len(value) <= limit else nodes.broken(f"{len(value)} characters, the limit is {limit}")


def generate(question):
    """Stands in for a model call."""
    body = f"An answer about {question}."
    text = json.dumps({"title": question.title(), "body": body})
    if FAULT and question == "tides":
        return text[:-12]          # the model stopped early: the JSON is cut off
    return text


@nodes.watch("app.answer", json_object, keys=("title", "body"))
def answer(question):
    return generate(question)


def summary(text):
    return nodes.check("app.summary", text[:80], max_chars, limit=120)


if __name__ == "__main__":
    for q in ("volcanoes", "tides", "comets"):
        a = answer(q)
        print(q, "->", summary(a))
