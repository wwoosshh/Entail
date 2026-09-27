# entail-nodes-basics

A workshop package for [entail](https://github.com/wwoosshh/entail)'s custom nodes: validators a small project puts on
points of its own program. entail finds it through the entry point group `entail.nodes` and registers its validators
by name.

| validator | checks | parameters |
|---|---|---|
| `json_object` | the text is a JSON object with these keys | `keys` |
| `max_chars` | the text is at most so many characters | `limit` |
| `within_context` | a prompt's tokens leave room for the answer in the context window | `context`, `reserve` |
| `same_size` | an image is the size that was asked for | `expected` |

It is not on PyPI; it installs from entail's repository, with entail 2.0 or later (`entail.nodes` is 2.0's):

```bash
pip install "git+https://github.com/wwoosshh/entail@v2.0.0#subdirectory=workshop/basics"
```

```python
from entail import nodes

answer = nodes.check("app.answer", answer, "json_object", keys=("title", "body"))
```

Run the program with `ENTAIL=load`. A check that fails is recorded (`entail_logs/`) and shown on the node "app.answer"
in `entail serve`; the program goes on. `ENTAIL_NODES=off` turns every custom node off; `ENTAIL_NODES=basics` attaches
only this package; the platform can turn a single node off.
