"""sites: where checks run, and what each may cost (LIBRARY_DESIGN.md 4.7, principle 6).

Always on: load (once, at start-up), container (host-side updates of a cache or a scheduler), request (per request).
Debug and CI only: operation-level propagation (propagate.py), about 2.1-2.2x in eager decode.
Never inside a compiled or captured region: a check placed there cost 47.7x and changed the output
(reinvestigation/feasibility.md 2.3). Before running: `entail check` makes the load decisions without a GPU.
"""
SITES = ("load", "container", "request", "debug")

# Cost targets (LIBRARY_DESIGN.md 8, S4). Off costs nothing at every site.
BUDGET = {
    "load": "at most 5% of start-up time",
    "container": "at most 1.02x decode time, on the CUDA Graph path too",
    "request": "microseconds per request",
    "debug": "at most 2x",
}


def check_static(model_path: str, engine: str, settings: dict):
    """`entail check`: the load decisions for a model, an engine and its settings, before anything runs."""
    raise NotImplementedError("M3.4: entail check")


def at_load(model_path: str, engine: str, choices: dict, policy):
    """Read the sources, verify declarations against the data, compare with the engine's choices."""
    raise NotImplementedError("M3.2: load-time contracts")


def at_container(extent, where: str, policy):
    """Range and time contracts on host-side numbers (KV extents, buffer epochs)."""
    raise NotImplementedError("M5.1-M5.2: container contracts")


def at_request(request_facts: dict, server_choice: dict, policy):
    """Chat template, reasoning history, tool-call format: facts that belong to one request."""
    raise NotImplementedError("M5.3: request contracts")


def debug_propagation():
    """Operation-level propagation for diagnosis (wraps propagate.RolePropagation)."""
    raise NotImplementedError("M7.1: diagnosis mode")
