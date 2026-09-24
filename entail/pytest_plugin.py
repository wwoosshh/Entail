"""pytest plugin: entail in a test suite, as a type checker runs in CI (LIBRARY_DESIGN.md 4.12; ROADMAP M7.2).

Inert until asked. `pytest --entail` turns entail on for the session (entail.enable: the adapters are installed and
processes the tests start inherit it) and runs every test as its own diagnosis run; `@pytest.mark.entail` does the
same for one test without the adapters:
  - debug mode by default: every declared boundary is checked and what breaks stops the test - CI fails where
    production only reports (LIBRARY_DESIGN.md 4.10). `--entail-mode=load` reports instead, and a boundary that
    broke while the test passed becomes a warning
  - operation-level propagation around the test (diagnose.propagating), in debug mode
  - the process's counts are cleared when the test starts, so its ledger is its own
  - a test that fails gets an "entail" section in its report: where meaning broke or, every checked boundary having
    held, that the fault lies inside a layer (record.locate, the failure standing for a wrong output)
Fixtures:
  entail_ledger     this test's decisions and layer comparisons, and .locate()
  entail_condition  one risk condition drawn from a model's declarations (testing.risk_conditions): the test runs
                    once per condition of @pytest.mark.entail_conditions(model=..., chunk_sizes=...)
  entail_problem    one test problem of --entail-problems FILE (or ENTAIL_PROBLEMS): the test runs once per problem
"""
import pytest


class EntailWarning(UserWarning):
    """Meaning broke at a boundary while the test went on (load mode reports what it cannot repair)."""


_RUN = pytest.StashKey()


def pytest_addoption(parser):
    group = parser.getgroup("entail")
    group.addoption("--entail", action="store_true", default=False,
                    help="turn entail on and run each test as a diagnosis run")
    group.addoption("--entail-mode", default="debug", choices=("debug", "load"),
                    help="debug (the default): what breaks stops the test; load: it is reported as a warning")
    group.addoption("--entail-problems", default=None, metavar="FILE",
                    help="a JSON list of test problems for the entail_problem fixture")


def pytest_configure(config):
    config.addinivalue_line("markers", "entail(mode='debug', propagate=True): run this test as an entail diagnosis "
                                       "run")
    config.addinivalue_line("markers", "entail_conditions(model=None, chunk_sizes=()): run the test once per risk "
                                       "condition the model declares (fixture entail_condition)")
    if config.getoption("entail"):
        import entail

        entail.enable(config.getoption("entail_mode"))


def _settings(item):
    marker = item.get_closest_marker("entail")
    if marker is None and not item.config.getoption("entail"):
        return None
    kw = dict(marker.kwargs) if marker is not None else {}
    return {"mode": kw.get("mode", item.config.getoption("entail_mode")), "propagate": kw.get("propagate", True)}


class Run:
    """What one test did: the ledger since it started, the counts since they were cleared, the layers compared."""

    def __init__(self):
        from . import load

        self.first = len(load.LEDGER.decisions)
        self.first_layer = len(load.LEDGER.layers)
        self.mode_before = None

    @property
    def decisions(self):
        from . import load

        return load.LEDGER.decisions[self.first:]

    @property
    def layers(self):
        from . import load

        return load.LEDGER.layers[self.first_layer:]

    def locate(self, output_wrong=None):
        from . import boundaries, record, tally

        passes = {}
        for (where, _), n in boundaries.PASSES.items():
            passes[f"boundary:{where}"] = passes.get(f"boundary:{where}", 0) + n
        for (b, _), n in tally.PASSES.items():
            passes[b] = passes.get(b, 0) + n
        skipped = [b for b, s in tally.STATS.items() if not s["checks"] and (s["skipped"] or s["deferred"])]
        return record.locate([record.decision_json(d) for d in self.decisions], passes, self.layers, output_wrong,
                             skipped)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    settings = _settings(item)
    if settings is None:
        return
    from . import boundaries, core, tally

    boundaries.REPEATS.clear()
    boundaries.PASSES.clear()
    tally.reset()
    run = Run()
    run.mode_before = core.mode()
    core.set_mode(settings["mode"])
    item.stash[_RUN] = run


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    settings = _settings(item)
    ctx = None
    if settings is not None and settings["propagate"]:
        from . import diagnose

        try:
            ctx = diagnose.propagating()
        except ImportError:   # no torch: nothing to propagate through
            ctx = None
    if ctx is None:
        yield
        return
    with ctx:
        yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    run = item.stash.get(_RUN, None)
    if run is not None and call.when in ("setup", "call"):
        found = run.locate(output_wrong=True if report.failed else None)
        if report.failed and (found.broken or found.all_intact or found.unchecked):
            report.sections.append(("entail", "\n".join(found.lines())))
        elif report.passed:   # broken and reported while the test went on (a refusal raised, and the test saw it)
            went_on = sorted({d.contract.boundary for d in run.decisions if d.verdict.value == "broken"})
            if went_on:
                item.warn(EntailWarning(f"meaning broke at {', '.join(went_on)} (reported, not stopped)"))


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    run = item.stash.get(_RUN, None)
    if run is not None and run.mode_before is not None:
        from . import core

        core.set_mode(run.mode_before)


@pytest.fixture
def entail_ledger(request):
    """This test's decisions, layer comparisons and localization so far (a run of its own when the test is not an
    entail run: it counts from here)."""
    run = request.node.stash.get(_RUN, None)
    if run is None:
        run = Run()
        request.node.stash[_RUN] = run
    return run


def pytest_generate_tests(metafunc):
    if not {"entail_condition", "entail_problem"} & set(metafunc.fixturenames):
        return   # every pytest session with entail installed passes here: nothing else is imported for it
    from . import testing

    if "entail_condition" in metafunc.fixturenames:
        marker = metafunc.definition.get_closest_marker("entail_conditions")
        kw = dict(marker.kwargs) if marker is not None else {}
        found = testing.risk_conditions(kw.get("model"), kw.get("facts"), kw.get("chunk_sizes", ()))
        metafunc.parametrize("entail_condition", found, ids=[c.id for c in found])
    if "entail_problem" in metafunc.fixturenames:
        found = testing.problems(metafunc.config.getoption("entail_problems"))
        metafunc.parametrize("entail_problem", found, ids=[p.id for p in found])
