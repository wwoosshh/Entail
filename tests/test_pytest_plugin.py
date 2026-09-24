"""Tests for the pytest plugin and the risk conditions (ROADMAP M7.2): real pytest runs on small generated test files,
in a subprocess with only this plugin loaded. Run: python tests/test_pytest_plugin.py"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from entail import load, testing  # noqa: E402
from entail.facts import Certainty, Fact, ModelProps, Rotary, Source, Template  # noqa: E402

HAVE_PYTEST = importlib.util.find_spec("pytest") is not None

BOUNDARIES = '''
    import pytest
    from entail import core
    from entail.core import boundary
    from entail.facts import Layout

    class W:   # any value that takes a weak reference can carry facts
        pass

    @boundary("pack", returns=Layout("dense"))
    def pack(*, w):
        return w

    @boundary("kernel", w=Layout("strided"))
    def kernel(*, w):
        return w
'''


def pytest_run(source, *args, files=None):
    """(exit code, output) of pytest on one generated test file, with only the entail plugin loaded."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "test_generated.py"), "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(source))
        for name, text in (files or {}).items():
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(text)
        env = dict(os.environ, PYTHONPATH=ROOT, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", ENTAIL_LOG_DIR="off")
        env.pop("ENTAIL", None)
        p = subprocess.run([sys.executable, "-m", "pytest", "-p", "entail.pytest_plugin", "-p", "no:cacheprovider",
                            "-q", "-rA", *args, d], capture_output=True, text=True, env=env, cwd=d)
        return p.returncode, p.stdout + p.stderr


def test_without_being_asked_it_does_nothing():
    if not HAVE_PYTEST:
        print("skip (pytest is not installed)")
        return
    code, out = pytest_run(BOUNDARIES + '''
    def test_plain():
        assert core.mode() == "off"
        kernel(w=pack(w=W()))   # off: nothing is checked
    ''')
    assert code == 0 and "1 passed" in out and "entail" not in out.split("short test summary")[0].lower(), out


def test_a_failing_test_says_where_meaning_broke():
    """--entail (debug mode): a boundary that breaks stops the test, and the report names it; a test that fails with
    every boundary intact is pointed inside a layer."""
    if not HAVE_PYTEST:
        print("skip (pytest is not installed)")
        return
    code, out = pytest_run(BOUNDARIES + '''
    def test_broken():
        kernel(w=pack(w=W()))   # dense handed to a kernel that takes strided: nothing converts it

    def test_wrong_but_intact():
        @boundary("kernel2", w=Layout("dense"))
        def kernel2(*, w):
            return w
        kernel2(w=pack(w=W()))
        assert False, "the output was wrong"
    ''', "--entail")
    assert code == 1 and "2 failed" in out, out
    assert "refused at boundary:kernel: Layout" in out, out
    assert "[entail] where: meaning broke at boundary:kernel" in out, out
    assert "[entail] where: every checked boundary kept its meaning (1 boundaries)" in out, out
    assert "suspect: the output is wrong but no checked boundary broke" in out, out


def test_load_mode_reports_and_the_test_goes_on():
    if not HAVE_PYTEST:
        print("skip (pytest is not installed)")
        return
    code, out = pytest_run(BOUNDARIES + '''
    from entail import load
    from entail.contracts import RULES, Contract, Decision, Verdict

    def test_reported():
        load.enforce([Decision(Contract("load:m", "engine.rotary", ("Rotary",)), "Rotary", Verdict.BROKEN,
                               RULES["no_resolution"])])
    ''', "--entail", "--entail-mode=load")
    assert code == 0 and "1 passed" in out, out
    assert "EntailWarning: meaning broke at load:m (reported, not stopped)" in out, out


def test_fixtures_conditions_problems_and_the_ledger():
    if not HAVE_PYTEST:
        print("skip (pytest is not installed)")
        return
    problems = json.dumps({"problems": [
        {"id": "rb-12", "fact": "KvExtent", "defect": "rolebench/cases/12_session_restore/case.py:defect",
         "fixed": "rolebench/cases/12_session_restore/case.py:fixed", "expected": "broken", "milestone": "M5",
         "site": "container"},
        {"id": "fd-rope", "fact": "Rotary", "defect": "issue_track/rope_override/", "fixed": None,
         "expected": "resolved", "milestone": "M3", "site": "load"}]})
    code, out = pytest_run(BOUNDARIES + '''
    from entail import load
    from entail.facts import Certainty, Fact, ModelProps, Source

    WINDOW = load.Declared()
    WINDOW.facts["ModelProps"] = [Fact("ModelProps", ModelProps(sliding_window=8), Source("file", "config.json"),
                                       Certainty.DECLARED)]

    @pytest.mark.entail_conditions(facts=WINDOW, chunk_sizes=(16,))
    def test_lengths(entail_condition):
        assert entail_condition.params["length"] in (7, 8, 9, 15, 16, 17)

    def test_problem(entail_problem):
        assert entail_problem.expected in ("broken", "resolved")

    @pytest.mark.entail
    def test_ledger(entail_ledger):
        with pytest.raises(core.RoleError):
            kernel(w=pack(w=W()))
        assert [d.contract.boundary for d in entail_ledger.decisions] == ["boundary:kernel"]
        assert entail_ledger.locate().broken_at == "boundary:kernel"

    def test_no_conditions(entail_condition):
        pass
    ''', "--entail-problems", "problems.json", files={"problems.json": problems})
    assert code == 0, out
    for test_id in ("test_lengths[length=7]", "test_lengths[length=16]", "test_lengths[length=17]",
                    "test_problem[rb-12]", "test_problem[fd-rope]", "test_ledger"):
        assert f"PASSED test_generated.py::{test_id}" in out, (test_id, out)
    assert "9 passed, 1 skipped in" in out and "got empty parameter set" in out, "no warning for a refusal it saw"


def fact(name, value):
    return Fact(name, value, Source("file", "config.json"), Certainty.DECLARED)


def test_risk_conditions_come_from_what_the_model_declares():
    d = load.Declared()
    d.facts["ModelProps"] = [fact("ModelProps", ModelProps(sliding_window=4096))]
    d.facts["Rotary"] = [fact("Rotary", Rotary("llama3", 500000.0, 32.0, 8192))]
    d.facts["Template"] = [fact("Template", Template("a" * 64, reasoning_history="keep", tool_call_format="hermes"))]
    found = testing.risk_conditions(facts=d, chunk_sizes=(4096,))
    assert [c.id for c in found] == ["length=4095", "length=4096", "length=4097", "length=8191", "length=8192",
                                     "length=8193", "second_turn", "tool_call"], [c.id for c in found]
    assert found[4].why == "exactly the context where llama3 RoPE scaling takes over (8192)", found[4]
    assert found[0].fact == "ModelProps.sliding_window" and found[6].params == {"turns": 2, "reasoning": True}
    plain = load.Declared()
    plain.facts["Rotary"] = [fact("Rotary", Rotary("default", 1000000.0))]
    assert testing.risk_conditions(facts=plain) == [], "an unscaled RoPE has no threshold"


def test_problems_come_from_a_file_or_nowhere():
    old = os.environ.pop("ENTAIL_PROBLEMS", None)
    try:
        assert testing.problems() == []
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"id": "rb-01", "fact": "Layout", "defect": "x", "fixed": "y", "expected": "refused",
                            "milestone": "M4", "unknown_column": 1}], f)
            (p,) = testing.problems(path)
            assert (p.id, p.expected, p.site) == ("rb-01", "refused", ""), p
    finally:
        if old is not None:
            os.environ["ENTAIL_PROBLEMS"] = old


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
