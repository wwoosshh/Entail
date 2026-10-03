"""CPU self-test of the harness's bookkeeping (M19 L5.4d), on synthetic run folders: python selftest.py

  aggregate.py   a mode is "passed" only when every scheduled case ran to its end in it; a missing case, a run that
                 exited non-zero, a run with steps not all recorded, a run of another case spec or of another freeze,
                 or a runner started on a part of the split makes it "partial"; something wrong in what ran makes it
                 "failed" whatever else is missing
  freeze_check   a code file changed, the entail commit moved, an environment key changed: each is a mismatch
No GPU, no vLLM: numpy only.
"""
import copy
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import aggregate  # noqa: E402
import common  # noqa: E402
import freeze_check  # noqa: E402

FREEZE = "f" * 64


def case(cid):
    return {"id": cid, "group": "normal", "N": 256, "K": 256, "out": "bfloat16", "layers": 1, "wseed": [1],
            "steps": [{"op": "call", "layer": 0, "M": 4, "seed": 2, "dist": "normal"},
                      {"op": "call", "layer": 0, "M": 4, "seed": 3, "dist": "normal"}], "split": "holdout"}


def defect(cid):
    c = case(cid)
    c.update(group="defect", steps=[dict(st, mut=1) for st in c["steps"]])
    return c


def write_run(root, c, label, *, exit_code=0, steps=None, ok=True, freeze=FREEZE, spec=None):
    d = os.path.join(root, c["id"], label)
    os.makedirs(d, exist_ok=True)
    if c["group"] == "defect" and label == "offA":
        ok = False                                   # the defect has its effect without entail
    repaired = {"repaired_delivered": 1} if c["group"] == "defect" and label == "guarantee" else {}
    ss = steps if steps is not None else [dict(s, delivered=True, exception=None, rng_same=True, entail=repaired,
                                               oracle={"ok": ok, "worst_ratio": 0.1 if ok else 9.0})
                                          for s in c["steps"]]
    common.write_json(os.path.join(d, "result.json"), {"case": spec or c, "steps": ss, "freeze_sha256": freeze})
    common.write_json(os.path.join(d, "run.json"), {"exit": exit_code})


def suite(root, cases, labels=("offA", "guarantee"), partial=False):
    common.write_json(os.path.join(root, "cases.json"), {
        "split": "holdout", "key": FREEZE, "freeze_sha256": FREEZE, "cases": cases,
        "scheduled": [c["id"] for c in cases], "selected": [c["id"] for c in cases], "labels": list(labels),
        "partial": partial})


def state(root):
    s = aggregate.summarize(root)
    return s["modes"]["guarantee"]["state"], s


def main():
    base = tempfile.mkdtemp(prefix="bfg_selftest_")
    try:
        cs = [case("N-a"), case("N-b"), defect("D-c")]

        def fresh(name):
            root = os.path.join(base, name)
            os.makedirs(root)
            suite(root, cs)
            for c in cs:
                write_run(root, c, "offA")
            return root

        root = fresh("complete")
        for c in cs:
            write_run(root, c, "guarantee")
        st, s = state(root)
        assert st == "passed" and s["complete"], (st, s["unfinished_runs"])
        print("ok every scheduled case ran to its end: passed, complete")

        root = fresh("missing")
        write_run(root, cs[0], "guarantee")
        write_run(root, cs[2], "guarantee")
        st, s = state(root)
        got = [tuple(x) for x in s["unfinished_runs"]["guarantee"]]
        assert st == "partial" and not s["complete"] and got == [("N-b", "missing")], (st, got)
        print("ok a scheduled case without results: partial (missing)")

        for name, kw, why in (("exit", {"exit_code": 1}, "exit 1"),
                              ("steps", {"steps": []}, "0 of 2 steps recorded"),
                              ("spec", {"spec": dict(cs[1], wseed=[9])}, "the run's case is not the scheduled one"),
                              ("freeze", {"freeze": "0" * 64}, "the run was bound to another freeze manifest")):
            root = fresh(name)
            write_run(root, cs[0], "guarantee")
            write_run(root, cs[2], "guarantee")
            write_run(root, cs[1], "guarantee", **kw)
            st, s = state(root)
            got = dict(map(tuple, s["unfinished_runs"]["guarantee"]))
            assert st in ("partial", "failed") and got.get("N-b") == why, (name, st, got)
        print("ok non-zero exit, missing steps, another case spec, another freeze: not complete, never passed")

        root = fresh("started_on_a_part")
        suite(root, cs, partial=True)
        for c in cs:
            write_run(root, c, "guarantee")
        st, s = state(root)
        assert st == "partial" and not s["complete"], st
        print("ok a runner started on a part of the split: partial")

        root = fresh("failed_and_missing")
        write_run(root, cs[0], "guarantee", ok=False)
        st, s = state(root)
        assert st == "failed", st
        print("ok a wrong value in what ran: failed, also when a case is missing")

        root = fresh("not_run_steps")                        # steps recorded as not run count as recorded
        write_run(root, cs[0], "guarantee")
        write_run(root, cs[2], "guarantee")
        steps = [dict(cs[1]["steps"][0], delivered=True, exception=None, rng_same=True,
                      oracle={"ok": True, "worst_ratio": 0.1}),
                 dict(cs[1]["steps"][1], not_run="the device was stopped at an earlier step")]
        write_run(root, cs[1], "guarantee", steps=steps)
        st, s = state(root)
        assert "N-b" not in dict(map(tuple, s["unfinished_runs"].get("guarantee", []))), s["unfinished_runs"]
        print("ok a step recorded as not run (after a stop) leaves the run complete")

        # freeze_check: code, commit and environment each compared
        f1, f2 = os.path.join(base, "a.txt"), os.path.join(base, "b.txt")
        for p in (f1, f2):
            with open(p, "w", encoding="utf-8") as f:
                f.write(p)
        rel = lambda p: os.path.relpath(p, common.ENTAIL_ROOT).replace(os.sep, "/")  # noqa: E731
        env = {"torch": "2.13.0", "gpu": "RTX"}
        manifest = {"code": {rel(f1): common.sha256_file(f1), rel(f2): "0" * 64},
                    "entail_git": common.git_head(common.ENTAIL_ROOT), "env": env}
        mm = freeze_check.mismatches(manifest, dict(env))
        assert [m["what"] for m in mm] == [f"code {rel(f2)}"], mm
        mm = freeze_check.mismatches(dict(manifest, entail_git={"head": "x", "dirty": False}),
                                     dict(env, torch="2.14.0"))
        whats = sorted(m["what"] for m in mm)
        assert whats == sorted([f"code {rel(f2)}", "entail commit", "environment torch"]), whats
        print("ok freeze_check: a changed code file, a moved commit and a changed environment key are mismatches")
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
