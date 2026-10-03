"""Judge and count a suite run: python aggregate.py <out_root> [summary.json]

Every decision is made from what the next operation read (the observed output, judged by the independent oracle),
from whether the call handed a value on, and - for graphs - from what the graph's own next operation read, not
from entail's records (those only label the path: normal or repaired). A case or a mode with missing results stays
in the denominator as incomplete.

Per step (calls, direct calls, replays):
  normal_delivered     handed on, within the oracle's tolerance, no repair or reference path used
  repaired_delivered   handed on, within the tolerance, after a repair or with the reference's values
  blocked              not handed on, refused before it left. A replay: the graph's next operation did not run (the
                       device stopped at the gate; v3 harness, which marks on the host whether it ran)
  wrong_escaped        a value beyond the tolerance was handed on, or read inside the graph, even once - NaN included
  refused_after_use    a replay refused after the graph's next operation had read a value (within the tolerance):
                       not a block, the refusal came too late
  stopped              not run: the device was stopped by a block at an earlier step of the same process
  unknown              nothing observed (no producer output, a step not reached)
  error                an exception that is not a refusal (crash, OOM)
Until 2026-10-03 (L5.4c) a replay whose next operation read only NaN counted as blocked; that is an escape of a value
without a permit, not a block. No step of the v1 and v2 runs read only NaN, so their numbers do not change.
Per defect or integrity case: the off baseline (off A) must be beyond the tolerance on a defective step
(mutation_effective); otherwise the case is incomplete, and it is never dropped.

Completeness (L5.4d): every case the split schedules is judged in every mode of the run, also when it has no results
(missing). A run of one case in one mode is complete when it exited 0, recorded every planned step (a step not run is
recorded with the reason), ran the scheduled case spec and was bound to the run's freeze manifest. A mode with an
unfinished run, or a run started on a part of the split (--only, --modes), is "partial", never "passed".
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LABELS = ("offA", "offB", "load", "guarantee", "structure")
COUNTS = ("scheduled", "reached", "mutation_effective", "normal_delivered", "repaired_delivered", "blocked",
          "wrong_escaped", "refused_after_use", "unknown", "error")
DEFECT_LIKE = ("defect", "integrity")


def load(out_root, cid, label):
    d = os.path.join(out_root, cid, label)
    p = os.path.join(d, "result.json")
    if not os.path.exists(p):
        return None, None
    with open(p, encoding="utf-8") as f:
        r = json.load(f)
    z = os.path.join(d, "tensors.npz")
    return r, (np.load(z) if os.path.exists(z) else None)


def defective(step, case):
    """Whether a step is one where the case's defect is on."""
    if case.get("group") == "integrity":
        return bool(step.get("corrupt"))
    if case.get("tile_k256"):
        return step.get("M", 0) >= 128
    if case.get("mechanism", "").startswith("consumer-side"):
        return bool(step.get("cache")) and step.get("layer", 0) == 0 and step.get("op") == "call" \
            and step.get("role") not in ("fills the cache", "another weight of the same shape fills the cache")
    return bool(step.get("mut"))


def repaired_by_entail(step, label):
    e = step.get("entail") or {}
    if label == "guarantee":
        return e.get("repaired_delivered", 0) > 0
    if label == "structure":        # the structural experiment: the reference handed on, or a repair at admission
        return e.get("reference_delivered", 0) > 0 or e.get("admission_repairs", 0) > 0 or \
            e.get("repaired_delivered", 0) > 0
    if label == "load":
        return e.get("definition_calls", 0) > 0 or e.get("sent_to_definition", 0) > 0
    return False


def step_outcome(step, label):
    if step.get("not_run"):
        return "stopped"
    if step["op"] in ("capture", "reload"):
        return None
    exc = step.get("exception")
    orc = step.get("oracle")
    if exc and exc.get("type") != "Refused":
        return "error"
    if step["op"] == "replay":
        ran = step.get("next_op_ran", True)     # v1/v2 harness: no device stop existed, the next operation ran
        if not ran:
            return "blocked" if exc else "unknown"
        if orc is None:
            return "unknown"
        if not orc["ok"]:
            return "wrong_escaped"        # the graph's next operation read a value beyond the tolerance (or NaN)
        if not step.get("delivered"):
            return "refused_after_use"
    else:
        if not step.get("delivered"):
            return "blocked" if exc else "unknown"
        if orc is None:
            return "unknown"
        if not orc["ok"]:
            return "wrong_escaped"
    return "repaired_delivered" if repaired_by_entail(step, label) else "normal_delivered"


def run_status(out_root, case, label, r, freeze_sha):
    """"complete", "missing", or why one case's run in one mode is not complete: a non-zero exit, steps not all
    recorded (a step not run is recorded with the reason), steps or a case spec other than the scheduled ones, a run
    bound to another freeze manifest."""
    if r is None:
        return "missing"
    run = os.path.join(out_root, case["id"], label, "run.json")
    if os.path.exists(run):
        with open(run, encoding="utf-8") as f:
            ex = json.load(f).get("exit")
        if ex != 0:
            return f"exit {ex}"
    planned = case["steps"]
    if len(r.get("steps", [])) != len(planned):
        return f"{len(r.get('steps', []))} of {len(planned)} steps recorded"
    if any(s.get("op") != p.get("op") for s, p in zip(r["steps"], planned)):
        return "the steps recorded are not the scheduled case's"
    got, want = dict(r.get("case") or {}), dict(case)
    got.pop("split", None)
    want.pop("split", None)
    if got != want:
        return "the run's case is not the scheduled one"
    if freeze_sha and r.get("freeze_sha256") != freeze_sha:
        return "the run was bound to another freeze manifest"
    return "complete"


def judge_case(out_root, case, freeze_sha=None):
    res = {"id": case["id"], "group": case["group"], "modes": {}}
    runs = {lab: load(out_root, case["id"], lab) for lab in LABELS}
    offA = runs["offA"][0]
    base_eff = None
    if case["group"] in DEFECT_LIKE and offA is not None:
        eff = [s for s in offA["steps"] if defective(s, case) and s.get("oracle")]
        base_eff = any(not s["oracle"]["ok"] for s in eff)
        res["off_baseline"] = {"defective_steps": len(eff),
                               "beyond_tolerance": sum(1 for s in eff if not s["oracle"]["ok"]),
                               "worst_ratio": max([s["oracle"]["worst_ratio"] for s in eff] or [0])}
    res["mutation_effective"] = base_eff
    # off A against off B: the engine's own run-to-run behaviour
    if runs["offA"][1] is not None and runs["offB"][1] is not None:
        za, zb = runs["offA"][1], runs["offB"][1]
        res["offA_equals_offB"] = all(np.array_equal(za[k], zb[k]) for k in za.files if k.endswith("_out")
                                      and k in zb.files)
    for lab in LABELS:
        r, z = runs[lab]
        m = {"present": r is not None, "run": run_status(out_root, case, lab, r, freeze_sha)}
        if r is None:
            res["modes"][lab] = m
            continue
        outs = [step_outcome(s, lab) for s in r["steps"]]
        # steps not run because an earlier step's block stopped the device count as blocked (nothing was handed on)
        if "stopped" in outs:
            first = outs.index("stopped")
            outs = [("blocked" if "blocked" in outs[:first] else "unknown") if o == "stopped" else o for o in outs]
            m["stopped_after_block"] = sum(1 for s in r["steps"] if s.get("not_run"))
        m["steps"] = outs
        m["rng_same"] = all(s.get("rng_same", True) is not False for s in r["steps"])
        m["exceptions"] = [((s.get("exception") or {}).get("kind") or (s.get("exception") or {}).get("type"))
                           for s in r["steps"]]
        m["next_op_ran"] = [s.get("next_op_ran") for s in r["steps"] if s.get("op") == "replay"]
        if case["group"] in DEFECT_LIKE:
            ds = [o for s, o in zip(r["steps"], outs) if o is not None and defective(s, case)]
            if sum(1 for s in case["steps"] if defective(s, case)) > len(ds):
                ds.append("unknown")         # a defective step that never ran
            if "wrong_escaped" in ds:
                verdict = "wrong_escaped"
            elif "error" in ds:
                verdict = "error"
            elif "refused_after_use" in ds:
                verdict = "refused_after_use"
            elif "unknown" in ds or not ds:
                verdict = "unknown"
            elif "blocked" in ds:
                verdict = "blocked"
            elif "repaired_delivered" in ds:
                verdict = "repaired_delivered"
            else:
                verdict = "normal_delivered"     # handed on right without a repair (the defect did nothing here)
            nd = [o for s, o in zip(r["steps"], outs) if o is not None and not defective(s, case)]
            m["non_defective_steps"] = nd
            m["verdict"] = verdict
            if case.get("expect"):
                m["refusal_kinds"] = [k for k in m["exceptions"] if k]
                m["expected_kind"] = case["expect"]
        elif case["group"] == "normal":
            bad = [o for o in outs if o is not None and o != "normal_delivered"]
            planned = sum(1 for s in case["steps"] if s["op"] not in ("capture", "reload"))
            seen = sum(1 for o in outs if o is not None)
            same = None
            if z is not None and runs["offA"][1] is not None and lab != "offA":
                za = runs["offA"][1]
                same = all(np.array_equal(za[k], z[k]) for k in za.files if k.endswith("_out") and k in z.files)
            m["bitwise_as_offA"] = same
            if "wrong_escaped" in outs:
                m["verdict"] = "wrong_escaped"
            elif "error" in outs:
                m["verdict"] = "error"
            elif "blocked" in outs or "refused_after_use" in outs:
                m["verdict"] = "wrong_refusal"
            elif seen < planned:
                m["verdict"] = "unknown"        # a planned step never ran (a capture failed): not a pass
            elif bad:
                m["verdict"] = "changed_without_need" if "repaired_delivered" in bad else "unknown"
            else:
                m["verdict"] = "normal_delivered"
        else:   # admission
            kinds = [((s.get("exception") or {}).get("kind")) for s in r["steps"]]
            m["verdict"] = ("blocked" if all(o == "blocked" for o in outs if o is not None) else
                            "wrong_escaped" if "wrong_escaped" in outs else
                            "delivered" if all(o in ("normal_delivered", "repaired_delivered") for o in outs
                                               if o is not None) else "other")
            m["refusal_kinds"] = kinds
            m["expected_kind"] = case.get("expect")
        res["modes"][lab] = m
    return res


def summarize(out_root):
    with open(os.path.join(out_root, "cases.json"), encoding="utf-8") as f:
        meta = json.load(f)
    scheduled = meta["cases"]                  # every case the split schedules (since L5.4d, also the ones not run)
    freeze_sha = meta.get("freeze_sha256") or (meta["key"] if meta["split"] == "holdout" else None)
    cases = [c for c in scheduled if not c["id"].startswith("dev-")]
    extra = [c for c in scheduled if c["id"].startswith("dev-")]
    judged = [judge_case(out_root, c, freeze_sha) for c in cases]
    judged_extra = [judge_case(out_root, c, freeze_sha) for c in extra]
    out = {"split": meta["split"], "key": meta["key"], "per_case": judged, "dev_extra": judged_extra, "modes": {}}
    labels = meta.get("labels") or [lab for lab in LABELS
                                    if any(j["modes"].get(lab, {}).get("present") for j in judged + judged_extra)]
    try:                                       # the scheduled list against what cases.py schedules now
        import cases as C

        now = C.cases(meta["split"], meta["key"])
        out["cases_py_now_matches"] = [c["id"] for c in now] == [c["id"] for c in scheduled] and now == scheduled
    except Exception as e:  # noqa: BLE001
        out["cases_py_now_matches"] = f"not compared: {type(e).__name__}: {e}"
    if "scheduled" not in meta:
        # written before L5.4d, the runner listed only the cases it ran: whether that was the whole split is read
        # from cases.py, with and without the integrity cases (v1/v2 schedules had none)
        try:
            import cases as C

            ids = {c["id"] for c in scheduled}
            fulls = [{c["id"] for c in C.cases(meta["split"], meta["key"], integrity=i)} for i in (False, True)]
            meta["partial"] = ids not in fulls
            out["partial_read_from_cases_py"] = True
        except Exception as e:  # noqa: BLE001
            out["partial_read_from_cases_py"] = f"not read: {type(e).__name__}: {e}"
    out["partial"] = bool(meta.get("partial", False))
    out["scheduled"] = [c["id"] for c in scheduled]
    out["selected"] = meta.get("selected", out["scheduled"])
    out["labels"] = labels
    unfinished_all = {}
    for lab in labels:
        unfinished = [(j["id"], j["modes"].get(lab, {}).get("run")) for j in judged + judged_extra
                      if j["modes"].get(lab, {}).get("run") != "complete"]
        if unfinished:
            unfinished_all[lab] = unfinished
    out["complete"] = not unfinished_all and not meta.get("partial", False)
    out["unfinished_runs"] = unfinished_all
    for lab in [x for x in ("load", "guarantee", "structure", "offA") if x in labels]:
        cnt = {k: 0 for k in COUNTS}
        cnt["scheduled"] = len(judged)
        normal_fail, adm_fail, incomplete = [], [], []
        for j in judged:
            m = j["modes"].get(lab, {})
            if not m.get("present"):
                incomplete.append(j["id"])
                continue
            cnt["reached"] += 1
            v = m.get("verdict")
            if j["group"] in DEFECT_LIKE:
                if j.get("mutation_effective"):
                    cnt["mutation_effective"] += 1
                else:
                    incomplete.append(j["id"])
                if v in cnt:
                    cnt[v] += 1
            elif j["group"] == "normal":
                if v == "normal_delivered":
                    cnt["normal_delivered"] += 1
                else:
                    normal_fail.append((j["id"], v))
                    if v in ("wrong_escaped", "error", "unknown"):
                        cnt[v] += 1
            else:
                ok = (v == "blocked" and all(k == j["modes"][lab].get("expected_kind")
                                             for k in m.get("refusal_kinds", []) if k))
                if not ok:
                    adm_fail.append((j["id"], v, m.get("refusal_kinds")))
                if v == "blocked":
                    cnt["blocked"] += 1
            if not m.get("rng_same", True):
                normal_fail.append((j["id"], "rng_changed"))
        by_group = {}
        for grp in DEFECT_LIKE:
            gj = [j for j in judged if j["group"] == grp]
            if not gj:
                continue
            eff = [j for j in gj if j.get("mutation_effective")]
            verdicts = {}
            for j in eff:
                v = j["modes"].get(lab, {}).get("verdict")
                verdicts[v] = verdicts.get(v, 0) + 1
            by_group[grp] = {"cases": len(gj), "effective": len(eff), "verdicts": verdicts,
                             "wrong_escaped": [j["id"] for j in eff if j["modes"].get(lab, {}).get("verdict") in
                                               ("wrong_escaped", "refused_after_use")]}
        effective = [j for j in judged if j["group"] in DEFECT_LIKE and j.get("mutation_effective")]
        escaped = [j["id"] for j in effective if j["modes"].get(lab, {}).get("verdict") in
                   ("wrong_escaped", "refused_after_use")]
        unresolved = [j["id"] for j in effective if j["modes"].get(lab, {}).get("verdict") in
                      ("unknown", "error", "normal_delivered", None)]
        lab_unfinished = unfinished_all.get(lab, [])
        # failed: something went wrong in what ran. partial: not every scheduled case ran to its end in this mode
        # (or the run was started on a part of the split). incomplete: all ran, but a defect had no effect in off or
        # a verdict is unresolved. passed: none of these.
        if escaped or normal_fail or (lab in ("guarantee", "structure") and adm_fail):
            state = "failed"
        elif lab_unfinished or meta.get("partial", False):
            state = "partial"
        elif incomplete or unresolved or not effective:
            state = "incomplete"
        else:
            state = "passed"
        defects = [j for j in effective if j["group"] == "defect"]
        out["modes"][lab] = {"counts": cnt, "state": state,
                             "complete": not lab_unfinished and not meta.get("partial", False),
                             "unfinished_runs": lab_unfinished, "wrong_escaped": escaped, "unresolved": unresolved,
                             "normal_failures": normal_fail, "admission_failures": adm_fail,
                             "incomplete": incomplete, "by_group": by_group,
                             "defects_effective": len(defects),
                             "repaired_of_effective": sum(1 for j in defects if j["modes"].get(lab, {})
                                                          .get("verdict") == "repaired_delivered"),
                             "blocked_of_effective": sum(1 for j in defects if j["modes"].get(lab, {})
                                                         .get("verdict") == "blocked")}
    return out


if __name__ == "__main__":
    root = sys.argv[1]
    s = summarize(root)
    path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    print("complete:", s["complete"], "| unfinished runs:", {k: len(v) for k, v in s["unfinished_runs"].items()},
          "| cases.py now matches the scheduled list:", s["cases_py_now_matches"])
    for lab, m in s["modes"].items():
        print(lab, m["state"], m["counts"], "escaped:", m["wrong_escaped"], "normal fail:", m["normal_failures"],
              "admission fail:", m["admission_failures"], "incomplete:", m["incomplete"],
              "groups:", {k: v["verdicts"] for k, v in m["by_group"].items()})
