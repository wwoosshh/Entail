"""Judge and count a suite run: python aggregate.py <out_root> [summary.json]

Every decision is made from what the next operation read (the observed output, judged by the independent oracle),
from whether the call handed a value on, and - for graphs - from what the graph's own next operation read, not
from entail's records (those only label the path: normal or repaired). A case or a mode with missing results stays
in the denominator as incomplete.

Per step (calls, direct calls, replays):
  normal_delivered     handed on, within the oracle's tolerance, no repair or reference path used
  repaired_delivered   handed on, within the tolerance, after a repair or with the reference's values
  blocked              not handed on, refused before it left (a replay: also nothing wrong read inside the graph -
                       the graph's next operation read only NaN)
  wrong_escaped        a value beyond the tolerance was handed on, or read inside the graph, even once
  unknown              nothing observed (no producer output, a step not reached)
  error                an exception that is not a refusal (crash, OOM)
Per defect case: the off baseline (off A) must be beyond the tolerance on a defective step (mutation_effective);
otherwise the case is incomplete, and it is never dropped.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LABELS = ("offA", "offB", "load", "guarantee")
COUNTS = ("scheduled", "reached", "mutation_effective", "normal_delivered", "repaired_delivered", "blocked",
          "wrong_escaped", "unknown", "error")


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
    if label == "load":
        return e.get("definition_calls", 0) > 0 or e.get("sent_to_definition", 0) > 0
    return False


def step_outcome(step, label):
    if step["op"] in ("capture", "reload"):
        return None
    exc = step.get("exception")
    orc = step.get("oracle")
    if exc and exc.get("type") != "Refused":
        return "error"
    if step["op"] == "replay":
        if orc is None:
            return "unknown"
        if not orc["ok"] and not step.get("poisoned"):
            return "wrong_escaped"        # the graph's next operation read a wrong value, refused or not
        if not step.get("delivered"):
            return "blocked"
    else:
        if not step.get("delivered"):
            return "blocked" if exc else "unknown"
        if orc is None:
            return "unknown"
        if not orc["ok"]:
            return "wrong_escaped"
    return "repaired_delivered" if repaired_by_entail(step, label) else "normal_delivered"


def judge_case(out_root, case):
    res = {"id": case["id"], "group": case["group"], "modes": {}}
    runs = {lab: load(out_root, case["id"], lab) for lab in LABELS}
    offA = runs["offA"][0]
    base_eff = None
    if case["group"] == "defect" and offA is not None:
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
        m = {"present": r is not None}
        if r is None:
            res["modes"][lab] = m
            continue
        outs = [step_outcome(s, lab) for s in r["steps"]]
        m["steps"] = outs
        m["rng_same"] = all(s.get("rng_same", True) for s in r["steps"])
        m["exceptions"] = [((s.get("exception") or {}).get("kind") or (s.get("exception") or {}).get("type"))
                           for s in r["steps"]]
        if case["group"] == "defect":
            ds = [o for s, o in zip(r["steps"], outs) if o is not None and defective(s, case)]
            if "wrong_escaped" in ds:
                verdict = "wrong_escaped"
            elif "error" in ds:
                verdict = "error"
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
        elif case["group"] == "normal":
            bad = [o for o in outs if o is not None and o != "normal_delivered"]
            same = None
            if z is not None and runs["offA"][1] is not None and lab != "offA":
                za = runs["offA"][1]
                same = all(np.array_equal(za[k], z[k]) for k in za.files if k.endswith("_out") and k in z.files)
            m["bitwise_as_offA"] = same
            if "wrong_escaped" in outs:
                m["verdict"] = "wrong_escaped"
            elif "error" in outs:
                m["verdict"] = "error"
            elif "blocked" in outs:
                m["verdict"] = "wrong_refusal"
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
    cases = [c for c in meta["cases"] if not c["id"].startswith("dev-")]
    extra = [c for c in meta["cases"] if c["id"].startswith("dev-")]
    judged = [judge_case(out_root, c) for c in cases]
    judged_extra = [judge_case(out_root, c) for c in extra]
    out = {"split": meta["split"], "key": meta["key"], "per_case": judged, "dev_extra": judged_extra, "modes": {}}
    for lab in ("load", "guarantee", "offA"):
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
            if j["group"] == "defect":
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
        defects = [j for j in judged if j["group"] == "defect"]
        effective = [j for j in defects if j.get("mutation_effective")]
        escaped = [j["id"] for j in effective if j["modes"].get(lab, {}).get("verdict") == "wrong_escaped"]
        unresolved = [j["id"] for j in effective if j["modes"].get(lab, {}).get("verdict") in
                      ("unknown", "error", "normal_delivered", None)]
        if escaped or normal_fail or (lab == "guarantee" and adm_fail):
            state = "failed"
        elif incomplete or unresolved or not effective:
            state = "incomplete"
        else:
            state = "passed"
        out["modes"][lab] = {"counts": cnt, "state": state, "wrong_escaped": escaped, "unresolved": unresolved,
                             "normal_failures": normal_fail, "admission_failures": adm_fail,
                             "incomplete": incomplete,
                             "defects_effective": len(effective),
                             "repaired_of_effective": sum(1 for j in effective if j["modes"].get(lab, {})
                                                          .get("verdict") == "repaired_delivered"),
                             "blocked_of_effective": sum(1 for j in effective if j["modes"].get(lab, {})
                                                         .get("verdict") == "blocked")}
    return out


if __name__ == "__main__":
    root = sys.argv[1]
    s = summarize(root)
    path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    for lab, m in s["modes"].items():
        print(lab, m["state"], m["counts"], "escaped:", m["wrong_escaped"], "normal fail:", m["normal_failures"],
              "admission fail:", m["admission_failures"], "incomplete:", m["incomplete"])
