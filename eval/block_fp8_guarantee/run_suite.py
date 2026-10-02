"""Run the scenarios, each mode in a fresh process: python run_suite.py <split> <out_root> [--freeze F] [--only ID,...]

split: dev (fixed seeds) or holdout (seeds from the freeze manifest's hash; --freeze is then required).
Modes per case: off A, off B (two fresh off runs: is off stable), load (entail as it ships, ENTAIL=load), guarantee
(ENTAIL=guarantee, the numeric guarantee) and, from v3 (L5.4c), structure (ENTAIL=structure, the structural check
experiment - reported apart, never as the guarantee). entail installs itself through its start-up hook
(adapters/autoinstall on PYTHONPATH), the way it does for a user. Every run's stdout, stderr and exit code are kept.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cases as C  # noqa: E402
import common  # noqa: E402

MODES = (("offA", "off"), ("offB", "off"), ("load", "load"), ("guarantee", "guarantee"), ("structure", "structure"))


def run_one(case, label, mode, out_root, freeze=None, plan_path=None):
    out_dir = os.path.join(out_root, case["id"], label)
    os.makedirs(out_dir, exist_ok=True)
    cpath = os.path.join(out_dir, "case.json")
    common.write_json(cpath, case)
    env = dict(os.environ)
    env["ENTAIL"] = mode
    auto = os.path.join(common.ENTAIL_ROOT, "entail", "adapters", "autoinstall")
    env["PYTHONPATH"] = os.pathsep.join([common.ENTAIL_ROOT, auto, HERE] + ([env["PYTHONPATH"]] if
                                                                             env.get("PYTHONPATH") else []))
    env["ENTAIL_LOG_DIR"] = os.path.join(out_dir, "entail_logs")
    env["ENTAIL_GUARANTEE_RECORD"] = os.path.join(out_dir, f"{mode}.jsonl")
    env["ENTAIL_QUIET"] = "start"
    if plan_path:
        env["ENTAIL_GUARANTEE_PLAN"] = plan_path
    if freeze:
        env["BFG_FREEZE"] = freeze
        env["BFG_FREEZE_SHA256"] = common.sha256_file(freeze)
    cmd = [sys.executable, os.path.join(HERE, "run_case.py"), cpath, mode, out_dir]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    meta = {"cmd": cmd, "exit": p.returncode, "seconds": round(time.time() - t0, 2),
            "env": {k: env[k] for k in ("ENTAIL", "PYTHONPATH", "ENTAIL_LOG_DIR", "ENTAIL_GUARANTEE_RECORD",
                                        "ENTAIL_GUARANTEE_PLAN", "BFG_FREEZE", "BFG_FREEZE_SHA256",
                                        "VLLM_DISABLED_KERNELS", "CUDA_VISIBLE_DEVICES") if k in env}}
    with open(os.path.join(out_dir, "stdout.txt"), "w", encoding="utf-8") as f:
        f.write(p.stdout)
    with open(os.path.join(out_dir, "stderr.txt"), "w", encoding="utf-8") as f:
        f.write(p.stderr)
    common.write_json(os.path.join(out_dir, "run.json"), meta)
    last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    print(f"{case['id']:28s} {label:9s} exit {p.returncode} {meta['seconds']:7.1f}s {last[:150]}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("dev", "holdout"))
    ap.add_argument("out_root")
    ap.add_argument("--freeze")
    ap.add_argument("--only")
    ap.add_argument("--modes", default=",".join(m[0] for m in MODES))
    a = ap.parse_args()
    key, plan_path = "", None
    if a.freeze:
        with open(a.freeze, encoding="utf-8") as f:
            fz = json.load(f)
        key = common.sha256_file(a.freeze)
        plan_path = a.freeze
        if a.split == "holdout":
            assert fz.get("frozen"), "the holdout runs only on a frozen manifest"
    elif a.split == "holdout":
        sys.exit("--freeze is required for the holdout")
    cs = C.cases(a.split, key)
    if a.only:
        want = set(a.only.split(","))
        cs = [c for c in cs if c["id"] in want]
    os.makedirs(a.out_root, exist_ok=True)
    common.write_json(os.path.join(a.out_root, "cases.json"), {"split": a.split, "key": key, "cases": cs})
    labels = a.modes.split(",")
    for case in cs:
        for label, mode in MODES:
            if label in labels:
                run_one(case, label, mode, a.out_root, a.freeze, plan_path)


if __name__ == "__main__":
    main()
