"""Before an evaluation bound to a freeze manifest: do this checkout and this environment match it?
python freeze_check.py <manifest.json> [out.json]        (exit 1 on a mismatch)

Compared (M19 L5.4d): the sha256 of every file the manifest lists under "code" (the profile and the harness), the
entail commit and whether its tree has uncommitted changes, and the environment record (env_record.py, made now: GPU,
driver, Python, torch, CUDA, Triton, vLLM and the RECORD hash of each distribution, precision settings, the block FP8
kernel vLLM chooses) key by key. The holdout does not start on a mismatch (run_suite.py); cost.py and engine_smoke.py
refuse too, unless BFG_POST_FREEZE=1 marks the run as a measurement after the freeze (then the mismatches are written
next to its results).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402


def mismatches(manifest, env_now):
    """What differs between a manifest (dict) and this checkout and environment (env_now: env_record.record())."""
    out = []
    for rel, frozen in sorted(manifest.get("code", {}).items()):
        path = os.path.join(common.ENTAIL_ROOT, rel)
        now = common.sha256_file(path) if os.path.exists(path) else None
        if now != frozen:
            out.append({"what": f"code {rel}", "frozen": frozen, "now": now})
    git = common.git_head(common.ENTAIL_ROOT)
    if git != manifest.get("entail_git"):
        out.append({"what": "entail commit", "frozen": manifest.get("entail_git"), "now": git})
    env = manifest.get("env", {})
    for key in sorted(set(env) | set(env_now)):
        if env.get(key) != env_now.get(key):
            out.append({"what": f"environment {key}", "frozen": env.get(key), "now": env_now.get(key)})
    return out


def check(manifest_path, out_path=None, env_now=None):
    """The comparison, written to out_path when given: {"matches": bool, "mismatches": [...], ...}."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if env_now is None:
        import env_record

        env_now = env_record.record()
    env_now = json.loads(json.dumps(env_now, default=str))     # as the manifest holds it (JSON)
    mm = mismatches(manifest, env_now)
    res = {"manifest": manifest_path, "manifest_sha256": common.sha256_file(manifest_path),
           "checked": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code_files": len(manifest.get("code", {})),
           "matches": not mm, "mismatches": mm}
    if out_path:
        common.write_json(out_path, res)
    return res


def require(manifest_path, out_dir, what):
    """For cost.py and engine_smoke.py: refuse to measure on a mismatch, unless BFG_POST_FREEZE=1 (then go on and
    say so). Returns the check written to out_dir."""
    res = check(manifest_path, os.path.join(out_dir, "freeze_check.json"))
    res["post_freeze"] = os.environ.get("BFG_POST_FREEZE") == "1"
    common.write_json(os.path.join(out_dir, "freeze_check.json"), res)
    if not res["matches"] and not res["post_freeze"]:
        sys.exit(f"{what} does not start: this checkout or environment differs from the freeze "
                 f"({len(res['mismatches'])} mismatches, {out_dir}/freeze_check.json); BFG_POST_FREEZE=1 measures "
                 f"anyway, as a measurement after the freeze")
    return res


if __name__ == "__main__":
    r = check(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    for m in r["mismatches"]:
        print("MISMATCH", m["what"])
    print("matches" if r["matches"] else f"{len(r['mismatches'])} mismatches")
    sys.exit(0 if r["matches"] else 1)
