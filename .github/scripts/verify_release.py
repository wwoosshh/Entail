"""After a release, install it from PyPI and try it (RELEASING.md).

PyPI's index - the list pip reads - can lag behind an upload: during PyPI's maintenance on 2026-09-28 it listed 2.1.1
about ten minutes after the upload and 2.1.2 not within twenty, and the old check, which read only the index, failed a
release that was complete. So:
  1. the upload: the version's own JSON (/pypi/<name>/<version>/json) names its files; none means the release failed;
  2. the index: once it lists the version (waited for up to --wait seconds), pip installs <name>==<version> from it;
  3. otherwise the wheel the version's JSON names, downloaded and held to its sha256 - the file pip will get once the
     index catches up - with a warning that the index was late;
then the same checks either way: the version, the start-up hook, and the page of `entail serve`. Run by the `publish`
workflow after the tag and the GitHub release exist.
  python .github/scripts/verify_release.py 2.1.2 [--wait 1200] [--venv /tmp/pypi]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

PYPI = "https://pypi.org"
SIMPLE_JSON = "application/vnd.pypi.simple.v1+json"


def say(text):
    print(text, flush=True)


def get(url, accept=None, tries=6, pause=20):
    """The body at `url`; None when PyPI answers 404. PyPI's web backend can answer 5xx for a while (it said "503
    Backend is unhealthy" during the maintenance), so errors other than 404 are retried."""
    for i in range(tries):
        req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            err = str(e)
        if i + 1 < tries:
            say(f"  {url}: {err}; again in {pause} s")
            time.sleep(pause)
    raise RuntimeError(f"{url}: {err} after {tries} tries")


def uploaded(name, version, wait, pause):
    """The files PyPI has for the version (its own JSON, which a new upload shows at once), waited for up to `wait`."""
    deadline = time.time() + wait
    while True:
        body = get(f"{PYPI}/pypi/{name}/{version}/json")
        files = json.loads(body)["urls"] if body else []
        if files or time.time() >= deadline:
            return files
        say(f"  PyPI has no files for {name} {version} yet; again in {pause} s")
        time.sleep(pause)


def listed(name, version):
    """Whether PyPI's index - what pip reads - lists the version (an index that cannot be read does not)."""
    try:
        body = get(f"{PYPI}/simple/{name}/", accept=SIMPLE_JSON)
    except RuntimeError as e:
        say(f"  {e}")
        return False
    return bool(body) and version in json.loads(body).get("versions", [])


def run(*args):
    """A command of the new environment, run away from the checkout: from the repository's root, `import entail` would
    find the checked-out package, not the installed one (a check that passed that way checked nothing)."""
    return subprocess.run(args, capture_output=True, text=True, cwd=tempfile.gettempdir())


def fresh_venv(path):
    shutil.rmtree(path, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "venv", path], check=True)
    return os.path.join(path, "bin", "python"), os.path.join(path, "bin", "entail")


def from_index(py, name, version, tries=3, pause=30):
    for i in range(tries):
        r = run(py, "-m", "pip", "install", "-q", "--no-cache-dir", f"{name}=={version}")
        if r.returncode == 0:
            return None
        if i + 1 < tries:
            time.sleep(pause)
    return (r.stderr or r.stdout).strip().splitlines()[-1:]


def from_file(py, wheel):
    """The uploaded wheel, by its URL, held to the sha256 PyPI gives for it."""
    body = get(wheel["url"])
    if body is None:
        return [f"{wheel['url']}: not found"]
    path = os.path.join(tempfile.mkdtemp(), wheel["filename"])
    with open(path, "wb") as f:
        f.write(body)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != wheel["digests"]["sha256"]:
        return [f"{wheel['filename']}: sha256 {digest}, PyPI says {wheel['digests']['sha256']}"]
    r = run(py, "-m", "pip", "install", "-q", "--no-cache-dir", path)
    return None if r.returncode == 0 else (r.stderr or r.stdout).strip().splitlines()[-1:]


def page_answers(entail, port):
    """`entail serve` starts, and its page and /api/runs answer."""
    logs = tempfile.mkdtemp()
    proc = subprocess.Popen([entail, "serve", "--dir", logs, "--port", str(port)], cwd=logs,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for path in ("/", "/api/runs"):
            for _ in range(20):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
                        if r.status == 200:
                            break
                except (urllib.error.URLError, OSError):
                    pass
                time.sleep(0.5)
            else:
                return f"{path} did not answer"
        return None
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def check(name, version, wait, poll, venv, port, upload_wait=300):
    """The release as PyPI serves it: (what was found, warnings, errors)."""
    report, warnings = [], []
    try:
        files = uploaded(name, version, upload_wait, 30)
    except RuntimeError as e:
        return report, warnings, [f"could not read what PyPI has for {version}: {e}"]
    if not files:
        return report, warnings, [f"PyPI has no files for {name} {version}: the upload did not happen"]
    report.append(f"PyPI has {name} {version}: " + ", ".join(
        f"{f['filename']} ({f['upload_time_iso_8601']})" for f in files) + ".")
    start = time.time()
    while True:
        in_index = listed(name, version)
        if in_index or time.time() - start >= wait:
            break
        say(f"  PyPI's index does not list {version} yet ({int(time.time() - start)} s); again in {poll} s")
        time.sleep(poll)
    py, entail = fresh_venv(venv)
    if in_index:
        report.append(f"PyPI's index lists it ({int(time.time() - start)} s into this check); installed with "
                      f"`pip install {name}=={version}`.")
        failed = from_index(py, name, version)
    else:
        warnings.append(f"PyPI's index did not list {version} within {wait} s of this check - PyPI's side: "
                        f"`pip install -U {name}` gets it once the index catches up. Checked the uploaded wheel "
                        f"instead, downloaded by its URL and held to its sha256")
        wheel = next((f for f in files if f["packagetype"] == "bdist_wheel"), None)
        failed = from_file(py, wheel) if wheel else ["PyPI names no wheel for this version"]
    if failed:
        return report, warnings, ["install: " + " ".join(failed)]
    errors = []
    got = run(py, "-I", "-c", "import entail; print(entail.__version__)").stdout.strip()
    if got != version:
        errors.append(f"the installed package says {got!r}, not {version}")
    hook = run(entail, "hook", "status")
    if hook.returncode != 0:
        errors.append(f"entail hook status failed: {(hook.stderr or hook.stdout).strip()}")
    page = page_answers(entail, port)
    if page:
        errors.append(f"entail serve: {page}")
    if not errors:
        report.append(f"{name} {version} installs, says its version, places its start-up hook, and its page answers.")
    return report, warnings, errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("version")
    ap.add_argument("--name", default="entail-ai")
    ap.add_argument("--wait", type=int, default=1200, help="seconds to wait for PyPI's index (default 1200)")
    ap.add_argument("--poll", type=int, default=60, help="seconds between looks at the index (default 60)")
    ap.add_argument("--venv", default=os.path.join(tempfile.gettempdir(), "pypi"))
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--upload-wait", type=int, default=300, help="seconds to wait for the upload to show")
    args = ap.parse_args(argv)
    report, warnings, errors = check(args.name, args.version, args.wait, args.poll, args.venv, args.port,
                                     args.upload_wait)
    for w in warnings:
        print(f"::warning::{w}")
        report.append(f"Warning: {w}.")
    for e in errors:
        print(f"::error::{e}")
        report.append(f"Error: {e}.")
    text = "\n\n".join(report)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
