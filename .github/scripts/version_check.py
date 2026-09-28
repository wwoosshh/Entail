"""The pull-request check of the release rule (RELEASING.md). A pull request into main is one of two kinds:

  an ordinary pull request   leaves the version as it is; it is merged and waits for a release. When it changes the
                             package, CHANGELOG.md says so under "## Unreleased".
  a release pull request     raises the version by exactly one step (2.3.5 -> 2.3.6, 2.4.0 or 3.0.0) and turns
                             "## Unreleased" into the new version's section. Merging it starts the release, which waits
                             for the researcher's approval of the `pypi` deployment.

It says which kind this is and what the merge will do (and the size since the last release, which the version rule
reads), and fails when the rule is broken.
  python .github/scripts/version_check.py <base ref> [<head ref>]
"""
import os
import re
import subprocess
import sys

VERSION = re.compile(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"\s*$', re.M)
PATCH_LINES = 500     # "a few hundred lines or fewer" (RELEASING.md): past this a PATCH release gets a warning
PACKAGE = ("entail/", "pyproject.toml", "setup.py", "README.md")


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, encoding="utf-8").stdout


def version_at(ref):
    m = VERSION.search(git("show", f"{ref}:entail/__init__.py"))
    if not m:
        sys.exit(f"no __version__ in entail/__init__.py at {ref}")
    return tuple(int(x) for x in m.groups())


def text(v):
    return ".".join(str(x) for x in v)


def last_tag():
    tags = [t for t in git("tag", "--list", "v*").split() if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    return max(tags, key=lambda t: tuple(int(x) for x in t[1:].split("."))) if tags else None


def size(since, head):
    stat = git("diff", "--shortstat", since, head)
    ins = re.search(r"(\d+) insertion", stat)
    dels = re.search(r"(\d+) deletion", stat)
    return stat.strip(), (int(ins.group(1)) if ins else 0) + (int(dels.group(1)) if dels else 0)


def main():
    base_ref = sys.argv[1]
    head_ref = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    base, head = version_at(base_ref), version_at(head_ref)
    changelog = git("show", f"{head_ref}:CHANGELOG.md")
    lines, warnings, errors = [], [], []
    if head == base:
        lines.append(f"**Ordinary pull request.** The version stays {text(base)}: merging does not release.")
        changed = git("diff", "--name-only", f"{base_ref}...{head_ref}").split()
        package = [p for p in changed if p.startswith(PACKAGE[0]) or p in PACKAGE[1:]]
        if package and not re.search(r"^## Unreleased\s*$", changelog, re.M):
            warnings.append("the package changes (" + ", ".join(package[:5]) + (", ..." if len(package) > 5 else "")
                            + ") but CHANGELOG.md has no '## Unreleased' section to say so")
    else:
        steps = {(base[0] + 1, 0, 0): "MAJOR", (base[0], base[1] + 1, 0): "MINOR", (base[0], base[1], base[2] + 1): "PATCH"}
        kind = steps.get(head)
        if kind is None:
            errors.append(f"{text(base)} -> {text(head)} is not one step; the next version is one of "
                          + ", ".join(text(v) for v in steps))
        if git("tag", "--list", f"v{text(head)}").strip():
            errors.append(f"v{text(head)} is already tagged")
        if not re.search(rf"^## {re.escape(text(head))}\s*$", changelog, re.M):
            errors.append(f"CHANGELOG.md has no '## {text(head)}' section")
        if re.search(r"^## Unreleased\s*$", changelog, re.M):
            errors.append(f"CHANGELOG.md still has '## Unreleased': its lines belong in '## {text(head)}'")
        lines.append(f"**Release pull request: {text(base)} -> {text(head)}" + (f" ({kind})" if kind else "") + ".** "
                     "Merging it starts the release, which waits for the researcher's approval of the `pypi` "
                     "deployment; then PyPI, the tag and the GitHub release.")
        tag = last_tag()
        if tag:
            stat, n = size(tag, head_ref)
            lines.append(f"Size since {tag}: {stat or 'no change'}.")
            if kind == "PATCH" and n > PATCH_LINES:
                warnings.append(f"a PATCH release of {n} changed lines: by the version rule a larger change raises "
                                f"the MINOR number ({text((base[0], base[1] + 1, 0))})")
    for w in warnings:
        print(f"::warning::{w}")
        lines.append(f"Warning: {w}.")
    for e in errors:
        print(f"::error::{e}")
        lines.append(f"Error: {e}.")
    report = "\n\n".join(lines)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
