"""The notes of a GitHub release: the version's section of CHANGELOG.md, its relative links pointed at that version's
tag (a relative link does not resolve on a release page). Fails when the section is missing.
  python .github/scripts/release_notes.py 2.1.1 > notes.md
"""
import os
import re
import sys


def notes(changelog, version, repo_url):
    m = re.search(rf"^## {re.escape(version)}\s*$(.*?)(?=^## |\Z)", changelog, re.M | re.S)
    if not m:
        raise SystemExit(f"CHANGELOG.md has no '## {version}' section")
    body = m.group(1).strip()
    return re.sub(r"\]\((?!https?://|mailto:|#)([^)\s]+)\)", rf"]({repo_url}/blob/v{version}/\1)", body) + "\n"


def main():
    version = sys.argv[1]
    repo = os.environ.get("GITHUB_REPOSITORY", "wwoosshh/entail")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    with open("CHANGELOG.md", encoding="utf-8") as f:
        sys.stdout.write(notes(f.read(), version, f"{server}/{repo}"))


if __name__ == "__main__":
    main()
