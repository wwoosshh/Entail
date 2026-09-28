"""After a release, say on each issue it closes that the fix reached PyPI (RELEASING.md). The issues are the ones the
pull requests merged since the previous release close ("Closes #N" in a pull request's description); an issue still
open, or one that already carries this release's comment, is left as it is. Run by the `publish` workflow with the
job's token, after the tag and the GitHub release exist; --dry-run only lists what it would do.
  python .github/scripts/release_issues.py 2.1.1 [--dry-run]
"""
import json
import os
import re
import subprocess
import sys

REPO = os.environ.get("GITHUB_REPOSITORY", "wwoosshh/entail")
MARK = "<!-- entail-release v{} -->"
QUERY = """query($owner: String!, $name: String!, $pr: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $pr) { closingIssuesReferences(first: 50) { nodes { number state } } }
  }
}"""


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, encoding="utf-8").stdout


def gh_json(*args):
    return json.loads(run("gh", *args) or "null")


def version_key(tag):
    return tuple(int(x) for x in tag[1:].split("."))


def previous_tag(version):
    tags = [t for t in run("git", "tag", "--list", "v*").split() if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    older = [t for t in tags if version_key(t) < version_key(f"v{version}")]
    return max(older, key=version_key) if older else None


def pull_requests(since, until):
    """The merged pull requests the commits since..until came from (a rebase merge keeps the link)."""
    prs = set()
    for sha in run("git", "rev-list", f"{since}..{until}").split():
        for pr in gh_json("api", f"repos/{REPO}/commits/{sha}/pulls") or []:
            if pr.get("merged_at"):
                prs.add(pr["number"])
    return sorted(prs)


def closed_issues(pr):
    owner, name = REPO.split("/")
    data = gh_json("api", "graphql", "-f", f"query={QUERY}", "-f", f"owner={owner}", "-f", f"name={name}",
                   "-F", f"pr={pr}")
    nodes = data["data"]["repository"]["pullRequest"]["closingIssuesReferences"]["nodes"]
    return [n["number"] for n in nodes if n["state"] == "CLOSED"]


def main():
    version, dry = sys.argv[1], "--dry-run" in sys.argv
    tag = f"v{version}"
    since = previous_tag(version)
    if since is None:
        print(f"no release before {tag}: nothing to say")
        return 0
    until = tag if run("git", "tag", "--list", tag).strip() else "HEAD"
    url = f"https://github.com/{REPO}/releases/tag/{tag}"
    report = [f"Issues closed by the pull requests of {tag} (since {since}):"]
    for pr in pull_requests(since, until):
        for issue in closed_issues(pr):
            mark = MARK.format(version)
            pages = run("gh", "api", f"repos/{REPO}/issues/{issue}/comments", "--paginate", "--jq",
                        f'[.[] | select((.body // "") | contains("{mark}"))] | length').split()
            if sum(int(n) for n in pages):
                report.append(f"- #{issue} (#{pr}): already said")
                continue
            body = (f"Released in **entail {version}** (fixed by #{pr}): `pip install -U entail-ai`. "
                    f"Release notes: {url}\n\n{MARK.format(version)}")
            if not dry:
                run("gh", "api", f"repos/{REPO}/issues/{issue}/comments", "-f", f"body={body}")
            report.append(f"- #{issue} (#{pr}): {'would comment' if dry else 'commented'}")
    if len(report) == 1:
        report.append("- none")
    text = "\n".join(report)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
