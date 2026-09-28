# Releasing

`pip install "git+https://github.com/wwoosshh/entail"` works from any commit of `main`. Publishing to PyPI as
`entail-ai` takes a one-time setup, done by the maintainer:

1. On pypi.org, *Your projects → Publishing → Add a new pending publisher*:
   PyPI project `entail-ai`, owner `wwoosshh`, repository `entail`, workflow `publish.yml`, environment `pypi`.
2. On GitHub, *Settings → Environments*: an environment named `pypi`, with the maintainer as its required reviewer
   and deployments from protected branches only. No token is stored anywhere (trusted publishing).

## Version numbers

`MAJOR.MINOR.PATCH`, chosen by what changed since the last release:

- **MAJOR** when the design's structure changes (2.0: the local platform around the checks).
- **MINOR** when the structure stays and the release has a purpose: new features, or a large scope of work.
- **PATCH** for bug fixes and small corrections of a few hundred lines or fewer. A larger change is a MINOR release.

The size is read against the last release's tag (`git diff --shortstat v2.1.0`, say). The packages outside the
core (`dlc/comfyui`, `workshop/basics`) keep their own numbers by the same rule.

## How a change reaches main

`main` is protected: every change comes in through a pull request, the checks must pass (`tests`, and `version` from
the `pull request` workflow), the branch must be up to date with `main`, and pull requests are merged by rebasing,
so `main` stays a straight line of the commits as written.

1. Bring the local `main` up to date (`git switch main && git pull --ff-only`) and branch off it.
2. Work on the branch, push it and open a pull request into `main`.
3. An **ordinary pull request** leaves `__version__` as it is. When it changes the package, it adds a line under
   `## Unreleased` at the top of CHANGELOG.md. Merged, it waits in `main` for the next release; nothing is published.

## How a release reaches PyPI

4. A **release pull request** raises `__version__` in `entail/__init__.py` by exactly one step from `main`'s
   (2.3.5 → 2.3.6, 2.4.0 or 3.0.0, by the rule above) and turns `## Unreleased` into `## 2.3.6`, with the date. Its
   title names the version: `Release 2.3.6`. The `version` check says which step it is and how much changed since the
   last tag, and fails when the step is not one, the section is missing or `## Unreleased` is left behind.
5. Merging it starts the `publish` workflow on `main`. It builds the wheel and checks that it installs, then **waits
   for the maintainer's approval** of the deployment to the `pypi` environment (*Actions → the run → Review
   deployments*). Approved, it publishes to PyPI, tags the commit `v2.3.6`, makes the GitHub release from the
   version's CHANGELOG section, and installs the release from PyPI to try it. Rejected, nothing is published; the
   version stays unreleased on `main` until the run is approved again (*Run workflow* on `publish`).
6. A push to `main` whose version is already tagged does nothing, so ordinary pull requests never publish. Never tag
   by hand: the workflow tags what it published.

Check a build locally before opening a release pull request:

```bash
python -m pip install build
python -m build
python -m pip install dist/entail_ai-*.whl   # into a fresh virtual environment
entail doctor
```
