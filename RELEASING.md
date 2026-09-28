# Releasing

`pip install "git+https://github.com/wwoosshh/entail"` works from any commit. Publishing to PyPI as `entail-ai`
takes a one-time setup on PyPI, done by the maintainer (it needs a PyPI account):

1. On pypi.org, *Your projects → Publishing → Add a new pending publisher*:
   PyPI project `entail-ai`, owner `wwoosshh`, repository `entail`, workflow `publish.yml`, environment `pypi`.
2. On GitHub, *Settings → Environments*: create an environment named `pypi`.

## Version numbers

`MAJOR.MINOR.PATCH`, chosen by what changed since the last release:

- **MAJOR** when the design's structure changes (2.0: the local platform around the checks).
- **MINOR** when the structure stays and the release has a purpose: new features, or a large scope of work.
- **PATCH** for bug fixes and small corrections of a few hundred lines or fewer. A larger change is a MINOR release.

The size is read against the last release's tag (`git diff --shortstat v2.0.0`, say). The packages outside the
core (`dlc/comfyui`, `workshop/basics`) keep their own numbers by the same rule.

## Each release

1. Pick the number by the rule above, bump `__version__` in `entail/__init__.py` and add the release's section
   to CHANGELOG.md.
2. Create a GitHub release with a tag such as `v0.1.1`. The `publish` workflow builds the wheel and uploads it;
   no token is stored anywhere (trusted publishing).

Check a build locally before tagging:

```bash
python -m pip install build
python -m build
python -m pip install dist/entail_ai-*.whl   # into a fresh virtual environment
entail doctor
```
