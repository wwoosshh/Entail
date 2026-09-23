# Releasing

`pip install "git+https://github.com/wwoosshh/entail"` works from any commit. Publishing to PyPI as `entail-ai`
takes a one-time setup on PyPI, done by the maintainer (it needs a PyPI account):

1. On pypi.org, *Your projects → Publishing → Add a new pending publisher*:
   PyPI project `entail-ai`, owner `wwoosshh`, repository `entail`, workflow `publish.yml`, environment `pypi`.
2. On GitHub, *Settings → Environments*: create an environment named `pypi`.

Then, for each release:

1. Bump `__version__` in `entail/__init__.py`.
2. Create a GitHub release with a tag such as `v0.1.1`. The `publish` workflow builds the wheel and uploads it;
   no token is stored anywhere (trusted publishing).

Check a build locally before tagging:

```bash
python -m pip install build
python -m build
python -m pip install dist/entail_ai-*.whl   # into a fresh virtual environment
entail doctor
```
