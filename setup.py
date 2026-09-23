"""Only here for one thing pyproject.toml cannot say: put the start-up hook at the top of the wheel.

A file at the root of a wheel is installed into site-packages itself, where Python reads .pth files at start-up.
The line it holds comes from entail/hook.py, so the file installed by pip and the one `entail hook install`
writes cannot drift apart. Everything else is in pyproject.toml.
"""
import os
import sys

from setuptools import setup
from setuptools.command.build_py import build_py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entail.hook import PTH_LINE, PTH_NAME  # noqa: E402


class build_py_with_hook(build_py):
    def run(self):
        super().run()
        target = os.path.join(self.build_lib, PTH_NAME)
        with open(target, "w", encoding="utf-8") as f:
            f.write(PTH_LINE)


setup(cmdclass={"build_py": build_py_with_hook})
