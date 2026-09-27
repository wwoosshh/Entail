# entail-dlc-comfyui

The official [entail](https://github.com/wwoosshh/entail) DLC for ComfyUI. It repairs one defect of ComfyUI itself,
[Comfy-Org/ComfyUI#16490](https://github.com/Comfy-Org/ComfyUI/issues/16490): ComfyUI 0.34.1's dynamic VRAM loader
writes a sampling node's schedule into the checkpoint's own object, so later runs without the node come out as another
image, then black.

Until entail 1.3 this repair lived in the core (`entail/adapters/comfyui_repair.py`). The core now holds no
engine-specific repair: engine-specific checks and repairs are DLCs, separate packages that attach through the entry
point group `entail.dlc`.

It is not on PyPI; it installs from entail's repository, with entail 2.0 or later as the core it attaches to:

```bash
pip install "git+https://github.com/wwoosshh/entail@v2.0.0#subdirectory=dlc/comfyui"
```

With `ENTAIL=load`, entail finds it and installs the repair when ComfyUI's modules are imported. `entail doctor` lists
the DLCs it found. `ENTAIL_DLC=off` leaves every DLC out; `ENTAIL_SKIP=repair:install_buffer_guard` leaves out one
entry. What the repair does is one line in the console and in `entail_logs/`, and the platform (`entail serve`) shows it
on the node "Sampling schedule (ComfyUI repair, #16490)".

Tests: `python tests/test_repair.py` (needs torch and entail).
