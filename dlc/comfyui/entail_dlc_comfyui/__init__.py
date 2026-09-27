"""entail-dlc-comfyui: the official DLC for ComfyUI (entail product track P4; LIBRARY_DESIGN.md 13.7).

It holds the repair of ComfyUI's own defect Comfy-Org/ComfyUI#16490 (repair.py; in the core as
entail/adapters/comfyui_repair.py until entail 1.3). entail finds this module through the entry point group
`entail.dlc` and reads what it says below: which entries to install once which ComfyUI module is imported, and the
platform's node for what the repair says. ENTAIL_DLC=off, or ENTAIL_SKIP=repair:install_buffer_guard for one entry,
leaves it out.
"""
name = "comfyui"
version = "0.1.0"
requires = ">=2.0,<3"                  # the entail core versions it works with
engines = {"comfyui": "0.34.1"}        # where the defect and the repair were measured
targets = {
    "comfy.model_sampling": ["entail_dlc_comfyui.repair:install_schedule_record"],
    "comfy.model_patcher": ["entail_dlc_comfyui.repair:install_buffer_guard"],
    "comfy.model_base": ["entail_dlc_comfyui.repair:install_schedule_check"],
}
facts = ()                             # it repairs; it decides no fact of the core's vocabulary
nodes = [{"id": "comfyui_schedule", "flow": "image", "step": 5, "ko": "샘플링 일정 (ComfyUI 수리)",
          "en": "Sampling schedule (ComfyUI repair, #16490)", "patterns": ["^dlc:comfyui\\."]}]
