# Installing entail, tool by tool

entail works from inside the Python process that runs your model. So one rule holds for every tool: **install
entail into the Python environment the tool itself uses**, then start the tool with `ENTAIL=load`. What entail finds
goes to `entail_logs/` in the folder the program starts from, and `entail serve` shows it.

한국어: [INSTALL.ko.md](INSTALL.ko.md)

## Who this is for

- People who build their own AI project on a Python engine - transformers, diffusers, vLLM, SGLang - or on ComfyUI.
- Not apps that run models in an engine compiled into the app: Ollama, LM Studio, llama.cpp, and the portable builds
  of text-generation-webui. entail has nothing to attach to there.

## At a glance

| tool | install entail into | turn it on | records go to | checked |
|---|---|---|---|---|
| your script (transformers, diffusers) | the environment that runs the script | `ENTAIL=load python app.py` | the folder you start it from | run |
| vLLM, with Open WebUI or any client in front | vLLM's environment | `ENTAIL=load vllm serve ...` | the folder you start it from | run |
| SGLang | SGLang's environment | `ENTAIL=load python -m sglang.launch_server ...` | the folder you start it from | run |
| ComfyUI installed with git | ComfyUI's `venv` | a launcher with `set ENTAIL=load` | ComfyUI's folder | run |
| ComfyUI portable (Windows) | `python_embeded` | a copy of `run_nvidia_gpu.bat` | the portable folder | source read |
| ComfyUI Desktop | `.venv` in its install folder | a user environment variable | the install folder | source read |
| text-generation-webui (full install) | its environment (`cmd_windows.bat`) | a launcher with `set ENTAIL=load` | its folder | source read |
| vLLM in Docker | an image built on vLLM's | `ENV ENTAIL=load` | a folder you mount | source read |

"run": done on this project's machine. "source read": the steps follow the tool's own source code, but were not run
here. [What was checked](#what-was-checked) has the details.

Setting the variable, by shell:

```bash
ENTAIL=load python app.py            # Linux and macOS: for this one command
```

```powershell
$env:ENTAIL = "load"                 # Windows PowerShell: for this window
python app.py
```

```bat
set ENTAIL=load
python app.py
```

In `cmd` and `.bat` files keep `set ENTAIL=load` on a line of its own: in `set ENTAIL=load && ...` the space before
`&&` becomes part of the value.

## Your own script (transformers, diffusers)

Install into the environment the script runs in, and start it with the variable set:

```bash
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install entail-ai
ENTAIL=load python app.py
```

Or turn it on from inside the script, before the model is loaded:

```python
import entail
entail.enable()
```

## vLLM, with Open WebUI or any OpenAI client in front

Install into the environment `vllm` is installed in. Open WebUI, or any other client, needs nothing: it talks to
vLLM's API, and entail works inside vLLM.

```bash
source ~/venvs/vllm/bin/activate     # wherever vllm is installed
pip install entail-ai
ENTAIL=load vllm serve Qwen/Qwen3-4B
```

vLLM starts its worker processes itself; entail reaches them through the one line it adds to `site-packages`
(README, "How it reaches engine worker processes").

### vLLM in Docker

Build an image on vLLM's and move the records to a folder you mount, so that the host can read them:

```dockerfile
FROM vllm/vllm-openai:v0.30.0
RUN uv pip install --system entail-ai
ENV ENTAIL=load ENTAIL_LOG_DIR=/entail_logs
```

```bash
docker build -t vllm-entail .
docker run --gpus all -p 8000:8000 -v "$PWD/entail_logs:/entail_logs" vllm-entail Qwen/Qwen3-4B
```

Add your usual options (a model cache, `--ipc=host`). The image starts `vllm serve` with the arguments you give it.
To see the records, install entail on the host as well and run `entail serve --dir ./entail_logs --open`.

## SGLang

```bash
pip install entail-ai                # in SGLang's environment
ENTAIL=load python -m sglang.launch_server --model-path Qwen/Qwen3-4B
```

## ComfyUI

### Installed with git and a virtual environment

In ComfyUI's folder, with the environment's own Python (the folder may be called `venv` or `.venv`):

```bat
.venv\Scripts\python.exe -m pip install entail-ai
```

Then a launcher next to the one you use, for example `run_entail.bat`:

```bat
@echo off
cd /d "%~dp0"
set ENTAIL=load
.venv\Scripts\python.exe main.py %*
pause
```

On Linux and macOS: `source .venv/bin/activate`, `pip install entail-ai`, then `ENTAIL=load python main.py`.
Records: `entail_logs` in ComfyUI's folder.

### Portable (ComfyUI_windows_portable)

Its Python is `python_embeded`. In the portable folder:

```bat
python_embeded\python.exe -m pip install entail-ai
```

Copy `run_nvidia_gpu.bat` to `run_nvidia_gpu_entail.bat` and put `set ENTAIL=load` above the line that starts
ComfyUI:

```bat
set ENTAIL=load
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build
pause
```

Records: `entail_logs` in the portable folder. (`-s` only leaves out your user-level packages; entail sits in
`python_embeded`.)

### ComfyUI Desktop

The app keeps its Python in `.venv` inside the folder you chose when you installed it - by default
`Documents\ComfyUI`. In PowerShell:

```powershell
& "$HOME\Documents\ComfyUI\.venv\Scripts\python.exe" -m pip install entail-ai
[Environment]::SetEnvironmentVariable("ENTAIL", "load", "User")
```

Quit the app completely (from the tray as well) and start it again. Records: `entail_logs` in that folder.
The variable reaches every program you start, but it does nothing where entail is not installed. To turn entail off,
remove it and restart the app:

```powershell
[Environment]::SetEnvironmentVariable("ENTAIL", $null, "User")
```

### The ComfyUI DLC (optional)

`entail-dlc-comfyui` repairs a defect of ComfyUI itself ([#16490](https://github.com/Comfy-Org/ComfyUI/issues/16490):
ComfyUI 0.34.1's dynamic VRAM loader writes a sampling node's schedule into the checkpoint, so later runs come out as
another image, then black). It is not on PyPI and installs with git, into the same Python as entail:

```bat
python -m pip install "git+https://github.com/wwoosshh/entail@v2.0.0#subdirectory=dlc/comfyui"
```

(Use `python_embeded\python.exe` or `.venv\Scripts\python.exe` in place of `python`, as above.)

## text-generation-webui (full install)

Only the full install runs models in Python; the portable builds run llama.cpp alone. Of its loaders entail checks
**Transformers**; llama.cpp, ExLlamaV3 and TensorRT-LLM are not checked.

1. In its folder run `cmd_windows.bat` (`cmd_linux.sh`, `cmd_macos.sh`) - a shell inside its own environment - and
   install:
   ```bat
   pip install entail-ai
   ```
2. Start it with entail on. A launcher next to `start_windows.bat`, for example `start_entail.bat`:
   ```bat
   @echo off
   set ENTAIL=load
   call "%~dp0start_windows.bat" %*
   ```
   On Linux and macOS: `ENTAIL=load ./start_linux.sh`.
3. Load the model with the Transformers loader. Records: `entail_logs` in its folder.

## See what it found

The records are in `entail_logs/` where the program started (or in `ENTAIL_LOG_DIR`). Open them with the same
environment's entail:

```bash
entail serve --dir entail_logs --open
```

For example in ComfyUI's portable folder: `python_embeded\python.exe -m entail serve --dir entail_logs --open`.
The page is at http://127.0.0.1:8765/ and answers this machine only.

## Check, turn off, remove

- `entail doctor` shows what entail sees in the environment; `entail hook status` shows whether its start-up line is
  in place.
- Without `ENTAIL` set, entail does nothing: its start-up line returns at once.
- `entail hook uninstall` removes that line; `pip uninstall entail-ai` removes entail.

## What was checked

- **Run, Windows 11, Python 3.12 (2026-09-28):** a new virtual environment, entail installed from a wheel; `ENTAIL`
  set the PowerShell way and the `cmd` way both turned it on, and unset it stayed off; a program wrote `entail_logs`
  in the folder it started from, `ENTAIL_LOG_DIR` moved it, and `entail serve` showed the run.
- **Run, Linux (WSL2):** vLLM 0.30.0, SGLang 0.5.20 and transformers 5.17.0 - the healthy runs and measurements in
  the README.
- **Run, ComfyUI installed with git, in a Windows venv:** entail installed with pip and started from a launcher with
  `set ENTAIL=load` (an earlier entail, 2026-09-23); ComfyUI 0.34.1 in that folder for the adapters' measurements
  (README).
- **Source read, not run here:** the portable build (its Python gets `import site` and pip, and `run_nvidia_gpu.bat`
  starts `python_embeded\python.exe -s ComfyUI\main.py`: Comfy-Org/ComfyUI, `.github/workflows` and `.ci`); ComfyUI
  Desktop (it starts ComfyUI with `.venv\Scripts\python.exe` from the install folder, hands its own environment on,
  and sets up pip there; the default folder is `Documents\ComfyUI`: Comfy-Org/desktop, `src/`); text-generation-webui
  (`cmd_windows.bat` opens its Conda environment in `installer_files`; `start_windows.bat` moves to its folder and
  the server is started with the environment it was given; the portable builds are llama.cpp only: its README,
  `start_windows.bat` and `one_click.py`); vLLM's image (`vllm serve` as the entry point, `uv` inside: vLLM's
  `docker/Dockerfile`).
