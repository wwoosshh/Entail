# 도구별 entail 설치 안내

entail은 모델을 돌리는 파이썬 프로세스 안에서 일한다. 그래서 어떤 도구든 규칙은 하나다. **도구가 실제로 쓰는 파이썬
환경에 entail을 설치하고**, 도구를 `ENTAIL=load`로 시작한다. entail이 찾은 것은 프로그램을 시작한 폴더의
`entail_logs/`에 남고, `entail serve`가 그것을 보여 준다.

English: [INSTALL.md](INSTALL.md)

## 누구를 위한 것인가

- 파이썬 엔진(transformers, diffusers, vLLM, SGLang)이나 ComfyUI로 자기 AI 프로젝트를 만드는 사람.
- 모델을 앱 안에 컴파일된 엔진으로 돌리는 앱은 대상이 아니다: Ollama, LM Studio, llama.cpp,
  text-generation-webui의 포터블 판. entail이 붙을 자리가 없다.

## 한눈에

| 도구 | entail을 설치할 곳 | 켜는 법 | 기록이 남는 곳 | 확인 |
|---|---|---|---|---|
| 내 스크립트(transformers, diffusers) | 스크립트를 돌리는 환경 | `ENTAIL=load python app.py` | 시작한 폴더 | 실행 |
| vLLM(앞에 Open WebUI나 다른 클라이언트) | vLLM의 환경 | `ENTAIL=load vllm serve ...` | 시작한 폴더 | 실행 |
| SGLang | SGLang의 환경 | `ENTAIL=load python -m sglang.launch_server ...` | 시작한 폴더 | 실행 |
| git으로 설치한 ComfyUI | ComfyUI의 `venv` | `set ENTAIL=load`를 넣은 실행 파일 | ComfyUI 폴더 | 실행 |
| ComfyUI 포터블(Windows) | `python_embeded` | `run_nvidia_gpu.bat`의 사본 | 포터블 폴더 | 소스 확인 |
| ComfyUI 데스크톱 | 설치 폴더의 `.venv` | 사용자 환경 변수 | 설치 폴더 | 소스 확인 |
| text-generation-webui(전체 설치) | 자체 환경(`cmd_windows.bat`) | `set ENTAIL=load`를 넣은 실행 파일 | 그 폴더 | 소스 확인 |
| Docker의 vLLM | vLLM 이미지를 바탕으로 만든 이미지 | `ENV ENTAIL=load` | 연결한 폴더 | 소스 확인 |

"실행": 이 프로젝트의 장비에서 해 봤다. "소스 확인": 도구의 소스 코드대로 적었지만 여기서 돌려 보지는 않았다.
자세한 것은 [확인한 것](#확인한-것)에 있다.

셸마다 변수를 주는 법:

```bash
ENTAIL=load python app.py            # Linux, macOS: 이 명령 하나에만
```

```powershell
$env:ENTAIL = "load"                 # Windows PowerShell: 이 창에서만
python app.py
```

```bat
set ENTAIL=load
python app.py
```

`cmd`와 `.bat` 파일에서는 `set ENTAIL=load`를 따로 한 줄에 쓴다. `set ENTAIL=load && ...`로 쓰면 `&&` 앞의 빈칸이
값에 들어간다.

콘솔에 `[entail] unknown`으로 시작하는 줄이 나오면, 문제를 찾았다는 뜻이 아니라 그 값을 확인하지 못했다는 뜻이다
(선언이 없거나 그 자리에서 검사할 수 없음). 이 줄을 콘솔에서 빼고 `entail_logs/`에만 남기려면 `ENTAIL=load`와 같은
방법으로 `ENTAIL_QUIET=unknown`을 준다.

## 내 스크립트(transformers, diffusers)

스크립트를 돌리는 환경에 설치하고, 변수를 준 채로 시작한다.

```bash
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install entail-ai
ENTAIL=load python app.py
```

스크립트 안에서 켜도 된다. 모델을 불러오기 전에 둔다.

```python
import entail
entail.enable()
```

## vLLM(앞에 Open WebUI나 OpenAI 클라이언트)

`vllm`이 설치된 환경에 설치한다. Open WebUI나 다른 클라이언트에는 아무것도 설치하지 않는다. 클라이언트는 vLLM의
API와 이야기하고, entail은 vLLM 안에서 일한다.

```bash
source ~/venvs/vllm/bin/activate     # vllm이 설치된 곳
pip install entail-ai
ENTAIL=load vllm serve Qwen/Qwen3-4B
```

vLLM은 작업 프로세스를 스스로 띄운다. entail은 `site-packages`에 더한 한 줄로 그 프로세스까지 닿는다(README의
"엔진의 작업 프로세스까지 닿는 방법").

### Docker의 vLLM

vLLM 이미지를 바탕으로 이미지를 만들고, 기록을 연결한 폴더로 옮겨서 호스트가 읽게 한다.

```dockerfile
FROM vllm/vllm-openai:v0.30.0
RUN uv pip install --system entail-ai
ENV ENTAIL=load ENTAIL_LOG_DIR=/entail_logs
```

```bash
docker build -t vllm-entail .
docker run --gpus all -p 8000:8000 -v "$PWD/entail_logs:/entail_logs" vllm-entail Qwen/Qwen3-4B
```

쓰던 옵션(모델 캐시, `--ipc=host`)을 더한다. 이미지는 준 인자로 `vllm serve`를 시작한다. 기록을 보려면 호스트에도
entail을 설치하고 `entail serve --dir ./entail_logs --open`을 돌린다.

## SGLang

```bash
pip install entail-ai                # SGLang의 환경에서
ENTAIL=load python -m sglang.launch_server --model-path Qwen/Qwen3-4B
```

## ComfyUI

### git과 가상 환경으로 설치한 경우

ComfyUI 폴더에서 그 환경의 파이썬으로 설치한다(폴더 이름은 `venv`나 `.venv`일 수 있다).

```bat
.venv\Scripts\python.exe -m pip install entail-ai
```

평소 쓰는 실행 파일 옆에 실행 파일을 하나 더 만든다. 예를 들어 `run_entail.bat`:

```bat
@echo off
cd /d "%~dp0"
set ENTAIL=load
.venv\Scripts\python.exe main.py %*
pause
```

Linux와 macOS에서는 `source .venv/bin/activate`, `pip install entail-ai`, 그리고 `ENTAIL=load python main.py`.
기록은 ComfyUI 폴더의 `entail_logs`에 남는다.

### 포터블(ComfyUI_windows_portable)

포터블의 파이썬은 `python_embeded`다. 포터블 폴더에서:

```bat
python_embeded\python.exe -m pip install entail-ai
```

`run_nvidia_gpu.bat`을 `run_nvidia_gpu_entail.bat`으로 복사하고, ComfyUI를 시작하는 줄 위에 `set ENTAIL=load`를
넣는다.

```bat
set ENTAIL=load
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build
pause
```

기록은 포터블 폴더의 `entail_logs`에 남는다. (`-s`는 사용자 단위로 설치한 패키지만 뺀다. entail은
`python_embeded` 안에 있다.)

### ComfyUI 데스크톱

앱은 설치할 때 고른 폴더(기본은 `문서\ComfyUI`, 실제 경로는 `C:\Users\<이름>\Documents\ComfyUI`) 안의 `.venv`에
파이썬을 둔다. PowerShell에서:

```powershell
& "$HOME\Documents\ComfyUI\.venv\Scripts\python.exe" -m pip install entail-ai
[Environment]::SetEnvironmentVariable("ENTAIL", "load", "User")
```

앱을 완전히 끄고(트레이에서도) 다시 켠다. 기록은 그 폴더의 `entail_logs`에 남는다.
이 변수는 내가 시작하는 모든 프로그램에 전해지지만, entail이 설치되지 않은 곳에서는 아무 일도 하지 않는다.
entail을 끄려면 변수를 지우고 앱을 다시 켠다.

```powershell
[Environment]::SetEnvironmentVariable("ENTAIL", $null, "User")
```

### ComfyUI DLC(선택)

`entail-dlc-comfyui`는 ComfyUI 자체의 결함 하나를 고친다([#16490](https://github.com/Comfy-Org/ComfyUI/issues/16490):
ComfyUI 0.34.1의 동적 VRAM 적재기가 샘플링 노드의 스케줄을 체크포인트에 써 넣어서, 그 뒤의 실행이 다른 그림이
되다가 검게 나온다). PyPI에는 없고 git으로 설치하며, entail과 같은 파이썬에 넣는다.

```bat
python -m pip install "git+https://github.com/wwoosshh/entail@v2.1.0#subdirectory=dlc/comfyui"
```

(`python` 대신 위와 같이 `python_embeded\python.exe`나 `.venv\Scripts\python.exe`를 쓴다.)

## text-generation-webui(전체 설치)

파이썬으로 모델을 돌리는 것은 전체 설치뿐이다. 포터블 판은 llama.cpp만 돌린다. 로더 가운데 entail이 검사하는
것은 **Transformers**다. llama.cpp, ExLlamaV3, TensorRT-LLM은 검사하지 않는다.

1. 그 폴더에서 `cmd_windows.bat`(`cmd_linux.sh`, `cmd_macos.sh`)을 실행한다. 자체 환경 안의 셸이 열린다. 거기서
   설치한다.
   ```bat
   pip install entail-ai
   ```
2. entail을 켜고 시작한다. `start_windows.bat` 옆에 실행 파일을 만든다. 예를 들어 `start_entail.bat`:
   ```bat
   @echo off
   set ENTAIL=load
   call "%~dp0start_windows.bat" %*
   ```
   Linux와 macOS에서는 `ENTAIL=load ./start_linux.sh`.
3. 모델을 Transformers 로더로 불러온다. 기록은 그 폴더의 `entail_logs`에 남는다.

## 찾은 것 보기

기록은 프로그램을 시작한 곳의 `entail_logs/`(또는 `ENTAIL_LOG_DIR`)에 있다. 같은 환경의 entail로 연다.

```bash
entail serve --dir entail_logs --open
```

예를 들어 ComfyUI 포터블 폴더에서는 `python_embeded\python.exe -m entail serve --dir entail_logs --open`.
화면은 http://127.0.0.1:8765/ 에 있고, 이 컴퓨터에서만 열린다.

## 확인, 끄기, 지우기

- `entail doctor`는 그 환경에서 entail이 보는 것을 보여 주고, `entail hook status`는 시작 줄이 놓였는지 보여 준다.
- `ENTAIL`을 주지 않으면 entail은 아무 일도 하지 않는다. 시작 줄은 바로 돌아간다.
- `entail hook uninstall`은 그 줄을 지우고, `pip uninstall entail-ai`는 entail을 지운다.

## 확인한 것

- **실행, Windows 11, Python 3.12(2026-09-28):** 새 가상 환경에 휠로 설치했다. PowerShell 방식과 `cmd` 방식으로
  `ENTAIL`을 주면 둘 다 켜졌고, 주지 않으면 꺼져 있었다. 프로그램이 시작한 폴더에 `entail_logs`를 썼고,
  `ENTAIL_LOG_DIR`가 그 자리를 옮겼으며, `entail serve`가 그 실행을 보여 주었다.
- **실행, Linux(WSL2):** vLLM 0.30.0, SGLang 0.5.20, transformers 5.17.0. README의 정상 실행과 측정이다.
- **실행, git으로 설치한 ComfyUI, Windows 가상 환경:** pip로 설치하고 `set ENTAIL=load`를 넣은 실행 파일로
  시작했다(예전 판 entail, 2026-09-23). 같은 폴더의 ComfyUI 0.34.1에서 어댑터를 측정했다(README).
- **소스 확인, 여기서 돌려 보지 않음:** 포터블(파이썬에 `import site`와 pip가 들어가고, `run_nvidia_gpu.bat`이
  `python_embeded\python.exe -s ComfyUI\main.py`를 시작한다: Comfy-Org/ComfyUI의 `.github/workflows`와 `.ci`),
  ComfyUI 데스크톱(설치 폴더의 `.venv\Scripts\python.exe`로 ComfyUI를 시작하고, 자기 환경 변수를 넘기며, 그
  환경에 pip를 둔다. 기본 폴더는 `문서\ComfyUI`다: Comfy-Org/desktop의 `src/`), text-generation-webui
  (`cmd_windows.bat`이 `installer_files`의 Conda 환경을 연다. `start_windows.bat`은 자기 폴더로 옮겨 가고, 서버는
  받은 환경 변수 그대로 시작된다. 포터블 판은 llama.cpp만 쓴다: README, `start_windows.bat`, `one_click.py`),
  vLLM 이미지(시작점이 `vllm serve`이고 안에 `uv`가 있다: vLLM의 `docker/Dockerfile`).
