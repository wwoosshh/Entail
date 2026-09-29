# entail

[![PyPI](https://img.shields.io/pypi/v/entail-ai)](https://pypi.org/project/entail-ai/) [![tests](https://github.com/wwoosshh/entail/actions/workflows/tests.yml/badge.svg)](https://github.com/wwoosshh/entail/actions/workflows/tests.yml)

**모델 파일은 자기를 어떻게 돌려야 하는지 적어 둔다. 엔진이 늘 그것을 읽지는 않는다.**

RoPE 밑과 스케일링, soft-capping, sliding window, 채팅 템플릿, 예측 방식. 이 선언 가운데 하나가 엔진에 닿지 않으면 출력은 경고 없이 틀린다. entail은 파일이 이미 선언한 것을 읽어 쓰이는 자리에서 대조하고, 고칠 수 있으면 첫 토큰 전에 고치고, 못 고치면 무엇이 깨졌는지 정확히 적는다. 설정 0줄. 요청 시간의 약 1%가 들고, vLLM에서는 적재가 약 1.7 s 늘어난다(작은 모델 적재의 13~15%; 대부분은 엔진 자신의 경로를 시작 때 대조하는 점검이고 `ENTAIL_NO_PATHS=1`로 끈다).

- **180개 중 64개.** Hugging Face에서 가장 많이 받는 LLM 300개 가운데 180개가 vLLM의 실행 때 `rope_scaling` 덮어쓰기(긴 문맥을 켜는 흔한 방법)에 해당한다. 그중 64개의 RoPE 밑이 경고 없이 바뀐다. entail을 켜면 180개 모두 원래 밑을 지킨다. (vLLM 0.30, 모델 파일을 vLLM 자신의 설정 코드로 대조; [E1](https://github.com/wwoosshh/entail-research/blob/main/testbed/results/m10/E1_SUMMARY.md))
- **379 → 273.** 끝까지 재면 Llama-3.2-3B-Instruct의 GSM8K가 그 경로에서 그렇게 떨어진다(entail 켬: 376). Qwen3-4B-Instruct-2507은 YaRN에서 183 → 175. 둘 다 경고가 없다.
- **평가 점수로는 다 안 보인다.** Gemma 2의 soft-capping을 버리는 백엔드는 500문제 중 198문제의 답을 바꾸지만 GSM8K는 3문제 차이다(p = 0.66). entail은 적재 때 그 성질을 지키는 백엔드로 보낸다.
- **정상 실행 102회, 깨뜨린 실행 0.** 인기 모델 38개를 transformers, vLLM, SGLang에서: 출력은 98번 비교 가운데 97번 entail 없는 실행과 같고(다른 하나는 엔진 자신의 비결정성), 1.3.0에서 더한 검사는 경보를 내지 않았다. 여섯 실행은 실제 토크나이저 차이를 알린다(transformers 5가 Llama-2 시절 토크나이저 둘에서 앞 공백 하나를 잃는다). 그 밖에서 드러난 틀린 경보 하나(vLLM의 인코더-디코더 모델)는 2.0.0에서 고쳤다. (1.0.0은 처음 81회 중 17회를 틀렸다. 원인은 고쳤고 [변경 기록](CHANGELOG.md)에 있다.)

내 모델 확인은 세 줄이다.

```bash
pip install entail-ai
entail preflight --model /path/to/model --engine vllm --list   # 백엔드마다 무엇을 버리는지, GPU 없이
ENTAIL=load vllm serve /path/to/model ...                       # 그리고 entail_logs/ 읽기
```

entail은 값의 뜻을 타입처럼 분명하게 만든다. 뜻을 만드는 곳에서 선언하고 쓰는 곳까지 잇고, 소비자의 선택과 실제 데이터와 대조하고, 어긋나면 먼저 해소하며(선언을 지키는 소비자에게 보내거나 소비자가 읽는 형태로 바꾼다), 해소할 방법이 없으면 알리고 실행은 잇고(멈추게 하려면 따로 켠다), 아무도 선언하지 않았으면 기본값이 조용히 대신하게 두지 않고 "모름"이라고 말한다. 이름은 논리학의 "함의(entail)"에서 왔다. 체크포인트가 선언한 뜻은 엔진이 실제로 실행하는 것을 반드시 함의해야 한다. ent·**AI**·**L**은 AI Library를 뜻한다.

**누구를 위한 것인가:** 파이썬 엔진(transformers, diffusers, vLLM, SGLang)이나 ComfyUI로 자기 AI 프로젝트를 만드는 사람이다. entail은 모델을 돌리는 파이썬 프로세스 안에서 일한다. 그래서 모델을 앱 안에 컴파일된 엔진으로 돌리는 앱(Ollama, LM Studio, llama.cpp)에는 붙을 자리가 없다. 도구별로 어디에 설치하는지(내 스크립트, Open WebUI 뒤의 vLLM, ComfyUI의 git 설치·포터블·데스크톱, text-generation-webui, Docker의 vLLM)는 [INSTALL.ko.md](https://github.com/wwoosshh/entail/blob/main/INSTALL.ko.md)에 있다.

> **상태: 2.1.2. 한 대의 장비에서 쟀다.** 2.1.2는 2.1.1의 현장 시험에서 나온 것을 고친 판이다(문서만 보고 하는 매니페스트 선언, 인기 모델에서 나던 `entail preflight`의 틀린 경보, entail이 바로잡은 것을 먼저 말하는 화면, Windows에서의 업그레이드). 2.1.1은 2.1.0의 첫 현장 시험에서 나온 것을 고친 판이다(더 알기 쉬운 콘솔 줄과 화면 문구, 경로 객체로 준 토크나이저 폴더, doctor의 엔진 목록). 2.1.0은 `entail serve`의 화면을 새로 만들고(쉬운 말, 노드 캔버스) 도구별 설치 안내([INSTALL.ko.md](https://github.com/wwoosshh/entail/blob/main/INSTALL.ko.md))를 더했고, 검사는 2.0과 같다. 2.0은 같은 검사 둘레에 플랫폼을 더했다: `entail serve`, 안전모드 둘, 공식 DLC, 커스텀 노드([플랫폼](#플랫폼-20)). 정상 실행 102회에서 1.3.0의 동결 코드와 판정이 정확히 같았다(깨뜨린 실행 0, 같은 토크나이저 실제 차이 여섯을 알림). entail을 켜면 요청이 많아야 1% 남짓 느려지고(번갈아 잰 10라운드에서 1.000~1.011배, 1.3.0과 같음), `entail serve`가 기록을 읽는 동안에도 같다. 색인 없이 설치한 휠은 entail 말고 아무것도 들이지 않는다. 아래 내용은 모두 "시험한 판"의 엔진과 버전으로, RTX 4070 Ti 한 장에서 쟀다. 평가는 "어떻게 쟀나"에, 평가에서 드러난 빈틈은 "알려진 빈틈"에 있다.
> **지금의 모습:** 가벼운 배포 전 점검이다. 모델 파일이 선언한 것을 읽고, 선언이 코드에만 있는 곳에서는 엔진 자신의 커널·경로·파서를 참조와 대조하고, 아는 부류는 고치고, 뜻이 어디서 깨졌는지 말한다. **아직 아닌 것:** 아무도 본 적 없는 버그를 막는 장치. 코드를 동결하고 실제 엔진 버그를 다시 돌린 사전 등록 재현 네 번에서, 재현된 부류 안 버그를 7건 중 0건, 8건 중 0건, 5건 중 0건, 6건 중 0건 잡았다. 대부분은 entail의 검사가 닿지 않는 곳(프로세서 설정, 스케줄러의 step, 가중치 적재기, KV 커넥터, C++에서 부르는 커널)에 있었다.
> 1.1.0은 1.0 평가에서 읽지 못한다고 드러난 사실 다섯을, 1.2.0은 1차 재현이 읽지 않는다고 드러낸 자리들을 더했다. 1.3.0은 참조 대조(선언된 토크나이저를 실제로 돌린 것, 커스텀 연산의 커널과 그 자신의 정의, 파서의 스트림과 전체 텍스트 파싱, 자리표의 출처, 응답의 logprobs), 엔진 함수 다섯의 정의, 엔진의 워밍업까지 닿는 검사(CUDA 그래프가 커널을 잡기 전), 시작 때 엔진 자신의 경로끼리의 대조를 더한다.
> entail은 모델, 컴파일러, 커널, 하드웨어 안쪽의 결함을 찾지 않는다. 검사한 모든 경계가 온전한데 출력이 틀리면, 그렇다고 말하고 살펴볼 곳을 좁힌다.

## 환경에 무엇을 남기나

- `pip install entail-ai`는 패키지 하나(import 이름은 `entail`, 의존성 없음)와 `site-packages`의 한 줄(`entail-autoinstall.pth`)을 더한다. 이 한 줄로 엔진이 띄우는 작업 프로세스까지 닿는다. `ENTAIL`이 없으면 그 줄은 바로 돌아온다. 파이썬 시작마다 0.2~0.3 ms이고 불러오는 모듈이 없다. `entail hook status`로 보고 `entail hook uninstall`로 지운다.
- `ENTAIL=load`이면 entail이 말한 것은 프로그램을 실행한 폴더의 `entail_logs/`에 남는다(날마다 로그와 JSON 기록, `.gitignore` 포함). 이 폴더에는 `said.txt`(이번 실행에서 이미 찍은 줄: 엔진의 여러 프로세스가 같은 줄을 되풀이하지 않게), 토크나이저 폴더에서 읽은 것의 작은 캐시(`tokenizer_ids.json`, `vocab_sources.json`), 화면의 설정 두 파일(`safe_mode.json`, `nodes.json`)도 생길 수 있다. `ENTAIL_LOG_DIR=off`면 아무것도 쓰지 않고, `ENTAIL_LOG_DIR=<폴더>`면 그곳에 쓴다. 시작할 때 stderr에 entail이 켜졌고 어디에 쓰는지 한 줄을 찍는다. `ENTAIL_QUIET=unknown`은 멈추지 않는 `unknown` 줄을, `ENTAIL_QUIET=start`는 그 첫 줄을 화면에서 뺀다(둘 다: `unknown,start`).
- 어댑터는 아래 표의 엔진 판에서 잰 내부 함수에 건다. 다른 판에서 설치되지 않는 어댑터는 한 번 알리고(`could not install ...`) 빠지며, 나머지는 돈다. `entail doctor`가 무엇이 설치돼 있고 무엇이 걸릴지 보여 준다.

| 엔진 | 잰 판 | 어댑터가 보는 것 |
|---|---|---|
| transformers | 5.12.1, 5.16.1, 5.17.0 | 어텐션 백엔드, 묶은 머리, 설정 키, RoPE 이름, KV 캐시, 채팅 템플릿 |
| vLLM | 0.30.0 | 어텐션 백엔드, 적재기, 다시 배치한 가중치의 자리, KV 캐시, OpenAI 서버, 파일과 가중치 대조, 커스텀 연산의 커널과 엔진 함수 넷을 정의와 대조, 시작 때 엔진 자신의 경로 |
| SGLang | 0.5.20 | 어텐션 백엔드, 적재기, KV 캐시, 서버, GDN 게이트를 정의와 대조(시작 때 경로 대조는 `ENTAIL_PATHS=1`일 때) |
| diffusers | 0.40.0 | 예측 방식, VAE 배율, LoRA가 닿는 곳 |
| ComfyUI | 0.34.1 | 예측 방식, VAE 배율, LoRA가 닿는 곳. ComfyUI 자신의 #16490 수리는 공식 DLC `entail-dlc-comfyui`(`dlc/comfyui`) |

## 왜 필요한가: 끝까지 돌려 잰 사례

모델 카드는 YaRN을 켜는 방법으로 실행 시점 덮어쓰기를 안내한다. 이 경로로 모델 **자신의** `rope_scaling` 값을 그대로 다시 넘기기만 해도, transformers 5에서는 `rope_theta`가 사라진다. 그러면 엔진은 RoPE 기준값을 10,000으로 조용히 바꿔 쓴다.

모델은 Llama-3.2-3B-Instruct이고 탐욕 디코딩이다. GSM8K는 vLLM에서 앞 500문항, SGLang에서 앞 200문항을 썼다.

| | 손대지 않음 | 같은 `rope_scaling`을 실행 시점에 다시 넘김 | entail 켬 |
|---|---|---|---|
| vLLM 0.30.0, transformers 5.17.0 (`hf_overrides`) | 379 / 500 | **273 / 500**, 경고 없음 | 376 / 500 |
| SGLang 0.5.20, transformers 5.12.1 (`json_model_override_args`) | 161 / 200 | **106 / 200** | 161 / 200 (출력까지 같음) |

- vLLM 줄은 1.0에서 다시 잰 값이고, SGLang 줄은 앞선 측정의 값이다. 둘 다 엔진의 오프라인 API(vLLM `LLM`, SGLang `Engine`)로 쟀다.
- SGLang 서버(`launch_server`, 같은 판)에서 5.4k 토큰 목록에서 코드 5개를 찾는 시험은, 손대지 않으면 5/5, 덮어쓰기를 주면 0/5, entail을 켜면 5/5였다([#26](https://github.com/wwoosshh/entail/issues/26). 떨어지지 않았다던 현장 시험은 뒤의 두 서버가 같은 포트를 잡지 못해 세 번 모두 첫 서버를 쟀다).
- 망가진 실행은 `rope_theta = 10000`을 직접 넣은 실행과 결과가 같았다.
- Qwen3 dense 모델은 우연히 안전하다. 그 모델 파일이 빠진 값을 1,000,000으로 채우기 때문이다. Llama, Qwen3-MoE, Gemma 등의 모델 파일은 채우지 않는다.

## 설치

```bash
pip install entail-ai
```

- 엔진(vLLM, SGLang, transformers)이 있는 그 환경에 설치한다. ComfyUI(git 설치, 포터블, 데스크톱), text-generation-webui, Docker의 vLLM에서 그 환경이 어디인지는 [INSTALL.ko.md](https://github.com/wwoosshh/entail/blob/main/INSTALL.ko.md)에 있다. entail 자체는 의존성이 없다.
- 가져올 때 쓰는 이름은 `entail`이다.
- `uv pip install entail-ai`도 같게 된다. 최신 커밋을 쓰려면 `pip install "git+https://github.com/wwoosshh/entail"`로 설치한다.

설치한 뒤에는 다음 명령으로 환경을 점검한다.

```bash
entail doctor
```

## 쓰는 법

환경 변수 하나만 켜면 된다. 다른 것은 바꾸지 않는다.

```bash
ENTAIL=load vllm serve meta-llama/Llama-3.2-3B-Instruct --hf-overrides '{"rope_scaling": {...}}'
ENTAIL=load python -m sglang.launch_server --model-path ... --json-model-override-args '{...}'
ENTAIL=load python your_transformers_script.py
ENTAIL=load python main.py            # ComfyUI는 그 폴더에서 실행한다(윈도우에서는 실행 .bat에 set ENTAIL=load)
```

스크립트 안에서 켤 때는 이렇게 한다.

```python
import entail
entail.enable()          # mode="load", policy="resolve". 자식 프로세스도 이어받는다
```

### 모델 파일이 뜻을 선언하지 않을 때

이미지 체크포인트는 예측 방식이나 잠재 배율을 선언하지 않는 경우가 많다. 그러면 entail은 추측하지 않고 "모름"으로 알린다. 선언 파일(매니페스트)은 그 뜻을 바깥에서 선언한다. 타입이 없는 JavaScript 라이브러리에 `.d.ts` 파일로 타입을 주는 것과 같다.

```bash
entail infer model.safetensors --out model.safetensors.entail.json   # 파일이 선언한 것과 빈칸
# 아는 빈칸을 채운 뒤 검토했다고 표시한다
entail pin model.safetensors.entail.json
```

모델 파일은 전체 경로로 준다. 초안의 빈칸마다 `form`에 값을 쓰는 형식이 적혀 있다. 마지막 단계의 SNR이 0인(zero terminal SNR) v-prediction 모델이면 `"value": {"kind": "v", "zsnr": true}`로 채운다(`kind`는 `eps`, `v`, `x0`, `flow`, `edm` 가운데 하나, `zsnr`는 `true`, `false`, `null` 가운데 하나). 잠재 배율은 `"value": {"scale": 0.13025, "shift": null}`로 채운다. `"v_prediction"`, `"epsilon"` 같은 낱말이나, 잠재 배율이면 숫자 하나만 써도 읽는다. 읽을 수 없는 값이면 `entail pin`이 쓰는 법을 알려 주고 아무것도 바꾸지 않는다.

파일 옆에 둔 선언 파일은 저절로 찾는다. 한 폴더에 모아 둔 선언 파일(`<sha256>.json`, 초안에 적힌 해시)은 `ENTAIL_MANIFESTS`로 찾는다. 검토 표시(pin)를 한 선언 파일만 선언으로 친다.

### 엔진의 작업 프로세스까지 닿는 방법

vLLM과 SGLang은 모델을 자기가 새로 띄운 프로세스에서 돌린다. 그래서 `pip install`은 site-packages에 `entail-autoinstall.pth` 파일 하나를 넣는다.

- 파이썬은 시작할 때마다 이 파일을 읽는다.
- 파일의 한 줄은 환경 변수만 확인하고, `ENTAIL`이 켜져 있지 않으면 아무것도 하지 않는다.
- `entail hook status|install|uninstall`로 상태를 보거나 관리한다.
- 편집 가능 설치(`pip install -e`)는 이 파일을 넣지 않으므로 `entail hook install`을 따로 실행한다.

## 플랫폼 (2.0)

entail 2.0은 AI 프로젝트의 안정성을 관리하는 로컬 플랫폼이다. 위의 검사에 더해, 의미가 어디서 지켜지고 어디서 깨졌는지 보여 주는 화면, 안전모드 둘, 노드 단위로 붙이고 떼는 검사(엔진용 공식 DLC, 내 코드용 커스텀 노드)가 있다. 모두 내 컴퓨터 안에서 돌고, 핵심은 여전히 의존성이 없다.

> **믿음의 경계:** DLC와 커스텀 노드는 내가 설치한 파이썬 코드이고, 내 프로그램 안에서 그 권한으로 돈다. entail은 그것이 실행을 깨지 않게 하고(실패, 느린 검증기, 멈춘 검증기는 기록하고 뗀다), entail의 규칙을 바꾸지 못하게 한다. 격리하지는 않는다. 믿는 것만 설치하고, 붙일 것을 목록으로 고른다(`ENTAIL_DLC`, `ENTAIL_NODES`).

### 실행을 본다: `entail serve`

```bash
ENTAIL=load python your_app.py        # (또는 vllm serve ..., ComfyUI ...) 전처럼 entail_logs/에 쓴다
entail serve --open                   # http://127.0.0.1:8765/ - 이 컴퓨터에서만
```

- **화면:** 실행마다 작업 흐름을 노드로 보인다.
  - LLM: 모델 파일과 설정, 토크나이저, 가중치, 어텐션과 회전, 엔진 자기 점검, 요청, 캐시, 커널, 응답
  - 이미지: 체크포인트, 예측 방식, VAE, LoRA
  - 내 코드
- **상태:** 노드마다 정상, 바로잡음, 어긋남, 멈춤, 확인 불가, 건너뜀이 보인다. 확인 불가와 건너뜀(`unknown`, `unchecked`)은 어긋남과 다른 색으로 보인다.
- **알림과 상세:** 머리줄이 무슨 일이 있었는지와 처음 어긋난 노드를 쉬운 말로 알린다. 노드를 누르면 선언된 값과 엔진이 쓴 값이 나란히 나오고, 다른 항목이 표시된다. 규칙, 해소, 메모도 나온다. 기록에 새 줄이 오면 바로 바뀐다.
- **언어:** 화면은 한국어나 영어로 나온다. 브라우저 언어를 따르고, 위 막대에서 바꾼다.
- **비용:** 서버는 따로 된 프로세스에서 기록 파일을 읽으므로 엔진의 비용을 늘리지 않는다.
- **짚기의 측정:** 심은 결함(M7.3의 열한 건과 정상 한 건) 12건 가운데 12건에서, 화면이 프로그램이 실행 중에 짚은 곳과 같은 곳을 가리켰다. 9건은 기록만으로, 3건은 진단 모드가 남긴 `located` 줄로 짚었다.
- **안전:** 127.0.0.1에만 묶고, Host가 자기 주소가 아니면 거부한다.
  - 기록 폴더의 파일 둘만 쓴다: 다음 시작의 안전모드와, 꺼 둔 커스텀 노드다.
  - 쓰려면 서버가 시작할 때 찍고 자기 화면에 넣어 주는 토큰이 있어야 한다(쓸 때마다 바뀜). Origin도 자기여야 한다.

### 안전모드 둘

`ENTAIL_SAFE`로 고른다. 화면의 설정 탭은 다음 시작을 위해 `entail_logs/safe_mode.json`에 쓴다.

- **auto(기본): 선택적 안전 경로.**
  - 시작할 때 엔진 자신의 경로끼리 대조한다. vLLM에서는 디코드 대 새 프리필, 단독 대 묶음, 냉 대 프리픽스 캐시 적중이다.
  - 어긋나면, 같은 설정의 다음 시작부터 그 어긋남에 걸린 최적화를 한 시작에 하나씩 끈다.
  - 경로가 일치하면 그것이 원인이고, 계속 끈다. 다 꺼도 어긋나면 후보 밖이라고 broken으로 한 번 알린다.
  - 멈추지 않는다. 어긋남을 찾은 실행은 그대로 간다.
- **all: 명시적 안전모드.**
  - 엔진이 결과를 바꾸지 않는다고 선언한 최적화를 모두 끈다.
    - vLLM: CUDA 그래프와 torch.compile, 프리픽스 캐시, 추측 디코딩, 커스텀 커널(custom ops와 IR 연산의 커널 우선순위)
    - SGLang: CUDA 그래프, radix 캐시, 추측 디코딩
  - 결함이 남으면 원인은 그 밖이라고, 사라졌으면 그 안이었다고 말한다.
- **off:** 둘 다 쓰지 않는다.

vLLM 0.30에서 잰 것:

- **선택적 안전 경로:** 경로가 어긋났던 두 설정(Qwen3.5-4B-NVFP4와 Nemotron-3-Nano-4B에 n-gram 추측 디코딩)이 둘째 시작에서 일치를 되찾았고, 원인으로 추측 디코딩을 짚었다.
  - 처리량은 Nemotron에서 0.775배, Qwen3.5-NVFP4에서 1.318배였다. Qwen3.5-NVFP4는 eager에서 추측이 원래 손해였다.
- **심은 결함:** 처음에는 vLLM의 RMSNorm 커널 결함을 "밖"이라고 틀리게 갈랐다. 그 커널은 IR 연산의 우선순위가 고르는데, 표가 아직 그것을 끄지 않았다.
  - 표를 고친 뒤 3건 중 3건을 맞게 갈랐다.
  - 다른 모델의 사전 등록 재현에서는 경로 점검이 본 3건 중 3건이 맞았다. 1건은 점검에 보이지 않을 만큼 작았다.
- **비용:** 시작할 때 자기 점검이 그 모델들에서 1.9~7.7초(적재의 9~26%) 걸린다. `ENTAIL_NO_PATHS=1`로 끈다.

### 공식 DLC

엔진 전용 검사와 수리는 핵심 밖의 패키지다.

- **붙는 방식:** 진입점 무리 `entail.dlc`로 찾는다. 핵심이 제 어댑터처럼 설치하고, 실패하면 기록하고 프로그램은 이어진다.
- **첫 DLC:** `entail-dlc-comfyui`(`dlc/comfyui`)다. ComfyUI 자신의 결함 #16490을 고치는 수리이고, 1.3까지는 핵심에 있었다.
  - DLC를 붙이면 M6 측정의 그림 12장 가운데 12장이 1.3의 핵심 수리와 화소 단위로 같았다. 떼면 누수가 돌아왔다.
  - 정상 워크플로 둘에서는 그림을 바꾸지 않았고 아무 말도 하지 않았다.
- **고르기:** `ENTAIL_DLC=off`는 모든 DLC를 뗀다. `ENTAIL_DLC=이름,이름`은 그것만 붙이고, 나머지는 import하지도 않는다.
- **설치:** PyPI에는 없고, 이 저장소에서 설치한다.

```bash
pip install "git+https://github.com/wwoosshh/entail@v2.1.0#subdirectory=dlc/comfyui"
```

### 커스텀 노드

내 프로그램의 한 자리에 내 검사를 붙인다.

```python
import json
from entail import nodes

@nodes.validator("json_object")
def json_object(value, keys=()):
    try:
        obj = json.loads(value)
    except ValueError as e:
        return nodes.broken(f"not JSON ({e})")
    missing = [k for k in keys if k not in obj]
    return nodes.broken(f"missing keys {missing}") if missing else nodes.ok()

@nodes.watch("app.answer", json_object, keys=("title", "body"))   # ask()가 돌려준 것을 검사
def ask(question): ...
```

- **판정과 화면:** 검증기가 찾은 것은 entail의 판정이다. 기록되고, 화면에 노드 "app.answer"로 보이며, 화면에서 그 노드를 끌 수 있다.
- **비용:** 검사 한 번에 약 5 µs다.
- **실행을 깨지 않음:** 검증기가 예외를 내거나, 느리거나(기본 50 ms, `budget_ms=`), 돌아오지 않으면(`hard=True`면 예산에서 기다림을 멈춤) 기록하고 뗀다. 내 프로그램을 깨지 않는다.
- **패키지와 예제:** 검증기 묶음은 진입점 무리 `entail.nodes`로 붙는다. `entail-nodes-basics`(`workshop/basics`)에 `json_object`, `max_chars`, `within_context`, `same_size`가 있다. `examples/custom_nodes`에 작은 앱 셋이 있다. 이 패키지도 PyPI에는 없다.

```bash
pip install "git+https://github.com/wwoosshh/entail@v2.1.0#subdirectory=workshop/basics"
```

entail은 DLC와 검증기를 격리하지 않는다. 그것들은 내 프로그램의 프로세스에서 그 권한으로 도는 코드다. entail이 막는 것은 실행을 깨는 것과 핵심의 규칙을 바꾸는 것이다. 무엇을 돌릴지는 무엇을 설치하고 목록에 올리느냐로 내가 정한다.

## 하는 일

**해소하는 것**

| 어긋남 | 해소 | 잰 결과 |
|---|---|---|
| 설정이 만들어진 뒤 transformers 4 이름으로 준 RoPE 값(`rope_theta`, `rope_scaling`). `from_pretrained` 인자, 속성, vLLM `--hf-overrides`, SGLang `--json-model-override-args` 모두 해당한다 | `config.json`이 넣었을 자리에 쓴다. 층 종류마다 다른 RoPE도 설정 클래스에게 물어 같은 자리에 쓴다 | 모델 4종 × 경로 2 × 값 3에서 `config.json` 경로와 같음. GSM8K가 원래로 돌아옴(위 표) |
| 모델이 선언한 성질을 버리는 어텐션 백엔드. 예: transformers `sdpa`와 SGLang `flashinfer`가 Gemma 2의 logit soft-capping을 버림 | 지킨다고 측정된 백엔드(`eager`, `triton`)로 바꾼다 | 기준 실행과 토큰이 같음. 비용은 transformers 1.18배, SGLang 1.13배이고, 이는 바뀐 백엔드 자체의 값이다 |
| **ComfyUI, diffusers:** 붙인 모델에 닿지 못하거나 일부만 닿는 LoRA(예: SDXL 워크플로에 Anima LoRA). ComfyUI는 모듈마다 콘솔에 한 줄을 남기고 건너뛰며, 실행은 "성공"으로 끝나지만 LoRA는 아무것도 하지 않는다 | 바꿀 방법이 없으므로 오류로 알리고 실행은 잇는다. 알림에는 모듈 가운데 몇 개가 모델에 닿는지와 LoRA가 선언한 학습 기반이 들어간다(`ENTAIL_ON_BROKEN=stop`이면 샘플링 전에 멈춘다) | ComfyUI 0.34.1: SDXL 모델에 붙인 Anima LoRA는 그림을 LoRA 없는 그림과 화소 단위로 같게 남겼다. entail은 LoRA를 붙이는 자리에서 알렸고, `ENTAIL_ON_BROKEN=stop`이면 샘플링 전에 멈췄다(3/3). 맞는 LoRA(그림을 15~31/255 바꿈)는 통과했고, entail을 켜고 끈 그림이 같았다. diffusers 0.40: 키를 읽지 못하는 LoRA가 적재된 뒤 아무것도 하지 않았다(그림이 화소 단위로 같음). 같은 방식으로 알렸다. 앞서 0.3.0의 검사로는 맞는 조합 22개가 오탐 없이 통과했다 |
| **ComfyUI, diffusers:** 엔진이 읽지 않는 방식으로 v-prediction을 선언한 체크포인트. ComfyUI는 `v_pred` 키만 읽어서, 메타데이터에 `modelspec.prediction_type = v`를 적은 체크포인트를 eps로 돌린다. diffusers의 단일 파일 적재는 둘 다 읽지 않고 epsilon으로 둔다. 실행은 "성공"이지만 그림은 망가진다 | 파일 자신의 선언(메타데이터, 표지 키)이나 고정한 선언 파일이 정한다. 샘플러를 그대로 다시 꾸린다. ModelSamplingDiscrete 노드나 다시 만든 스케줄러가 하는 것과 같다. 선언이 말하지 않는 것(zero-terminal SNR)은 엔진의 값을 그대로 둔다. diffusers에서는 파이프라인이 샘플링할 때, 그때 쓰는 스케줄러를 두고 정한다(이미 선언대로 맞춘 프로그램은 건드리지 않는다). 모델을 자기 샘플링 루프에서 돌리는 프로그램에는 알리기만 하고 바꾸지 않는다. 사용자가 넣은 샘플링 노드나 스케줄러는 덮어쓰지 않고, 어긋남을 알린다. 아무것도 선언하지 않은 체크포인트는 "모름"으로 알린다. 모델의 동작은 바꾸는 근거가 되지 않으므로, 표지를 잃은 체크포인트는 선언 파일이 필요하다 | ComfyUI 0.34.1, AstolfoCarmix-VPredXL(메타데이터에 v를 선언, 표지 키 없음): entail 없이는 작성자의 기준 설정과 83~95/255 달랐고, entail을 켜면 시드 셋에서 같음, 0.16, 0.14/255였다. diffusers 0.40 단일 파일: NoobAI-XL-Vpred는 entail 없이 기준과 55~83/255 달랐고 켜면 화소 단위로 같았다. AstolfoCarmix는 90~95/255에서 화소 단위로 같아졌다. entail 0.3.0은 첫 모델 호출로 판정해서 AstolfoCarmix를 놓쳤다(가장 잡음이 큰 단계에서 eps처럼 동작함). 표지를 뺀 체크포인트는 이제 선언 파일이 없으면 "모름"으로 알린다 |
| **ComfyUI:** 워크플로가 끝난 뒤에도 남는 샘플링 노드의 설정. ComfyUI의 동적 VRAM 로더는 모델 버퍼를 속성 경로 이름으로 백업한다. 그래서 ModelSamplingDiscrete 같은 노드로 한 번 돌리면, 노드를 뺀 뒤에도 체크포인트가 그 노드의 일정으로 샘플링된다. 거꾸로, 노드 없이 먼저 돌린 뒤에 쓴 노드는 조용히 원래 일정을 받는다 | 샘플링 객체마다 자기 setter가 등록한 일정의 사본을 둔다. 로더의 백업은 다른 객체에 넣지 않고 원래 객체에 돌려준다. 버퍼가 바뀐 뒤 첫 모델 호출에서 사본과 대조하고, 다르면 되돌린다 | ComfyUI 0.34.1에서 waiIllustrious에 ModelSamplingDiscrete(v_prediction, zsnr) 노드로 한 번 돌리면, 이후 노드 없는 실행이 다른 그림(55.8/255)을 거쳐 검은 화면이 됐다. entail을 켜든 끄든 재시작 전까지 그랬다. 고친 뒤에는 새 세션과 화소 단위로 같았다(3/3). NoobAI-XL-Vpred를 그대로 돌린 것과 zsnr=false 노드를 건 것을 두 순서로 돌리면, entail이 없을 때 뒤에 돈 쪽이 앞의 설정을 화소 단위로 그대로 받았다. entail을 켜면 12장 모두 새 세션과 같았다. 첫 호출 확인만으로는(가드를 뺀 경우) 검은 화면은 막았지만 2~11/255가 남았다. Anima 워크플로와 다른 실행은 켜고 끈 그림이 같았고 속도도 같았다 |

| **vLLM 서버:** 모델이 선언한 형식을 읽지 못하는 도구 호출 파서(hermes 형식으로 부르는 Qwen3 모델을 `--tool-call-parser pythonic`으로 띄움). 도구 호출이 본문 텍스트로 돌아온다 | 선언된 형식을 읽는다고 측정된 파서로 바꾼다 | vLLM 0.30.0, Qwen3-4B: 도구 호출이 다시 구조화된 호출로 돌아옴 |
| **diffusers:** 따로 읽은 VAE가 다른 모델의 잠재 배율을 받음(SDXL VAE를 SD1.5의 것으로 읽음) | 선언 파일이 그 모델에 선언한 배율을 쓴다 | entail 없이는 기준 그림과 15~16/255 달랐고, 켜면 화소 단위로 같았다(시드 셋) |
| **vLLM:** 만들어진 토큰과 더는 맞지 않는 접두사 캐시 블록 해시. 스트리밍 세션 갱신이 요청의 토큰을 잘라도 블록 해시는 덧붙이기만 해서, 버린 토큰까지 엮인 해시가 살아남고, 옛 토큰과 맞는 뒤 요청이 새 토큰의 KV를 받는다(vllm#49377, #49449; 0.30.0에 살아 있음) | 낡은 첫 블록부터 해시를 잊고 엔진이 현재 토큰으로 다시 만들게 한다 | vLLM 0.30.0, SmolLM2-135M-Instruct: entail 없이는 다시 만든 세션이 16토큰짜리 거짓 캐시 적중과 틀린 이어짓기를 받았고, 켜면 갱신 자리에서 잡아 다시 계산해 맞는 출력이 나온다 |
| **SGLang:** K 타일이 가중치 양자화 블록의 약수가 아닌 블록 FP8 커널. 스케일이 타일마다 한 번 움직여 블록을 건너뛴다(손으로 준 설정, sglang#39626; E=512, N=256 H100 fused-MoE 배포 설정 하나도 블록 128에 BLOCK_SIZE_K 256) | 타일을 블록으로 묶는다. 엔진의 기본값과 같다 | SGLang 0.5.20: 밀집 커널이 288이 맞는 자리에서 64를 냈고 묶으면 288; 배포된 MoE 설정은 커널 수준에서 512가 맞는 자리에서 256, 묶으면 512. 나머지 배포 블록 FP8 항목 1,538개는 모두 나누어떨어져 보통 실행은 아무것도 결정하지 않는다 |
| 엔진이 정지 id를 읽는 파일이 그 id를 선언한 파일이 아니어서 답이 끝난 뒤에도 이어지는 생성. generation_config.json, config.json, 토크나이저가 저마다 끝을 선언하는데 transformers는 첫째만, vLLM은 첫째와 토크나이저, SGLang은 앞의 둘을 읽는다(2024년 4월 Llama 3의 모양: config.json은 한 끝을 말했고 모델은 다른 끝을 냈다) | 다른 파일이 선언한 id를 적재 때 엔진의 정지 집합에 더한다 | transformers 5.17, generation_config.json에 `<\|end_of_text\|>`만 적은 Llama-3.2-3B-Instruct: entail 없이는 시험 답 셋이 모두 `<\|eot_id\|>`를 지나 160토큰 한도까지 달렸고, 켜면 config.json이 선언한 끝을 적재 때 더해 8, 18, 37토큰에서 멈췄다 |
| LoRA 어댑터가 잘못된 스케일로 서빙되는 것. `adapter_config.json`이 선언한 설정을 엔진이 읽지 않아서다(`use_rslora`: PEFT는 `lora_alpha / sqrt(r)`로, SGLang은 언제나 `lora_alpha / r`로 스케일하므로 r=16이면 4배, r=64면 8배 약해진다; 같은 파일의 `rank_pattern`, `alpha_pattern`, `lora_bias`, `modules_to_save`도 vLLM과 SGLang이 조용히 버린다) | 파일의 모든 키를 엔진마다 읽는 키의 표(코드 줄 근거)와 대조한다. 버려진 키는 알리고, 엔진에 자리가 있으면 나른다: SGLang의 어댑터 스케일을 PEFT가 쓸 값으로 놓는다 | SGLang 0.5.20, sglang#40835의 스크립트(PEFT rsLoRA 어댑터, r=64 alpha=128): entail 없이는 PEFT가 16.0인 자리에 SGLang이 2.0을 들었고, 켜면 16.0. Qwen2.5-3B-Instruct에 PEFT 어댑터(r=16 alpha=32)를 얹은 실제 Engine: 8.0으로 해소. `use_rslora` 없는 같은 어댑터와 vLLM 0.30의 둘: 판정 없이 통과 |
| 요청의 `chat_template_kwargs` 설정을 채팅 템플릿은 한 이름으로 받고 추론 파서는 다른 이름으로 읽는 것: vLLM 0.2x의 Kimi K2에 `{"enable_thinking": false}` — 템플릿은 생각 토큰을 넣지 않고, 파서는 `thinking`을 읽어 기본값(켬)으로 돌아 `content: null`과 `reasoning` 아래의 답을 돌려준다(vllm#43728) | 추론 파서마다 읽는 이름을 vLLM 판별로 적은 표(코드 줄 근거)를 요청의 이름과 템플릿의 변수에 대조하고, 요청의 값을 파서가 읽는 이름으로 넘긴다 | 표의 0.22.0 행으로 규칙을 돌림: 해소(`enable_thinking`에서 `thinking`을 놓음); vLLM 0.30.0은 Kimi K2에서 두 이름을 다 읽으므로 같은 요청이 통과. 회고이지 검출이 아님 |
| 캐시된 블록을 만든 필드가 접두 캐시 키에 빠진 것: vLLM 0.30의 블록 해시는 토큰, 프롬프트 임베딩의 digest, 멀티모달 해시, LoRA 이름, cache salt를 담지만 `prompt_is_token_ids`(임베딩을 쓰는 위치)는 담지 않아, 그 마스크만 다른 요청이 앞선 요청의 KV를 받는다(vllm#56655, 수정 미병합) | 요청의 입력 필드를 해시가 읽는 필드의 표(코드 줄 근거)에 대조하고, 빠진 필드의 블록별 digest를 해시의 extra key에 더한 뒤 요청의 해시를 다시 만든다 | vLLM 0.30.0, Qwen3-0.6B, 보고의 스크립트: entail 없이는 B after A가 캐시 32토큰을 적중해 A의 출력을 냈고, 켜면 0토큰 적중에 자신의 출력, A after A는 여전히 32토큰 적중. 회고 |
| 이름을 아는 캐시만 재정렬하는 빔 서치: transformers 5.12.1은 `past_key_values`만 재정렬해 Mamba의 `cache_params`가 그대로 남고 빔이 다른 빔의 상태에서 이어졌다(transformers#46612) | 모델 forward의 캐시 인자 이름을 재정렬이 만지는 이름(transformers 판별)에 대조 | 5.12.1: 빔 서치 경계에서 보고(해소 없음, 출력은 그대로); 5.17.0은 모든 이름을 재정렬: 통과. 회고 |
| 안쪽 차원이 strided인 텐서를 연속인 줄 알고 읽는 Triton 커널: SGLang의 `fused_gdn_gating`이 나눈 뷰의 두 반쪽(stride 2)을 받아 섞인 값으로 게이트를 계산했다(sglang#21843) | 프로세스에서 eager로 호출되는 모든 `@triton.jit` 커널을 커널·stride 패턴마다 한 번(커널마다 처음 strided 패턴 여덟까지), 엔진과 무관하게: strided인 안쪽 차원을 커널 자신의 인자에 대조한다. 어떤 이름으로든 그 stride를 받은 커널은 통과; stride류 이름은 있지만 이 값을 받지 않은 커널은 `unknown`, 한 번; stride류 인자도 같은 정수도 없는 커널은 알린다(`broken`, 알 길이 없음) | SGLang 0.5.20, sglang#21843의 텐서: 커널이 행 stride만 받으므로 entail은 커널 경계에서 텐서마다 `unknown`을 말하고 실행은 이어진다. 보통 실행의 vLLM 0.30 커널 19개와 SGLang 커널 5개: 통과. 회고, 알리기만 |
| 차원을 잘못 짝지어 적용된 회전 임베딩: GLM 계열은 `2i`와 `2i+1`을 짝짓고(interleaved) Llama는 `i`와 `i + d/2`를 짝짓는데(split), vLLM의 Triton MRoPE 커널은 0.27 전까지 층이 무엇을 들든 split로 짝지어 GLM-OCR이 쓰레기를 냈다(vllm#42016; #49290, 목표 모델의 짝짓기를 물려받지 않은 초안 모델 #53063도) | config 키나 아키텍처의 참조 구현이 선언한 짝짓기(config·모델링 파일 줄 근거의 표)를 vLLM이 언어 모델에 만든 회전 모듈의 `is_neox_style`과 대조하고, 다르면 선언대로 놓는다(이 해소는 단위 시험에서만 돌려 봤다). 층을 무시하는 커널 경로(0.27 전 vLLM의 MRoPE 커널, 모듈이 그리로 디스패치할 때)는 알린다. 자기 키로 짝짓는 DSA indexer와 두 관례를 다 든 언어 모델은 손대지 않는다 | vLLM 0.30.0: GLM-OCR의 텍스트 회전 모듈 둘은 선언대로 interleaved, 비전 타워의 26개는 제 참조대로 split: 통과, 손댄 것 없음, 출력은 entail 켜고 끄고 같음; Qwen3-0.6B와 Qwen3-4B(split): 통과. vLLM 0.22.0(커널 수정 전), 보고의 이미지 그대로: GLM-OCR이 쓰레기를 내고 entail은 적재 경계에서 커널 경로를 짚어 `broken`을 알린다. 거기엔 해소가 없어 출력은 틀린 채 실행이 이어진다. 회고; 첫 실제 실행은 비전 타워를 텍스트 선언에 대조했고 이 행을 쓰기 전에 고쳤다 |
| 맞는 파일에서 틀린 클래스로 만들어져, 내는 id가 폴더의 tokenizer.json의 id와 다른 토크나이저: transformers 5.10.2는 deepseek-coder의 토크나이저를 `LlamaTokenizer`로 만들었고(transformers#46489), 5.8.0은 Granite를 `GPT2Tokenizer`로 만들어 pre-tokenizer를 잃었고(#45812), 5.4.0은 Kimi-K2.5의 tiktoken 토크나이저를 변환하며 선언된 추가 토큰 23개 중 18개에 다른 id를 주었고(`</think>`가 `<\|media_end\|>`의 id를 받음; #45356), 5.12.1은 DeepSeek-R1-Distill이 선언한 클래스를 바꿔치웠다(#46710). 넷 다 어휘 크기 검사는 통과했다 | 토크나이저 적재 때 알린다: 선언된 토크나이저를 고정 탐침 10개에 실제로 돌리고 선언된 추가 토큰의 id를 찾아, id가 같아야 한다(고침 없음: 결정이 클래스와 처음 다른 텍스트를 적으므로 선언된 토크나이저를 직접 적재할 수 있다). 폴더 자신의 플래그가 설명하는 차이(legacy 내보내기 tokenizer.json 옆의 `legacy: false`, 특수 토큰 뒤의 텍스트)는 선언끼리 어긋난 것으로 두 id 목록과 함께 말한다 | 회고(규칙은 이 네 버그에서 썼다; 검출률이 아니다): 보고된 판에서 넷 다 토크나이저 경계에서 `broken`(탐침 10개 중 9·4·9개, 추가 토큰 23개 중 18개); 5.17.0에서는 셋이 pass, Kimi-K2.5는 `unknown`(tiktoken: 추가 토큰 23개는 일치, 텍스트는 비교 안 함). 인기 폴더 38개(5.17.0; 서로 다른 토크나이저 11개, 23개는 Qwen2의 것): pass 36, `broken` 2 — 둘 다 그 집합의 유일한 Llama-2 시절 토크나이저(TinyLlama-1.1B-Chat과 작은 시험 폴더)로, transformers 5는 그런 tokenizer.json을 Metaspace로 다시 만들며 공백으로 시작하는 텍스트 앞의 `▁`를 `legacy`와 무관하게 겹치지 않아 `"   leading spaces"`가 파일과도 sentencepiece와도 다른 id를 받는다. 정적 폴더 300개(5.17.0): pass 201, `broken` 9, `unknown` 5, 비교할 토크나이저 없음 85. 9건은 같은 Llama-2 모양 여섯(CodeLlama-7b-hf, EuroLLM-22B-Instruct, MiniCPM-SALA, ...)과, 바이트 수준 BPE tokenizer.json 위에 `LlamaTokenizerFast`를 선언한 폴더 셋 — **DeepSeek-R1-0528-Qwen3-8B**, deepseek-coder-7b-instruct-v1.5, MLX 내보내기 하나 — 로, 5.17.0이 Llama 파이프라인으로 만들어 `"How are you doing?"`이 `How are y oud o ing ?`가 되고 `Howareyoudoing?`으로 복원된다(#46710의 부류가 이 폴더들에서는 5.17.0에 살아 있음; 초안은 연구 작업공간에, 게시 안 함). 비용: 폴더의 첫 프로세스 중앙값 +241 ms(최대 +894 ms: 참조를 만든다), 다음 프로세스 중앙값 +2 ms, 90번째 백분위 +36 ms |
| 연산 자신의 정의와 다른 값을 내는 커스텀 연산의 커널: vLLM 0.27 전의 Triton MRoPE 커널은 interleaved로 짝짓는 모델(GLM-OCR, vllm#42016)의 회전 차원을 split로 짝지었고, 그 층은 맞는 `forward_native`와 틀린 `forward_cuda`를 함께 들고 있었다 | 엔진의 워밍업 호출로 판정하고(엔진 자신의 모양에 지어낸 토큰 값, 크기 부류마다 한 번), 그렇지 못하면 연산마다(클래스·구성·입력 패턴) 첫 실제 호출에서 판정한다: 커널이 인자를 손대기 전에 자른 같은 입력의 64행 슬라이스로 커널과 정의를 돌려, 값마다 정의 자신의 그 값의 반올림 잡음 + ulp 몇 개 안에서 같아야 한다. 어긋나면 그 구성의 모든 모듈을 연산 자신의 정의로 보내 고친다. 단 CUDA 그래프가 아직 확인하지 않은 크기에서 커널을 잡지 않았을 때만이고, 그 밖에는 알린다. 엔진 상태를 든 연산, 텐서 병렬·torch.compile 아래의 연산, 모듈에서 닿지 않는 연산은 한 번 `unknown`으로 말한다 | vLLM 0.22.0, GLM-OCR: `MRotaryEmbedding`에서 커널이 정의와 다름(출력 크기 10.9에 최대 차이 10.5, 허용 0.588), 아키텍처 표 없이; eager 모드에서는 해소되고 출력이 native 모드의 것과 같아진다; vLLM 0.30.0: pass. 인기 모델 8개, vLLM 0.30.0 `enforce_eager`: 결정 35건, pass 34·`unknown` 1(모듈이 아닌 도우미가 든 연산), 모든 값이 허용치의 1/10 안; 34건 중 독립 커널을 비교한 것은 13건(rotary, 활성 함수)이고 나머지 21건은 정의를 정의와 비교한 것으로 그렇게 적힌다(eager 모드에서 vLLM의 RMSNorm 경로는 정의 자체이고 융합 커널은 torch.compile 아래라 비교하지 않는다). 회고: 규칙은 이 버그에서 썼다; 허용치의 배수는 여유값이고 데이터로 정할 것이다 |
| 스트리밍으로 낸 메시지가 같은 전체 텍스트를 푼 것과 다르거나, 도구 호출의 인자가 요청의 도구 스키마에 없는 채팅 파서: vLLM의 kimi_k2 파서는 스트리밍에서 스키마의 형 강제를 건너뛰었고(`"3"` 대 `3`, vllm#49316), qwen3 파서는 두 길에서 도구 호출 주변 content가 달랐고(#49412), deepseek_v4 파서는 한 도구의 인자를 다른 도구의 스키마로 풀었다(#47986) | 스트림이 끝날 때 알린다: 델타를 모으고 같은 클래스의 새 파서로 전체 텍스트를 풀어 둘이 정확히 같아야 하며(스스로 끝나지 않은 출력의 차이는 `unknown`), 도구 호출마다 선언된 도구를 부르고 필수 파라미터를 들고 추가 키를 금한 도구에는 다른 키가 없어야 한다(고침 없음: 클라이언트는 이미 스트림을 받았다) | vLLM 0.30.0의 파서를 서버가 모는 대로 몰아: 세 보고의 텍스트 4/4, 2/3(셋째는 양끝 공백만 달라 메모로 남김), 1/1에서 `broken`, 정상 텍스트 6개에서는 없음; 모든 어댑터를 켠 실제 vLLM 서버(Qwen3-0.6B, qwen3 추론 파서, hermes 도구 파서, 스트리밍·비스트리밍 18 요청: 평문, thinking on/off, 도구 호출, `tool_choice: none`, n=2, 길이 잘림)에서 broken 없음. 회고: 규칙은 이 버그들에서 썼다 |
| 모델이 선언한 마크업 밖에 묶인 멀티모달 자리표: 사용자 메시지에 친 `<\|image_pad\|>`가 자리표 id로 토큰화되고 vLLM이 첫 구간에 이미지를 묶어, 템플릿의 `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` 자리에는 pad 하나만 남고 모델이 엉뚱한 구간에 답한다(Qwen2.5-VL, vllm#57740) | 프로세서가 자리표를 놓은 뒤 알린다: 이미지 구간 바로 앞 토큰이 모델 설정이 선언한 `vision_start_token_id`여야 한다(마크업을 선언하지 않는 모델은 판정하지 않음; 고침 없음: 프롬프트는 사용자의 것) | vLLM 0.30.0, Qwen2.5-VL-3B-Instruct, 보고의 두 메시지 순서: 공격 순서 `broken`("the image placeholder bound at tokens 20..275 is preceded by id 220, not by the declared start of the markup"), 대조 순서 pass; vLLM의 프로파일링 프롬프트는 판정하지 않는다. 회고 |
| logprobs가 content가 아닌 텍스트를 덮는 채팅 응답: SGLang은 `separate_reasoning`일 때 `logprobs.content`를 `<think>` 구간과 마커까지 포함한 원문 전체에 걸쳐 돌려주고 `message.content`에는 푼 답만 두어 둘을 맞출 수 없다(sglang#25055) | 응답을 만들 때 알린다: logprob 토큰을 이어 붙인 것이 메시지의 content(양끝 공백 제외)여야 한다 | SGLang 0.5.20, Qwen3-0.6B, qwen3 추론 파서, `logprobs: true, separate_reasoning: true`: `broken`("the 155 logprob tokens cover the reasoning span (545 characters and its markers) as well as the content (12 characters)"). 회고 |

**검사하는 것** (해소할 수 없으면 오류로 알린다)

- 어텐션 성질을 엔진 백엔드와 대조한다. 가중치를 읽기 전에 한다.
- 삼켜질 설정 키가 있는지, 묶인 임베딩 선언이 체크포인트와 맞는지 본다.
- vLLM이 재포장한 가중치의 배치, stride, 값 표본을 선언된 변환과 대조한다.
- 메모리의 가중치를 체크포인트 파일과 대조한다(`ENTAIL_SOURCE=1`).
- KV 캐시 계약을 본다. 요청이 필요한 만큼 가지고 있는지, 줄어든 것이 없는지다. transformers, vLLM 페이지 캐시, SGLang에서 돈다.
- 이미지 모델(ComfyUI, diffusers): 체크포인트, 폴더, 선언 파일이 선언한 예측 방식과 잠재 배율을 샘플러와 VAE에 대조하고, LoRA의 모듈이 붙인 모델에 닿는지 본다.
- vLLM의 OpenAI 서버에서 요청마다: 선언과 다른 채팅 템플릿, 모델이 유지한다고 선언한 사고 기록을 뺀 요청, 아무도 읽지 않는 요청 필드와 템플릿 설정을 본다.
- transformers가 채팅 템플릿을 적용하는 자리(스크립트의 `apply_chat_template`, SGLang 서버)에서도 템플릿과 사고 기록을 같은 방식으로 보고, SGLang 자신의 대화 템플릿(`--chat-template chatml`)도 본다.
- 엔진이 만든 토크나이저를 모델의 어휘와 대조한다. 임베딩 행을 넘는 id, 그리고 폴더가 두 어휘를 들었는데(100,000짜리 vocab.txt 옆의 32,000짜리 tokenizer.json; transformers#48967) 엔진이 모델의 것이 아닌 쪽을 만든 경우다. 토크나이저 적재 때 알리고, `ENTAIL_ON_BROKEN=stop`이면 첫 id 전에 거부하며, `entail check`는 exit 1이다.
- 엔진이 만든 토크나이저를 폴더가 선언한 토크나이저를 실제로 돌려 대조한다. 고정 탐침 10개를 양쪽으로 인코딩하고(tokenizer.json은 `tokenizers` 라이브러리로; sentencepiece만 있는 폴더는 그 참조를 잴 때까지 `unknown`) 선언된 추가 토큰마다 id를 찾아 정확히 같아야 한다. 제 pre-tokenizer로 토크나이저를 다시 만드는 클래스는 적재 때 알리고, `ENTAIL_ON_BROKEN=stop`이면 첫 id 전에 거부하며, `entail check`는 exit 1이다. 돌릴 수 없는 선언은 한 번 `unknown`으로 말하고, 폴더 자신의 플래그가 설명하는 차이(legacy 모드 tokenizer.json 옆의 `legacy: false`, 특수 토큰 뒤의 텍스트)는 선언끼리 어긋난 것으로 두 id 목록과 함께 말한다.
- vLLM에서 커널로 디스패치되는 모든 커스텀 연산을 연산 자신의 정의와 대조한다. 연산 클래스·구성·입력 패턴마다 한 번, vLLM 자신의 워밍업 호출을 탐침으로 써서(지어낸 토큰 값, 엔진의 모양, 크기 부류마다 한 번) 또는 첫 실제 입력의 64행 슬라이스로 한다. 어긋나면 CUDA 그래프가 커널을 잡지 않았을 때 정의로 보낸다(정의는 입력 dtype과 float32로 돌리고 연산의 텐서는 되돌리며, 커널 출력의 값마다 정의 자신의 그 값의 반올림 잡음 + ulp 몇 개 안이어야 한다). forward를 덮어쓴 연산과 엔진 상태를 든 연산, 텐서 병렬·torch.compile 아래의 모든 연산, 모듈에서 닿지 않는 연산, CUDA 그래프 캡처 중의 호출, 입력을 거부하는 정의는 비교하지 않고 한 번씩 말한다.
- vLLM 채팅 서버에서 파서가 스트리밍으로 낸 메시지를 스트림이 끝날 때 같은 전체 텍스트를 푼 것과 대조하고(토큰 한도에 닿았거나 reasoning 블록이 열린 채 끝난 출력의 차이는 broken이 아니라 `unknown`), 도구 호출마다 인자를 요청이 선언한 도구와 대조한다(필수 파라미터가 빠짐, 추가 키를 금한 도구에 다른 키).
- vLLM에서 멀티모달 항목의 자리표가 묶인 자리를 모델 설정이 선언한 마크업(`vision_start_token_id` 뒤의 이미지 토큰)과 대조하고, SGLang 채팅 서버에서 응답의 logprobs를 content와 대조한다.
- vLLM의 점수 경로에서 cross-encoder의 패딩이 받은 토큰 종류를 토크나이저가 선언한 패딩 종류와 대조한다(vllm#58138: 패딩이 문서의 세그먼트를 받아 /rerank 점수가 움직였다). vLLM 0.30은 토큰 종류를 해소를 담을 수 없는 형태로 들어서 알리기만 하고, `ENTAIL_ON_BROKEN=stop`이면 거부한다.
- 정적으로, 엔진마다 LoRA 어댑터 폴더의 `adapter_config.json`을 그 엔진이 읽는 키와 대조한다(`entail check <어댑터 폴더>`: 버려진 키는 알리고, 엔진의 어댑터가 적재 때 나르는 키는 해소로 말한다).
- 정적으로, 엔진마다 모델 폴더에서 만들 정지 집합을 파일들이 선언한 모든 끝과 대조한다(`entail check`).
- 프로세스의 모든 Triton 커널 호출을 엔진과 무관하게, 커널·텐서 배치마다 한 번: 안쪽 차원이 strided인 텐서를 커널 자신의 인자 이름과 대조한다(stride 인자가 없는 커널은 알 길이 없으므로 알린다; stride를 받지만 이 값을 받지 않은 커널은 `unknown`, 한 번). 그런 호출은 사본으로 두 번 더 돌린다. 받은 그대로와, 그 텐서를 다시 배치한 것이다. 커널이 다른 값을 쓰면 그 배치 패턴은 그 뒤로 다시 배치한 사본으로 부른다(해소). 컴파일만 하는 워밍업은 보지 않고, 커널마다 처음 여덟 배치까지만 본다.
- 자기 정의가 없는 엔진 함수 다섯을 entail의 정의와 대조한다: vLLM의 `fused_experts`, `w8a8_triton_block_scaled_mm`, `prepare_pos_seq_lens`, `BlockTables.compute_slot_mappings`(위치와 KV 슬롯은 정확히 같아야 한다), SGLang의 `fused_gdn_gating`. 어긋나면 정의로 보낸다.
- 시작 때 고정 탐침 요청 셋으로 엔진 자신의 경로끼리 대조한다: 같은 토큰을 새로 prefill한 것과 decode, 혼자 보낸 요청과 같은 요청을 묶어 보낸 것, 처음 돌린 것과 접두 캐시를 읽은 것(확신하던 예측이 바뀌거나 확률이 0.25 넘게 움직이면 알린다). vLLM은 기본으로 켜져 있고(`ENTAIL_NO_PATHS=1`로 끈다), SGLang은 `ENTAIL_PATHS=1`일 때만 돈다. SGLang은 시작 때 prefill을 한 번도 돌리지 않아서, 탐침이 엔진 결함을 처음 만나는 요청이 될 수 있기 때문이다.
- 모델의 회전 임베딩이 차원을 짝짓는 방식(split: `i`와 `i + d/2`; interleaved: `2i`와 `2i+1`)을 config 키나 아키텍처의 참조 구현이 선언한 대로, vLLM이 언어 모델에 만든 층과 대조하고, 층이 무엇을 들든 split로 짝짓는 커널 경로(0.27 전 vLLM의 Triton MRoPE 커널: GLM-OCR이 쓰레기를 냈다, vllm#42016)와도 대조한다.
- 디버그 모드에서는 사용자가 자기 코드에 선언한 경계(`@entail.boundary`: 인자마다의 뜻)를 본다. strided 배치, 양자화된 값, 청크 상대 위치는 읽는 쪽이 필요한 형태로 바꿔 넘긴다.

### 그래도 출력이 틀리면

entail은 검사한 경계마다 판정을 `entail_logs/`에 남긴다. `entail locate`는 이 기록을 읽어 의미가 깨진 곳, 곧 의미를 지키지 못한 첫 경계를 짚는다. 검사한 모든 경계가 온전한데 출력이 틀렸다면(`entail locate --wrong`), 문제는 계층 사이에서 넘긴 것이 아니라 계층 안쪽에 있다. 모델 자체, 컴파일러, 커널, 하드웨어가 여기에 든다. entail이 검사하지 못한 경계는 양옆 계층과 함께 의심 구간으로 남는다.

계층까지 좁히려면 디버그 모드에서 돌리면서 계층을 같은 입력으로 참조 구현과 비교한다. 층마다 첫 호출을 비교하고(`calls=`로 더 늘린다), 참조가 출력을 재현하지 못하는 계층을 짚는다.

```python
import torch, entail
from entail import diagnose
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

def mlp_in_float32(self, x):   # 참조: 같은 MLP를 float32로 계산한다
    f = torch.nn.functional
    gate, up = (f.linear(x.float(), p.weight.float()) for p in (self.gate_proj, self.up_proj))
    return f.linear(f.silu(gate) * up, self.down_proj.weight.float()).to(x.dtype)

entail.enable("debug")         # 모델을 올리기 전에 켠다. 적재도 검사된다
with diagnose.propagating(), diagnose.watch(Qwen3MLP, "forward", mlp_in_float32, label="mlp"):
    model.generate(**inputs, max_new_tokens=8)
print("\n".join(entail.locate(output_wrong=True).lines()))
```

`diagnose.propagating()`은 두 경계 사이에서 선언된 사실을 무효로 만든 연산도 짚는다. 예를 들어 배치를 선언한 값을 transpose한 경우다. Qwen3-4B와 gemma-2-2b-it(transformers 5.17)에 결함을 심어 쟀다. 심은 곳은 적재·캐시·코드 경계, 어텐션과 MLP 커널의 안쪽, 검사하지 못한 경계 뒤였고, 11건 모두 심은 곳을 짚었다. 진단 비용은 64토큰 복호에서 1.72배(eager)와 1.90배(sdpa)였고, 토큰은 같았다.

시험에서는 `pytest --entail`이 시험마다 이렇게 돈다. 깨지면 시험이 실패하고, 실패한 시험의 보고가 그 곳을 짚는다. `entail_condition` 픽스처를 받고 `@pytest.mark.entail_conditions(model="...")`를 단 시험은 모델의 선언이 걸리는 조건마다 한 번씩 돈다. 슬라이딩 창의 한 토큰 아래·같음·위, 스케일된 RoPE가 넘겨받는 문맥 둘레, 이전 사고의 처리를 선언한 모델의 두 번째 차례가 그 예다.

### 새 코드에는: 역할 타입 프런트엔드 (실험)

`entail.frontend`는 처음부터 새로 쓰는 코드를 위한 것이다. 예를 들어 모델의 디코드 스텝이나 커널을 부르는 쪽이다. 값마다 타입이 있다. 이름 붙은 차원, dtype, 무엇인지(질의, 키, 값), 지닌 사실이 여기에 든다. 연산은 인자를 키워드로 받고, 프로그램은 입력의 타입으로 한 번 추적된다. 값 자리에 넘긴 키, 마지막 키 번호 자리에 쓴 길이, 어느 커널도 읽지 않는 형식의 가중치, 쓰기 전 판본으로 읽은 캐시, 두 번 합산한 합은 실행 전에 거부된다. 고칠 수 있는 것은 그때 고치고 알린다. 오프셋이 있는 청크 상대 위치, 스케일이 있는 양자화 값, 고른 커널이 무시하는 softcap이 그렇다.

```python
from entail.frontend import qwen3

program = qwen3.trace_decode(model.config, batch=8, slots=640, attention="triton")    # 모든 검사가 여기서 돈다
step = qwen3.bind(program, model, cache, tokens, positions, until)          # 적재 계약
logits = step()["logits"]                                                              # 남은 검사가 없다
```

이렇게 쓴 Qwen3-4B 디코드 스텝은 torch 낮춤에서 transformers와 로짓이 비트 단위로 같다. 컴파일해 CUDA 그래프로 잡으면(int4, 배치 8), 같은 커널로 손수 짠 스텝의 1.005~1.007배(손 커널), 1.020~1.024배(FlexAttention) 시간이 든다. 재현 사례 16건 가운데 9건은 추적 때 거부되거나 고쳐지고, 2건은 프로그램을 텐서에 묶을 때 거부된다. 수정 판이 거부된 것은 없다.

## 어떻게 쟀나

1.0을 내면서 개발 단계의 측정을 모두 최종 코드로 다시 쟀다. 장비는 RTX 4070 Ti 한 장이다.

- **정상 실행**(Qwen3-4B, Llama-3.2-3B-Instruct, gemma-2-2b-it을 transformers, vLLM, SGLang의 기본 설정으로): 오탐이 없었다. 해소는 두 번이었고, 둘 다 백엔드가 Gemma 2의 soft-capping을 버리는 자리였다. 모델 폴더가 선언한 사실은 쓰인 자리에서 모두 판정 자리에 닿았다. 채팅 템플릿도 포함된다.
- **더 넓은 정상 실행, 1.0.1을 위해**(12 GB 카드 한 장에 맞는 인기 모델 38개를 내려받기 순위로 골라 같은 세 엔진에서; 유효 실행 102회): 1.0.0은 처음 81회 가운데 17회에서 틀린 것을 알렸고, 그 가운데 실제 손실은 없었다. 원인 다섯은 변경 기록에 있다. 1.0.1은 `broken`도 `refused`도 없고, 해소는 둘(다시 Gemma 2의 soft-capping, 실측 행), 출력은 entail 없는 실행과 모두 같고, 적재 비용 중앙값 0.7~0.9%다. 판정하지 못하는 것은 `unknown`으로 말한다(81회에 69줄, 경계마다 한 줄).
- **실제 버그, 1.1.0을 위해:** 엔진 이슈 229건에서 무작위로 뽑아 재현한 8건 가운데 4건이 entail의 부류였고 1.0은 넷 다 통과시켰다. 1.1.0은 둘을 고치고(vLLM의 낡은 블록 해시, SGLang의 커널 타일) 둘을 그 경계에서 알린다(패딩의 토큰 종류, 두 번째 어휘). 이 넷은 1.1.0의 사실을 만든 출처인 버그들이므로 "4/4"는 검출률이 아니다. 부류 밖 셋(CUDA Graph 약한 참조, 파서의 스트리밍 논리, 스케줄러의 산술)은 설계대로 잡지 않는다. 다섯째 부류(정지 id)는 transformers에서 재현했다(위 표).
- **처음 보는 버그의 검출률(사전 등록, 어휘를 1.1.0에 동결):** 같은 네 저장소의 이슈 150건을 정해진 규칙으로 심사해 17건이 통과했고 15건이 여기서 재현됐다. 눈가림 평정자 둘이 그 15건 가운데 7건을 entail의 부류로 봤다(일곱 범주 카파 0.86, 부류 안팎 0.95). entail은 **그 7건 가운데 0건**을 잡았고, 부류 밖 8건에서 틀린 경보는 없었다. 못 잡은 7건은 모두 어휘 밖의 사실이었고, 그중 5건은 아직 어떤 어댑터도 보지 않는 자리(커널 호출, 요청 파서, LoRA 설정 파일, 접두 캐시 키, 빔 재정렬)에서 생겼다. 부류 주장은 이렇게 읽어야 한다. 위의 다섯 사실은 그것을 낳은 버그로 쟀고, 새 버그는 대개 entail이 아직 읽지 않는 사실이다. 규약·심사 기록·평정·사례 스크립트는 연구 작업공간(github.com/wwoosshh/entail-research)의 `testbed/M16_PROTOCOL.md`와 `testbed/results/m16/`에 있다. 인기 모델 38개를 세 엔진에서 다시(유효 102회): `broken` 0, `refused` 0, 같은 해소 둘에 더해 선언된 끝을 transformers의 정지 집합에 더한 해소 둘(배포된 그대로의 Nemotron-3-Nano-4B, 작은 시험 모델 하나), 해소 없는 실행의 출력은 98번 가운데 97번 entail 없는 실행과 같음(다른 하나는 엔진 자신의 비결정성), `unknown` 줄 69, 라이브러리의 적재 시간 비중 중앙값 1.4%, 90번째 백분위 8.2%(위의 여섯 행을 더한 뒤 다시 잰 값: 같은 실행, 같은 해소, 같은 69줄, 새로 말한 것 없음). 인기 모델 폴더 230개 정적: 새 사실들의 거짓 `broken` 0.
- **2차 재현(사전 등록, 어휘를 1.2.0의 코드 `ce79b19`에 동결):** 같은 순서의 다음 이슈 150건을 같은 규칙으로 심사해 20건이 통과했고 15건이 여기서 재현됐다(보고된 판의 환경 여섯을 그것을 위해 만들었다). 눈가림 평정자 둘이 그 15건 가운데 8건을 entail의 부류로 봤다(일곱 범주 카파 0.72, 부류 안팎 0.68; 86건 가운데 갈린 18건은 셋째 눈가림 평정자가 정했다). entail은 **그 8건 가운데 0건**을 잡았고(3의 법칙: 95%에서 8건 중 많아야 3건), 부류 밖 재현 7건에서 틀린 경보는 없었으며, 실행 하나를 스스로 깨뜨렸다(vLLM 0.23.0에서 서명을 고정한 감싸기; 1.2.0에서 고침). 못 잡은 8건은 폴더의 tokenizer.json과 다른 id를 내는 토크나이저 셋(Vocab 검사는 크기를 비교하지 id를 비교하지 않는다; 셋 가운데 하나는 토크나이저 경계에서 `unknown`으로 말함)과 경계가 없는 자리 다섯(커널의 스케일 배치, 선형 어텐션 커널의 입력 배치, 도구 파서, 멀티모달 자리표의 바인딩, 응답의 logprobs)이다. 평정된 86건 가운데 부류 비중은 31건(36%)이다. 평정자는 1차와 같이 연구 세션에서 띄운 에이전트다. 규약 7절·심사·평정·사례는 연구 작업공간의 `testbed/results/m17/replay2/`에 있다.
- **정상 실행, 1.3.0을 위해**(같은 모델 38개를 같은 세 엔진에서, 유효 실행 102회, 코드를 `2aa975b`에 동결): entail이 깨뜨린 실행은 없고, 1.2.0 뒤에 더한 검사가 `broken`이나 `refused`를 말한 것도 없으며, 해소 없는 실행의 출력은 98번 비교 가운데 97번 같았다. 여섯 실행은 토크나이저 경계에서 `broken`을 말한다. 모두 Llama-2 시절 폴더 둘에서 엔진 셋 모두에 났고, 틀린 경보가 아니라 실제 차이다(transformers 5가 이 토크나이저들을 공백으로 시작하는 텍스트에서 공백 하나를 잃게 만든다. 폴더의 tokenizer.json, 모델의 sentencepiece 파일, transformers 4.57은 서로 같다; transformers#47700이 이것을 적었다). 동결 전에 entail을 켠 실행 하나가 깨졌다(SGLang, Phi-3.5-mini-instruct). 시작 탐침이 엔진의 첫 prefill이 되어 엔진 결함을 건드렸다(flashinfer의 상태 병합이 head_dim 96을 받지 않는다; entail을 꺼도 128토큰 이상 프롬프트면 스케줄러가 멈춘다). 그 뒤로 SGLang의 경로 점검은 요청할 때만 돈다.
- **3차와 4차 재현(사전 등록, 코드를 `07fceac`과 1.3.0의 `2aa975b`에 동결):** 3차는 같은 순서의 다음 이슈 150건을 심사했고(통과 10, 재현 8), 4차는 1차 모집단 바로 앞 여섯 달의 새 모집단 437건에서 150건을 심사했다(통과 10, 재현 9). 눈가림 평정자 둘이 재현된 것 가운데 5건과 6건을 entail의 부류로 봤다(갈린 것은 셋째가 정함; 일곱 범주 카파 0.735와 0.826, 부류 안팎 0.776과 0.916). 평정자는 결함이 데이터와 계산 자체에 있는지 그 둘레에 있는지도 매겼다. entail은 **5건 가운데 0건**, **6건 가운데 0건**을 잡았고(데이터와 계산 자체의 것은 2건 중 0건, 5건 중 0건), 틀린 경보는 3차에 없고 4차에 하나다(2.0.0에서 고침). 평정된 보고 가운데 부류 비중은 96건 중 36건(37.5%), 91건 중 25건(27.5%)이다. 네 번의 재현을 합치면 26건 중 0건이다. 규약 8·9절, 심사, 평정, 사례는 `testbed/results/m18/replay3/`와 `testbed/results/m19/replay4/`에 있다.
- **시험 문제 31건**(재현 사례 16, 실제 환경 사례 8, 모사한 시장 사례 7): 결함마다 고쳐졌다. 고칠 방법이 없는 결함은 그것이 일어난 경계와 사실을 짚어 알리고 실행을 이었고, `ENTAIL_ON_BROKEN=stop`이면 멈췄다. 수정 판을 잘못 짚은 것은 없었다. (ComfyUI 사례 둘은 1.0 전에 쟀고 다시 돌리지 않았다.)
- **비용, 1.3.0을 위해:** vLLM 기본 경로(torch.compile과 CUDA 그래프, Qwen3-4B)에서 모두 켠 것과 entail을 설치하지 않은 것의 요청 처리 속도 비는 배치 1·8·32에서 1.0007·1.0028·1.0110배다(같은 상태 둘의 차이가 0.6%까지 난다; 배치 32는 네 라운드 모두 entail을 켠 쪽이 0.5~2.0% 느렸다). 적재는 설치한 사본으로 vLLM의 0.6~3B 모델에서 1.6~1.7 s 는다(적재의 13~15%). 대부분은 시작 경로 점검이고 `ENTAIL_NO_PATHS=1`로 끈다. 훅만으로는 -0.03~+0.25 s다.
- **비용, 1.0을 위해:** 적재 때 적재 시간의 0.3~2.6%. 상시 모드에서 vLLM의 CUDA Graph 경로는 모든 어댑터를 켜고 배치 1·8·32에서 1.008·1.003·1.006배(entail 없이 두 번 돌린 대조는 0.997~1.004배; 요청마다 도는 경계를 더하기 전에는 0.999~1.000배), transformers 동적 KV 캐시는 eager 디코드의 1.017~1.022배. vLLM 서버에서 요청마다 약 60 µs. 진단 모드는 1.74배(eager), 1.85배(sdpa). `ENTAIL`을 켜지 않으면 파이썬 시작마다 0.2~0.3 ms이고 불러오는 모듈이 없다.
- **사후 탐지와 나란히:** GSM8K(500문항, 탐욕 디코딩)는 위의 RoPE 손실과 심어 둔 가중치 밀림은 잡았다. 그러나 Gemma 2의 soft-capping을 버리는 백엔드는 잡지 못했다. SGLang `torch_native` 313 대 `triton` 316(McNemar p = 0.66)이었고, 1.0 전에 잰 transformers `sdpa` 대 `eager`는 337 대 339(2B), 442 대 443(9B)이었다. 정상 실행과 출력을 비교하면 드러났다(500문항 가운데 198문항이 다름, 정상 실행 둘 사이에서는 0). 다만 그런 정상 실행이 있을 때의 이야기다. entail은 적재 때 고친다.
- **위치 짚기:** 심어 둔 결함 11건을 모두 짚었다.

## 알려진 빈틈

- **처음 보는 버그.** 위의 사전 등록 재현에서 부류 안 재현 7건 가운데 0건을 잡았다. 그 뒤 재현이 드러낸 사실과 자리를 더했다(고치는 것 표의 마지막 여섯 행: 어댑터 설정 파일, 추론 파서의 설정 이름, 접두 캐시 키, 빔 재정렬, Triton 호출의 strides, 회전 짝짓기). 라우팅 가중치가 속한 행은 아직 읽지 않는다. 새 어휘로 보고된 판에서 다시 돌리면 7건은 이렇게 나온다: 고침 둘(LoRA 스케일, 접두 캐시 키), 알림 둘(transformers 5.12.1의 빔 재정렬, vLLM 0.22.0의 GLM-OCR 짝짓기), 커널에서 `unknown` 하나(strides), 규칙으로는 고치지만 vLLM 0.22.0에 그 자리가 없는 것 하나(파서의 설정 이름), 전혀 읽지 않는 것 하나(Marlin MoE의 행). 이것은 회고이지 검출률이 아니다. 이 어휘를 동결한 위의 2차 사전 등록 재현은 8건 가운데 0건을 잡았다. 그것이 드러낸 읽지 않는 것: 커널의 스케일 배치, 선형 어텐션 커널의 입력 배치, 도구 파서의 슬롯, 멀티모달 자리표의 출처, 응답의 logprobs가 덮는 범위. 만들어진 토크나이저가 내는 id(여덟 가운데 셋)는 1.3.0부터 위 토크나이저 검사가 읽는다. 그 검사는 그 버그들에서 썼으므로 검출률이 아니다. 3차와 4차 사전 등록 재현(코드를 `07fceac`과 1.3.0의 `2aa975b`에 동결)은 5건 중 0건, 6건 중 0건을 잡았고, 1.3.0을 위해 마지막에 더한 검사(워밍업 탐침, Triton 호출을 두 번 돌리기, 시작 경로 점검, 인덱스의 정확한 정의)는 4차 재현의 결함 자리 어디에도 닿지 않았다. 두 재현이 드러낸 읽지 않는 것: 멀티모달 프로세서의 합쳐진 설정, 추론 파서의 끝나지 않은 출력(규칙대로 그 자리에서 `unknown`), transformers Qwen2.5-Omni DiT 안의 회전 배치(짝짓기 규칙은 vLLM 쪽에만 있다), 탐침 텍스트 열 개 어디에도 없는 글자(결합 문자)에서만 pre-tokenization이 다른 토크나이저, KV 커넥터의 재계산 경로, 선언된 timestep 간격을 쓰지 않는 스케줄러 step, tied-weights 매핑 탓에 적재되지 않은 체크포인트 가중치, LoRA 어댑터의 모듈 경로, C++ 커널(Marlin)의 입력 스케일, GGUF 토크나이저가 선언한 종류, CUDA 그래프에 잡힌 cross-attention 메타데이터. 멀티모달 모델의 비전 타워는 회전 짝짓기를 대조하지 않는다(참조 구현이 제 나름으로 짝짓는다). 언어 모델만 본다.
- **대조가 닿지 않는 곳:** vLLM 기본 경로(torch.compile)에서는 커스텀 연산이 컴파일되어 정의와 대조하지 않고(대조는 eager 모드에서 돈다), C++에서 부르는 커널(Marlin)에는 아예 닿지 않는다. SGLang의 시작 경로 점검은 `ENTAIL_PATHS=1`일 때만 돈다.
- **허브 id로 적재한 모델**은 엔진이 로컬 캐시에 내려받은 스냅샷에서 읽는다(1.2.0에서 고침: huggingface_hub가 그런 스냅샷을 불완전하다고 거부해 허브 id 적재마다 Vocab·Stops가 "검사할 수 없음"이었다; vLLM 0.30에 허브 id로 적재한 GLM-OCR이 이제 둘 다 통과). 캐시에 아예 없는 모델은 여전히 `unknown`이다.
- **자기 샘플링 루프를 쓰는 앱**(2.1.1 현장 시험): diffusers의 예측 방식은 파이프라인이 샘플링할 때 정한다. 그래서 모델을 자기 샘플링 루프에서, 자기 모델 설정으로 만든 스케줄러로 돌리는 프로그램(InvokeAI)은 고치지 않는다. entail은 그 스케줄러를 보지 못하므로, 선언된 예측 방식과 함께 `unknown`으로 한 번 알리고 아무것도 바꾸지 않는다. 2.1.2까지는 적재 때 고쳤고, 그 때문에 SD.Next의 그림이 달라졌고([#21](https://github.com/wwoosshh/entail/issues/21)) InvokeAI의 기본 샘플러가 멈췄다([#24](https://github.com/wwoosshh/entail/issues/24)).

- transformers 동적 캐시의 상시 KV 계약은 목표의 경계에 있다. Qwen3-4B eager 디코드의 1.017~1.022배이고, 실행을 짝짓는 방식에 따라 다르다(목표 1.02배, 이번 수정 전 1.041배). vLLM의 CUDA Graph 경로에서는 잡음 안이다.
- 멀티모달 프로세서의 `apply_chat_template`은 검사하지 않는다. `ENTAIL_ON_BROKEN=stop`에서 SGLang 서버가 거부한 요청은 SGLang 자신의 오류(500)를 받는다. vLLM 서버는 400으로 답한다.
- 어휘가 담지 못하는 RoPE 선언은 대조하지 않았다고 알린다. 인기 폴더 230개에서는 full/sliding attention 밖 이름의 층 종류별 분할(DeepSeek-V4), 국소 층만의 `partial_rotary_factor`(Laguna), 어느 엔진도 읽지 않는 이름 `attn_factor`다.
- vLLM 점수 경로의 패딩 토큰 종류는 고치지 않고 알린다. vLLM 0.30은 토큰 종류를 첫 1의 위치로만 들어서 문서 뒤의 패딩 종류를 담을 수 없다. transformers의 `PreTrainedTokenizerBase.from_pretrained` 밖에서 만든 토크나이저(Mistral의 파일, tiktoken)는 실행 시 검사하지 않는다. GGUF 파일로 만든 토크나이저는 토크나이저 경계에서 `unknown`이다. GGUF 파일 자신의 토크나이저 선언을 읽지 않기 때문이다(transformers#41494: 다른 토크나이저 종류로 만들어진 Gemma GGUF를 잡지 못한다).
- 정지 집합 검사는 토크나이저의 끝을 tokenizer_config.json(과 tokenizer.json)에서 토크나이저를 만들지 않고 읽는다. 파일 어디에도 `eos_token`의 id가 없는 토크나이저는 거기서 못 본다. 해소 뒤 저장한 모델(`save_pretrained`)은 고쳐진 목록을 generation_config.json에 쓴다.
- 능력표가 엔진 코드를 읽어서만 아는 어긋남(SGLang flashinfer의 sliding window)은 추론했다고 알리고 고치지 않는다. 실측 행만 백엔드를 바꾼다. 어휘 밖의 설정 키를 모델의 설정 클래스가 받지 않으면 잃었다가 아니라 읽히지 않았다고 알린다.
- SGLang의 KV 계약은 추측 복호 배치를 건너뛰고(스케줄러가 초안 토큰 칸을 미리 잡는다) 프로세스마다 한 번 알린다.
- GPU 한 장이다. 합산 계약(여러 랭크에서 두 번 합산한 값)은 한 프로세스가 두 랭크를 대신해서 쟀다.

## 설정

| 변수 | 값 | 뜻 |
|---|---|---|
| `ENTAIL` | `off`(기본), `load`, `debug` | `load`는 시작 검사와 해소기를 켠다. `debug`는 선언된 모든 경계까지 보고, 덮지 못하는 경우를 오류로 만든다 |
| `ENTAIL_POLICY` | `resolve`(기본), `refuse` | `refuse`는 아무것도 고치지 않고, 어긋남을 알리기만 한다 |
| `ENTAIL_ON_BROKEN` | `report`(기본), `stop` | 고칠 수 없는 어긋남을 로그(`entail_logs/`)에 남기고 실행을 잇거나, 결과가 나오기 전에 멈춘다(요청이면 서버의 오류 응답). `ENTAIL_FACT_POLICY=Layout=stop`처럼 사실 종류별로도 멈출 수 있다 |
| `ENTAIL_UNKNOWN` | `report`(기본), `require`, `stop` | 아무도 선언하지 않은 뜻을 바꾸는 사실: 알리거나, 선언될 때까지 진행하지 않는다 |
| `ENTAIL_LOG_DIR` | 폴더, 또는 `off` | entail이 말한 것을 남기는 곳이다. 정하지 않으면, entail이 켜져 있을 때 프로그램을 실행한 폴더의 `entail_logs/`에 날마다 로그(`entail-<날짜>.log`, 줄마다 시각과 프로세스)와 기록(`record-<날짜>.jsonl`, 모든 판정의 JSON)을 남긴다. 프로젝트 기록에 섞이지 않게 `.gitignore`도 둔다. 엔진이 띄우는 모든 프로세스도 같은 곳에 쓴다 |
| `ENTAIL_RECORD` | 파일 | JSON 기록을 `record-<날짜>.jsonl` 대신 이 파일에 적는다 |
| `ENTAIL_RESPONSE_NOTE` | `1` | vLLM 서버: 요청에서 깨진 것을 응답에도 적는다(`entail` 필드, 스트림이면 데이터 앞의 SSE 주석 줄) |
| `ENTAIL_ONLY` | 예: `rope_alias,sglang_adapter` | 적은 어댑터만 설치한다 |
| `ENTAIL_SKIP` | 예: `comfyui:install_nodes` | 적은 항목만 빼고 설치한다(어댑터 이름만 적으면 그 어댑터 전체를 뺀다). 나머지가 그것 없이 무엇을 하는지 잴 때 쓴다 |
| `ENTAIL_VERBOSE` | `1` | 어댑터가 설치될 때마다 알린다 |
| `ENTAIL_QUIET` | `unknown`, `start`(쉼표로 여럿) | `unknown`: 멈추지 않는 `unknown` 판정을 화면에 찍지 않는다. 로그와 기록에는 남고, 프로세스마다 한 번 그렇다고 알린다. `start`: entail이 켜졌다는 첫 줄을 뺀다 |
| `ENTAIL_SOURCE` | `1` | 적재한 가중치를 체크포인트 파일과도 대조한다(vLLM, 시작 때 약간의 입출력) |
| `ENTAIL_NO_PATHS` | `1` | vLLM: 시작 때 엔진 자신의 경로끼리 대조하는 점검을 뺀다(거기서 entail 적재 비용의 대부분) |
| `ENTAIL_PATHS` | `1` | SGLang: 시작 때 엔진 자신의 경로끼리 대조하는 점검을 돌린다(기본은 끔: 탐침이 SGLang의 첫 prefill이 된다) |
| `ENTAIL_SAFE` | `auto`(기본), `all`, `off` | 안전모드. `auto`는 엔진 자신의 경로가 어긋나면 다음 시작부터 최적화를 하나씩 꺼서 원인을 좁힌다. `all`은 엔진이 결과를 바꾸지 않는다고 선언한 최적화를 모두 끈다. 정하지 않으면 `entail_logs/safe_mode.json`(화면의 설정 탭)이 정한다 |
| `ENTAIL_DLC` | `off`, 또는 이름들 | 공식 DLC. `off`가 아니면 설치된 것이 모두 붙는다. 이름을 적으면 그것만 붙고, 나머지는 import하지 않는다 |
| `ENTAIL_NODES` | `off`, 또는 이름들 | 커스텀 노드. `off`면 아무것도 돌지 않는다. 이름을 적으면 그 창작마당 패키지만 붙는다. 노드 하나는 화면에서 끈다(`entail_logs/nodes.json`) |
| `ENTAIL_MANIFESTS` | 폴더들, `:`로 구분(윈도우는 `;`) | 뜻을 선언하지 않는 모델 파일의 선언 파일(`<sha256>.json`)을 찾을 곳이다. 어떤 선언 파일의 대상일 수 있는 파일만 해시를 계산하고, 그 해시는 첫 폴더의 `entail_hashes.json`에 남긴다 |

## 시험한 판

- transformers 5.12.1, 5.16.1, 5.17.0, vLLM 0.30.0, SGLang 0.5.20, diffusers 0.40.0, ComfyUI 0.34.1(윈도우), torch 2.13~2.14, Python 3.12, RTX 4070 Ti(12 GB)에서 시험했다.
- 다른 버전에서도 될 수 있다. `entail doctor`가 설치된 버전을 보여 준다.
- transformers 4.x에서는 RoPE 해소기가 할 일이 없으므로 끼어들지 않는다.
- 능력표(어느 백엔드가 무엇을 지키는가)는 항목마다 근거를 적는다. 해소 대상으로는 "measured"로 표시된 항목만 쓴다.
- 측정 스크립트와 원자료는 저자의 연구 작업 공간에 있고, 이 저장소에는 아직 없다.

## 개발

```bash
git clone https://github.com/wwoosshh/entail && cd entail
pip install -e .
entail hook install          # 편집 가능 설치는 시작 훅을 넣지 않는다
PYTHON=python bash tests/run_all.sh
```

로컬 모델이 필요한 시험은 `ENTAIL_TEST_MODELS`(기본 `~/models`)에서 모델을 찾는다. 없으면 건너뛴다.

## 라이선스

[LICENSE](LICENSE)를 본다.
