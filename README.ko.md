# entail

[![PyPI](https://img.shields.io/pypi/v/entail-ai)](https://pypi.org/project/entail-ai/) [![tests](https://github.com/wwoosshh/entail/actions/workflows/tests.yml/badge.svg)](https://github.com/wwoosshh/entail/actions/workflows/tests.yml)

**LLM 추론 스택의 경계에서 값의 뜻이 사라지지 않게 지킨다.**

추론 스택은 여러 부품이 이어진 사슬이다. 체크포인트와 설정, 로더, 엔진, 커널, 양자화, 캐시가 그 부품이다. 부품마다 제 규약 안에서는 맞게 계산하는데도, 두 부품 사이에서 값의 **뜻**이 사라질 수 있다. 몇 가지 예를 들면 이렇다.

- 모델이 선언한 성질을 고른 커널이 무시한다.
- 설정값이 이제 아무도 읽지 않는 옛 이름으로 들어온다.
- 캐시가 토큰 하나를 조용히 잃는다.

그러면 출력은 문장이 매끄럽게 틀리고, 경고도 없다.

entail의 목표는 그 뜻을 타입처럼 분명하게 만드는 것이다.

- 뜻을 만드는 곳에서 선언하고, 쓰는 곳까지 잇는다.
- 소비자의 선택과, 그리고 실제 데이터와 대조한다.
- 어긋나면 **먼저 해소한다.** 선언을 지키는 소비자에게 보내거나, 소비자가 읽는 형태로 바꾼다. 해소할 방법이 없을 때만 멈춘다.
- 아무도 선언하지 않았으면 기본값이 조용히 대신하게 두지 않고 "모름"이라고 알린다.

이름은 논리학의 "함의(entail)"에서 왔다. 체크포인트가 선언한 뜻은 엔진이 실제로 실행하는 것을 반드시 함의해야 한다. ent·**AI**·**L**은 AI Library를 뜻한다.

> **상태: 연구용 시제품(알파). 쓰기 전에 이 부분을 먼저 읽어 주세요.**
> 0.3.0은 아직 위의 범용 장치가 아니다. 측정한 사례마다 만든 검사와 해소의 모음이고, 목록은 아래 "하는 일"에 있다.
> - 옛 이름으로 넘긴 RoPE 설정
> - 선언된 성질을 버리는 어텐션 백엔드
> - ComfyUI 사례 몇 가지
> - LLM 엔진 셋의 시작 점검과 캐시 계약
>
> 목록에 없는 것은 검사하지 않는다. 범용 층은 다음 주요 판에서 만든다. 모델 파일과 설정에서 읽은 사실, 적재·캐시·요청 경계의 계약, 뜻이 깨진 곳을 보여 주는 장부가 그 내용이다. 측정은 RTX 4070 Ti 한 장에서, 아래 "시험한 판"의 버전으로 했다.

## 왜 필요한가: 끝까지 돌려 잰 사례

모델 카드는 YaRN을 켜는 방법으로 실행 시점 덮어쓰기를 안내한다. 이 경로로 모델 **자신의** `rope_scaling` 값을 그대로 다시 넘기기만 해도, transformers 5에서는 `rope_theta`가 사라진다. 그러면 엔진은 RoPE 기준값을 10,000으로 조용히 바꿔 쓴다.

모델은 Llama-3.2-3B-Instruct이고 탐욕 디코딩이다. GSM8K는 vLLM에서 앞 500문항, SGLang에서 앞 200문항을 썼다.

| | 손대지 않음 | 같은 `rope_scaling`을 실행 시점에 다시 넘김 | entail 켬 |
|---|---|---|---|
| vLLM 0.30.0 `--hf-overrides` | 379 / 500 | **279 / 500**, 경고 없음 | 378 / 500 |
| SGLang 0.5.20 `--json-model-override-args` | 161 / 200 | **106 / 200** | 161 / 200 (출력까지 같음) |

- 망가진 실행은 `rope_theta = 10000`을 직접 넣은 실행과 결과가 같았다.
- Qwen3 dense 모델은 우연히 안전하다. 그 모델 파일이 빠진 값을 1,000,000으로 채우기 때문이다. Llama, Qwen3-MoE, Gemma 등의 모델 파일은 채우지 않는다.

## 설치

```bash
pip install entail-ai
```

- 엔진(vLLM, SGLang, transformers)이 있는 그 환경에 설치한다. entail 자체는 의존성이 없다.
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

### 엔진의 작업 프로세스까지 닿는 방법

vLLM과 SGLang은 모델을 자기가 새로 띄운 프로세스에서 돌린다. 그래서 `pip install`은 site-packages에 `entail-autoinstall.pth` 파일 하나를 넣는다.

- 파이썬은 시작할 때마다 이 파일을 읽는다.
- 파일의 한 줄은 환경 변수만 확인하고, `ENTAIL`이 켜져 있지 않으면 아무것도 하지 않는다.
- `entail hook status|install|uninstall`로 상태를 보거나 관리한다.
- 편집 가능 설치(`pip install -e`)는 이 파일을 넣지 않으므로 `entail hook install`을 따로 실행한다.

## 하는 일

**해소하는 것**

| 어긋남 | 해소 | 잰 결과 |
|---|---|---|
| 설정이 만들어진 뒤 transformers 4 이름으로 준 RoPE 값(`rope_theta`, `rope_scaling`). `from_pretrained` 인자, 속성, vLLM `--hf-overrides`, SGLang `--json-model-override-args` 모두 해당한다 | `config.json`이 넣었을 자리에 쓴다. 층 종류마다 다른 RoPE도 설정 클래스에게 물어 같은 자리에 쓴다 | 모델 4종 × 경로 2 × 값 3에서 `config.json` 경로와 같음. GSM8K가 원래로 돌아옴(위 표) |
| 모델이 선언한 성질을 버리는 어텐션 백엔드. 예: transformers `sdpa`와 SGLang `flashinfer`가 Gemma 2의 logit soft-capping을 버림 | 지킨다고 측정된 백엔드(`eager`, `triton`)로 바꾼다 | 기준 실행과 토큰이 같음. 비용은 transformers 1.18배, SGLang 1.13배이고, 이는 바뀐 백엔드 자체의 값이다 |
| **ComfyUI:** 붙인 모델에 닿지 못하는 LoRA(예: SDXL 워크플로에 Anima LoRA). ComfyUI는 모듈마다 콘솔에 한 줄을 남기고 건너뛰며, 실행은 "성공"으로 끝나지만 LoRA는 아무것도 하지 않는다 | 바꿀 방법이 없으므로 샘플링 전에 멈추고 이유를 보여 준다. 이유에는 LoRA가 선언한 학습 기반과 실제로 만난 모델이 들어간다. 일부만 닿으면 알리고 계속한다 | 실제 ComfyUI 0.34.1에서 잘못된 조합은 그림을 평균 0.8/255만 바꿨다(맞는 LoRA는 35.2). 그 뒤에서 콘솔 경고가 840줄 나왔다. entail을 켜면 잘못된 조합이 두 방향 모두 멈췄다. 맞는 조합 22개(SDXL LoRA 21개, 텍스트 인코더 포함, Anima 1개)는 오탐 없이 통과했다. entail을 켜고 끈 그림은 같았다 |
| **ComfyUI:** 병합이나 변환 과정에서 `v_pred` 표지를 잃은 v-prediction 체크포인트. ComfyUI는 이를 eps로 돌리고, 실행은 "성공"이지만 그림은 색 잡음이나 검은 화면이 된다 | 샘플링의 첫 모델 호출로 모델이 실제로 어떻게 동작하는지 판정한다. eps 모델은 받은 잡음을 되돌려 주고 v 모델은 그러지 않는다. 계산을 추가로 돌리지 않는다. 판정에 맞게 샘플링 방식을 바꾸며, ModelSamplingDiscrete 노드가 하는 것과 같다. 워크플로에 넣은 샘플링 노드가 모델과 어긋나면 해소하지 않고 멈춘다 | 표지를 뺀 NoobAI-XL-Vpred에서, entail이 없을 때는 정상 그림과의 차이가 67~102/255였고 entail을 켜면 12~20이었다. 남은 차이는 ztsnr 설정 때문인데, 이것은 동작으로 알 수 없다. eps 체크포인트는 0.9997~0.9999, v 체크포인트는 0.01로 나뉘었다. 서버를 켠 뒤 첫 그림까지 포함해 entail을 켜고 끈 그림은 같았다. eps 체크포인트 앞에 `v_prediction` 노드가 켜진 채 남은 경우는 샘플러에서 멈췄다(3/3). entail이 없으면 "성공"인데 회색 단색 그림이 나온다 |
| **ComfyUI:** 워크플로가 끝난 뒤에도 남는 샘플링 노드의 설정. ComfyUI의 동적 VRAM 로더는 모델 버퍼를 속성 경로 이름으로 백업한다. 그래서 ModelSamplingDiscrete 같은 노드로 한 번 돌리면, 노드를 뺀 뒤에도 체크포인트가 그 노드의 일정으로 샘플링된다. 거꾸로, 노드 없이 먼저 돌린 뒤에 쓴 노드는 조용히 원래 일정을 받는다 | 샘플링 객체마다 자기 setter가 등록한 일정의 사본을 둔다. 로더의 백업은 다른 객체에 넣지 않고 원래 객체에 돌려준다. 버퍼가 바뀐 뒤 첫 모델 호출에서 사본과 대조하고, 다르면 되돌린다 | ComfyUI 0.34.1에서 waiIllustrious에 ModelSamplingDiscrete(v_prediction, zsnr) 노드로 한 번 돌리면, 이후 노드 없는 실행이 다른 그림(55.8/255)을 거쳐 검은 화면이 됐다. entail을 켜든 끄든 재시작 전까지 그랬다. 고친 뒤에는 새 세션과 화소 단위로 같았다(3/3). NoobAI-XL-Vpred를 그대로 돌린 것과 zsnr=false 노드를 건 것을 두 순서로 돌리면, entail이 없을 때 뒤에 돈 쪽이 앞의 설정을 화소 단위로 그대로 받았다. entail을 켜면 12장 모두 새 세션과 같았다. 첫 호출 확인만으로는(가드를 뺀 경우) 검은 화면은 막았지만 2~11/255가 남았다. Anima 워크플로와 다른 실행은 켜고 끈 그림이 같았고 속도도 같았다 |

**검사하는 것** (해소할 수 없으면 멈춘다)

- 어텐션 성질을 엔진 백엔드와 대조한다. 가중치를 읽기 전에 한다.
- 삼켜질 설정 키가 있는지, 묶인 임베딩 선언이 체크포인트와 맞는지 본다.
- vLLM이 재포장한 가중치의 배치, stride, 값 표본을 선언된 변환과 대조한다.
- 메모리의 가중치를 체크포인트 파일과 대조한다(`ENTAIL_SOURCE=1`).
- KV 캐시 계약을 본다. 요청이 필요한 만큼 가지고 있는지, 줄어든 것이 없는지다. transformers, vLLM 페이지 캐시, SGLang에서 돈다.

## 설정

| 변수 | 값 | 뜻 |
|---|---|---|
| `ENTAIL` | `off`(기본), `load`, `debug` | `load`는 시작 검사와 해소기를 켠다. `debug`는 선언된 모든 경계까지 보고, 덮지 못하는 경우를 오류로 만든다 |
| `ENTAIL_POLICY` | `resolve`(기본), `refuse` | `refuse`는 해소하지 않고 첫 어긋남에서 멈춘다 |
| `ENTAIL_ONLY` | 예: `rope_alias,sglang_adapter` | 적은 어댑터만 설치한다 |
| `ENTAIL_SKIP` | 예: `comfyui:install_buffer_guard` | 적은 항목만 빼고 설치한다(어댑터 이름만 적으면 그 어댑터 전체를 뺀다). 나머지가 그것 없이 무엇을 하는지 잴 때 쓴다 |
| `ENTAIL_VERBOSE` | `1` | 어댑터가 설치될 때마다 알린다 |
| `ENTAIL_SOURCE` | `1` | 적재한 가중치를 체크포인트 파일과도 대조한다(vLLM, 시작 때 약간의 입출력) |
| `ENTAIL_SEED` | 시험 이름 | 검사 자체를 시험하는 결함 주입이다. 실제 서비스에서는 켜지 않는다 |

## 시험한 판

- transformers 5.12.1, 5.16.1, 5.17.0, vLLM 0.30.0, SGLang 0.5.20, ComfyUI 0.34.1(윈도우), torch 2.13~2.14, Python 3.12, RTX 4070 Ti(12 GB)에서 시험했다.
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
