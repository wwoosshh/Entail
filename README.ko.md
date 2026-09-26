# entail

[![PyPI](https://img.shields.io/pypi/v/entail-ai)](https://pypi.org/project/entail-ai/) [![tests](https://github.com/wwoosshh/entail/actions/workflows/tests.yml/badge.svg)](https://github.com/wwoosshh/entail/actions/workflows/tests.yml)

**모델 파일은 자기를 어떻게 돌려야 하는지 적어 둔다. 엔진이 늘 그것을 읽지는 않는다.**

RoPE 밑과 스케일링, soft-capping, sliding window, 채팅 템플릿, 예측 방식. 이 선언 가운데 하나가 엔진에 닿지 않으면 출력은 경고 없이 틀린다. entail은 파일이 이미 선언한 것을 읽어 쓰이는 자리에서 대조하고, 고칠 수 있으면 첫 토큰 전에 고치고, 못 고치면 무엇이 깨졌는지 정확히 적는다. 설정 0줄, 적재 시간의 약 1%.

- **180개 중 64개.** Hugging Face에서 가장 많이 받는 LLM 300개 가운데 180개가 vLLM의 실행 때 `rope_scaling` 덮어쓰기(긴 문맥을 켜는 흔한 방법)에 해당한다. 그중 64개의 RoPE 밑이 경고 없이 바뀐다. entail을 켜면 180개 모두 원래 밑을 지킨다. (vLLM 0.30, 모델 파일을 vLLM 자신의 설정 코드로 대조; [E1](https://github.com/wwoosshh/entail-research/blob/main/testbed/results/m10/E1_SUMMARY.md))
- **379 → 273.** 끝까지 재면 Llama-3.2-3B-Instruct의 GSM8K가 그 경로에서 그렇게 떨어진다(entail 켬: 376). Qwen3-4B-Instruct-2507은 YaRN에서 183 → 175. 둘 다 경고가 없다.
- **평가 점수로는 다 안 보인다.** Gemma 2의 soft-capping을 버리는 백엔드는 500문제 중 198문제의 답을 바꾸지만 GSM8K는 3문제 차이다(p = 0.66). entail은 적재 때 그 성질을 지키는 백엔드로 보낸다.
- **102회에서 틀린 경보 0.** 인기 모델 38개를 transformers, vLLM, SGLang에서: 출력은 entail 없는 실행과 모두 같고, 적재 비용 중앙값 0.7~0.9%. (1.0.0은 처음 81회 중 17회를 틀렸다. 원인 다섯은 고쳤고 [변경 기록](CHANGELOG.md)에 있다.)

내 모델 확인은 세 줄이다.

```bash
pip install entail-ai
entail preflight --model /path/to/model --engine vllm --list   # 백엔드마다 무엇을 버리는지, GPU 없이
ENTAIL=load vllm serve /path/to/model ...                       # 그리고 entail_logs/ 읽기
```

entail은 값의 뜻을 타입처럼 분명하게 만든다. 뜻을 만드는 곳에서 선언하고 쓰는 곳까지 잇고, 소비자의 선택과 실제 데이터와 대조하고, 어긋나면 먼저 해소하며(선언을 지키는 소비자에게 보내거나 소비자가 읽는 형태로 바꾼다), 해소할 방법이 없으면 알리고 실행은 잇고(멈추게 하려면 따로 켠다), 아무도 선언하지 않았으면 기본값이 조용히 대신하게 두지 않고 "모름"이라고 말한다. 이름은 논리학의 "함의(entail)"에서 왔다. 체크포인트가 선언한 뜻은 엔진이 실제로 실행하는 것을 반드시 함의해야 한다. ent·**AI**·**L**은 AI Library를 뜻한다.

> **상태: 1.1.0. 한 대의 장비에서 쟀다.** 아래 내용은 모두 "시험한 판"의 엔진과 버전으로, RTX 4070 Ti 한 장에서 쟀다. 평가는 "어떻게 쟀나"에, 평가에서 드러난 빈틈은 "알려진 빈틈"에 있다. 1.1.0은 1.0 평가에서 읽지 못한다고 드러난 사실 다섯(낡은 캐시 정체, 패딩의 토큰 종류, 양자화 블록 대 커널 타일, 토크나이저의 어휘, 생성이 끝나는 자리)을 더했고, 각각 그 출처인 실제 버그로 쟀다.
> entail은 모델, 컴파일러, 커널, 하드웨어 안쪽의 결함을 찾지 않는다. 검사한 모든 경계가 온전한데 출력이 틀리면, 그렇다고 말하고 살펴볼 곳을 좁힌다.

## 환경에 무엇을 남기나

- `pip install entail-ai`는 패키지 하나(import 이름은 `entail`, 의존성 없음)와 `site-packages`의 한 줄(`entail-autoinstall.pth`)을 더한다. 이 한 줄로 엔진이 띄우는 작업 프로세스까지 닿는다. `ENTAIL`이 없으면 그 줄은 바로 돌아온다. 파이썬 시작마다 0.2~0.3 ms이고 불러오는 모듈이 없다. `entail hook status`로 보고 `entail hook uninstall`로 지운다.
- `ENTAIL=load`이면 entail이 말한 것은 프로그램을 실행한 폴더의 `entail_logs/`에 남는다(날마다 로그와 JSON 기록, `.gitignore` 포함). `ENTAIL_LOG_DIR=off`면 아무것도 쓰지 않고, `ENTAIL_LOG_DIR=<폴더>`면 그곳에 쓴다. `ENTAIL_QUIET=unknown`은 멈추지 않는 `unknown` 줄을 화면에서 뺀다.
- 어댑터는 아래 표의 엔진 판에서 잰 내부 함수에 건다. 다른 판에서 설치되지 않는 어댑터는 한 번 알리고(`could not install ...`) 빠지며, 나머지는 돈다. `entail doctor`가 무엇이 설치돼 있고 무엇이 걸릴지 보여 준다.

| 엔진 | 잰 판 | 어댑터가 보는 것 |
|---|---|---|
| transformers | 5.12.1, 5.16.1, 5.17.0 | 어텐션 백엔드, 묶은 머리, 설정 키, RoPE 이름, KV 캐시, 채팅 템플릿 |
| vLLM | 0.30.0 | 어텐션 백엔드, 적재기, 다시 배치한 가중치의 자리, KV 캐시, OpenAI 서버, 파일과 가중치 대조 |
| SGLang | 0.5.20 | 어텐션 백엔드, 적재기, KV 캐시, 서버 |
| diffusers | 0.40.0 | 예측 방식, VAE 배율, LoRA가 닿는 곳 |
| ComfyUI | 0.34.1 | 예측 방식, VAE 배율, LoRA가 닿는 곳, 그리고 엔진 전용 수리 하나(그렇게 표시됨) |

## 왜 필요한가: 끝까지 돌려 잰 사례

모델 카드는 YaRN을 켜는 방법으로 실행 시점 덮어쓰기를 안내한다. 이 경로로 모델 **자신의** `rope_scaling` 값을 그대로 다시 넘기기만 해도, transformers 5에서는 `rope_theta`가 사라진다. 그러면 엔진은 RoPE 기준값을 10,000으로 조용히 바꿔 쓴다.

모델은 Llama-3.2-3B-Instruct이고 탐욕 디코딩이다. GSM8K는 vLLM에서 앞 500문항, SGLang에서 앞 200문항을 썼다.

| | 손대지 않음 | 같은 `rope_scaling`을 실행 시점에 다시 넘김 | entail 켬 |
|---|---|---|---|
| vLLM 0.30.0 `--hf-overrides` | 379 / 500 | **273 / 500**, 경고 없음 | 376 / 500 |
| SGLang 0.5.20 `--json-model-override-args` | 161 / 200 | **106 / 200** | 161 / 200 (출력까지 같음) |

- vLLM 줄은 1.0에서 다시 잰 값이고, SGLang 줄은 앞선 측정의 값이다.
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

### 모델 파일이 뜻을 선언하지 않을 때

이미지 체크포인트는 예측 방식이나 잠재 배율을 선언하지 않는 경우가 많다. 그러면 entail은 추측하지 않고 "모름"으로 알린다. 선언 파일(매니페스트)은 그 뜻을 바깥에서 선언한다. 타입이 없는 JavaScript 라이브러리에 `.d.ts` 파일로 타입을 주는 것과 같다.

```bash
entail infer model.safetensors --out model.safetensors.entail.json   # 파일이 선언한 것과 빈칸
# 아는 빈칸을 채운 뒤 검토했다고 표시한다
entail pin model.safetensors.entail.json
```

파일 옆에 둔 선언 파일은 저절로 찾는다. 한 폴더에 모아 둔 선언 파일(`<sha256>.json`, 초안에 적힌 해시)은 `ENTAIL_MANIFESTS`로 찾는다. 검토 표시(pin)를 한 선언 파일만 선언으로 친다.

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
| **ComfyUI, diffusers:** 붙인 모델에 닿지 못하거나 일부만 닿는 LoRA(예: SDXL 워크플로에 Anima LoRA). ComfyUI는 모듈마다 콘솔에 한 줄을 남기고 건너뛰며, 실행은 "성공"으로 끝나지만 LoRA는 아무것도 하지 않는다 | 바꿀 방법이 없으므로 오류로 알리고 실행은 잇는다. 알림에는 모듈 가운데 몇 개가 모델에 닿는지와 LoRA가 선언한 학습 기반이 들어간다(`ENTAIL_ON_BROKEN=stop`이면 샘플링 전에 멈춘다) | ComfyUI 0.34.1: SDXL 모델에 붙인 Anima LoRA는 그림을 LoRA 없는 그림과 화소 단위로 같게 남겼다. entail은 LoRA를 붙이는 자리에서 알렸고, `ENTAIL_ON_BROKEN=stop`이면 샘플링 전에 멈췄다(3/3). 맞는 LoRA(그림을 15~31/255 바꿈)는 통과했고, entail을 켜고 끈 그림이 같았다. diffusers 0.40: 키를 읽지 못하는 LoRA가 적재된 뒤 아무것도 하지 않았다(그림이 화소 단위로 같음). 같은 방식으로 알렸다. 앞서 0.3.0의 검사로는 맞는 조합 22개가 오탐 없이 통과했다 |
| **ComfyUI, diffusers:** 엔진이 읽지 않는 방식으로 v-prediction을 선언한 체크포인트. ComfyUI는 `v_pred` 키만 읽어서, 메타데이터에 `modelspec.prediction_type = v`를 적은 체크포인트를 eps로 돌린다. diffusers의 단일 파일 적재는 둘 다 읽지 않고 epsilon으로 둔다. 실행은 "성공"이지만 그림은 망가진다 | 파일 자신의 선언(메타데이터, 표지 키)이나 고정한 선언 파일이 정한다. 샘플러를 그대로 다시 꾸린다. ModelSamplingDiscrete 노드나 다시 만든 스케줄러가 하는 것과 같다. 선언이 말하지 않는 것(zero-terminal SNR)은 엔진의 값을 그대로 둔다. 사용자가 넣은 샘플링 노드나 스케줄러는 덮어쓰지 않고, 어긋남을 알린다. 아무것도 선언하지 않은 체크포인트는 "모름"으로 알린다. 모델의 동작은 바꾸는 근거가 되지 않으므로, 표지를 잃은 체크포인트는 선언 파일이 필요하다 | ComfyUI 0.34.1, AstolfoCarmix-VPredXL(메타데이터에 v를 선언, 표지 키 없음): entail 없이는 작성자의 기준 설정과 83~95/255 달랐고, entail을 켜면 시드 셋에서 같음, 0.16, 0.14/255였다. diffusers 0.40 단일 파일: NoobAI-XL-Vpred는 entail 없이 기준과 55~83/255 달랐고 켜면 화소 단위로 같았다. AstolfoCarmix는 90~95/255에서 화소 단위로 같아졌다. entail 0.3.0은 첫 모델 호출로 판정해서 AstolfoCarmix를 놓쳤다(가장 잡음이 큰 단계에서 eps처럼 동작함). 표지를 뺀 체크포인트는 이제 선언 파일이 없으면 "모름"으로 알린다 |
| **ComfyUI:** 워크플로가 끝난 뒤에도 남는 샘플링 노드의 설정. ComfyUI의 동적 VRAM 로더는 모델 버퍼를 속성 경로 이름으로 백업한다. 그래서 ModelSamplingDiscrete 같은 노드로 한 번 돌리면, 노드를 뺀 뒤에도 체크포인트가 그 노드의 일정으로 샘플링된다. 거꾸로, 노드 없이 먼저 돌린 뒤에 쓴 노드는 조용히 원래 일정을 받는다 | 샘플링 객체마다 자기 setter가 등록한 일정의 사본을 둔다. 로더의 백업은 다른 객체에 넣지 않고 원래 객체에 돌려준다. 버퍼가 바뀐 뒤 첫 모델 호출에서 사본과 대조하고, 다르면 되돌린다 | ComfyUI 0.34.1에서 waiIllustrious에 ModelSamplingDiscrete(v_prediction, zsnr) 노드로 한 번 돌리면, 이후 노드 없는 실행이 다른 그림(55.8/255)을 거쳐 검은 화면이 됐다. entail을 켜든 끄든 재시작 전까지 그랬다. 고친 뒤에는 새 세션과 화소 단위로 같았다(3/3). NoobAI-XL-Vpred를 그대로 돌린 것과 zsnr=false 노드를 건 것을 두 순서로 돌리면, entail이 없을 때 뒤에 돈 쪽이 앞의 설정을 화소 단위로 그대로 받았다. entail을 켜면 12장 모두 새 세션과 같았다. 첫 호출 확인만으로는(가드를 뺀 경우) 검은 화면은 막았지만 2~11/255가 남았다. Anima 워크플로와 다른 실행은 켜고 끈 그림이 같았고 속도도 같았다 |

| **vLLM 서버:** 모델이 선언한 형식을 읽지 못하는 도구 호출 파서(hermes 형식으로 부르는 Qwen3 모델을 `--tool-call-parser pythonic`으로 띄움). 도구 호출이 본문 텍스트로 돌아온다 | 선언된 형식을 읽는다고 측정된 파서로 바꾼다 | vLLM 0.30.0, Qwen3-4B: 도구 호출이 다시 구조화된 호출로 돌아옴 |
| **diffusers:** 따로 읽은 VAE가 다른 모델의 잠재 배율을 받음(SDXL VAE를 SD1.5의 것으로 읽음) | 선언 파일이 그 모델에 선언한 배율을 쓴다 | entail 없이는 기준 그림과 15~16/255 달랐고, 켜면 화소 단위로 같았다(시드 셋) |
| **vLLM:** 만들어진 토큰과 더는 맞지 않는 접두사 캐시 블록 해시. 스트리밍 세션 갱신이 요청의 토큰을 잘라도 블록 해시는 덧붙이기만 해서, 버린 토큰까지 엮인 해시가 살아남고, 옛 토큰과 맞는 뒤 요청이 새 토큰의 KV를 받는다(vllm#49377, #49449; 0.30.0에 살아 있음) | 낡은 첫 블록부터 해시를 잊고 엔진이 현재 토큰으로 다시 만들게 한다 | vLLM 0.30.0, SmolLM2-135M-Instruct: entail 없이는 다시 만든 세션이 16토큰짜리 거짓 캐시 적중과 틀린 이어짓기를 받았고, 켜면 갱신 자리에서 잡아 다시 계산해 맞는 출력이 나온다 |
| **SGLang:** K 타일이 가중치 양자화 블록의 약수가 아닌 블록 FP8 커널. 스케일이 타일마다 한 번 움직여 블록을 건너뛴다(손으로 준 설정, sglang#39626; E=512, N=256 H100 fused-MoE 배포 설정 하나도 블록 128에 BLOCK_SIZE_K 256) | 타일을 블록으로 묶는다. 엔진의 기본값과 같다 | SGLang 0.5.20: 밀집 커널이 288이 맞는 자리에서 64를 냈고 묶으면 288; 배포된 MoE 설정은 커널 수준에서 512가 맞는 자리에서 256, 묶으면 512. 나머지 배포 블록 FP8 항목 1,538개는 모두 나누어떨어져 보통 실행은 아무것도 결정하지 않는다 |
| 엔진이 정지 id를 읽는 파일이 그 id를 선언한 파일이 아니어서 답이 끝난 뒤에도 이어지는 생성. generation_config.json, config.json, 토크나이저가 저마다 끝을 선언하는데 transformers는 첫째만, vLLM은 첫째와 토크나이저, SGLang은 앞의 둘을 읽는다(2024년 4월 Llama 3의 모양: config.json은 한 끝을 말했고 모델은 다른 끝을 냈다) | 다른 파일이 선언한 id를 적재 때 엔진의 정지 집합에 더한다 | transformers 5.17, generation_config.json에 `<\|end_of_text\|>`만 적은 Llama-3.2-3B-Instruct: entail 없이는 시험 답 셋이 모두 `<\|eot_id\|>`를 지나 160토큰 한도까지 달렸고, 켜면 config.json이 선언한 끝을 적재 때 더해 8, 18, 37토큰에서 멈췄다 |

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
- vLLM의 점수 경로에서 cross-encoder의 패딩이 받은 토큰 종류를 토크나이저가 선언한 패딩 종류와 대조한다(vllm#58138: 패딩이 문서의 세그먼트를 받아 /rerank 점수가 움직였다). vLLM 0.30은 토큰 종류를 해소를 담을 수 없는 형태로 들어서 알리기만 하고, `ENTAIL_ON_BROKEN=stop`이면 거부한다.
- 정적으로, 엔진마다 모델 폴더에서 만들 정지 집합을 파일들이 선언한 모든 끝과 대조한다(`entail check`).
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
- **처음 보는 버그의 검출률(사전 등록, 어휘를 1.1.0에 동결):** 같은 네 저장소의 이슈 150건을 정해진 규칙으로 심사해 17건이 통과했고 15건이 여기서 재현됐다. 눈가림 평정자 둘이 그 15건 가운데 7건을 entail의 부류로 봤다(일곱 범주 카파 0.86, 부류 안팎 0.95). entail은 **그 7건 가운데 0건**을 잡았고, 부류 밖 8건에서 틀린 경보는 없었다. 못 잡은 7건은 모두 어휘 밖의 사실이었고, 그중 5건은 아직 어떤 어댑터도 보지 않는 자리(커널 호출, 요청 파서, LoRA 설정 파일, 접두 캐시 키, 빔 재정렬)에서 생겼다. 부류 주장은 이렇게 읽어야 한다. 위의 다섯 사실은 그것을 낳은 버그로 쟀고, 새 버그는 대개 entail이 아직 읽지 않는 사실이다. 규약·심사 기록·평정·사례 스크립트는 연구 작업공간(github.com/wwoosshh/entail-research)의 `testbed/M16_PROTOCOL.md`와 `testbed/results/m16/`에 있다. 인기 모델 38개를 세 엔진에서 다시(유효 102회): `broken` 0, `refused` 0, 같은 해소 둘에 더해 선언된 끝을 transformers의 정지 집합에 더한 해소 둘(배포된 그대로의 Nemotron-3-Nano-4B, 작은 시험 모델 하나), 해소 없는 실행의 출력은 98번 가운데 97번 entail 없는 실행과 같음(다른 하나는 엔진 자신의 비결정성), `unknown` 줄 69, 라이브러리의 적재 시간 비중 중앙값 1.2%, 90번째 백분위 9.1%. 인기 모델 폴더 230개 정적: 새 사실들의 거짓 `broken` 0.
- **시험 문제 31건**(재현 사례 16, 실제 환경 사례 8, 모사한 시장 사례 7): 결함마다 고쳐졌다. 고칠 방법이 없는 결함은 그것이 일어난 경계와 사실을 짚어 알리고 실행을 이었고, `ENTAIL_ON_BROKEN=stop`이면 멈췄다. 수정 판을 잘못 짚은 것은 없었다. (ComfyUI 사례 둘은 1.0 전에 쟀고 다시 돌리지 않았다.)
- **비용:** 적재 때 적재 시간의 0.3~2.6%. 상시 모드에서 vLLM의 CUDA Graph 경로는 0.999~1.000배(entail 없이 두 번 돌린 대조는 0.997~0.999배), transformers 동적 KV 캐시는 eager 디코드의 1.017~1.022배. vLLM 서버에서 요청마다 약 60 µs. 진단 모드는 1.74배(eager), 1.85배(sdpa). `ENTAIL`을 켜지 않으면 파이썬 시작마다 0.2~0.3 ms이고 불러오는 모듈이 없다.
- **사후 탐지와 나란히:** GSM8K(500문항, 탐욕 디코딩)는 위의 RoPE 손실과 심어 둔 가중치 밀림은 잡았다. 그러나 Gemma 2의 soft-capping을 버리는 백엔드는 잡지 못했다. SGLang `torch_native` 313 대 `triton` 316(McNemar p = 0.66)이었고, 1.0 전에 잰 transformers `sdpa` 대 `eager`는 337 대 339(2B), 442 대 443(9B)이었다. 정상 실행과 출력을 비교하면 드러났다(500문항 가운데 198문항이 다름, 정상 실행 둘 사이에서는 0). 다만 그런 정상 실행이 있을 때의 이야기다. entail은 적재 때 고친다.
- **위치 짚기:** 심어 둔 결함 11건을 모두 짚었다.

## 알려진 빈틈

- **처음 보는 버그.** 위의 사전 등록 재현에서 부류 안 재현 7건 가운데 0건을 잡았다. entail이 아직 읽지 않는 사실: 접두 캐시 키의 구성, LoRA 어댑터의 스케일 규칙(`use_rslora`), 커널이 가정하는 입력 strides, 요청 설정이 오가는 이름, 회전 임베딩의 짝짓기 방식, 라우팅 가중치가 속한 행. 7건 가운데 5건은 어댑터가 없는 자리(맨 커널 호출, 요청 파서, 어댑터 설정 파일, 접두 캐시 키, 빔 재정렬)에서 생겼다.
- **허브 id로 적재한 모델.** 로컬 폴더 없이 허브에서 바로 적재하면 transformers와 vLLM의 Vocab·Stops 검사는 판정 대신 `unknown`("검사할 수 없음")을 낸다.

- transformers 동적 캐시의 상시 KV 계약은 목표의 경계에 있다. Qwen3-4B eager 디코드의 1.017~1.022배이고, 실행을 짝짓는 방식에 따라 다르다(목표 1.02배, 이번 수정 전 1.041배). vLLM의 CUDA Graph 경로에서는 잡음 안이다.
- 멀티모달 프로세서의 `apply_chat_template`은 검사하지 않는다. `ENTAIL_ON_BROKEN=stop`에서 SGLang 서버가 거부한 요청은 SGLang 자신의 오류(500)를 받는다. vLLM 서버는 400으로 답한다.
- 어휘가 담지 못하는 RoPE 선언은 대조하지 않았다고 알린다. 인기 폴더 230개에서는 full/sliding attention 밖 이름의 층 종류별 분할(DeepSeek-V4), 국소 층만의 `partial_rotary_factor`(Laguna), 어느 엔진도 읽지 않는 이름 `attn_factor`다.
- vLLM 점수 경로의 패딩 토큰 종류는 고치지 않고 알린다. vLLM 0.30은 토큰 종류를 첫 1의 위치로만 들어서 문서 뒤의 패딩 종류를 담을 수 없다. transformers의 `PreTrainedTokenizerBase.from_pretrained` 밖에서 만든 토크나이저(Mistral의 파일, tiktoken, GGUF)는 실행 시 검사하지 않는다.
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
| `ENTAIL_SKIP` | 예: `comfyui_repair:install_buffer_guard` | 적은 항목만 빼고 설치한다(어댑터 이름만 적으면 그 어댑터 전체를 뺀다). 나머지가 그것 없이 무엇을 하는지 잴 때 쓴다 |
| `ENTAIL_VERBOSE` | `1` | 어댑터가 설치될 때마다 알린다 |
| `ENTAIL_QUIET` | `unknown` | 멈추지 않는 `unknown` 판정을 화면에 찍지 않는다. 로그와 기록에는 남고, 프로세스마다 한 번 그렇다고 알린다 |
| `ENTAIL_SOURCE` | `1` | 적재한 가중치를 체크포인트 파일과도 대조한다(vLLM, 시작 때 약간의 입출력) |
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
