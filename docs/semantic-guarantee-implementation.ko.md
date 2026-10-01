# entail 의미 보존 보증: 블록 FP8 시범 구현

2026년 10월 1일. 설계안(`semantic-guarantee-design.ko.md`)과 시험 규약(`semantic-guarantee-gpu-protocol.ko.md`)에 따라 만든 첫 구현이다. 연구 로드맵의 M19 L5.4a 항목이다. 기존 off, load, debug 모드와 기본 report 정책은 바꾸지 않았다. 보증 프로파일은 따로 켜는 별도 모드다.

## 켜는 방법과 이름

| 이름 | 뜻 |
|---|---|
| `ENTAIL=guarantee` | 보증 프로파일만 설치한다. 다른 어댑터, DLC, 알림은 설치하지 않는다. 시작 hook(`.pth` 또는 `adapters/autoinstall`)이 엔진의 하위 프로세스까지 적용한다. |
| `ENTAIL_GUARANTEE_PLAN` | 실행 계획 JSON(동결 manifest의 `plan`). 지원 범위, 허용치 상수, 예산을 정한다. 없으면 `guarantee.Plan()` 기본값을 쓴다. |
| `ENTAIL_GUARANTEE_RECORD` | 호출 기록 JSONL 경로. 지정하지 않으면 `entail_logs/guarantee-<날짜>.jsonl`에 쓴다. `off`이면 쓰지 않는다. |

코드:

- `entail/guarantee.py`: 계획, 생산자 issue, 소비 gate, 그래프 판정, 기록을 맡는다. 규칙은 모두 여기에 있다.
- `entail/adapters/vllm_block_fp8_guarantee.py`: vLLM 0.30의 연결 지점과 처리 핸들을 둔다. 규칙은 두지 않는다.

## 보호 범위와 경계

- **소비자:** `fp8_utils.w8a8_triton_block_scaled_mm`이다. vLLM의 custom op `w8a8_triton_block_scaled_mm_func`가 호출마다 이 이름을 import하므로, 엔진 경로도 이 래퍼를 지난다.
- **활성값 생산자:** `fp8_utils.per_token_group_quant_fp8`다(`QuantFP8.forward_cuda`가 부른다). 값과 그룹 스케일을 짝으로 발급한다.
- **가중치 생산자:** `Fp8BlockScaledMMLinearKernel.process_weights_after_loading`이다. checkpoint의 `quantization_config.weight_block_size`를 읽어 커널 블록과 대조한 뒤, 가중치와 블록 스케일을 짝으로 발급한다.
- **표지 내용:** 역할, 짝, 블록 또는 그룹, 스케일 배치, storage와 epoch, version, 바이트 checksum, 출처다. 소비 지점에서 shape로 뜻을 추정하는 일은 하지 않는다. 발급받지 않은 값은 받지 않는다.
- **전달 허가:** 다음 둘을 모두 통과한 호출에만 허가를 낸다. 실패하면 `guarantee.Refused`(`RoleError`)로 전달 전에 차단한다.
  - admission: 필수 hook, issue, 역할, 짝, 블록, epoch, 지원 범위, 메모리 예산
  - 이 호출의 출력 전체를 참조와 비교. 출력 열 단위로 나눠 계산하지만 모든 값을 비교한다.

## 판정 결과의 구분

| 기록의 `outcome` | 뜻 |
|---|---|
| `normal_delivered` | 커널 출력이 허용치 안이라 그대로 전달 |
| `repaired_delivered` | 전달 전에 교정한 경우. 두 가지가 있다. ① 생산자가 짝지은 스케일이나 선언된 블록으로 바꾼 뒤 커널 결과가 허용치 안이었다(`repairs`). ② 커널 출력이 허용치를 넘어 참조의 출력을 전달했다(`path_after: reference`). |
| `blocked` | 전달하지 않았다. `blocked_kind`: `hook`, `declaration`, `pairing`, `contract`, `unsupported`, `epoch`, `integrity`, `scales`, `reference`, `budget`, `checker` |
| `error` | 커널 자체의 예외. 그대로 다시 던진다. |

- **허용치:** `|커널 − 참조| ≤ ulps·ulp(출력 자료형, |참조|) + c_acc·Σ_블록 |a_블록|·|b_블록|`이다.
  - 참조는 블록 역양자화한 피연산자의 float32 행렬곱이다(TF32 끔).
  - 상수는 정상 calibration으로 동결했다(`c_acc = 2^-11`).
- **CUDA 그래프:** 같은 검사를 캡처 안에 넣는다. 장치에서 커널 출력과 참조 출력 중 하나를 고른다. 무결성이나 스케일 검사에 실패하면 출력을 NaN으로 오염시킨다.
  - replay가 끝날 때마다 hook이 플래그를 읽고, 실패하면 그 replay를 거부한다.
  - replay 전에는 그래프가 읽는 가중치가 다시 발급되거나 쓰였는지 확인한다.
- **torch.compile:** 컴파일된 코드에서는 생산자 hook이 돌지 않는다. 그래서 첫 호출이 `declaration`으로 거부된다(명시적 거부).

## 지원하지 않는 것

UE8M0 스케일, 128×128이 아닌 블록, bfloat16·float16이 아닌 출력, 연속이 아닌 A, 전문가 병렬, multi-GPU, fnuz FP8, torch.compile 경로. 모두 차단하며, 통과로 세지 않는다.

## 한계

- 보증은 조건부다. 참조 구현(PyTorch eager 연산), CUDA 런타임, 장치 메모리, 이 검사기가 정상이라는 가정 위에 선다.
- producer와 버전 카운터를 모두 우회해 storage에 쓴 값은, 그 값을 읽는 다음 호출의 checksum에서 잡힌다. 그 전에는 잡히지 않는다.
- 첫 시범의 범위는 블록 FP8 행렬곱 경계 하나다. 모델 전체나 다른 커널의 보증으로 표시하지 않는다.

시험 도구는 `eval/block_fp8_guarantee/`에 있고, 결과는 연구 작업공간 `lowlevel/l5/guarantee/`에 있다.
