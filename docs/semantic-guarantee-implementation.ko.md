# entail 의미 보존 보증: 블록 FP8 시범 구현

2026년 10월 1일. 설계안(`semantic-guarantee-design.ko.md`)과 시험 규약(`semantic-guarantee-gpu-protocol.ko.md`)에 따라 만든 첫 구현이다. 연구 로드맵의 M19 L5.4a 항목이다. 기존 off, load, debug 모드와 기본 report 정책은 바꾸지 않았다. 보증 프로파일은 따로 켜는 별도 모드다.

## 켜는 방법과 이름

| 이름 | 뜻 |
|---|---|
| `ENTAIL=guarantee` | 보증 프로파일만 설치한다. 다른 어댑터, DLC, 알림은 설치하지 않는다. 시작 hook(`.pth` 또는 `adapters/autoinstall`)이 엔진의 하위 프로세스까지 적용한다. |
| `ENTAIL_GUARANTEE_PLAN` | 실행 계획 JSON(동결 manifest의 `plan`). 지원 범위, 허용치 상수, 예산을 정한다. 없으면 `guarantee.Plan()` 기본값을 쓴다. |
| `ENTAIL_GUARANTEE_RECORD` | 호출 기록 JSONL 경로. 지정하지 않으면 `entail_logs/guarantee-<날짜>.jsonl`에 쓴다. `off`이면 쓰지 않는다. |
| 계획의 `check` | `output`(L5.4a, 기본값): 매 호출 출력 전체를 참조와 비교. `static`(L5.4b): 소비 커널의 TTIR을 실행 설정마다 한 번 읽어 증명된 실행만 내보냄(아래 "검사 방식 둘"). |
| 계획의 `integrity`, `records` | `checksum`·`epoch`, `all`·`changes`(아래) |

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

## 검사 방식 둘: `check`

계획의 `check` 값으로 고른다.

### `output` (L5.4a, 기본값)

위 "판정 결과의 구분"에서 설명한 방식이다. 매 호출 출력 전체를 참조와 비교하고, 피연산자의 바이트 checksum을 발급 때와 대조한다(`integrity: "checksum"`). 실제 결과까지 수치로 확인하지만, 비용은 함수 단위 13~17배, 엔진 약 21배였다.

### `static` (L5.4b)

`entail/kernel_ir.py`가 소비 커널의 Triton 중간 표현(TTIR)을 읽는다. Triton이 컴파일해 내주는 것을 읽기만 하고, 컴파일러는 고치지 않는다. 실행 설정(그리드, 정수 인자, constexpr 설정)마다 한 번, 모든 프로그램과 모든 루프 단계의 주소 계산을 데이터 없이 따라가며 다음을 확인한다.

- 각 원소에 곱하는 활성값 스케일이 그 원소의 `As[m, k // group_k]`이고, 가중치 스케일이 `Bs[n // block_n, k // block_k]`이다. 그 스케일은 생산자가 그 값과 짝지어 발급한 것이어야 한다.
- 두 피연산자를 같은 k에서 읽는다.
- 결과를 자기 행·열 `C[m, n]`에 저장한다.

판정은 넷이다.

| 판정 | 뜻 |
|---|---|
| `proven` | 모든 경우에 성립한다 |
| `violation` | 남의 스케일을 곱하는 원소가 있다 |
| `possible` | 실행 중 읽는 데이터로 스케일을 고르는데, 그중 한 갈래가 틀린다 |
| `unproven` | 모델에 없는 연산이나 데이터에 달린 주소가 있다 |

- **실행:** Triton 실행 진입점(`JITFunction.run`)의 hook이 gate 안에서 소비 커널의 실행을 받는다. 증명된 설정만 실행하고, 나머지는 실행하지 않고 참조의 출력을 전달한다(`repaired_delivered`, 실행 전 교정). 증명된 호출 뒤에는 아무것도 비교하지 않고, 장치를 기다리지도 않는다.
- **CUDA 그래프:** 캡처 때 판정한다. 증명된 커널은 그대로 캡처되어 replay 비용이 없다. 증명되지 않은 실행은 참조 경로가 캡처된다.
- **판정 캐시:** 실행 설정과 크기마다 한 번 판정하고 저장한다. 값은 축별 벡터로 나누어 계산하고(프로그램마다 스칼라 하나와 축마다 벡터 하나), 나눌 수 없을 때만 원소마다 계산한다. 19,456개 프로그램인 실행 하나를 약 2초에 판정한다.
- **함께 쓰는 설정:** `integrity: "epoch"`는 storage, layout, epoch, version만 호스트에서 확인한다(장치 작업 없음). `records: "changes"`는 실행 설정마다의 첫 정상 전달과 정상이 아닌 결과만 기록한다.

**보증의 종류가 다르다.** `static`이 증명하는 것은 뜻의 대응이다. 어느 원소를 어느 스케일과 곱하고, 어디에 누적하고, 어디에 저장하는가다. 곱셈과 누산의 산술 자체(내적, 반올림)는 Triton 컴파일러와 장치를 신뢰한다. `output`처럼 매 호출의 실제 수치를 확인하지는 않는다. 또 `integrity: "epoch"`는 버전 카운터와 생산자를 모두 우회한 쓰기를 보지 못한다(`checksum`은 본다).

## 지원하지 않는 것

UE8M0 스케일, 128×128이 아닌 블록, bfloat16·float16이 아닌 출력, 연속이 아닌 A, 전문가 병렬, multi-GPU, fnuz FP8, torch.compile 경로. 모두 차단하며, 통과로 세지 않는다.

## 한계

- 보증은 조건부다. 참조 구현(PyTorch eager 연산), CUDA 런타임, 장치 메모리, 이 검사기가 정상이라는 가정 위에 선다.
- producer와 버전 카운터를 모두 우회해 storage에 쓴 값은, 그 값을 읽는 다음 호출의 checksum에서 잡힌다. 그 전에는 잡히지 않는다.
- 첫 시범의 범위는 블록 FP8 행렬곱 경계 하나다. 모델 전체나 다른 커널의 보증으로 표시하지 않는다.

시험 도구는 `eval/block_fp8_guarantee/`에 있고, 결과는 연구 작업공간 `lowlevel/l5/guarantee/`에 있다.
