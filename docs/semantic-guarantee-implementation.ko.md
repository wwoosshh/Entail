# entail 의미 보존 보증: 블록 FP8 시범 구현

2026년 10월 1일. 설계안(`semantic-guarantee-design.ko.md`)과 시험 규약(`semantic-guarantee-gpu-protocol.ko.md`)에 따라 만든 첫 구현이다. 연구 로드맵의 M19 L5.4a 항목이다. 기존 off, load, debug 모드와 기본 report 정책은 바꾸지 않았다. 보증 프로파일은 따로 켜는 별도 모드다.

2026년 10월 3일(L5.4c, v3)에 검토 지적에 따라 고쳤다. 아래 "보정 이력"을 본다. 수치 보증과 구조 검사 실험은 이제 다른 모드다.

## 켜는 방법과 이름

| 이름 | 뜻 |
|---|---|
| `ENTAIL=guarantee` | 보증 프로파일만 설치한다. 매 호출 출력 전체를 참조와 비교하는 수치 보증(`check: "output"`)이다. 다른 어댑터, DLC, 알림은 설치하지 않는다. 시작 hook(`.pth` 또는 `adapters/autoinstall`)이 엔진의 하위 프로세스까지 적용한다. |
| `ENTAIL=structure` | 구조 검사 실험이다. 보증이 아니다(아래 "구조 검사 실험"). 생산자 issue와 admission은 보증과 같고, 소비 커널의 실행을 IR 증명으로 정한다. |
| `ENTAIL_GUARANTEE_PLAN` | 실행 계획 JSON. 동결 manifest의 `plan`은 보증이, `structure_plan`은 구조 실험이 읽는다. 지원 범위, 허용치 상수, 예산을 정한다. 없으면 모드별 기본값을 쓴다. |
| `ENTAIL_GUARANTEE_RECORD` | 호출 기록 JSONL 경로. 지정하지 않으면 `entail_logs/guarantee-<날짜>.jsonl`(구조 실험은 `structure-<날짜>.jsonl`)에 쓴다. `off`이면 쓰지 않는다. |
| 계획의 `check` | 모드가 정한다. `guarantee`는 `output`만, `structure`는 `static`만 돈다. 다른 값을 담은 계획은 `plan`으로 거부한다. |
| 계획의 `integrity`, `records` | `checksum`·`epoch`, `all`·`changes`. 보증의 기본값은 `checksum`·`all`, 구조 실험은 `epoch`·`changes`다. |

코드:

- `entail/guarantee.py`: 계획, 생산자 issue, 소비 gate, 그래프 판정, 기록을 맡는다. 규칙은 모두 여기에 있다.
- `entail/kernel_ir.py`: 구조 검사 실험이 쓰는 TTIR 판정이다.
- `entail/adapters/vllm_block_fp8_guarantee.py`: vLLM 0.30의 연결 지점과 처리 핸들을 둔다. 규칙은 두지 않는다.

## 보호 범위와 경계

- **소비자:** `fp8_utils.w8a8_triton_block_scaled_mm`이다. vLLM의 custom op `w8a8_triton_block_scaled_mm_func`가 호출마다 이 이름을 import하므로, 엔진 경로도 이 래퍼를 지난다.
- **활성값 생산자:** `fp8_utils.per_token_group_quant_fp8`다(`QuantFP8.forward_cuda`가 부른다). 값과 그룹 스케일을 짝으로 발급한다.
- **가중치 생산자:** `Fp8BlockScaledMMLinearKernel.process_weights_after_loading`이다. checkpoint의 `quantization_config.weight_block_size`를 읽어 커널 블록과 대조한 뒤, 가중치와 블록 스케일을 짝으로 발급한다.
- **표지 내용:** 역할, 짝, 블록 또는 그룹, 스케일 배치, storage와 epoch, version, 바이트 checksum, 출처다. 소비 지점에서 shape로 뜻을 추정하는 일은 하지 않는다. 발급받지 않은 값은 받지 않는다.
- **전달 허가(보증):** 다음 둘을 모두 통과한 호출에만 허가를 낸다. 실패하면 `guarantee.Refused`(`RoleError`)로 전달 전에 차단한다.
  - admission: 필수 hook, issue, 역할, 짝, 블록, epoch, 지원 범위, 메모리 예산
  - 이 호출의 출력 전체를 참조와 비교. 출력 열 단위로 나눠 계산하지만 모든 값을 비교한다.

## 보증(`ENTAIL=guarantee`)의 판정

| 기록의 `outcome` | 뜻 |
|---|---|
| `normal_delivered` | 커널 출력이 허용치 안이라 그대로 전달 |
| `repaired_delivered` | 전달 전에 교정한 경우. 두 가지가 있다. ① 생산자가 짝지은 스케일이나 선언된 블록으로 바꾼 뒤 커널 결과가 허용치 안이었다(`repairs`). ② 커널 출력이 허용치를 넘어 참조의 출력을 전달했다(`path_after: reference`). |
| `blocked` | 전달하지 않았다. `blocked_kind`: `hook`, `declaration`, `pairing`, `contract`, `unsupported`, `epoch`, `integrity`, `scales`, `reference`, `budget`, `checker`, `plan` |
| `error` | 커널 자체의 예외. 그대로 다시 던진다. 그래프에서 검사가 실패했는데 장치가 멈추지 않은 경우(`error_kind: stop_failed`)도 여기에 든다. 이는 다음 연산이 그 값을 읽었다는 뜻이고, 차단으로 세지 않는다. |

- **허용치:** `|커널 − 참조| ≤ ulps·ulp(출력 자료형, |참조|) + c_acc·Σ_블록 |a_블록|·|b_블록|`이다.
  - 참조는 블록 역양자화한 피연산자의 float32 행렬곱이다(TF32 끔).
  - 상수는 정상 calibration으로 동결했다(`c_acc = 2^-11`).
- **CUDA 그래프:** 같은 검사를 캡처 안에 넣는다.
  - 장치에서 커널 출력과 참조 출력 중 하나를 고른다(허용치 초과는 출력 전체를 참조로 교정).
  - 무결성·스케일·참조 비유한 검사가 실패하면 넘길 올바른 값이 없다. gate의 플래그를 고정(pinned) 호스트 메모리에 복사한 뒤, 장치 단언(`torch._assert_async`)으로 그 자리에서 장치를 멈춘다. 그래프의 다음 연산은 스트림 순서상 그 뒤라서 돌지 않는다.
  - replay가 끝나면 hook이 호스트 복사본에서 플래그를 읽는다. 멈춤은 `blocked`(`stopped: true`)로 기록하고 그 replay를 거부한다. 멈춤과 함께 그 프로세스의 CUDA 컨텍스트를 잃는다. eager의 거부가 엔진을 멈추는 것과 같은 결과다.
  - replay 전에는 그래프가 읽는 가중치가 다시 발급되거나 쓰였는지 확인한다.
  - 고정 메모리는 캡처 시작 hook에서 그래프마다 미리 만든다(gate 4,096개분). 캡처 중 할당은 안전하지 않기 때문이다.
- **torch.compile:** 컴파일된 코드에서는 생산자 hook이 돌지 않는다. 그래서 첫 호출이 `declaration`으로 거부된다(명시적 거부).
- **비용:** 매 호출 전체를 다시 계산하므로 비싸다. v1 측정에서 함수 단위 13~17배, 엔진 eager 약 21배였다. 보존 판정과 비용 판정은 따로 낸다.

## 구조 검사 실험(`ENTAIL=structure`): 보증이 아니다

`entail/kernel_ir.py`가 소비 커널의 Triton 중간 표현(TTIR)을 읽는다. Triton이 컴파일해 내주는 것을 읽기만 하고, 컴파일러는 고치지 않는다. 실행 설정(그리드, 정수 인자, constexpr 설정)마다 한 번, 모든 프로그램과 모든 루프 단계의 주소 계산을 데이터 없이 따라간다.

`proven`은 이 실행이 계약 전체를 계산한다는 뜻이다(2026-10-03부터):

- 출력의 모든 원소 C[m, n](M×N)가 실행 전체에서 정확히 한 번 저장된다. 출력 밖이나 다른 텐서에는 쓰지 않는다.
- 저장하는 값은 정확한 0에서 시작한 누적이다. 각 항은 dot 하나(0으로 누적)에 활성값 스케일 하나와 가중치 스케일 하나만 곱한 것이다. 상수 인수가 1이 아니면 `violation`이다.
- 그 누적의 k가 [0, K)를 정확히 한 번씩 덮는다. 빠지거나 두 번 더한 k가 있으면 `violation`이다.
- 각 원소에 곱하는 스케일은 생산자가 그 값과 짝지어 발급한 `As[m, k // group_k]`, `Bs[n // block_n, k // block_k]`다.
- 두 피연산자를 같은 k에서 읽고, k 방향 마스크가 같으며, 마스크된 자리는 0으로 읽힌다.
- 정수 주소 계산이 i32 범위 안에 있다(IR은 넘치면 감긴다. 이 모듈은 그것을 모델링하지 않는다).

판정은 넷이다.

| 판정 | 뜻 |
|---|---|
| `proven` | 위 조건이 데이터와 무관하게 모두 성립한다 |
| `violation` | 이 실행이 계약을 계산하지 않는다. 남의 스케일, 빠지거나 겹친 k, 저장되지 않는 원소, 다른 곳에 쓰기, 추가 인수 |
| `possible` | 실행 중 읽는 데이터로 스케일을 고르는데, 그중 한 갈래가 틀린다 |
| `unproven` | 모델에 없는 연산, 데이터에 달린 주소, 누적이 아닌 저장 값 등. 아무것도 주장하지 않는다 |

- **실행:** Triton 실행 진입점(`JITFunction.run`)의 hook이 gate 안에서 소비 커널의 실행을 받는다. 증명된 설정만 실행하고(`kernel_delivered`), 나머지는 실행하지 않고 참조의 출력을 전달한다(`reference_delivered`). 증명된 호출 뒤에는 아무것도 비교하지 않고, 장치를 기다리지도 않는다. 허가(permit)는 내지 않는다.
- **CUDA 그래프:** 캡처 때 판정한다. 증명된 커널은 그대로 캡처되어 replay 비용이 없다. 증명되지 않은 실행은 참조 경로가 캡처된다.
- **판정 캐시:** 실행 설정과 크기마다 한 번 판정하고 저장한다. 값은 축별 벡터로 나누어 계산하고, 나눌 수 없을 때만 원소마다 계산한다.
- **함께 쓰는 설정:** `integrity: "epoch"`는 storage, layout, epoch, version만 호스트에서 확인한다(장치 작업 없음). `records: "changes"`는 실행 설정마다의 첫 전달과 그 밖의 결과만 기록한다.
- **보증이 아닌 까닭:** 곱셈과 누산의 산술(내적, 반올림)은 Triton 컴파일러와 장치를 신뢰한다. 매 호출의 실제 수치를 확인하지 않는다. `integrity: "epoch"`는 버전 카운터와 생산자를 모두 우회한 쓰기를 보지 못한다(시험의 I1·I2가 그 경우다).

## 지원하지 않는 것

UE8M0 스케일, 128×128이 아닌 블록, bfloat16·float16이 아닌 출력, 연속이 아닌 A, 전문가 병렬, multi-GPU, fnuz FP8, torch.compile 경로. 모두 차단하며, 통과로 세지 않는다.

## 한계

- 보증은 조건부다. 참조 구현(PyTorch eager 연산), CUDA 런타임, 장치 메모리, 이 검사기가 정상이라는 가정 위에 선다.
- producer와 버전 카운터를 모두 우회해 storage에 쓴 값은, 그 값을 읽는 다음 호출의 checksum에서 잡힌다. 그 전에는 잡히지 않는다.
- 첫 시범의 범위는 블록 FP8 행렬곱 경계 하나다. 모델 전체나 다른 커널의 보증으로 표시하지 않는다.

## 보정 이력

- **2026-10-03(L5.4c, v3), 설계 세션의 코드 검토에 따라:**
  - 그래프에서 교정할 수 없는 실패가 출력을 NaN으로 채워 내보냈다. 그래프의 다음 연산이 그것을 읽은 뒤에야 replay를 거부했다. 시험 집계도 다음 연산이 NaN만 읽은 replay를 차단으로 셌다. 이제 장치를 그 자리에서 멈추고, 집계는 다음 연산이 돌지 않았음이 관찰된 경우만 차단으로 센다. v1·v2 holdout에서 이 경로에 걸린 replay는 없었다(고친 집계로 다시 세어도 수치가 같다).
  - 구조 검사의 `proven`은 찾아낸 곱셈항의 스케일 대응만 보았다. 저장을 지운 커널과 K 20그룹 중 첫 그룹만 더하는 커널도 `proven`이었다. 이제 위의 계약 전체를 요구한다. 반례는 `tests/test_kernel_ir.py`의 회귀 시험이다.
  - 수치 보증과 구조 검사가 같은 보증 프로파일 이름 아래 있었다. 구조 검사를 `ENTAIL=structure`로 옮겼다. 보증은 `output` 검사만 돈다.

시험 도구는 `eval/block_fp8_guarantee/`에 있고, 결과는 연구 작업공간 `lowlevel/l5/guarantee/`에 있다.
