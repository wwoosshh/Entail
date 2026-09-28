// entail serve: the node view (LIBRARY_DESIGN.md 13.5; ROADMAP product track P2, P7.1). Everything read from the
// records is put in the page as text (textContent), never as markup: a record can hold any string a model or a user
// produced. The page speaks one language at a time (Korean or English, chosen by the browser and switchable).
"use strict";

// ---------------------------------------------------------------- words
const WORDS = {
  ko: {
    folder: "기록 폴더",
    folderTip: (p) => `기록 폴더: ${p}\n누르면 경로를 복사합니다.`,
    copied: "경로를 복사했습니다.",
    copyFailed: "경로를 복사하지 못했습니다.",
    live: "실시간",
    offline: "연결 끊김",
    connecting: "연결 중",
    liveTip: "기록 파일에 새 줄이 생기면 화면이 바로 바뀝니다.",
    offlineTip: "서버와 연결이 끊겼습니다. entail serve가 돌고 있는지 확인하세요.",
    safeChip: (m) => `안전 모드: ${m}`,
    safeSearching: "안전 모드: 원인 찾는 중",
    safeChipTip: "안전 모드 설정 열기",
    tab_runs: "실행 기록",
    tab_settings: "설정",
    tab_help: "도움말",
    runsTitle: "실행 기록",
    settingsTitle: "설정",
    helpTitle: "도움말",
    today: "오늘",
    yesterday: "어제",
    justNow: "방금",
    minutesAgo: (n) => `${n}분 전`,
    noTime: "시각 없음",
    userCode: "사용자 코드",
    places: (n) => `${n}곳`,
    unreadable: (n) => `읽지 못한 기록 파일 ${n}개`,
    unreadableHelp: "링크가 깨졌거나 읽을 권한이 없는 파일입니다. 이 파일들의 기록은 화면에 없습니다.",
    noFolderTitle: "기록 폴더를 찾지 못했습니다",
    noFolderBody: "entail은 프로그램을 시작한 폴더의 entail_logs에 기록을 남깁니다. 그 폴더를 지정해서 다시 여세요.",
    noFolderCode: (p) => `# 찾아본 폴더: ${p}\nentail serve --dir <기록 폴더>`,
    noRunsTitle: "아직 기록이 없습니다",
    noRunsBody: "entail을 켜고 AI 프로그램을 실행하면 이 화면에 실시간으로 나타납니다.",
    noRunsCode: "ENTAIL=load python your_app.py\nentail serve --open",
    emptyRunTitle: "이 실행에는 아직 확인 결과가 없습니다",
    emptyRunBody: "프로그램이 모델을 불러오거나 요청을 처리하면 여기에 단계가 나타납니다.",
    st: {
      pass: "정상", resolved: "바로잡음", unknown: "확인 불가", unchecked: "건너뜀", broken: "어긋남",
      refused: "멈춤", none: "결과 없음", absent: "기록 없음", off: "꺼 둠",
    },
    sx: {
      pass: "선언된 값을 엔진이 그대로 썼습니다.",
      resolved: "엔진이 선언과 다른 값을 쓰려 해서, entail이 선언대로 바로잡았습니다.",
      broken: "엔진이 선언과 다른 값을 썼습니다. 실행은 계속됐지만, 이 뒤의 결과는 틀렸을 수 있습니다.",
      refused: "값이 어긋나서 결과를 내기 전에 실행을 멈췄습니다.",
      unknown: "선언이 없거나 엔진이 어떤 값을 쓰는지 알 수 없어서 확인하지 못했습니다. 문제가 있다는 뜻은 아닙니다.",
      unchecked: "검사할 수 없는 자리라서 건너뛰었습니다(예: CUDA 그래프 안에서 돈 호출). 문제가 있다는 뜻은 아닙니다.",
      none: "이 단계의 기록은 있지만 확인할 값은 없었습니다.",
      absent: "이 실행에서 이 단계는 기록이 없습니다.",
      off: "이 커스텀 노드의 검사를 꺼 두었습니다.",
    },
    headBroken: (w) => `${w}에서 값이 어긋났습니다`,
    headBrokenSomewhere: "값이 어긋난 곳이 있습니다",
    headRefused: (w) => `${w}에서 값이 어긋나 실행을 멈췄습니다`,
    headLost: "값이 전달되는 도중에 사라졌습니다",
    headSaid: "프로그램이 이상을 알렸습니다",
    headClean: "어긋난 곳은 없습니다",
    headOk: "모두 정상입니다",
    headEmpty: "아직 확인한 곳이 없습니다",
    subBroken: (n) => (n > 1 ? `선언과 다른 값이 쓰인 곳이 ${n}곳 있습니다. ` : "") +
      "실행은 계속됐지만, 이 뒤의 결과는 틀렸을 수 있습니다. 노드를 누르면 무엇이 달랐는지 볼 수 있습니다.",
    subRefused: "결과를 내기 전에 멈췄으므로 틀린 결과는 나가지 않았습니다.",
    subLost: (w) => `값을 잃은 곳: ${w}`,
    subClean: (n, u) => `확인한 ${n}곳은 모두 정상이고, ${u}곳은 확인하지 못했습니다.`,
    subOk: (n) => `확인한 ${n}곳 모두 선언된 값 그대로 쓰였습니다.`,
    subResolved: (n) => ` 그 가운데 entail이 바로잡은 결과가 ${n}건 있습니다.`,
    subEmpty: "모델을 불러오거나 요청을 처리하면 결과가 나타납니다.",
    processes: (n) => `프로세스 ${n}개`,
    points: (n) => `지점 ${n}곳`,
    checks: (n) => `검사 ${n.toLocaleString("ko-KR")}회`,
    customNode: "커스텀 노드",
    flowSteps: (shown, total) => (shown < total ? `${total}단계 중 ${shown}단계` : `${total}단계`),
    showAll: "빈 단계 보기",
    showAllTip: "이 실행에서 기록이 없는 단계도 흐름 안에 보입니다.",
    panel: "자세히",
    panelTip: "오른쪽 패널 열기·닫기",
    overview: "실행 요약",
    back: "실행 요약으로",
    close: "닫기",
    toLook: "살펴볼 곳",
    nothingToLook: "살펴볼 곳이 없습니다. 확인한 곳은 모두 정상입니다.",
    runInfo: "실행 정보",
    info: { start: "시작", duration: "걸린 시간", engines: "엔진", processes: "프로세스", results: "기록된 결과", run: "실행 ID" },
    resultsN: (n) => `${n}건`,
    clickHint: "노드를 누르면 어떤 값을 어디서 확인했는지 볼 수 있습니다.",
    loading: "불러오는 중…",
    noRunYet: "실행을 고르면 요약이 여기에 나옵니다.",
    results: "확인 결과",
    passedN: (n) => `정상으로 확인된 결과 ${n}건`,
    noDecisions: "따로 남은 결과가 없습니다. 정상인 검사는 결과를 남기지 않고 횟수만 셉니다(아래 지점별 검사).",
    where: "지점",
    consumer: "쓰는 쪽",
    declared: "선언된 값",
    chosen: "엔진이 쓴 값",
    observed: "관찰한 값",
    unknownValue: "알 수 없음",
    differs: "선언과 다른 항목",
    rule: "근거",
    note: "자세히",
    resolution: "바로잡은 방법",
    lostBy: "값을 잃은 곳",
    blocking: "이 결과에서 실행을 멈췄습니다.",
    conflict: "출처마다 다른 값",
    perPoint: "지점별 검사",
    pChecks: "검사",
    pPassed: "통과",
    pSkipped: "건너뜀",
    pTime: "시간",
    calls: (n) => `${n}회`,
    said: "entail이 남긴 말",
    source: {
      user: "사용자 코드", manifest: "선언 파일", boundary: "코드 경계", file: "모델 파일", config: "설정 파일",
      probe: "시험 입력", default: "기본값", engine: "엔진", data: "데이터",
    },
    certainty: { declared: "선언", verified: "확인됨", inferred: "추정", defaulted: "기본값", unknown: "모름" },
    nodeSwitch: "이 노드 검사",
    nodeSwitchOn: "켜져 있습니다. 끄면 실행 중인 프로그램도 1초 안에 이 검사를 멈춥니다.",
    nodeSwitchOff: "꺼 두었습니다. 켜면 실행 중인 프로그램도 1초 안에 다시 검사합니다.",
    askNodeOff: (n) => `'${n}' 검사를 끌까요?`,
    askNodeOn: (n) => `'${n}' 검사를 다시 켤까요?`,
    askNodeBody: "실행 중인 프로그램은 1초 안에 따릅니다.",
    fileChanged: "바뀌는 파일",
    turnOff: "끄기",
    turnOn: "켜기",
    cancel: "취소",
    change: "바꾸기",
    nodeOffDone: (n) => `'${n}' 검사를 껐습니다.`,
    nodeOnDone: (n) => `'${n}' 검사를 다시 켰습니다.`,
    writeFailed: (e) => `바꾸지 못했습니다: ${e}`,
    safeTitle: "안전 모드",
    safeNote: "엔진을 다음에 시작할 때 쓰입니다. 지금 도는 엔진은 바뀌지 않습니다.",
    safe: {
      auto: ["자동", "경로 점검에서 어긋남이 나오면, 다음 시작부터 관련된 최적화를 하나씩 꺼 보며 원인을 찾습니다."],
      all: ["최적화 모두 끄기", "결과를 바꾸지 않는다고 알려진 최적화(CUDA 그래프, 프리픽스 캐시, 추측 디코딩, 커스텀 커널)를 모두 끄고 시작합니다. 문제가 사라지면 원인은 그 최적화에 있습니다."],
      off: ["사용 안 함", "어긋남을 기록만 하고 설정은 바꾸지 않습니다."],
    },
    defaultTag: "기본값",
    settingFile: "설정 파일",
    noSettingFile: "(아직 없어서 기본값을 씁니다)",
    envNote: "엔진에 환경 변수 ENTAIL_SAFE가 있으면 그 값이 먼저입니다.",
    askSafe: "안전 모드를 바꿀까요?",
    askSafeNext: "다음에 시작하는 엔진부터 쓰입니다. 지금 도는 엔진은 바뀌지 않습니다.",
    safeDone: (m) => `안전 모드: ${m}`,
    pathsTitle: "자동 모드 진행 상황",
    noPaths: "원인을 찾고 있는 엔진이 없습니다.",
    pathSearching: (i, n, f) => `원인 찾는 중 (${i}/${n}) · 지금 꺼 둔 기능: ${f}`,
    pathFound: (f) => `원인을 찾았습니다: ${f}. 이 설정은 그 기능을 끄고 시작합니다.`,
    pathOutside: (f) => `후보(${f})를 하나씩 꺼도 경로가 어긋났습니다. 원인은 그 밖에 있고, 기능은 다시 켰습니다.`,
    pathOutsideNone: "어긋난 경로에 걸린 최적화가 켜져 있지 않았습니다.",
    pathKey: "설정 열쇠",
    pathPairs: "어긋난 짝",
    offNodesTitle: "꺼 둔 커스텀 노드",
    noOffNodes: "꺼 둔 노드가 없습니다.",
    features: {
      cuda_graphs: "CUDA 그래프", prefix_cache: "프리픽스 캐시", speculative_decoding: "추측 디코딩",
      custom_kernels: "커스텀 커널", attention_kernels: "어텐션 커널",
    },
    helpStates: "상태",
    helpTerms: "말뜻",
    terms: [
      ["선언된 값", "모델 파일, 설정, 코드가 '이 값은 이렇다'고 적어 둔 값입니다."],
      ["엔진이 쓴 값", "엔진(transformers, vLLM, ComfyUI 등)이 실제로 받아서 쓴 값입니다."],
      ["지점", "entail이 값을 확인하는 자리입니다. 한 단계에 여러 지점이 있을 수 있습니다."],
    ],
    helpGraph: "그래프 보는 법",
    helpGraphItems: [
      "노드 하나는 AI 프로그램이 거치는 단계 하나입니다.",
      "선은 단계의 순서입니다. 점선은 그 사이에 기록 없는 단계가 있다는 뜻입니다.",
      "위의 상태 버튼을 누르면 그 상태의 노드만 밝게 보입니다.",
      "노드를 누르면 오른쪽에 선언된 값과 엔진이 쓴 값이 나옵니다.",
    ],
    helpAbout: "entail은",
    helpAboutBody: "AI 실행 스택에서 값의 뜻(저장 형식, 위치 기준, 유효 범위 같은 것)이 부품 사이를 지나며 바뀌지 않았는지 확인하는 라이브러리입니다. 이 화면은 기록 폴더를 읽기만 하고, 안전 모드와 커스텀 노드 설정 두 파일만 씁니다.",
    helpDocs: "문서 보기 (GitHub)",
    docsUrl: "https://github.com/wwoosshh/entail/blob/main/README.ko.md",
  },
  en: {
    folder: "Log folder",
    folderTip: (p) => `Log folder: ${p}\nClick to copy the path.`,
    copied: "Path copied.",
    copyFailed: "Could not copy the path.",
    live: "Live",
    offline: "Disconnected",
    connecting: "Connecting",
    liveTip: "The page updates as soon as a line is added to the records.",
    offlineTip: "Lost the connection to the server. Check that entail serve is running.",
    safeChip: (m) => `Safe mode: ${m}`,
    safeSearching: "Safe mode: finding the cause",
    safeChipTip: "Open the safe mode settings",
    tab_runs: "Runs",
    tab_settings: "Settings",
    tab_help: "Help",
    runsTitle: "Runs",
    settingsTitle: "Settings",
    helpTitle: "Help",
    today: "Today",
    yesterday: "Yesterday",
    justNow: "just now",
    minutesAgo: (n) => `${n} min ago`,
    noTime: "No time",
    userCode: "User code",
    places: (n) => (n === 1 ? "1 step" : `${n} steps`),
    unreadable: (n) => (n === 1 ? "1 log file could not be read" : `${n} log files could not be read`),
    unreadableHelp: "A broken link, or no permission to read. What these files hold is not on this page.",
    noFolderTitle: "The log folder was not found",
    noFolderBody: "entail writes its records to entail_logs in the folder the program started from. Open the page on that folder.",
    noFolderCode: (p) => `# looked in: ${p}\nentail serve --dir <log folder>`,
    noRunsTitle: "No runs yet",
    noRunsBody: "Run your AI program with entail turned on and it shows up here, live.",
    noRunsCode: "ENTAIL=load python your_app.py\nentail serve --open",
    emptyRunTitle: "Nothing has been checked in this run yet",
    emptyRunBody: "Steps appear here once the program loads a model or serves a request.",
    st: {
      pass: "Pass", resolved: "Fixed", unknown: "Unverified", unchecked: "Skipped", broken: "Broken",
      refused: "Stopped", none: "No result", absent: "No records", off: "Off",
    },
    sx: {
      pass: "The engine used the declared value.",
      resolved: "The engine was about to use a different value, and entail put the declared one back.",
      broken: "The engine used a value other than the declared one. The run went on, but what came after may be wrong.",
      refused: "A value did not match, so the run stopped before it produced output.",
      unknown: "Nothing declared the value, or what the engine uses is not known, so it could not be checked. This does not mean something is wrong.",
      unchecked: "The check was skipped where it cannot run (for example, a call inside a CUDA graph). This does not mean something is wrong.",
      none: "This step left records, but had no value to check.",
      absent: "This step left no records in this run.",
      off: "Checks of this custom node are turned off.",
    },
    headBroken: (w) => `A value broke at ${w}`,
    headBrokenSomewhere: "A value broke somewhere",
    headRefused: (w) => `A value broke at ${w}, and the run stopped`,
    headLost: "A value was lost on the way",
    headSaid: "The program reported a fault",
    headClean: "Nothing broke",
    headOk: "All clear",
    headEmpty: "Nothing checked yet",
    subBroken: (n) => (n > 1 ? `${n} points used a value other than the declared one. ` : "") +
      "The run went on, but what came after may be wrong. Click the node to see what differed.",
    subRefused: "It stopped before producing output, so no wrong output went out.",
    subLost: (w) => `Lost at: ${w}`,
    subClean: (n, u) => `All ${n} checked points kept their values; ${u} could not be checked.`,
    subOk: (n) => `All ${n} checked points used the declared values.`,
    subResolved: (n) => ` entail repaired ${n} ${n === 1 ? "result" : "results"} among them.`,
    subEmpty: "Results appear once the program loads a model or serves a request.",
    processes: (n) => `${n} processes`,
    points: (n) => (n === 1 ? "1 point" : `${n} points`),
    checks: (n) => `${n.toLocaleString("en-US")} ${n === 1 ? "check" : "checks"}`,
    customNode: "Custom node",
    flowSteps: (shown, total) => (shown < total ? `${shown} of ${total} steps` : `${total} steps`),
    showAll: "Empty steps",
    showAllTip: "Also show the steps that left no records in this run.",
    panel: "Details",
    panelTip: "Open or close the right panel",
    overview: "Run summary",
    back: "Back to the summary",
    close: "Close",
    toLook: "Worth a look",
    nothingToLook: "Nothing to look at. Every checked point passed.",
    runInfo: "About this run",
    info: { start: "Started", duration: "Took", engines: "Engines", processes: "Processes", results: "Recorded results", run: "Run ID" },
    resultsN: (n) => String(n),
    clickHint: "Click a node to see which values were checked, and where.",
    loading: "Loading…",
    noRunYet: "Pick a run to see its summary here.",
    results: "Results",
    passedN: (n) => (n === 1 ? "1 result passed" : `${n} results passed`),
    noDecisions: "No results were kept. Passing checks are counted, not recorded (see the points below).",
    where: "Point",
    consumer: "Used by",
    declared: "Declared",
    chosen: "Used by the engine",
    observed: "Observed",
    unknownValue: "Not known",
    differs: "Differs from the declaration",
    rule: "Why",
    note: "Details",
    resolution: "How it was fixed",
    lostBy: "Lost at",
    blocking: "The run stopped at this result.",
    conflict: "Values by source",
    perPoint: "Checks by point",
    pChecks: "checks",
    pPassed: "passed",
    pSkipped: "skipped",
    pTime: "time",
    calls: (n) => (n === 1 ? "1 call" : `${n} calls`),
    said: "What entail said",
    source: {
      user: "Your code", manifest: "Manifest", boundary: "Code boundary", file: "Model files", config: "Config",
      probe: "Probe", default: "Default", engine: "Engine", data: "The data",
    },
    certainty: { declared: "declared", verified: "verified", inferred: "inferred", defaulted: "default", unknown: "unknown" },
    nodeSwitch: "Check this node",
    nodeSwitchOn: "On. Turned off, a running program stops this check within a second.",
    nodeSwitchOff: "Off. Turned on, a running program checks again within a second.",
    askNodeOff: (n) => `Turn off the checks of '${n}'?`,
    askNodeOn: (n) => `Turn the checks of '${n}' back on?`,
    askNodeBody: "A running program follows within a second.",
    fileChanged: "File changed",
    turnOff: "Turn off",
    turnOn: "Turn on",
    cancel: "Cancel",
    change: "Change",
    nodeOffDone: (n) => `Checks of '${n}' turned off.`,
    nodeOnDone: (n) => `Checks of '${n}' turned on.`,
    writeFailed: (e) => `Could not change it: ${e}`,
    safeTitle: "Safe mode",
    safeNote: "Used the next time an engine starts. An engine already running is not changed.",
    safe: {
      auto: ["Automatic", "If the path check finds a mismatch, the next starts turn the optimizations involved off one at a time to find the cause."],
      all: ["All optimizations off", "Starts with every optimization that is declared not to change results turned off (CUDA graphs, prefix cache, speculative decoding, custom kernels). If the problem goes away, the cause is among them."],
      off: ["Off", "Mismatches are only recorded; settings are not changed."],
    },
    defaultTag: "Default",
    settingFile: "Settings file",
    noSettingFile: "(not there yet, so the default is used)",
    envNote: "ENTAIL_SAFE in an engine's environment takes precedence.",
    askSafe: "Change the safe mode?",
    askSafeNext: "Engines started from now on use it. An engine already running is not changed.",
    safeDone: (m) => `Safe mode: ${m}`,
    pathsTitle: "Automatic mode, in progress",
    noPaths: "No engine is looking for a cause.",
    pathSearching: (i, n, f) => `Finding the cause (${i}/${n}) · turned off now: ${f}`,
    pathFound: (f) => `Found the cause: ${f}. This configuration starts with it turned off.`,
    pathOutside: (f) => `The paths still disagreed with each candidate (${f}) turned off. The cause is elsewhere; they were turned back on.`,
    pathOutsideNone: "No optimization tied to the mismatched paths was on.",
    pathKey: "Configuration key",
    pathPairs: "Mismatched pairs",
    offNodesTitle: "Custom nodes turned off",
    noOffNodes: "No node is turned off.",
    features: {
      cuda_graphs: "CUDA graphs", prefix_cache: "prefix cache", speculative_decoding: "speculative decoding",
      custom_kernels: "custom kernels", attention_kernels: "attention kernels",
    },
    helpStates: "States",
    helpTerms: "Terms",
    terms: [
      ["Declared", "The value the model files, a config or your code say a thing has."],
      ["Used by the engine", "The value the engine (transformers, vLLM, ComfyUI, ...) actually took and used."],
      ["Point", "A place where entail checks a value. A step can have several."],
    ],
    helpGraph: "Reading the graph",
    helpGraphItems: [
      "Each node is one step your AI program goes through.",
      "Lines show the order of the steps. A dotted line means steps with no records lie in between.",
      "Click a state above the graph to highlight only the nodes in that state.",
      "Click a node to see the declared value next to the one the engine used.",
    ],
    helpAbout: "About entail",
    helpAboutBody: "entail checks that the meaning of values (storage format, position base, valid range and the like) does not change as they pass between the parts of an AI stack. This page only reads the log folder; it writes two files there: the safe mode and which custom nodes are off.",
    helpDocs: "Documentation (GitHub)",
    docsUrl: "https://github.com/wwoosshh/entail#readme",
  },
};

const ORDER = ["refused", "broken", "unknown", "unchecked", "resolved", "pass", "none", "absent"];   // worst first
const VERDICTS = new Set(["pass", "resolved", "unknown", "unchecked", "broken", "refused"]);
const ENGINES = {
  transformers: "transformers", vllm: "vLLM", sglang: "SGLang", comfyui: "ComfyUI", diffusers: "Diffusers",
  torch: "PyTorch", triton: "Triton",
};
// the canvas: node size (the width grows between min and max to fill a row), gaps, group padding (px)
const GEO = { minW: 176, maxW: 240, h: 92, pill: 36, gapX: 36, gapY: 52, gapV: 40, pad: 20, head: 40, top: 22,
              bottom: 22, edge: 20, flowGap: 20, socket: 18 };
const ICONS = {
  runs: [[12, 12, 8.5], "M12 7.5V12l3 2"],
  settings: ["M4 7h8", "M16 7h4", "M4 17h3", "M11 17h9", [14, 7, 2], [9, 17, 2]],
  help: [[12, 12, 8.5], "M9.8 9.6a2.3 2.3 0 1 1 3.2 2.1c-.6.3-1 .8-1 1.4v.3", "M12 16.4v.01"],
  shield: ["M12 3.5l7 2.7v5.4c0 4.2-2.9 7.8-7 8.9-4.1-1.1-7-4.7-7-8.9V6.2z"],
  pass: [[12, 12, 8.5], "M8.3 12.3l2.5 2.5 4.9-5"],
  resolved: ["M19.5 12a7.5 7.5 0 1 1-2.2-5.3L19.5 8", "M19.5 4v4h-4", "M9 12.2l2.2 2.2 4-4.2"],
  unknown: [[12, 12, 8.5], "M9.8 9.6a2.3 2.3 0 1 1 3.2 2.1c-.6.3-1 .8-1 1.4v.3", "M12 16.4v.01"],
  unchecked: [[12, 12, 8.5], "M8.5 12h7"],
  broken: ["M12 4.2L21 19.5H3z", "M12 10v4", "M12 16.9v.01"],
  refused: [[12, 12, 8.5], "M9.5 9.5h5v5h-5z"],
  none: [[12, 12, 8.5]],
  chevronLeft: ["M14.5 6l-6 6 6 6"],
  chevronRight: ["M9.5 6l6 6-6 6"],
  close: ["M6.5 6.5l11 11", "M17.5 6.5l-11 11"],
  steps: ["M3.5 6.5h5v5h-5z", "M15.5 12.5h5v5h-5z", "M8.5 9h2.5a2 2 0 0 1 2 2v2a2 2 0 0 0 2 2"],
  panel: ["M4 5h16v14H4z", "M14.5 5v14"],
  info: [[12, 12, 8.5], "M12 11v5", "M12 7.9v.01"],
  external: ["M13.5 5.5h5v5", "M18.5 5.5l-7.5 7.5", "M17 13.5v5H5.5V7H10"],
  inbox: ["M4 13l2.5-7h11l2.5 7v5H4z", "M4 13h4.5l1 2h5l1-2H20"],
  folderMissing: ["M3.5 7.5a2 2 0 0 1 2-2h4l2 2h7a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z", "M10 11.5l4 4", "M14 11.5l-4 4"],
};

let lang = pickLang();
const view = {
  runs: [], folder: "", folderFound: true, unreadable: [], run: null, graph: null, node: null, detail: null,
  source: null, live: null, tab: "runs", filter: null, showAll: readPref("entail.showAll") === "1",
  sideOpen: true, detailOpen: true, mode: null, nodesFile: "", sig: {}, width: 0,
};
// The two writes (LIBRARY_DESIGN.md 13.2, 13.6) need this server's token (in the page; a new one comes back with
// each write).
const safe = { token: (document.querySelector('meta[name="entail-token"]') || {}).content || "", mode: null, file: "",
               set: false, paths: [] };
let customOff = new Set();      // custom nodes turned off (nodes.json; entail/nodes.py)

// ---------------------------------------------------------------- small helpers
function readPref(key) {
  try { return localStorage.getItem(key); } catch (e) { return null; }
}

function writePref(key, value) {
  try { localStorage.setItem(key, value); } catch (e) { /* not kept; the page works without it */ }
}

function pickLang() {
  const saved = readPref("entail.lang");
  if (saved === "ko" || saved === "en") return saved;
  return (navigator.language || "").toLowerCase().startsWith("ko") ? "ko" : "en";
}

function w(key, ...args) {
  const v = WORDS[lang][key] !== undefined ? WORDS[lang][key] : WORDS.en[key];
  return typeof v === "function" ? v(...args) : v;
}

const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = String(text);
  return e;
}

function svgEl(tag, attrs) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

function icon(name) {
  const s = svgEl("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 1.8,
                           "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" });
  for (const p of ICONS[name] || []) {
    s.append(typeof p === "string" ? svgEl("path", { d: p }) : svgEl("circle", { cx: p[0], cy: p[1], r: p[2] }));
  }
  return s;
}

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(url + " " + r.status);
  return r.json();
}

async function post(url, body) {
  let r, d = {};
  try {
    r = await fetch(url, {
      method: "POST", cache: "no-store", body: JSON.stringify(body),
      headers: { "Content-Type": "application/json", "X-Entail-Token": safe.token },
    });
    d = await r.json();
  } catch (e) {
    return { ok: false, error: r ? String(r.status) : String(e.message || e) };
  }
  if (!r.ok) return { ok: false, error: d.error || String(r.status) };
  if (d.token) safe.token = d.token;
  return { ok: true, data: d };
}

function locale() { return lang === "ko" ? "ko-KR" : "en-US"; }
function engineName(e) { return ENGINES[e] || e; }
function vstate(v) { return VERDICTS.has(v) ? v : "unknown"; }
function stateOf(n) { return n.state === "none" && !(n.boundaries || []).length ? "absent" : n.state; }
function isOff(n) { return !!(n && n.custom && customOff.has(n.ko)); }
function flowName(f) { return (lang === "ko" ? f.ko : f.en) || f.id; }

function splitName(s) {
  const m = /^(.+?)\s*\((.+)\)$/.exec(s || "");
  return m ? [m[1], m[2]] : [s || "", ""];
}

// a node's name and, from "Request (template, settings, multimodal)", what it covers
function nodeNames(n) {
  if (n.custom) return [n.ko || String(n.id).replace(/^node:/, ""), ""];
  return splitName((lang === "ko" ? n.ko : n.en) || n.id);
}

function fmtMs(ms) {
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return (ms < 10 ? ms.toFixed(1) : Math.round(ms)) + " ms";
  return (ms / 1000).toFixed(ms < 10000 ? 2 : 1) + " s";
}

function fmtDuration(a, b) {
  const s = Math.max(0, b - a);
  if (lang === "ko") {
    if (s < 1) return "1초 미만";
    if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + "초";
    if (s < 3600) return Math.floor(s / 60) + "분 " + Math.round(s % 60) + "초";
    return Math.floor(s / 3600) + "시간 " + Math.round((s % 3600) / 60) + "분";
  }
  if (s < 1) return "under 1 s";
  if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + " s";
  if (s < 3600) return Math.floor(s / 60) + " min " + Math.round(s % 60) + " s";
  return Math.floor(s / 3600) + " h " + Math.round((s % 3600) / 60) + " min";
}

function dayKey(t) {
  const d = new Date(t * 1000);
  return d.getFullYear() + "-" + d.getMonth() + "-" + d.getDate();
}

function dayLabel(t) {
  const now = new Date();
  if (dayKey(t) === dayKey(now / 1000)) return w("today");
  if (dayKey(t) === dayKey(now / 1000 - 86400)) return w("yesterday");
  return new Date(t * 1000).toLocaleDateString(locale(), { month: "long", day: "numeric", weekday: "short" });
}

function clock(t) {
  return new Date(t * 1000).toLocaleTimeString(locale(), { hour: "2-digit", minute: "2-digit" });
}

function relTime(t) {
  const s = Date.now() / 1000 - t;
  if (s < 60) return w("justNow");
  if (s < 3600) return w("minutesAgo", Math.floor(s / 60));
  return clock(t);
}

function fullTime(t) {
  const d = new Date(t * 1000);
  return d.toLocaleString(locale(), {
    year: d.getFullYear() === new Date().getFullYear() ? undefined : "numeric", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

function shortPath(p) {
  const parts = String(p || "").split(/[\\/]+/).filter(Boolean);
  return parts.slice(-2).join(" / ") || String(p || "");
}

function features(list) {
  const f = w("features");
  return (list || []).map((x) => f[x] || x).join(", ");
}

let toastTimer = 0;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2800);
}

// A question before a write, in the page's own dialog (the browser's confirm() where <dialog> is missing).
function ask({ title, lines, file, okText }) {
  const dlg = $("dialog");
  if (typeof dlg.showModal !== "function") {
    return Promise.resolve(window.confirm([title, ...lines, file || ""].join("\n\n")));
  }
  $("dialog-title").textContent = title;
  const body = $("dialog-body");
  body.replaceChildren();
  for (const line of lines) body.append(el("p", "", line));
  if (file) body.append(el("div", "file", file));
  $("dialog-ok").textContent = okText;
  $("dialog-cancel").textContent = w("cancel");
  return new Promise((resolve) => {
    dlg.returnValue = "";
    dlg.addEventListener("close", () => resolve(dlg.returnValue === "ok"), { once: true });
    dlg.showModal();
  });
}

// ---------------------------------------------------------------- frame: panels, tabs, language
const WIDE = window.matchMedia("(min-width: 1181px)");
const NARROW = window.matchMedia("(max-width: 820px)");

function layoutMode() { return WIDE.matches ? "wide" : NARROW.matches ? "narrow" : "medium"; }

// Wide windows dock both panels; narrower ones turn the detail panel (and on a phone the side panel) into drawers.
function applyPanels() {
  const m = layoutMode();
  if (m !== view.mode) {
    view.mode = m;
    view.sideOpen = m !== "narrow";
    view.detailOpen = m === "wide";
  }
  const app = $("app");
  app.classList.toggle("side-closed", m !== "narrow" && !view.sideOpen);
  app.classList.toggle("drawer-side", m === "narrow" && view.sideOpen);
  app.classList.toggle("detail-closed", m === "wide" && !view.detailOpen);
  app.classList.toggle("drawer-detail", m !== "wide" && view.detailOpen);
  $("scrim").hidden = !((m === "narrow" && view.sideOpen) || (m !== "wide" && view.detailOpen));
  for (const b of document.querySelectorAll(".rail-btn")) {
    b.setAttribute("aria-pressed", String(view.sideOpen && b.dataset.tab === view.tab));
  }
  $("toggle-detail").setAttribute("aria-pressed", String(view.detailOpen));
}

function showTab(tab) {
  if (view.tab === tab && view.sideOpen) {
    view.sideOpen = false;          // pressing the open tab again folds the panel away
  } else {
    view.tab = tab;
    view.sideOpen = true;
    if (layoutMode() === "narrow") view.detailOpen = false;
  }
  for (const p of ["runs", "settings", "help"]) $("pane-" + p).hidden = p !== view.tab;
  applyPanels();
}

function openDetail() {
  view.detailOpen = true;
  if (layoutMode() === "narrow") view.sideOpen = false;
  applyPanels();
}

function applyStatic() {
  document.documentElement.lang = lang;
  for (const e of document.querySelectorAll("[data-t]")) e.textContent = w(e.dataset.t);
  for (const b of document.querySelectorAll("#lang button")) b.setAttribute("aria-pressed", String(b.dataset.lang === lang));
  for (const b of document.querySelectorAll(".rail-btn")) {
    b.title = w("tab_" + b.dataset.tab);
    b.setAttribute("aria-label", b.title);
  }
  $("show-all").title = w("showAllTip");
  $("toggle-detail").title = w("panelTip");
  $("safe-chip").title = w("safeChipTip");
}

function setLang(next) {
  if (next === lang) return;
  lang = next;
  writePref("entail.lang", next);
  applyStatic();
  renderFolder();
  setLive(view.live);
  renderRuns(true);
  if (view.graph) {
    renderRunHead();
    renderBoard();
  } else if (!view.runs.length) {
    renderEmpty();
  }
  if (view.node && view.detail) renderNodeDetail(view.detail, true);
  else renderOverview();
  if (safe.mode) renderSafe({ mode: safe.mode, file: safe.file, set: safe.set, paths: safe.paths }, true);
  renderOffNodes();
  renderHelp();
}

function setLive(on) {
  view.live = on;
  const l = $("live");
  l.classList.toggle("on", on === true);
  l.classList.toggle("off", on === false);
  $("live-text").textContent = on === true ? w("live") : on === false ? w("offline") : w("connecting");
  l.title = on === false ? w("offlineTip") : w("liveTip");
}

function renderFolder() {
  $("folder-path").textContent = shortPath(view.folder);
  $("folder").title = w("folderTip", view.folder);
}

// ---------------------------------------------------------------- runs
async function loadRuns() {
  let data;
  try {
    data = await getJSON("/api/runs");
  } catch (e) {
    setLive(false);
    return;
  }
  view.folder = data.folder;
  // what could not be read is said, never left to look like "no records yet" (P6)
  view.folderFound = data.folder_found !== false;
  view.unreadable = data.unreadable || [];
  view.runs = data.runs || [];
  renderFolder();
  if (!view.runs.length) {
    if (view.source) view.source.close();
    view.source = null;
    view.run = null;
    view.graph = null;
    renderRuns();
    renderEmpty();
    return;
  }
  if (!view.run || !view.runs.some((r) => r.run === view.run)) selectRun(view.runs[0].run);
  else renderRuns();
}

function runTitle(r) {
  const names = (r.engines || []).map(engineName);
  if (names.length) return names.join(" · ");
  return (r.flows || []).includes("user") ? w("userCode") : "entail";
}

function runStatus(r) {
  const label = w("st")[r.state] || r.state;
  if ((r.state === "broken" || r.state === "refused") && r.where) return [label, lang === "ko" ? r.where.ko : r.where.en];
  const n = (r.states || {})[r.state];
  if (n && r.state !== "pass") return [label, w("places", n)];
  return [label, ""];
}

function renderRuns(force) {
  const sig = JSON.stringify([lang, view.run, view.folderFound, view.unreadable, view.runs,
                              Math.floor(Date.now() / 60000)]);
  if (!force && sig === view.sig.runs) return;
  view.sig.runs = sig;
  $("run-count").textContent = view.runs.length ? String(view.runs.length) : "";
  const notices = $("notices");
  notices.replaceChildren();
  if (view.unreadable.length) {
    const box = el("div", "notice");
    box.append(el("strong", "", w("unreadable", view.unreadable.length)), el("span", "", w("unreadableHelp")));
    const ul = el("ul");
    for (const u of view.unreadable) {
      const li = el("li", "", u.file);
      li.title = u.why;
      ul.append(li);
    }
    box.append(ul);
    notices.append(box);
  }
  const list = $("runlist");
  list.replaceChildren();
  let day = null;
  for (const r of view.runs) {
    const key = r.start == null ? "none" : dayKey(r.start);
    if (key !== day) {
      day = key;
      list.append(el("div", "run-day", r.start == null ? w("noTime") : dayLabel(r.start)));
    }
    list.append(runItem(r));
  }
}

function runItem(r) {
  const [label, extra] = runStatus(r);
  const b = el("button", "run st-" + r.state + (r.run === view.run ? " selected" : ""));
  b.type = "button";
  const main = el("span", "run-main");
  const sub = el("span", "run-sub");
  sub.append(el("span", "s", label));
  if (extra) sub.append(document.createTextNode(" · " + extra));
  main.append(el("span", "run-title", runTitle(r)), sub);
  b.append(el("span", "run-dot"), main, el("span", "run-time", r.start == null ? "" : relTime(r.start)));
  b.title = (r.start == null ? "" : fullTime(r.start) + "\n") + r.run;
  if (r.run === view.run) b.setAttribute("aria-current", "true");
  b.onclick = () => {
    if (r.run !== view.run) selectRun(r.run);
    if (layoutMode() === "narrow") {
      view.sideOpen = false;
      applyPanels();
    }
  };
  return b;
}

function selectRun(run) {
  if (view.source) view.source.close();
  Object.assign(view, { run, graph: null, node: null, detail: null, filter: null });
  renderRuns(true);
  $("runhead").replaceChildren();
  $("board").replaceChildren();
  $("empty").hidden = true;
  const src = new EventSource("/api/events?run=" + encodeURIComponent(run));
  src.addEventListener("graph", (e) => { if (view.source === src) onGraph(JSON.parse(e.data)); });
  src.onopen = () => setLive(true);
  src.onerror = () => setLive(false);
  view.source = src;
  renderOverview();
}

function onGraph(g) {
  view.graph = g;
  renderRunHead();
  renderBoard();
  if (view.node && g.nodes.some((n) => n.id === view.node)) loadDetail(view.node, true);
  else renderOverview();
}

// ---------------------------------------------------------------- the run's verdict, in plain words
function nodeOf(g, boundary) { return g.nodes.find((n) => (n.boundaries || []).includes(boundary)); }

function verdict(g) {
  const loc = g.locate || {};
  const name = (n) => (n ? nodeNames(n)[0] : null);
  const repaired = g.nodes.reduce((a, n) => a + ((n.verdicts || {}).resolved || 0), 0);
  const firstBad = loc.broken_at ? nodeOf(g, loc.broken_at)
    : g.nodes.find((n) => n.state === "refused") || g.nodes.find((n) => n.state === "broken");
  if (loc.broken_at || g.state === "broken" || g.state === "refused") {
    const where = name(firstBad);
    if (g.state === "refused") {
      return { state: "refused", title: where ? w("headRefused", where) : w("headBrokenSomewhere"), sub: w("subRefused") };
    }
    return { state: "broken", title: where ? w("headBroken", where) : w("headBrokenSomewhere"),
             sub: w("subBroken", (loc.broken || []).length) };
  }
  if ((loc.lost_by || []).length) return { state: "broken", title: w("headLost"), sub: w("subLost", loc.lost_by[0]) };
  if (g.located && (g.located.suspects || []).length) {
    return { state: "broken", title: w("headSaid"), sub: String((g.located.why || [])[0] || g.located.suspects[0]) };
  }
  const intact = (loc.intact || []).length, unverified = (loc.unchecked || []).length;
  const extra = repaired ? w("subResolved", repaired) : "";
  if (unverified) return { state: "unknown", title: w("headClean"), sub: w("subClean", intact, unverified) + extra };
  if (intact) return { state: repaired ? "resolved" : "pass", title: w("headOk"), sub: w("subOk", intact) + extra };
  return { state: "none", title: w("headEmpty"), sub: w("subEmpty") };
}

function renderRunHead() {
  const g = view.graph, head = $("runhead");
  head.replaceChildren();
  if (!g) return;
  const v = verdict(g);
  const top = el("div", "rh-top");
  const badge = el("div", "rh-icon st-" + v.state);
  badge.append(icon(v.state));
  const text = el("div", "rh-text");
  text.append(el("h1", "rh-title", v.title), el("p", "rh-sub", v.sub));
  const meta = el("div", "rh-meta");
  const when = [];
  if (g.start != null) when.push(fullTime(g.start));
  if (g.start != null && g.end != null) when.push(fmtDuration(g.start, g.end));
  if ((g.pids || []).length > 1) when.push(w("processes", g.pids.length));
  meta.append(el("div", "rh-when", when.join(" · ") || w("noTime")));
  if ((g.engines || []).length) {
    const tags = el("div", "tags");
    for (const e of g.engines) tags.append(el("span", "tag", engineName(e)));
    meta.append(tags);
  }
  top.append(badge, text, meta);
  head.append(top);
  // how many nodes are in each state; a button per state picks those nodes out on the canvas
  const counts = {};
  for (const n of g.nodes) {
    const s = stateOf(n);
    if (s !== "absent") counts[s] = (counts[s] || 0) + 1;
  }
  const stats = el("div", "rh-stats");
  for (const s of ORDER) {
    if (!counts[s]) continue;
    const c = el("button", "chip st-" + s);
    c.type = "button";
    c.title = w("sx")[s];
    c.setAttribute("aria-pressed", String(view.filter === s));
    c.append(el("span", "dot"), el("span", "", w("st")[s]), el("span", "n", counts[s]));
    c.onclick = () => {
      view.filter = view.filter === s ? null : s;
      renderRunHead();
      applyFilter();
    };
    stats.append(c);
  }
  if (stats.childNodes.length) head.append(stats);
  document.title = v.title + " · entail";
}

// ---------------------------------------------------------------- the canvas
function renderEmpty() {
  $("runhead").replaceChildren();
  document.title = "entail";
  if (!view.folderFound) showEmpty("folderMissing", w("noFolderTitle"), w("noFolderBody"), w("noFolderCode", view.folder));
  else showEmpty("inbox", w("noRunsTitle"), w("noRunsBody"), w("noRunsCode"));
  renderOverview();
}

function showEmpty(iconName, title, body, code) {
  const board = $("board");
  board.replaceChildren();
  board.style.height = "";
  const box = $("empty");
  const card = el("div", "empty-card");
  const ic = el("div", "empty-icon");
  ic.append(icon(iconName));
  card.append(ic, el("h2", "empty-title", title), el("p", "empty-body", body));
  if (code) card.append(el("pre", "codeblock", code));
  box.replaceChildren(card);
  box.hidden = false;
}

function nodeHeight(n) { return stateOf(n) === "absent" ? GEO.pill : GEO.h; }

// Each flow is a group; its nodes run left to right in rows as wide as the canvas allows, the wire going back to
// the start of the next row. A canvas too narrow for two nodes stacks them top to bottom.
function renderBoard() {
  const g = view.graph, board = $("board");
  if (!g) return;
  $("empty").hidden = true;
  board.replaceChildren();
  const byId = Object.fromEntries(g.nodes.map((n) => [n.id, n]));
  const flows = g.flows.map((f) => ({
    f,
    items: f.nodes.map((id, i) => ({ n: byId[id], i })).filter((x) => x.n && (view.showAll || stateOf(x.n) !== "absent")),
  })).filter((x) => x.items.length);
  if (!flows.length) {
    showEmpty("inbox", w("emptyRunTitle"), w("emptyRunBody"));
    return;
  }
  const width = $("canvas").clientWidth;
  view.width = width;
  const inner = width - 2 * GEO.edge - 2 * GEO.pad;
  const maxCols = Math.max(1, Math.floor((inner + GEO.gapX) / (GEO.minW + GEO.gapX)));
  const vertical = maxCols === 1;
  board.classList.toggle("vertical", vertical);
  for (const x of flows) {
    const n = x.items.length;
    x.rows = Math.ceil(n / Math.min(n, maxCols));
    x.cols = Math.ceil(n / x.rows);          // rows as even as they can be
  }
  const cols = vertical ? 1 : Math.max(...flows.map((x) => x.cols));
  GEO.w = Math.max(GEO.minW, Math.min(GEO.maxW, Math.floor((inner - (cols - 1) * GEO.gapX) / cols)));
  const groupW = cols * GEO.w + (cols - 1) * GEO.gapX + 2 * GEO.pad;
  const left = GEO.edge;
  const wires = svgEl("svg", { class: "wires" });
  const nodes = [];
  let y = GEO.edge;
  for (const x of flows) {
    const top = y + GEO.head + GEO.top;
    const pos = [];
    let bodyH;
    if (vertical) {
      let ny = top;
      for (const it of x.items) {
        const h = nodeHeight(it.n);
        pos.push({ ...it, r: pos.length, x: left + GEO.pad, y: ny, h });
        ny += h + GEO.gapV;
      }
      bodyH = ny - GEO.gapV - top;
    } else {
      x.items.forEach((it, k) => {
        const r = Math.floor(k / x.cols), c = k % x.cols;
        pos.push({ ...it, r, x: left + GEO.pad + c * (GEO.w + GEO.gapX), y: top + r * (GEO.h + GEO.gapY), h: nodeHeight(it.n) });
      });
      bodyH = x.rows * GEO.h + (x.rows - 1) * GEO.gapY;
    }
    const gh = GEO.head + GEO.top + bodyH + GEO.bottom;
    board.append(groupEl(x.f, left, y, groupW, gh, x.items.length, x.f.nodes.length));
    for (let k = 1; k < pos.length; k++) {
      const a = pos[k - 1], b = pos[k];
      const path = svgEl("path", { d: wirePath(a, b, vertical) });
      if (b.i - a.i > 1) path.setAttribute("class", "gap");     // steps with no records lie in between
      wires.append(path);
    }
    pos.forEach((p, k) => nodes.push(nodeEl(p, k > 0, k < pos.length - 1)));
    y += gh + GEO.flowGap;
  }
  const height = y - GEO.flowGap + GEO.edge + 56;    // room under the last group for the canvas tools
  board.style.height = height + "px";
  wires.setAttribute("width", width);
  wires.setAttribute("height", height);
  board.append(wires, ...nodes);
}

function wirePath(a, b, vertical) {
  if (vertical) {
    const x = a.x + GEO.w / 2;
    return `M${x} ${a.y + a.h}V${b.y}`;
  }
  const sx = a.x + GEO.w, sy = a.y + GEO.socket, ex = b.x, ey = b.y + GEO.socket;
  if (a.r === b.r) return `M${sx} ${sy}H${ex}`;
  const r = 8, ox = sx + 16, ix = ex - 16, cy = a.y + GEO.h + GEO.gapY / 2;
  return `M${sx} ${sy}H${ox - r}Q${ox} ${sy} ${ox} ${sy + r}V${cy - r}Q${ox} ${cy} ${ox - r} ${cy}` +
         `H${ix + r}Q${ix} ${cy} ${ix} ${cy + r}V${ey - r}Q${ix} ${ey} ${ix + r} ${ey}H${ex}`;
}

function groupEl(f, x, y, gw, gh, shown, total) {
  const g = el("div", "group");
  Object.assign(g.style, { left: x + "px", top: y + "px", width: gw + "px", height: gh + "px" });
  const t = el("div", "group-title");
  t.append(el("span", "group-name", flowName(f)), el("span", "group-meta", w("flowSteps", shown, total)));
  g.append(t);
  return g;
}

function nodeMeta(n) {
  const parts = [];
  if (n.custom) parts.push(w("customNode"));
  else if ((n.boundaries || []).length) parts.push(w("points", n.boundaries.length));
  if (n.checks) parts.push(w("checks", n.checks));
  return parts.join(" · ");
}

function nodeEl(p, hasIn, hasOut) {
  const n = p.n, st = stateOf(n), off = isOff(n);
  const cls = ["node", "st-" + (off ? "off" : st)];
  if (st === "absent") cls.push("collapsed");
  if (hasIn) cls.push("in");
  if (hasOut) cls.push("out");
  if (n.id === view.node) cls.push("selected");
  if (view.filter && st !== view.filter) cls.push("dim");
  const b = el("button", cls.join(" "));
  b.type = "button";
  b.dataset.id = n.id;
  Object.assign(b.style, { left: p.x + "px", top: p.y + "px", width: GEO.w + "px" });
  const [name, covers] = nodeNames(n);
  const label = off ? w("st").off : w("st")[st];
  const title = el("span", "node-title");
  title.append(el("span", "node-dot"), el("span", "node-name" + (n.custom ? " mono" : ""), name));
  b.append(title);
  if (st !== "absent") {
    const line = el("span", "node-state");
    line.append(el("span", "", label));
    if (n.ms > 0) line.append(el("span", "node-ms", fmtMs(n.ms)));
    const body = el("span", "node-body");
    body.append(line, el("span", "node-meta", nodeMeta(n)));
    b.append(body);
  }
  b.title = (covers ? name + " (" + covers + ")" : name) + "\n" + (off ? w("sx").off : w("sx")[st]);
  b.setAttribute("aria-label", name + ", " + label);
  b.onclick = () => selectNode(n.id);
  return b;
}

function applyFilter() {
  if (!view.graph) return;
  for (const b of document.querySelectorAll("#board .node")) {
    const n = view.graph.nodes.find((x) => x.id === b.dataset.id);
    b.classList.toggle("dim", !!view.filter && !!n && stateOf(n) !== view.filter);
  }
}

function markSelected() {
  for (const b of document.querySelectorAll("#board .node")) b.classList.toggle("selected", b.dataset.id === view.node);
}

// ---------------------------------------------------------------- the right panel: the run's summary
// The panel's head: the summary's title, or (on a node) the way back to it
function detailHead(withBack) {
  const head = $("dhead");
  head.replaceChildren();
  if (withBack) {
    const back = el("button", "crumb");
    back.type = "button";
    back.title = w("back");
    back.append(icon("chevronLeft"), el("span", "", w("overview")));
    back.onclick = () => renderOverview();
    head.append(back, el("span", "spacer"));
  } else {
    head.append(el("h2", "", w("overview")));
  }
  const close = el("button", "icon-btn");
  close.type = "button";
  close.title = w("close");
  close.setAttribute("aria-label", w("close"));
  close.append(icon("close"));
  close.onclick = () => {
    view.detailOpen = false;
    applyPanels();
  };
  head.append(close);
}

function renderOverview() {
  view.node = null;
  view.detail = null;
  markSelected();
  detailHead(false);
  const body = $("dbody");
  body.replaceChildren();
  const g = view.graph;
  if (!g) {
    body.append(el("p", "muted-line", view.runs.length ? w("loading") : w("noRunYet")));
    return;
  }
  body.append(el("h3", "sec-label", w("toLook")));
  const rank = (s) => ORDER.indexOf(s);
  const issues = g.nodes.filter((n) => ["refused", "broken", "unknown", "unchecked", "resolved"].includes(n.state))
    .sort((a, b) => rank(a.state) - rank(b.state));
  if (issues.length) {
    const list = el("div", "issues");
    for (const n of issues) {
      const b = el("button", "issue st-" + n.state);
      b.type = "button";
      b.append(el("span", "issue-dot"), el("span", "issue-name", nodeNames(n)[0]),
               el("span", "issue-state", w("st")[n.state]), icon("chevronRight"));
      b.onclick = () => selectNode(n.id);
      list.append(b);
    }
    body.append(list);
  } else {
    const ok = el("div", "allgood");
    ok.append(icon("pass"), el("span", "", w("nothingToLook")));
    body.append(ok);
  }
  body.append(el("h3", "sec-label", w("runInfo")));
  const I = w("info");
  const dl = el("dl", "dl");
  const add = (k, v, mono) => dl.append(el("dt", "", k), el("dd", mono ? "mono" : "", v));
  if (g.start != null) add(I.start, fullTime(g.start));
  if (g.start != null && g.end != null) add(I.duration, fmtDuration(g.start, g.end));
  if ((g.engines || []).length) add(I.engines, g.engines.map(engineName).join(", "));
  if ((g.pids || []).length) add(I.processes, g.pids.join(", "), true);
  add(I.results, w("resultsN", g.decisions));
  add(I.run, view.run, true);
  body.append(dl);
  const hint = el("p", "hint");
  hint.append(icon("info"), el("span", "", w("clickHint")));
  body.append(hint);
}

// ---------------------------------------------------------------- the right panel: one node
function selectNode(id) {
  view.node = id;
  markSelected();
  openDetail();
  const n = view.graph && view.graph.nodes.find((x) => x.id === id);
  detailHead(true);
  $("dbody").replaceChildren(el("p", "muted-line", w("loading")));
  loadDetail(id, false);
}

async function loadDetail(id, quiet) {
  let d;
  try {
    d = await getJSON("/api/node?run=" + encodeURIComponent(view.run) + "&node=" + encodeURIComponent(id));
  } catch (e) {
    return;
  }
  if (view.node !== id) return;        // another node was picked meanwhile
  view.detail = d;
  renderNodeDetail(d, quiet);
}

function renderNodeDetail(d, quiet) {
  const g = view.graph;
  const n = (g && g.nodes.find((x) => x.id === d.id)) ||
    { id: d.id, ko: d.ko, en: d.en, state: "none", boundaries: [], custom: d.custom, verdicts: {}, checks: 0, ms: 0 };
  const body = $("dbody");
  const keep = quiet ? { top: body.scrollTop, open: [...body.querySelectorAll("details[open]")].map((x) => x.dataset.key) } : null;
  const [name, covers] = nodeNames(n);
  const st = stateOf(n), off = isOff(n), shown = off ? "off" : st;
  detailHead(true);
  body.replaceChildren();
  const flow = g && g.flows.find((f) => f.nodes.includes(n.id));
  if (flow) body.append(el("div", "nd-flow", flowName(flow)));
  body.append(el("h2", "nd-title" + (n.custom ? " mono" : ""), name));
  if (covers) body.append(el("div", "nd-sub", covers));
  const pills = el("div", "nd-pills");
  pills.append(el("span", "pill state st-" + shown, w("st")[shown]));
  if ((n.boundaries || []).length) pills.append(el("span", "pill", w("points", n.boundaries.length)));
  if (n.checks) pills.append(el("span", "pill", w("checks", n.checks)));
  if (n.ms > 0) pills.append(el("span", "pill mono", fmtMs(n.ms)));
  body.append(pills, el("p", "nd-explain st-" + shown, w("sx")[shown]));
  if (n.custom) body.append(nodeSwitch(n, off));

  body.append(el("h3", "sec-label", w("results")));
  const decs = d.decisions.slice().sort((a, b) => ORDER.indexOf(vstate(a.verdict)) - ORDER.indexOf(vstate(b.verdict)));
  const main = decs.filter((x) => vstate(x.verdict) !== "pass");
  const passed = decs.filter((x) => vstate(x.verdict) === "pass");
  if (!decs.length) body.append(el("p", "muted-line", w("noDecisions")));
  for (const x of main) body.append(decisionEl(x));
  if (passed.length) {
    const det = el("details", "passed");
    det.dataset.key = "passed";
    const sm = el("summary");
    sm.append(icon("chevronRight"), el("span", "", w("passedN", passed.length)));
    det.append(sm);
    for (const x of passed) det.append(decisionEl(x));
    if (!main.length && !keep) det.open = true;        // nothing else to read here
    body.append(det);
  }

  const points = [...new Set([...Object.keys(d.counts || {}), ...Object.keys(d.timing || {})])].sort();
  if (points.length) {
    body.append(el("h3", "sec-label", w("perPoint")));
    const box = el("div", "points");
    for (const b of points) {
      const row = el("div", "point");
      const stats = el("div", "point-stats");
      const stat = (k, v) => {
        const s = el("span", "", k + " ");
        s.append(el("b", "", v));
        stats.append(s);
      };
      const c = (d.counts || {})[b], t = (d.timing || {})[b];
      if (c) {
        stat(w("pChecks"), c.checks.toLocaleString(locale()));
        stat(w("pPassed"), c.passed.toLocaleString(locale()));
        if (c.skipped) stat(w("pSkipped"), c.skipped.toLocaleString(locale()));
      }
      if (t) stat(w("pTime"), fmtMs(t.ms) + " / " + w("calls", t.calls));
      row.append(el("div", "point-name", b), stats);
      box.append(row);
    }
    body.append(box);
  }
  if ((d.said || []).length) {
    body.append(el("h3", "sec-label", w("said")));
    for (const s of d.said) {
      const b = el("div", "said");
      b.append(el("span", "where", s.where), el("span", "", s.text == null ? "" : s.text));
      body.append(b);
    }
  }
  if (keep) {
    for (const x of body.querySelectorAll("details")) x.open = keep.open.includes(x.dataset.key);
    body.scrollTop = keep.top;
  }
}

// "Tokenization(digest='53..', probes=10)" -> {name, kv: [["digest", "'53..'"], ["probes", "10"]]}, else null
function parseValue(s) {
  if (typeof s !== "string") return null;
  const m = /^([A-Za-z_][\w.]*)\(([\s\S]*)\)$/.exec(s.trim());
  if (!m) return null;
  const parts = [];
  let depth = 0, quote = null, cur = "";
  for (const ch of m[2]) {
    if (quote) {
      cur += ch;
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === "'" || ch === '"') quote = ch;
    else if ("([{".includes(ch)) depth++;
    else if (")]}".includes(ch)) depth--;
    if (ch === "," && depth === 0) {
      parts.push(cur.trim());
      cur = "";
      continue;
    }
    cur += ch;
  }
  if (cur.trim()) parts.push(cur.trim());
  const kv = [];
  for (const p of parts) {
    const i = p.indexOf("=");
    if (i <= 0 || !/^\w+$/.test(p.slice(0, i))) return null;
    kv.push([p.slice(0, i), p.slice(i + 1)]);
  }
  return kv.length ? { name: m[1], kv } : null;
}

function diffKeys(a, b) {
  const out = new Set();
  if (!a || !b || a.name !== b.name) return out;
  const other = new Map(b.kv);
  for (const [k, v] of a.kv) if (other.has(k) && other.get(k) !== v) out.add(k);
  return out;
}

function valueText(v) { return typeof v === "object" ? JSON.stringify(v) : String(v); }

function valueBlock(label, fact, parsed, diffs) {
  const wrap = el("div", "val-block");
  wrap.append(el("div", "val-label", label));
  const box = el("div", "val");
  if (fact.value === null || fact.value === undefined) {
    box.classList.add("unknown");
    box.textContent = w("unknownValue");
  } else if (parsed) {
    box.append(el("div", "vname", parsed.name));
    const kv = el("div", "kv");
    for (const [k, v] of parsed.kv) {
      const d = diffs.has(k) ? " diff" : "";
      const ks = el("span", "k" + d, k), vs = el("span", "v" + d, v);
      if (d) ks.title = vs.title = w("differs");
      kv.append(ks, vs);
    }
    box.append(kv);
  } else {
    box.textContent = valueText(fact.value);
  }
  wrap.append(box);
  if (fact.source) {
    const src = el("div", "src");
    src.append(el("span", "kind", w("source")[fact.source.kind] || fact.source.kind || "?"));
    if (fact.source.where) src.append(document.createTextNode(" · " + fact.source.where));
    if (fact.certainty) src.append(el("span", "cert", w("certainty")[fact.certainty] || fact.certainty));
    wrap.append(src);
  }
  return wrap;
}

function decRow(k, v, mono) {
  const r = el("div", "dec-row");
  r.append(el("span", "k", k), el("span", "v" + (mono ? " mono" : ""), v));
  return r;
}

function decisionEl(x) {
  const st = vstate(x.verdict);
  const box = el("article", "dec st-" + st);
  const head = el("div", "dec-head");
  head.append(el("span", "dec-name", x.name || "?"), el("span", "pill state st-" + st, w("st")[st]));
  box.append(head);
  const where = el("dl", "dec-where");
  where.append(el("dt", "", w("where")), el("dd", "", x.boundary || "-"));
  if (x.consumer) where.append(el("dt", "", w("consumer")), el("dd", "", x.consumer));
  box.append(where);
  const dv = parseValue(x.declared && x.declared.value);
  const cv = parseValue(x.chosen && x.chosen.value);
  const ov = parseValue(x.observed && x.observed.value);
  if (x.declared) box.append(valueBlock(w("declared"), x.declared, dv, diffKeys(cv, dv)));
  if (x.chosen) box.append(valueBlock(w("chosen"), x.chosen, cv, diffKeys(dv, cv)));
  if (x.observed) box.append(valueBlock(w("observed"), x.observed, ov, diffKeys(dv, ov)));
  if (x.resolution) box.append(decRow(w("resolution"), x.resolution + (x.handle ? " (" + x.handle + ")" : "")));
  if (x.lost_by) box.append(decRow(w("lostBy"), x.lost_by, true));
  if (x.rule) box.append(decRow(w("rule"), x.rule));
  if (x.note) box.append(decRow(w("note"), x.note));
  if ((x.conflict || []).length > 1) {
    const c = el("div", "conflict");
    c.append(el("div", "val-label", w("conflict")));
    for (const f of x.conflict) {
      const item = el("div", "item");
      const src = f.source ? (w("source")[f.source.kind] || f.source.kind) + (f.source.where ? " · " + f.source.where : "") : "";
      item.append(document.createTextNode(src), el("code", "", f.value == null ? w("unknownValue") : valueText(f.value)));
      c.append(item);
    }
    box.append(c);
  }
  if (x.blocking) box.append(el("div", "dec-block", w("blocking")));
  return box;
}

function nodeSwitch(n, off) {
  const card = el("div", "switch-card");
  const text = el("div");
  text.append(el("div", "t", w("nodeSwitch")), el("div", "d", off ? w("nodeSwitchOff") : w("nodeSwitchOn")));
  const t = el("button", "toggle");
  t.type = "button";
  t.setAttribute("role", "switch");
  t.setAttribute("aria-checked", String(!off));
  t.setAttribute("aria-label", w("nodeSwitch"));
  t.onclick = () => switchNode(n.ko, off);
  card.append(text, t);
  return card;
}

// ---------------------------------------------------------------- the two writes: custom nodes, safe mode
async function loadNodes() {
  try {
    const d = await getJSON("/api/nodes");
    view.nodesFile = d.file || "";
    const next = new Set(d.off);
    const changed = next.size !== customOff.size || [...next].some((n) => !customOff.has(n));
    customOff = next;
    if (changed) {
      renderOffNodes();
      if (view.graph) renderBoard();
      if (view.node && view.detail) renderNodeDetail(view.detail, true);
    }
  } catch (e) { /* the next poll tries again */ }
}

async function switchNode(name, on) {
  const ok = await ask({
    title: on ? w("askNodeOn", name) : w("askNodeOff", name), lines: [w("askNodeBody")],
    file: w("fileChanged") + ": " + (view.nodesFile || "nodes.json"), okText: on ? w("turnOn") : w("turnOff"),
  });
  if (!ok) return;
  const r = await post("/api/nodes", { node: name, on });
  if (!r.ok) {
    toast(w("writeFailed", r.error));
    return;
  }
  customOff = new Set(r.data.off);
  toast(on ? w("nodeOnDone", name) : w("nodeOffDone", name));
  renderOffNodes();
  if (view.graph) renderBoard();
  if (view.node && view.detail) renderNodeDetail(view.detail, true);
}

function renderOffNodes() {
  const box = $("off-nodes");
  box.replaceChildren();
  if (!customOff.size) {
    box.append(el("p", "muted-line", w("noOffNodes")));
    return;
  }
  for (const name of [...customOff].sort()) {
    const row = el("div", "off-node");
    const b = el("button", "btn btn-secondary btn-sm", w("turnOn"));
    b.type = "button";
    b.onclick = () => switchNode(name, true);
    row.append(el("span", "name", name), b);
    box.append(row);
  }
}

async function loadSafe() {
  try { renderSafe(await getJSON("/api/safe-mode")); } catch (e) { /* the next poll tries again */ }
}

function pathLine(p) {
  const cands = p.candidates || [], tried = p.tried || [];
  if (p.status === "searching") {
    const left = cands.filter((f) => !tried.includes(f));
    return w("pathSearching", tried.length + 1, cands.length, features(left.slice(0, 1)));
  }
  if (p.status === "found") return w("pathFound", features(p.off));
  if (p.status === "outside") return cands.length ? w("pathOutside", features(cands)) : w("pathOutsideNone");
  return String(p.status);
}

function renderSafe(s, force) {
  const sig = JSON.stringify([lang, s]);
  if (!force && sig === view.sig.safe) return;
  view.sig.safe = sig;
  Object.assign(safe, { mode: s.mode, file: s.file, set: s.set, paths: s.paths || [] });
  const S = w("safe");
  const searching = safe.paths.some((p) => p.status === "searching");
  $("safe-chip").classList.toggle("searching", searching);
  $("safe-chip-text").textContent = searching ? w("safeSearching") : w("safeChip", S[safe.mode] ? S[safe.mode][0] : safe.mode);
  const box = $("safe-options");
  box.replaceChildren();
  for (const m of ["auto", "all", "off"]) {
    const b = el("button", "radio-card");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(m === safe.mode));
    const title = el("span", "rc-title");
    title.append(el("span", "", S[m][0]));
    if (m === "auto") title.append(el("span", "tag", w("defaultTag")));
    b.append(el("span", "rc-radio"), title, el("span", "rc-desc", S[m][1]));
    b.onclick = () => changeSafe(m);
    box.append(b);
  }
  const file = $("safe-file");
  file.replaceChildren(document.createTextNode(w("settingFile") + " "), el("code", "", safe.file));
  if (!safe.set) file.append(document.createTextNode(" " + w("noSettingFile")));
  file.append(el("div", "env-note", w("envNote")));
  const paths = $("safe-paths");
  paths.replaceChildren();
  if (!safe.paths.length) paths.append(el("p", "muted-line", w("noPaths")));
  for (const p of safe.paths) {
    const st = { searching: "unknown", found: "resolved", outside: "broken" }[p.status] || "none";
    const card = el("div", "path-card st-" + st);
    card.append(el("div", "path-who", engineName(p.engine || "?") + " · " + String(p.model || "?").split("/").pop()),
                el("div", "path-status", pathLine(p)));
    card.title = w("pathKey") + " " + p.key + (p.pairs && p.pairs.length ? "\n" + w("pathPairs") + ": " + p.pairs.join(", ") : "");
    paths.append(card);
  }
}

async function changeSafe(mode) {
  if (mode === safe.mode) return;
  const S = w("safe");
  const ok = await ask({
    title: w("askSafe"), lines: [S[mode][0] + ": " + S[mode][1], w("askSafeNext"), w("envNote")],
    file: w("fileChanged") + ": " + safe.file, okText: w("change"),
  });
  if (!ok) return;
  const r = await post("/api/safe-mode", { mode });
  if (!r.ok) {
    toast(w("writeFailed", r.error));
    return;
  }
  renderSafe(r.data, true);
  toast(w("safeDone", S[mode][0]));
}

// ---------------------------------------------------------------- help
function renderHelp() {
  const body = $("help-body");
  body.replaceChildren(el("h3", "sec-label", w("helpStates")));
  const legend = el("div", "legend");
  for (const s of ["pass", "resolved", "broken", "refused", "unknown", "unchecked", "none"]) {
    const r = el("div", "legend-row st-" + s);
    r.append(el("span", "dot"), el("span", "name", w("st")[s]), el("span", "desc", w("sx")[s]));
    legend.append(r);
  }
  body.append(legend, el("h3", "sec-label", w("helpTerms")));
  const terms = el("dl", "terms");
  for (const [k, v] of w("terms")) {
    const d = el("div");
    d.append(el("dt", "", k), el("dd", "", v));
    terms.append(d);
  }
  body.append(terms, el("h3", "sec-label", w("helpGraph")));
  const ul = el("ul", "help-list");
  for (const t of w("helpGraphItems")) ul.append(el("li", "", t));
  body.append(ul, el("h3", "sec-label", w("helpAbout")), el("p", "help-p", w("helpAboutBody")));
  const a = el("a", "text-link", w("helpDocs"));
  a.href = w("docsUrl");
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  a.append(icon("external"));
  body.append(a);
}

// ---------------------------------------------------------------- start
function init() {
  for (const s of document.querySelectorAll(".icon-slot[data-icon]")) s.replaceChildren(icon(s.dataset.icon));
  applyStatic();
  setLive(null);
  applyPanels();
  renderHelp();
  renderOffNodes();
  renderOverview();
  $("show-all").setAttribute("aria-pressed", String(view.showAll));
  for (const b of document.querySelectorAll(".rail-btn")) b.onclick = () => showTab(b.dataset.tab);
  for (const b of document.querySelectorAll("#lang button")) b.onclick = () => setLang(b.dataset.lang);
  $("safe-chip").onclick = () => {
    view.sideOpen = false;
    showTab("settings");
  };
  $("folder").onclick = async () => {
    try {
      await navigator.clipboard.writeText(view.folder);
      toast(w("copied"));
    } catch (e) {
      toast(w("copyFailed"));
    }
  };
  $("show-all").onclick = () => {
    view.showAll = !view.showAll;
    writePref("entail.showAll", view.showAll ? "1" : "0");
    $("show-all").setAttribute("aria-pressed", String(view.showAll));
    renderBoard();
  };
  $("toggle-detail").onclick = () => {
    if (view.detailOpen) {
      view.detailOpen = false;
      applyPanels();
    } else {
      openDetail();
    }
  };
  $("scrim").onclick = () => {
    view.detailOpen = false;
    if (layoutMode() === "narrow") view.sideOpen = false;
    applyPanels();
  };
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || $("dialog").open) return;
    if (view.node) renderOverview();
    else if (layoutMode() !== "wide" && view.detailOpen) {
      view.detailOpen = false;
      applyPanels();
    }
  });
  let frame = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      if (view.graph && $("canvas").clientWidth !== view.width) renderBoard();
    });
  }).observe($("canvas"));
  for (const mq of [WIDE, NARROW]) mq.addEventListener("change", applyPanels);
  loadRuns();
  loadSafe();
  loadNodes();
  setInterval(() => { loadRuns(); loadSafe(); loadNodes(); }, 3000);
}

init();
