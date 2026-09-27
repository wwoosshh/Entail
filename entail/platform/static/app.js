// entail's node view (LIBRARY_DESIGN.md 13.5; ROADMAP product track P2). Everything read from the records is put in
// the page as text (textContent), never as markup: a record can hold any string a model or a user produced.
"use strict";

const STATES = {
  pass: ["통과", "pass"], resolved: ["해소됨", "resolved"], unknown: ["보증 못 함", "unknown: not declared or not decided"],
  unchecked: ["검사 못 함", "unchecked: every check skipped"], broken: ["깨짐", "broken: reported, the run went on"],
  refused: ["멈춤", "refused: stopped before output"], none: ["판정 없음", "nothing decided here"],
};
let current = { run: null, node: null, graph: null, source: null };
// The safety mode of the next start (LIBRARY_DESIGN.md 13.6): the page's one write, which needs this server's token
// (in the page; a new one comes back with each write).
const SAFE = {
  auto: ["자동: 선택적 안전 경로", "엔진의 경로 점검이 어긋나면, 다음 시작부터 그 어긋남에 걸린 최적화를 하나씩 꺼서 원인을 찾는다. 지금 도는 실행은 그대로 간다."],
  all: ["명시적 안전모드", "결과를 바꾸지 않는다고 선언된 최적화(CUDA 그래프, 프리픽스 캐시, 추측 디코딩, 커스텀 커널)를 다음 시작부터 모두 끈다. 문제가 남으면 원인은 그 밖에 있고, 사라지면 그 안에 있다."],
  off: ["끔", "안전모드를 쓰지 않는다. 어긋남은 그대로 기록만 된다."],
};
const FEATURES = {
  cuda_graphs: "CUDA 그래프", prefix_cache: "프리픽스 캐시", speculative_decoding: "추측 디코딩",
  custom_kernels: "커스텀 커널", attention_kernels: "어텐션 커널",
};
let safe = { token: (document.querySelector('meta[name="entail-token"]') || {}).content || "", mode: null, file: "" };

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = String(text);
  return e;
}

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(url + " " + r.status);
  return r.json();
}

function when(t) {
  if (t === null || t === undefined) return "시각 없음 (판본 1 기록)";
  return new Date(t * 1000).toLocaleString();
}

function badge(state) {
  const [ko, en] = STATES[state] || [state, state];
  const b = el("span", "badge b-" + state, ko);
  b.title = en;
  return b;
}

async function loadRuns() {
  let data;
  try { data = await getJSON("/api/runs"); } catch (e) { return; }
  document.getElementById("folder").textContent = data.folder;
  const list = document.getElementById("runlist");
  list.replaceChildren();
  if (!data.runs.length) {
    list.append(el("li", "hint", "아직 기록이 없다. ENTAIL=load로 프로그램을 돌리면 여기에 나온다."));
    return;
  }
  for (const r of data.runs) {
    const li = el("li", r.run === current.run ? "sel" : "");
    li.append(el("div", "", (r.engines.join(", ") || "entail")), badge(r.state));
    li.append(el("div", "when", when(r.start) + " · 판정 " + r.decisions));
    li.title = r.run;
    li.onclick = () => selectRun(r.run);
    list.append(li);
  }
  if (!current.run || !data.runs.some(r => r.run === current.run)) selectRun(data.runs[0].run);
}

function selectRun(run) {
  if (current.source) current.source.close();
  current = { run, node: null, graph: null, source: null };
  for (const li of document.querySelectorAll("#runlist li")) li.classList.toggle("sel", li.title === run);
  const src = new EventSource("/api/events?run=" + encodeURIComponent(run));
  src.addEventListener("graph", e => renderGraph(JSON.parse(e.data)));
  src.onopen = () => document.getElementById("live").classList.add("on");
  src.onerror = () => document.getElementById("live").classList.remove("on");
  current.source = src;
  document.getElementById("dbody").replaceChildren(el("span", "hint", "노드를 누르면 어디서, 왜가 여기에 나온다."));
}

function nodeName(g, id) {
  const n = g.nodes.find(x => x.id === id);
  return n ? n.ko : id;
}

function renderGraph(g) {
  current.graph = g;
  const banner = document.getElementById("banner");
  const loc = g.locate || {};
  if (loc.broken_at) {
    const n = g.nodes.find(x => x.boundaries.includes(loc.broken_at));
    banner.className = "banner broken";
    banner.textContent = "처음 깨진 곳: " + (n ? n.ko : "?") + " — " + loc.broken_at +
      (loc.broken.length > 1 ? " (그 밖에 " + (loc.broken.length - 1) + "곳)" : "");
  } else if (loc.lost_by && loc.lost_by.length) {
    banner.className = "banner broken";
    banner.textContent = "옮기는 도중에 뜻을 잃음: " + loc.lost_by[0];
  } else if (g.located && g.located.suspects && g.located.suspects.length) {
    banner.className = "banner broken";
    banner.textContent = "판정한 경계는 뜻을 지켰다. 프로그램이 짚은 곳: " + (g.located.why[0] || g.located.suspects[0]);
  } else if (loc.all_intact) {
    banner.className = "banner intact";
    banner.textContent = "판정한 경계는 모두 뜻을 지켰다 (" + loc.intact.length + "곳)" +
      (loc.unchecked.length ? " · 검사 못 한 경계 " + loc.unchecked.length + "곳" : "");
  } else {
    banner.className = "banner";
    banner.textContent = "판정한 경계가 아직 없다.";
  }
  const flows = document.getElementById("flows");
  flows.replaceChildren();
  for (const f of g.flows) {
    const box = el("div", "flow");
    box.append(el("h3", "", f.ko + " · " + f.en));
    const row = el("div", "row");
    f.nodes.forEach((id, i) => {
      const n = g.nodes.find(x => x.id === id);
      if (!n) return;
      if (i > 0) row.append(el("span", "arrow", "→"));
      const card = el("div", "node st-" + n.state + (current.node === id ? " sel" : ""));
      card.append(el("div", "name", n.ko), el("div", "en", n.en), badge(n.state));
      const decided = Object.values(n.verdicts).reduce((a, b) => a + b, 0);
      const parts = ["판정 " + decided];
      if (n.checks) parts.push("검사 " + n.checks);
      if (n.passed) parts.push("통과 " + n.passed);
      if (n.skipped) parts.push("건너뜀 " + n.skipped);
      if (n.timed_calls) parts.push(n.ms.toFixed(1) + " ms");
      card.append(el("div", "counts", parts.join(" · ")));
      card.title = n.boundaries.join("\n") || "이 실행에서 판정한 경계가 없다";
      card.onclick = () => selectNode(id);
      row.append(card);
    });
    box.append(row);
    flows.append(box);
  }
  if (current.node) selectNode(current.node, true);
}

function kv(label, value) {
  const d = el("div", "kv");
  d.append(el("b", "", label), document.createTextNode(value === null || value === undefined ? "-" : String(value)));
  return d;
}

function fact(f) {
  if (!f) return null;
  const src = f.source ? (f.source.kind + (f.source.where ? ": " + f.source.where : "")) : "";
  return (f.value === null ? "모름" : f.value) + (src ? "  (" + src + ", " + f.certainty + ")" : "");
}

async function selectNode(id, quiet) {
  current.node = id;
  for (const c of document.querySelectorAll(".node")) c.classList.remove("sel");
  if (!quiet && current.graph) renderGraph(current.graph);
  let d;
  try {
    d = await getJSON("/api/node?run=" + encodeURIComponent(current.run) + "&node=" + encodeURIComponent(id));
  } catch (e) { return; }
  document.getElementById("dtitle").textContent = d.ko + " · " + d.en;
  const body = document.getElementById("dbody");
  body.replaceChildren();
  if (!d.decisions.length) body.append(el("div", "hint", "판정으로 남은 것이 없다. 통과는 세기만 하고 적지 않는다(아래 집계)."));
  const order = { refused: 0, broken: 1, unknown: 2, resolved: 3, pass: 4 };
  const decs = d.decisions.slice().sort((a, b) => (order[a.verdict] ?? 9) - (order[b.verdict] ?? 9));
  for (const x of decs) {
    const box = el("div", "dec");
    const head = el("div", "head");
    head.append(badge(x.verdict in STATES ? x.verdict : "unknown"), el("strong", "", x.name), el("span", "bnd", x.boundary));
    box.append(head);
    box.append(kv("규칙", x.rule));
    if (x.declared) box.append(kv("선언", fact(x.declared)));
    if (x.chosen) box.append(kv("선택", fact(x.chosen)));
    if (x.observed) box.append(kv("관찰", fact(x.observed)));
    if (x.resolution) box.append(kv("해소", x.resolution + (x.handle ? " (" + x.handle + ")" : "")));
    if (x.lost_by) box.append(kv("잃은 곳", x.lost_by));
    if (x.note) box.append(kv("메모", x.note));
    if (x.blocking) box.append(kv("멈춤", "이 판정에서 멈췄다"));
    body.append(box);
  }
  const bs = Object.entries(d.counts);
  if (bs.length) {
    const t = el("table", "small");
    const hr = el("tr");
    for (const h of ["경계", "검사", "통과", "건너뜀"]) hr.append(el("th", "", h));
    t.append(hr);
    for (const [b, c] of bs) {
      const tr = el("tr");
      for (const v of [b, c.checks, c.passed, c.skipped]) tr.append(el("td", "", v));
      t.append(tr);
    }
    body.append(el("h2", "", "집계 · counts"), t);
  }
  const ts = Object.entries(d.timing);
  if (ts.length) {
    const t = el("table", "small");
    for (const [b, c] of ts) {
      const tr = el("tr");
      for (const v of [b, c.calls + "회", c.ms.toFixed(2) + " ms"]) tr.append(el("td", "", v));
      t.append(tr);
    }
    body.append(el("h2", "", "검사 시간 · time spent"), t);
  }
  for (const s of d.said) body.append(kv("말한 것 (" + s.where + ")", s.text));
}

function features(list) {
  return (list || []).map(f => FEATURES[f] || f).join(", ");
}

function pathLine(p) {
  const cands = p.candidates || [], tried = p.tried || [];
  if (p.status === "searching") {
    const left = cands.filter(f => !tried.includes(f));
    return "원인 찾는 중 (" + (tried.length + 1) + "/" + cands.length + "), 꺼 두는 기능: " + features(left.slice(0, 1));
  }
  if (p.status === "found") return "원인: " + features(p.off) + " (이 설정은 그것을 끄고 시작한다)";
  if (p.status === "outside") {
    return cands.length ? "후보 밖: " + features(cands) + "을(를) 하나씩 꺼도 경로가 어긋났다 (다시 켰다)"
                        : "어긋난 짝에 걸린 최적화가 켜져 있지 않았다";
  }
  return String(p.status);
}

function renderSafe(s) {
  const sel = document.getElementById("safemode");
  if (s.mode !== safe.mode) sel.value = s.mode;      // only when the file changed: never under a choice being made
  safe.mode = s.mode;
  safe.file = s.file;
  document.getElementById("safenote").textContent = SAFE[s.mode][1] +
    (s.set ? "" : " (설정 파일이 없어 기본값이다.)") + " 엔진에 환경 변수 ENTAIL_SAFE가 있으면 그것이 먼저다.";
  const list = document.getElementById("safepaths");
  list.replaceChildren();
  for (const p of s.paths) {
    const li = el("li");
    li.append(el("div", "who", (p.engine || "?") + " · " + String(p.model || "?").split("/").pop()),
              el("div", "s-" + p.status, pathLine(p)));
    li.title = "설정 열쇠 " + p.key + (p.pairs && p.pairs.length ? "\n어긋난 짝: " + p.pairs.join(", ") : "");
    list.append(li);
  }
}

async function loadSafe() {
  try { renderSafe(await getJSON("/api/safe-mode")); } catch (e) { /* the next poll tries again */ }
}

async function changeSafe() {
  const sel = document.getElementById("safemode");
  const mode = sel.value;
  if (mode === safe.mode) return;
  const ok = window.confirm("안전모드를 '" + SAFE[mode][0] + "'(으)로 바꾼다.\n\n" + SAFE[mode][1] +
    "\n\n다음에 시작하는 엔진부터 쓰인다. 지금 도는 엔진은 바뀌지 않는다. 엔진에 환경 변수 ENTAIL_SAFE가 있으면 그것이 먼저다." +
    "\n\n바꾸는 파일: " + safe.file);
  if (!ok) { sel.value = safe.mode; return; }
  let r, d = {};
  try {
    r = await fetch("/api/safe-mode", {
      method: "POST", cache: "no-store", body: JSON.stringify({ mode }),
      headers: { "Content-Type": "application/json", "X-Entail-Token": safe.token },
    });
    d = await r.json();
  } catch (e) { r = r || { ok: false, status: "?" }; }
  if (!r.ok) {
    sel.value = safe.mode;
    document.getElementById("safenote").textContent = "바꾸지 못했다: " + (d.error || r.status);
    return;
  }
  safe.token = d.token;
  renderSafe(d);
}

function legend() {
  const box = document.getElementById("legend");
  for (const [state, [ko, en]] of Object.entries(STATES)) {
    const s = el("span");
    s.append(badge(state), document.createTextNode(" " + en));
    box.append(s);
  }
}

legend();
document.getElementById("safemode").onchange = changeSafe;
loadRuns();
loadSafe();
setInterval(() => { loadRuns(); loadSafe(); }, 3000);
