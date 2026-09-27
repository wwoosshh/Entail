// entail's node view (LIBRARY_DESIGN.md 13.5; ROADMAP product track P2). Everything read from the records is put in
// the page as text (textContent), never as markup: a record can hold any string a model or a user produced.
"use strict";

const STATES = {
  pass: ["통과", "pass"], resolved: ["해소됨", "resolved"], unknown: ["보증 못 함", "unknown: not declared or not decided"],
  unchecked: ["검사 못 함", "unchecked: every check skipped"], broken: ["깨짐", "broken: reported, the run went on"],
  refused: ["멈춤", "refused: stopped before output"], none: ["판정 없음", "nothing decided here"],
};
let current = { run: null, node: null, graph: null, source: null };

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

function legend() {
  const box = document.getElementById("legend");
  for (const [state, [ko, en]] of Object.entries(STATES)) {
    const s = el("span");
    s.append(badge(state), document.createTextNode(" " + en));
    box.append(s);
  }
}

legend();
loadRuns();
setInterval(loadRuns, 3000);
