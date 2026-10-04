/* ML From Scratch: local course app. Talks to course.py (same origin). */
"use strict";

const S = { units: [], progress: null, today: "", git: { changes: [] }, python: "" };
const view = { unit: null, cleanup: [] };
const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
const $ = (sel, el = document) => el.querySelector(sel);

// ------------------------------------------------------------------ helpers

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

function fmtMinutes(m) {
  if (!m) return "0 min";
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`;
}

function addDays(iso, n) {
  // Local-date arithmetic (toISOString would shift to UTC and change the day in non-UTC timezones).
  const [y, m, d] = iso.split("-").map(Number);
  const date = new Date(y, m - 1, d + n);
  const pad = v => String(v).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function unitById(id) { return S.units.find(u => u.id === id); }
function uprog(id) { return (S.progress.units[id] = S.progress.units[id] || { checks: {}, attempts: [] }); }

function stepCounts(unit) {
  const checks = uprog(unit.id).checks;
  const done = unit.steps.filter(s => checks[s.id]).length;
  return { done, total: unit.steps.length };
}

function unitStatus(unit) {
  const c = uprog(unit.id).checks;
  const { done, total } = stepCounts(unit);
  if (unit.kind !== "algorithm") return done === total ? ["Done", "ok"] : done ? ["In progress", "accent"] : ["Not started", ""];
  if (c.ready) return ["Interview-ready", "ok"];
  if (c.spaced) return ["Spaced repeat", "accent"];
  if (c.timed) return ["Timed", "accent"];
  if (c.blank) return ["Blank file", "accent"];
  if (c.practice) return ["Practised", "accent"];
  if (done) return ["Learning", "warn"];
  return ["Not started", ""];
}

function lastPractised(unitId) {
  const runs = S.progress.runs.filter(r => r.unit === unitId && r.target !== "solution" && r.target !== "explained");
  return runs.length ? runs[runs.length - 1].at.slice(0, 10) : "";
}

function bestTime(unitId) {
  const ok = uprog(unitId).attempts.filter(a => a.passed && !a.peeked).map(a => a.minutes);
  return ok.length ? Math.min(...ok) : null;
}

function nextReview(unit) {
  const passed = uprog(unit.id).attempts.filter(a => a.passed);
  if (!passed.length) return "";
  const last = passed[passed.length - 1].date;
  return addDays(last, uprog(unit.id).checks.spaced ? 7 : 3);
}

// ------------------------------------------------------------------ markdown

function renderMarkdown(src) {
  const code = [], math = [];
  src = src.replace(/```[\s\S]*?```|`[^`\n]*`/g, m => { code.push(m); return `CODEPH${code.length - 1}END`; });
  src = src.replace(/\$\$([\s\S]+?)\$\$/g, (m, t) => { math.push([t, true]); return `MATHPH${math.length - 1}END`; });
  src = src.replace(/\$([^$\n]+?)\$/g, (m, t) => { math.push([t, false]); return `MATHPH${math.length - 1}END`; });
  src = src.replace(/CODEPH(\d+)END/g, (m, i) => code[+i]);
  let html = marked.parse(src);
  html = html.replace(/MATHPH(\d+)END/g, (m, i) => {
    const [t, display] = math[+i];
    try { return katex.renderToString(t, { displayMode: display, throwOnError: false }); }
    catch (e) { return esc(t); }
  });
  return html;
}

function resolvePath(baseFile, rel) {
  const parts = baseFile.split("/").slice(0, -1);
  for (const p of rel.split("/")) {
    if (p === "..") parts.pop();
    else if (p && p !== ".") parts.push(p);
  }
  return parts.join("/");
}

function routeForFile(path) {
  const [unitId, ...rest] = path.split("/");
  const unit = unitById(unitId);
  if (!unit) return null;
  const file = rest.join("/");
  if (!file) return `#/unit/${unitId}/learn`;
  if (file.endsWith(".md")) {
    if (file === "LECTURE.md") return `#/unit/${unitId}/learn`;
    if (file === "LOG.md") return `#/unit/${unitId}/notes`;
    return `#/unit/${unitId}/learn/${file}`;
  }
  const stem = file.replace(/\.py$/, "");
  if (["solution", "explained", "example"].includes(stem)) return `#/unit/${unitId}/code/${stem}`;
  if (["practice", "my_practice"].includes(stem)) return `#/unit/${unitId}/lab/practice`;
  return null;
}

function mountMarkdown(el, src, filePath) {
  el.innerHTML = renderMarkdown(src);
  el.querySelectorAll("a[href]").forEach(a => {
    const href = a.getAttribute("href");
    if (/^(https?:|mailto:|#)/.test(href)) {
      if (/^https?:/.test(href)) a.target = "_blank";
      return;
    }
    const target = resolvePath(filePath, href.split("#")[0]);
    const route = routeForFile(target.replace(/\/$/, ""));
    if (route) a.setAttribute("href", route);
    else { a.removeAttribute("href"); a.title = target; }
  });
}

// ------------------------------------------------------------------ chrome

function renderSidebar() {
  const route = location.hash;
  let html = `<a class="nav-item ${route === "#/" || route === "" ? "active" : ""}" href="#/"><div class="title">Dashboard</div></a>`;
  let section = null;
  for (const u of S.units) {
    const sec = u.kind === "foundations" ? "Start here" : u.kind === "explain" ? "Theory only" : u.priority ? "Priority set" : "Later";
    if (sec !== section) { html += `<div class="nav-section">${sec}</div>`; section = sec; }
    const { done, total } = stepCounts(u);
    const pct = Math.round(100 * done / total);
    const active = view.unit === u.id ? "active" : "";
    html += `<a class="nav-item ${active}" href="#/unit/${u.id}">
      <div class="week">${esc(u.week)}</div>
      <div class="title"><span>${esc(u.title)}</span><span class="muted">${done}/${total}</span></div>
      <div class="bar ${done === total ? "done" : ""}"><span style="width:${pct}%"></span></div></a>`;
  }
  $("#sidebar").innerHTML = html;
}

function renderTopbar() {
  const mins = Object.values(S.progress.time[S.today] || {}).reduce((a, b) => a + b, 0);
  $("#today-time").textContent = `Today: ${fmtMinutes(mins)}`;
  const n = S.git.changes ? S.git.changes.length : 0;
  $("#commit-btn").innerHTML = `Commit progress${n ? `<span class="badge">${n}</span>` : ""}`;
}

async function refresh() {
  const data = await api("/api/state");
  Object.assign(S, data);
  renderSidebar();
  renderTopbar();
}

// ------------------------------------------------------------------ dashboard

function renderDashboard() {
  view.unit = null;
  const all = S.units.reduce((acc, u) => { const c = stepCounts(u); acc.d += c.done; acc.t += c.total; return acc; }, { d: 0, t: 0 });
  const algos = S.units.filter(u => u.kind === "algorithm");
  const ready = algos.filter(u => uprog(u.id).checks.ready).length;
  const days = [];
  for (let i = 13; i >= 0; i--) days.push(addDays(S.today, -i));
  const perDay = days.map(d => Object.values(S.progress.time[d] || {}).reduce((a, b) => a + b, 0));
  const week = perDay.slice(-7).reduce((a, b) => a + b, 0);
  let streak = 0;
  for (let i = days.length - 1; i >= 0 && perDay[i] > 0; i--) streak++;
  const maxDay = Math.max(30, ...perDay);

  // What to do next: the first incomplete step of the first incomplete unit, or a due review.
  const due = algos.filter(u => { const r = nextReview(u); return r && r <= S.today && !uprog(u.id).checks.ready; });
  let next = null;
  for (const u of S.units) {
    const step = u.steps.find(s => !uprog(u.id).checks[s.id]);
    if (step) { next = { unit: u, step }; break; }
  }
  const nextHref = next ? hrefForStep(next.unit, next.step) : "#/";

  const rows = S.units.map(u => {
    const [label, cls] = unitStatus(u);
    const best = bestTime(u.id);
    const review = u.kind === "algorithm" ? nextReview(u) : "";
    const reviewCell = review ? (review <= S.today ? `<span class="pill warn">due ${review}</span>` : review) : "";
    return `<tr>
      <td><a href="#/unit/${u.id}">${esc(u.title)}</a></td>
      <td><span class="pill ${cls}">${label}</span></td>
      <td>${stepCounts(u).done}/${stepCounts(u).total}</td>
      <td>${best !== null ? best + " min" : ""}</td>
      <td>${u.target_minutes ? u.target_minutes + " min" : ""}</td>
      <td>${lastPractised(u.id)}</td>
      <td>${reviewCell}</td></tr>`;
  }).join("");

  $("#main").innerHTML = `
    <h1>Dashboard</h1>
    <p class="muted">Goal: write each algorithm from a blank file within its target time, without looking anything up.</p>
    ${next ? `<div class="card continue"><div><div class="muted">Up next · ${esc(next.unit.title)}</div>
      <strong>${esc(next.step.label)}</strong></div><a class="btn primary" href="${nextHref}">Continue →</a></div>` : ""}
    ${due.length ? `<div class="card continue" style="border-left-color:var(--warn)"><div><div class="muted">Spaced repetition</div>
      <strong>Review due: ${due.map(u => esc(u.short)).join(", ")}</strong></div>
      <a class="btn" href="#/unit/${due[0].id}/lab/attempt">Start a timed attempt →</a></div>` : ""}
    <div class="grid">
      <div class="card stat"><div class="num">${all.d}/${all.t}</div><div class="lbl">steps complete</div></div>
      <div class="card stat"><div class="num">${ready}/${algos.length}</div><div class="lbl">algorithms interview-ready</div></div>
      <div class="card stat"><div class="num">${fmtMinutes(week)}</div><div class="lbl">studied in the last 7 days</div></div>
      <div class="card stat"><div class="num">${streak} day${streak === 1 ? "" : "s"}</div><div class="lbl">current streak</div></div>
    </div>
    <h2>Progress</h2>
    <div class="card table-wrap"><table class="data">
      <thead><tr><th>Unit</th><th>Status</th><th>Steps</th><th>Best time</th><th>Target</th><th>Last practised</th><th>Next review</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <h2>Study time, last 14 days</h2>
    <div class="card"><div class="chart">${perDay.map((m, i) => `<div class="col" title="${days[i]}: ${fmtMinutes(m)}">
      <div class="b ${m ? "" : "zero"}" style="height:${Math.max(2, 90 * m / maxDay)}px"></div>
      <div class="d">${days[i].slice(8)}</div></div>`).join("")}</div>
      <p class="muted" style="margin:8px 0 0;font-size:13px">Counted automatically while this app is open and you're active.</p></div>
    <h2>Journal</h2>
    <p class="muted">Free-form daily notes, saved to <code>LOG.md</code>.</p>
    <textarea class="notes" id="journal"></textarea>
    <div class="row" style="margin-top:8px"><button class="btn" id="journal-save">Save journal</button><span class="muted" id="journal-status"></span></div>
    <p class="muted" style="margin-top:32px;font-size:13px">Checks run with <code>${esc(S.python)}</code>. Progress lives in <code>progress.json</code>; use <b>Commit progress</b> to save it in git.</p>`;

  api("/api/file?path=LOG.md").then(d => { $("#journal").value = d.content; }).catch(() => {});
  $("#journal-save").onclick = async () => {
    await api("/api/file", { path: "LOG.md", content: $("#journal").value });
    $("#journal-status").textContent = "Saved.";
    refresh();
  };
}

function hrefForStep(unit, step) {
  const id = step.id;
  if (id.startsWith("read:") || id.startsWith("explain:")) return `#/unit/${unit.id}/learn/${id.split(":")[1]}`;
  return {
    lecture: `#/unit/${unit.id}/learn`, tutorial: `#/unit/${unit.id}/learn`, explained: `#/unit/${unit.id}/code/explained`,
    practice: `#/unit/${unit.id}/lab/practice`, drills: `#/unit/${unit.id}/lab/practice`, blank: `#/unit/${unit.id}/lab/blank`,
  }[id] || `#/unit/${unit.id}/lab/attempt`;
}

// ------------------------------------------------------------------ unit page

function tabsFor(unit) {
  if (unit.kind === "explain") return [["learn", "Notes"]];
  const tabs = [["learn", unit.kind === "foundations" ? "Lectures" : "Lecture"], ["code", "Code walkthrough"], ["lab", "Practice lab"]];
  if (unit.kind === "algorithm") tabs.push(["notes", "Notes & history"]);
  return tabs;
}

function renderUnit(unitId, tab, sub) {
  const unit = unitById(unitId);
  if (!unit) return renderDashboard();
  view.unit = unitId;
  const tabs = tabsFor(unit);
  tab = tabs.some(t => t[0] === tab) ? tab : "learn";
  const checks = uprog(unitId).checks;

  const steps = unit.steps.map(s => `
    <label class="step ${checks[s.id] ? "done" : ""}" title="${s.auto ? "Ticked automatically when the checks pass" : "Tick when done"}">
      <input type="checkbox" data-step="${esc(s.id)}" ${checks[s.id] ? "checked" : ""} ${s.auto ? "disabled" : ""}>
      <span>${esc(s.label)}${checks[s.id] ? `<span class="date">${checks[s.id]}</span>` : ""}</span>
      ${s.auto ? `<span class="auto">auto</span>` : ""}
    </label>`).join("");

  $("#main").innerHTML = `
    <div class="unit-head"><div class="week">${esc(unit.week)}${unit.target_minutes ? ` · target ${unit.target_minutes} min` : ""}</div>
      <h1>${esc(unit.title)}</h1></div>
    <div class="steps">${steps}</div>
    <div class="tabs">${tabs.map(([id, label]) => `<a class="tab ${id === tab ? "active" : ""}" href="#/unit/${unitId}/${id}">${label}</a>`).join("")}</div>
    <div id="tab-body"></div>`;

  $("#main").querySelectorAll("input[data-step]").forEach(cb => {
    cb.onchange = async () => {
      await api("/api/check", { unit: unitId, step: cb.dataset.step, done: cb.checked });
      await refresh();
      renderUnit(unitId, tab, sub);
    };
  });

  const body = $("#tab-body");
  if (tab === "learn") renderLearn(unit, body, sub);
  else if (tab === "code") renderCode(unit, body, sub || "explained");
  else if (tab === "lab") renderLab(unit, body, sub);
  else if (tab === "notes") renderNotes(unit, body);
}

async function renderLearn(unit, body, sub) {
  let file = "LECTURE.md";
  let nav = "";
  if (unit.readings) {
    file = unit.readings.some(r => r.file === sub) ? sub : unit.readings[0].file;
    nav = `<div class="reading-nav">${unit.readings.map(r =>
      `<a class="btn ${r.file === file ? "primary" : ""}" href="#/unit/${unit.id}/learn/${r.file}">${esc(r.title)}</a>`).join("")}</div>`;
  }
  const path = `${unit.id}/${file}`;
  const editable = unit.kind === "explain";
  body.innerHTML = `${nav}${editable ? `<div class="row" style="margin-bottom:12px"><button class="btn" id="edit-md">Edit notes</button>
    <span class="muted">Fill these in yourself, in your own words.</span></div>` : ""}<article class="md" id="md"></article>`;
  const { content } = await api(`/api/file?path=${encodeURIComponent(path)}`);
  mountMarkdown($("#md"), content, path);
  if (editable) {
    $("#edit-md").onclick = () => {
      body.querySelector(".row").innerHTML = `<button class="btn primary" id="save-md">Save</button><button class="btn ghost" id="cancel-md">Cancel</button>`;
      $("#md").outerHTML = `<textarea class="notes" id="md-edit">${esc(content)}</textarea>`;
      $("#cancel-md").onclick = () => renderLearn(unit, body, file);
      $("#save-md").onclick = async () => {
        await api("/api/file", { path, content: $("#md-edit").value });
        toast("Notes saved.", "ok");
        await refresh();
        renderLearn(unit, body, file);
      };
    };
  }
}

function makeEditor(host, value, readOnly) {
  const cm = CodeMirror(host, {
    value, mode: "python", lineNumbers: true, indentUnit: 4, tabSize: 4, matchBrackets: true, readOnly,
    theme: dark ? "material-darker" : "default",
    extraKeys: { Tab: c => c.somethingSelected() ? c.indentSelection("add") : c.replaceSelection("    ", "end") },
  });
  setTimeout(() => cm.refresh(), 0);
  return cm;
}

async function renderCode(unit, body, which) {
  const files = [["explained", "explained.py · line by line"], ["solution", "solution.py"], ["example", "example.py · usage & checks"]];
  if (!files.some(f => f[0] === which)) which = "explained";
  body.innerHTML = `
    <div class="row" style="margin-bottom:12px">
      <div class="seg">${files.map(([id, label]) => `<button data-f="${id}" class="${id === which ? "active" : ""}">${label}</button>`).join("")}</div>
      <span class="spacer"></span>
      <button class="btn" id="run-solution">Run example on the solution</button>
    </div>
    <div class="editor-wrap readonly"><div id="viewer"></div></div>
    <div id="run-out"></div>`;
  body.querySelectorAll(".seg button").forEach(b => b.onclick = () => { location.hash = `#/unit/${unit.id}/code/${b.dataset.f}`; });
  const { content } = await api(`/api/file?path=${encodeURIComponent(`${unit.id}/${which}.py`)}`);
  makeEditor($("#viewer"), content, true);
  $("#run-solution").onclick = () => runChecks(unit, "solution", $("#run-out"), $("#run-solution"));
}

async function runChecks(unit, target, outEl, btn, peeked = false) {
  if (btn) { btn.disabled = true; btn.textContent = "Running…"; }
  outEl.innerHTML = `<p class="muted">Running <code>python ${esc(unit.id)}/example.py ${esc(target)}</code>…</p>`;
  try {
    const r = await api("/api/run", { unit: unit.id, target, peeked });
    outEl.innerHTML = `<div class="result-line ${r.passed ? "pass" : "fail"}">${r.passed ? "✓ All checks passed" : "✗ Not passing yet"} · ${r.seconds}s</div>
      <pre class="output ${r.passed ? "pass" : "fail"}">${esc(r.output)}</pre>`;
    r.events.forEach(e => toast(e, "ok"));
    await refresh();
    return r;
  } catch (e) {
    outEl.innerHTML = `<pre class="output fail">${esc(e.message)}</pre>`;
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "Run checks"; }
  }
}

function renderLab(unit, body, sub) {
  const files = unit.files || {};
  const active = S.progress.active_attempt && S.progress.active_attempt.unit === unit.id ? S.progress.active_attempt : null;
  const modes = unit.kind === "foundations" ? [["practice", "NumPy drills"]]
    : [["practice", "1 · Practice (hints)"], ["blank", "2 · Blank rewrite"], ["attempt", "3 · Timed attempts"]];
  let mode = (sub || "").split(":")[0];
  if (!modes.some(m => m[0] === mode)) mode = active ? "attempt" : "practice";

  let target = null;
  if (mode === "practice" && files.my_practice) target = "my_practice";
  if (mode === "blank" && files.day2_blank) target = "day2_blank";
  if (mode === "attempt") {
    const named = (sub || "").split(":")[1];
    if (named && files.attempts.includes("attempts/" + named)) target = "attempts/" + named;
    else if (active) target = active.file;
  }

  const seg = `<div class="seg">${modes.map(([id, label]) => `<button data-m="${id}" class="${id === mode ? "active" : ""}">${label}</button>`).join("")}</div>`;
  const attemptsList = mode === "attempt" && files.attempts.length
    ? `<select id="attempt-pick" class="btn">${["<option value=''>Past attempts…</option>"].concat(files.attempts.slice().reverse().map(a =>
      `<option value="${esc(a.slice(9))}" ${"attempts/" + a.slice(9) === target ? "selected" : ""}>${esc(a.slice(9))}</option>`)).join("")}</select>` : "";
  body.innerHTML = `<div class="row" style="margin-bottom:12px">${seg}<span class="spacer"></span>${attemptsList}</div><div id="lab-body"></div>`;
  body.querySelectorAll(".seg button").forEach(b => b.onclick = () => { location.hash = `#/unit/${unit.id}/lab/${b.dataset.m}`; });
  if ($("#attempt-pick")) $("#attempt-pick").onchange = e => { if (e.target.value) location.hash = `#/unit/${unit.id}/lab/attempt:${e.target.value}`; };

  const labBody = $("#lab-body");
  if (!target) return renderLabStart(unit, mode, labBody);
  renderEditorLab(unit, target, labBody, active && active.file === target ? active : null);
}

function renderLabStart(unit, mode, el) {
  const info = {
    practice: ["Practice with hints", unit.kind === "foundations"
      ? "Twelve one-line NumPy functions. You get your own copy of the drills (my_practice.py); the template stays untouched."
      : "You get your own copy of practice.py (saved as my_practice.py). Write one line under each numbered hint, then run the checks.", "Start practice"],
    blank: ["Blank-file rewrite", "Write the whole implementation from memory in day2_blank.py. Peek at the solution only when stuck, and note where in Notes & history.", "Start blank rewrite"],
    attempt: ["Timed attempt", `A fresh file in attempts/ and a running timer. No peeking: if you open the solution, the attempt is marked as peeked. Pass the checks under ${unit.target_minutes} minutes to become interview-ready.`, "Start timed attempt"],
  }[mode];
  el.innerHTML = `<div class="card empty-state"><h3>${info[0]}</h3><p class="muted">${info[1]}</p>
    <button class="btn primary" id="start-work">${info[2]}</button></div>`;
  $("#start-work").onclick = async () => {
    const r = await api("/api/workfile", { unit: unit.id, kind: mode });
    await refresh();
    location.hash = mode === "attempt" ? `#/unit/${unit.id}/lab/attempt:${r.target.slice(9)}` : `#/unit/${unit.id}/lab/${mode}`;
    if (mode === "attempt") toast("Timer started. Good luck!");
  };
}

async function renderEditorLab(unit, target, el, attempt) {
  const path = `${unit.id}/${target}.py`;
  const isAttempt = target.startsWith("attempts/");
  el.innerHTML = `
    ${isAttempt && !attempt ? `<p class="muted">This is a past attempt (not timed). You can still edit and run it.</p>` : ""}
    <div class="lab" id="lab">
      <div class="editor-wrap">
        <div class="editor-bar">
          <span class="file">${esc(path)}</span>
          ${attempt ? `<span class="timer" id="timer">00:00</span>` : ""}
          <span class="muted" id="save-state"></span>
          <span class="spacer"></span>
          ${target === "my_practice" ? `<button class="btn ghost danger" id="reset-btn">Reset</button>` : ""}
          ${attempt ? `<button class="btn ghost" id="cancel-attempt">Abandon attempt</button>` : ""}
          <button class="btn ghost" id="peek-btn">${attempt ? "Peek (marks as peeked)" : "Show solution"}</button>
          <button class="btn primary" id="run-btn" title="Ctrl+Enter">Run checks</button>
        </div>
        <div id="editor"></div>
      </div>
      <div class="editor-wrap readonly hidden" id="peek-wrap">
        <div class="editor-bar"><span class="file">${esc(unit.id)}/explained.py</span><span class="spacer"></span>
          <button class="btn ghost" id="peek-close">Hide</button></div>
        <div id="peek"></div>
      </div>
    </div>
    <div id="run-out"><p class="muted">Ctrl+S saves (it also autosaves), Ctrl+Enter saves and runs <code>python ${esc(unit.id)}/example.py ${esc(target)}</code>.</p></div>`;

  const { content } = await api(`/api/file?path=${encodeURIComponent(path)}`);
  const cm = makeEditor($("#editor"), content, false);
  let saved = content, saveTimer = null;
  const save = async () => {
    clearTimeout(saveTimer);
    const value = cm.getValue();
    if (value === saved) return;
    await api("/api/file", { path, content: value });
    saved = value;
    $("#save-state") && ($("#save-state").textContent = "Saved");
    refresh();
  };
  cm.on("change", () => {
    $("#save-state").textContent = "Unsaved…";
    clearTimeout(saveTimer);
    saveTimer = setTimeout(save, 1500);
  });
  view.cleanup.push(save);
  let peeked = false;
  const run = async () => { await save(); runChecks(unit, target, $("#run-out"), $("#run-btn"), peeked); };
  cm.setOption("extraKeys", Object.assign({}, cm.getOption("extraKeys"), { "Ctrl-S": save, "Cmd-S": save, "Ctrl-Enter": run, "Cmd-Enter": run }));
  $("#run-btn").onclick = run;

  if ($("#reset-btn")) $("#reset-btn").onclick = async () => {
    if (!confirm("Replace my_practice.py with a fresh copy of the hints template? Your current code will be lost (unless committed).")) return;
    await api("/api/reset", { unit: unit.id });
    renderEditorLab(unit, target, el, attempt);
  };

  let peekEditor = null;
  $("#peek-btn").onclick = async () => {
    if (attempt && !confirm("Peeking marks this timed attempt as 'peeked' (it won't count towards interview-ready). Peek anyway?")) return;
    if (attempt) { peeked = true; await api("/api/peek", {}); }
    $("#peek-wrap").classList.remove("hidden");
    $("#lab").classList.add("split");
    if (!peekEditor) {
      const r = await api(`/api/file?path=${encodeURIComponent(unit.id + "/explained.py")}`);
      peekEditor = makeEditor($("#peek"), r.content, true);
    }
    cm.refresh();
  };
  $("#peek-close").onclick = () => { $("#peek-wrap").classList.add("hidden"); $("#lab").classList.remove("split"); cm.refresh(); };

  if (attempt) {
    if (attempt.peeked) peeked = true;
    const started = new Date(attempt.started);
    const tick = () => {
      const t = $("#timer");
      if (!t) return;
      const s = Math.max(0, Math.floor((Date.now() - started) / 1000));
      t.textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
      t.classList.toggle("over", s / 60 > unit.target_minutes);
    };
    tick();
    const id = setInterval(tick, 1000);
    view.cleanup.push(() => clearInterval(id));
    $("#cancel-attempt").onclick = async () => {
      if (!confirm("Stop the timer? The file stays in attempts/, but this attempt won't be recorded.")) return;
      await api("/api/attempt/cancel", {});
      await refresh();
      renderUnit(unit.id, "lab", "attempt:" + target.slice(9));
    };
  }
  cm.focus();
}

async function renderNotes(unit, body) {
  const attempts = uprog(unit.id).attempts.slice().reverse();
  const runs = S.progress.runs.filter(r => r.unit === unit.id).slice(-15).reverse();
  body.innerHTML = `
    <h2 style="margin-top:0">Timed attempts</h2>
    ${attempts.length ? `<div class="card table-wrap"><table class="data"><thead><tr><th>Date</th><th>File</th><th>Minutes</th><th>Peeked?</th></tr></thead><tbody>
      ${attempts.map(a => `<tr><td>${a.date}</td><td><a href="#/unit/${unit.id}/lab/attempt:${esc(a.file.slice(9))}">${esc(a.file)}</a></td>
      <td>${a.minutes}${a.minutes <= unit.target_minutes && !a.peeked ? ` <span class="pill ok">under target</span>` : ""}</td>
      <td>${a.peeked ? "yes" : "no"}</td></tr>`).join("")}</tbody></table></div>`
      : `<p class="muted">No timed attempts yet. They're recorded automatically when one passes.</p>`}
    <h2>Recent check runs</h2>
    ${runs.length ? `<div class="card table-wrap"><table class="data"><thead><tr><th>When</th><th>File</th><th>Result</th></tr></thead><tbody>
      ${runs.map(r => `<tr><td>${r.at.replace("T", " ")}</td><td>${esc(r.target)}.py</td><td>${r.passed ? `<span class="pill ok">passed</span>` : `<span class="pill">failed</span>`}</td></tr>`).join("")}
      </tbody></table></div>` : `<p class="muted">No runs yet.</p>`}
    <h2>Your notes (LOG.md)</h2>
    <p class="muted">Key ideas, what you got stuck on, interview gotchas.</p>
    <textarea class="notes" id="log"></textarea>
    <div class="row" style="margin-top:8px"><button class="btn" id="log-save">Save notes</button><span class="muted" id="log-status"></span></div>`;
  const path = `${unit.id}/LOG.md`;
  api(`/api/file?path=${encodeURIComponent(path)}`).then(d => { $("#log").value = d.content; }).catch(() => {});
  $("#log-save").onclick = async () => {
    await api("/api/file", { path, content: $("#log").value });
    $("#log-status").textContent = "Saved.";
    refresh();
  };
}

// ------------------------------------------------------------------ router

async function route() {
  for (const fn of view.cleanup.splice(0)) { try { await fn(); } catch (e) { /* ignore */ } }
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  if (parts[0] === "unit") renderUnit(parts[1], parts[2], decodeURIComponent(parts.slice(3).join("/")));
  else renderDashboard();
  renderSidebar();
  $("#sidebar").classList.remove("open");
  window.scrollTo(0, 0);
}

// ------------------------------------------------------------------ commit dialog

function setupCommit() {
  const dlg = $("#commit-dialog");
  $("#commit-btn").onclick = async () => {
    for (const fn of view.cleanup.filter(f => f.name === "save")) await fn();
    await refresh();
    const ch = S.git.changes || [];
    $("#commit-changes").innerHTML = ch.length ? ch.map(c => `<li>${esc(c)}</li>`).join("") : "<li>No changes since the last commit.</li>";
    $("#commit-message").value = "";
    $("#commit-output").classList.add("hidden");
    $("#commit-go").disabled = !ch.length;
    dlg.showModal();
  };
  $("#commit-go").onclick = async e => {
    e.preventDefault();
    $("#commit-go").disabled = true;
    const r = await api("/api/commit", { message: $("#commit-message").value });
    $("#commit-output").textContent = r.output;
    $("#commit-output").classList.remove("hidden");
    if (r.ok) toast("Progress committed to git.", "ok");
    await refresh();
  };
}

// ------------------------------------------------------------------ study-time heartbeat

function setupHeartbeat() {
  let last = Date.now();
  ["mousemove", "keydown", "scroll", "click"].forEach(ev => window.addEventListener(ev, () => { last = Date.now(); }, { passive: true }));
  setInterval(async () => {
    if (document.visibilityState !== "visible" || Date.now() - last > 120000) return;
    try {
      await api("/api/heartbeat", { unit: view.unit });
      const day = (S.progress.time[S.today] = S.progress.time[S.today] || {});
      const key = view.unit || "dashboard";
      day[key] = (day[key] || 0) + 1;
      renderTopbar();
    } catch (e) { /* server stopped */ }
  }, 60000);
}

// ------------------------------------------------------------------ boot

(async function boot() {
  marked.setOptions({ gfm: true, mangle: false, headerIds: true });
  $("#menu-btn").onclick = () => $("#sidebar").classList.toggle("open");
  setupCommit();
  setupHeartbeat();
  try {
    await refresh();
  } catch (e) {
    $("#main").innerHTML = `<div class="pad"><h1>Can't reach the course server</h1><p>Start it with <code>python course.py</code> from the repo folder.</p></div>`;
    return;
  }
  window.addEventListener("hashchange", route);
  route();
})();
