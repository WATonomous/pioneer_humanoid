/* Pioneer telemetry dashboard. No dependencies: the robot PC may be offline.
 * Data: /api/config once, /api/history to backfill the charts, then /api/stream (SSE, 10 Hz). */
"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const fmt = (v, d = 1) => {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  const text = Number(v).toFixed(d);
  return Number(text) === 0 ? text.replace("-", "") : text;
};
const gain = (v) => fmt(v, v !== null && Math.abs(v) < 10 ? 2 : 0);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const STATUS = {
  ok: ["●", "OK"], idle: ["◌", "Idle"], warn: ["▲", "Warning"], fault: ["✖", "Fault"],
  stale: ["◆", "No feedback"], offline: ["○", "Offline"],
};
const OVERALL = {
  nominal: ["●", "Nominal"], warning: ["▲", "Warning"], degraded: ["◆", "Degraded"],
  fault: ["✖", "Fault"], offline: ["○", "Offline"],
};
const SERIES = ["pos", "sp", "req", "tau", "ff", "temp"];
const HISTORY_MAX_S = 900;

const state = {
  cfg: null, snap: null, byId: new Map(), rows: new Map(), hist: new Map(),
  selected: null, range: 120, paused: false, events: [], lastSeq: 0, lastMsg: 0,
  view: { yaw: 0.5, pitch: 0.28 }, hits: [], dirty: false,
};

/* ------------------------------------------------------------------ theme */
let cssCache = {};
function css(name) {
  if (!(name in cssCache)) {
    cssCache[name] = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  return cssCache[name];
}
function setTheme(theme) {
  if (theme === "light") document.documentElement.dataset.theme = "light";
  else delete document.documentElement.dataset.theme;
  cssCache = {};
  try { localStorage.setItem("pioneer-telemetry-theme", theme); } catch (e) { /* private mode */ }
  state.dirty = true;
}
function statusColor(st) {
  return { ok: css("--good"), warn: css("--warn"), stale: css("--serious"), fault: css("--crit") }[st]
    || css("--neutral");
}

/* ------------------------------------------------------------------ history */
function histFor(id) {
  if (!state.hist.has(id)) {
    const h = { t: [] };
    SERIES.forEach((k) => { h[k] = []; });
    state.hist.set(id, h);
  }
  return state.hist.get(id);
}
function pushHist(id, t, values) {
  const h = histFor(id);
  if (h.t.length && t <= h.t[h.t.length - 1]) return;
  h.t.push(t);
  SERIES.forEach((k, i) => h[k].push(values[i] === undefined ? null : values[i]));
  if (h.t.length > 12000 || t - h.t[0] > HISTORY_MAX_S + 30) {
    let cut = 0;
    while (cut < h.t.length && t - h.t[cut] > HISTORY_MAX_S) cut++;
    cut = Math.max(cut, h.t.length - 12000);
    if (cut > 0) { h.t.splice(0, cut); SERIES.forEach((k) => h[k].splice(0, cut)); }
  }
}

/* ------------------------------------------------------------------ build */
function jointLabel(j) { return `${j.limb} ${j.name}`; }

function buildTable() {
  const body = $("joint-rows");
  body.textContent = "";
  let limb = null;
  for (const j of state.cfg.joints) {
    if (j.limb !== limb) {
      limb = j.limb;
      const tr = el("tr", "limb");
      const td = el("td");
      td.colSpan = 14;
      td.textContent = limb + " ";
      const note = el("span", "muted");
      note.dataset.limb = limb;
      td.appendChild(note);
      tr.appendChild(td);
      body.appendChild(tr);
    }
    const tr = el("tr", "joint");
    tr.dataset.id = j.id;
    const c = {};
    const td = (cls) => { const d = el("td", cls); tr.appendChild(d); return d; };

    c.st = el("span", "st"); c.stIcon = el("i"); c.stText = el("span");
    c.st.append(c.stIcon, c.stText);
    td().appendChild(c.st);

    const name = td();
    name.appendChild(el("span", "jname", j.name));
    if (j.active === false) name.appendChild(el("span", "tag", "inactive"));
    if (!j.calibrated) name.appendChild(el("span", "tag", "uncalibrated"));
    if (j.model) name.appendChild(el("span", "tag", j.model));
    c.why = el("span", "why");
    name.appendChild(c.why);

    td("num").textContent = j.id;

    const bar = el("div", "rbar");
    bar.appendChild(el("div", "trk"));
    c.zero = el("div", "zero"); c.sp = el("div", "sp"); c.pos = el("div", "pos");
    bar.append(c.zero, c.sp, c.pos);
    bar.title = j.lower !== null ? `limits ${fmt(j.lower)} to ${fmt(j.upper)} deg` : "no limits configured";
    td().appendChild(bar);

    c.posV = td("num"); c.spV = td("num"); c.errV = td("num"); c.velV = td("num");

    c.tauM = el("div", "meter center"); c.tauFill = el("span"); c.tauM.appendChild(c.tauFill);
    c.tauM.title = j.max_torque ? `rated ±${j.max_torque} N·m` : "no torque cap configured";
    td().appendChild(c.tauM);
    c.tauV = td("num"); c.curV = td("num");

    const tempTd = td();
    c.tempM = el("div", "meter"); c.tempFill = el("span"); c.tempM.appendChild(c.tempFill);
    c.tempV = el("span", "mval");
    c.tempM.title = j.max_temp ? `limit ${j.max_temp} °C` : "no temperature limit configured";
    tempTd.append(c.tempM, c.tempV);

    c.mode = td(); c.hz = td("num");
    tr.addEventListener("click", () => select(j.id));
    body.appendChild(tr);
    state.rows.set(j.id, { tr, c });
  }
}

function buildStatic() {
  const cfg = state.cfg;
  $("robot-name").textContent = cfg.robot;
  $("demo-banner").hidden = !cfg.demo;
  const f = cfg.files;
  $("foot-files").textContent =
    `git ${cfg.git} · calibration ${f.calibration} · limits ${f.actuators || "not found"}`;

  const legend = $("skeleton-legend");
  const key = (cls, color, text) => {
    const s = el("span"); const i = el("i", cls); i.style.background = color;
    s.append(i, document.createTextNode(text)); legend.appendChild(s);
  };
  key("", "var(--s1)", "measured");
  key("", "var(--s2)", "setpoint");
  key("dotkey", "var(--good)", "● OK");
  key("dotkey", "var(--warn)", "▲ warning");
  key("dotkey", "var(--crit)", "✖ fault");
  key("dotkey", "var(--neutral)", "○ idle / offline");

  const lg = (id, items) => {
    for (const [color, text] of items) {
      const s = el("span"); const i = el("i"); i.style.background = color;
      s.append(i, document.createTextNode(text)); $(id).appendChild(s);
    }
  };
  lg("lg-pos", [["var(--s1)", "measured"], ["var(--s2)", "setpoint sent"], ["var(--s3)", "requested"]]);
  lg("lg-tau", [["var(--s1)", "measured"], ["var(--s2)", "feed-forward"]]);

  const cams = Object.keys(cfg.cameras || {});
  $("camera-card").hidden = cams.length === 0;
  for (const name of cams) {
    const fig = el("figure");
    fig.appendChild(el("figcaption", "", `${name} `)).appendChild(el("span", "unit", cfg.cameras[name]));
    const img = el("img"); img.src = `/camera/${name}`; img.alt = `${name} camera`;
    fig.appendChild(img);
    $("cameras").appendChild(fig);
  }
}

function select(id) {
  state.selected = id;
  for (const [jid, row] of state.rows) row.tr.classList.toggle("sel", jid === id);
  state.dirty = true;
  if (state.paused) renderDetail();
}

/* ------------------------------------------------------------------ tiles */
function tile(label, value, unit, note, level) {
  const t = el("div", "tile");
  t.appendChild(el("div", "tile-label", label));
  const v = el("div", "tile-value", value);
  if (unit) v.appendChild(el("small", "", unit));
  if (level === "warn" || level === "fault") {
    const flag = el("span", "tile-flag", level === "fault" ? "✖ over limit" : "▲ near limit");
    flag.style.color = level === "fault" ? css("--crit") : css("--warn");
    v.appendChild(flag);
  }
  t.appendChild(v);
  const n = el("div", "tile-note", note || "");
  n.title = note || "";
  t.appendChild(n);
  return t;
}

function renderTiles(snap) {
  const joints = snap.joints.map((s) => ({ s, j: state.byId.get(s.id) }));
  const online = joints.filter((x) => x.s.st !== "offline" && x.s.st !== "stale");
  const driven = joints.filter((x) => x.s.sp !== null).length;
  const bad = joints.filter((x) => x.s.st === "fault" || x.s.st === "stale");
  const warn = joints.filter((x) => x.s.st === "warn");
  const best = (list, score) => list.reduce((a, x) => {
    const v = score(x);
    return v !== null && (a === null || v > a.v) ? { v, x } : a;
  }, null);

  const hot = best(online, (x) => x.s.temp);
  const load = best(online, (x) => (x.s.tau !== null && x.j.max_torque ? Math.abs(x.s.tau) / x.j.max_torque : null));
  const track = best(online, (x) => (x.s.err !== null ? Math.abs(x.s.err) : null));
  const slow = online.length ? online.reduce((a, x) => (x.s.hz < a.s.hz ? x : a)) : null;
  const meanHz = online.length ? online.reduce((a, x) => a + x.s.hz, 0) / online.length : null;
  const streams = Object.entries(snap.streams || {});
  const live = streams.filter(([, s]) => s.live);
  const can = (snap.system || {}).can || {};

  const tiles = $("tiles");
  tiles.textContent = "";
  tiles.append(
    tile("Joints online", `${online.length}`, `/ ${joints.length}`,
      `${driven} driven · ${joints.filter((x) => x.j.active === false).length} set inactive`),
    tile("Active faults", `${bad.length}`, warn.length ? `+ ${warn.length} warning${warn.length > 1 ? "s" : ""}` : "",
      bad.length ? `${jointLabel(bad[0].j)}: ${bad[0].s.why}` : warn.length ? `${jointLabel(warn[0].j)}: ${warn[0].s.why}` : "none",
      bad.length ? "fault" : null),
    tile("Hottest drive", hot ? fmt(hot.v, 0) : "–", "°C",
      hot ? `${jointLabel(hot.x.j)}${hot.x.j.max_temp ? ` · limit ${hot.x.j.max_temp} °C` : ""}` : "no temperature readings",
      hot && hot.x.j.max_temp ? (hot.v >= hot.x.j.max_temp ? "fault" : hot.v >= hot.x.j.max_temp - state.cfg.thresholds.warnTempMarginC ? "warn" : null) : null),
    tile("Peak torque load", load ? fmt(load.v * 100, 0) : "–", "% of rated",
      load ? `${jointLabel(load.x.j)} · ${fmt(Math.abs(load.x.s.tau), 2)} of ${load.x.j.max_torque} N·m` : "no torque readings",
      load ? (load.v > 1 ? "fault" : load.v >= state.cfg.thresholds.warnTorqueFraction ? "warn" : null) : null),
    tile("Worst tracking error", track ? fmt(track.v, 1) : "–", "deg",
      track ? `${jointLabel(track.x.j)}${track.x.j.max_track_err ? ` · limit ${track.x.j.max_track_err} deg` : ""}` : "no joint is being driven",
      track && track.x.j.max_track_err ? (track.v > track.x.j.max_track_err ? "fault" : track.v >= state.cfg.thresholds.warnTrackFraction * track.x.j.max_track_err ? "warn" : null) : null),
    tile("Feedback rate", meanHz !== null ? fmt(meanHz, 0) : "–", "Hz per joint",
      slow ? `slowest ${jointLabel(slow.j)} at ${fmt(slow.s.hz, 0)} Hz` : "no feedback"),
    tile("Target stream", live.length ? fmt(live.reduce((a, [, s]) => a + s.hz, 0) / live.length, 0) : "–", "Hz",
      live.length ? live.map(([side, s]) => `${side} arm, ${fmt((s.age || 0) * 1000, 0)} ms old`).join(" · ")
        : "no arm targets arriving"),
    tile("CAN bus", can.present ? fmt(can.rxHz, 0) : "–", "frames/s in",
      can.present ? `${can.iface} ${can.state || can.operstate || ""} · ${can.rx_errors} rx / ${can.tx_errors} tx errors`
        : `${can.iface || state.cfg.canInterface} is not on this machine`),
  );
}

/* ------------------------------------------------------------------ table */
function renderTable(snap) {
  const perLimb = {};
  for (const s of snap.joints) {
    const j = state.byId.get(s.id);
    const row = state.rows.get(s.id);
    if (!j || !row) continue;
    const c = row.c;
    const [icon, label] = STATUS[s.st] || STATUS.offline;
    c.st.dataset.st = s.st; c.stIcon.textContent = icon; c.stText.textContent = label;
    c.why.textContent = s.st === "idle" || s.st === "ok" ? "" : s.why;
    c.why.title = s.why || "";
    row.tr.classList.toggle("dim", s.st === "offline");
    (perLimb[j.limb] = perLimb[j.limb] || []).push(s.st);

    const lo = j.lower !== null ? j.lower : -180, hi = j.upper !== null ? j.upper : 180;
    const pct = (v) => clamp(((v - lo) / (hi - lo)) * 100, 0, 100);
    c.pos.style.display = s.pos === null ? "none" : "";
    if (s.pos !== null) c.pos.style.left = pct(s.pos) + "%";
    c.sp.style.display = s.sp === null ? "none" : "";
    if (s.sp !== null) c.sp.style.left = `calc(${pct(s.sp)}% - 1px)`;
    c.zero.style.display = lo < 0 && hi > 0 ? "" : "none";
    c.zero.style.left = pct(0) + "%";

    c.posV.textContent = fmt(s.pos); c.spV.textContent = fmt(s.sp);
    c.errV.textContent = s.err === null ? "–" : (s.err > 0 ? "+" : "") + fmt(s.err, 2);
    c.velV.textContent = fmt(s.vel, 0);

    const cap = j.max_torque;
    if (s.tau !== null && cap) {
      const frac = clamp(s.tau / cap, -1, 1);
      c.tauFill.style.left = (frac < 0 ? 50 + frac * 50 : 50) + "%";
      c.tauFill.style.width = Math.abs(frac) * 50 + "%";
      const a = Math.abs(s.tau) / cap;
      c.tauM.dataset.lv = a > 1 ? "fault" : a >= state.cfg.thresholds.warnTorqueFraction ? "warn" : "";
    } else { c.tauFill.style.width = "0"; c.tauM.dataset.lv = ""; }
    c.tauV.textContent = fmt(s.tau, 2);
    c.curV.textContent = fmt(s.cur, 2);

    if (s.temp !== null) {
      const top = j.max_temp || 80;
      c.tempFill.style.width = clamp(((s.temp - 20) / (top - 20)) * 100, 2, 100) + "%";
      c.tempM.dataset.lv = !j.max_temp ? "" : s.temp >= j.max_temp ? "fault"
        : s.temp >= j.max_temp - state.cfg.thresholds.warnTempMarginC ? "warn" : "";
      c.tempV.textContent = `${fmt(s.temp, 0)} °C`;
    } else { c.tempFill.style.width = "0"; c.tempV.textContent = "–"; }

    c.mode.textContent = s.mode
      ? (s.kp !== null && s.mode === "MIT" ? `MIT · kp ${gain(s.kp)} kd ${fmt(s.kd, 2)}` : s.mode) : "–";
    c.hz.textContent = s.age === null ? "–" : s.st === "stale" ? `${fmt(s.age, 1)} s ago` : `${fmt(s.hz, 0)} Hz`;
  }
  document.querySelectorAll("[data-limb]").forEach((node) => {
    const list = perLimb[node.dataset.limb] || [];
    const on = list.filter((st) => st !== "offline" && st !== "stale").length;
    node.textContent = `· ${on} of ${list.length} online`;
  });
  const unknown = Object.keys(snap.unknownIds || {});
  $("unknown-ids").hidden = unknown.length === 0;
  if (unknown.length) {
    $("unknown-ids").textContent =
      `▲ Feedback from motor id ${unknown.join(", ")} that is in no calibration file (wrong id on a drive, or a missing config entry).`;
  }
}

/* ------------------------------------------------------------------ pipeline + thermal */
function renderPipeline(snap) {
  const ros = snap.ros || {}, topics = ros.topics || {}, nodes = ros.nodes || {};
  const can = (snap.system || {}).can || {};
  const streams = Object.entries(snap.streams || {});
  const live = streams.filter(([, s]) => s.live);
  const cmdHz = snap.joints.reduce((a, s) => a + (s.cmdHz || 0), 0);
  const fbHz = snap.joints.reduce((a, s) => a + (s.hz || 0), 0);
  const online = snap.joints.filter((s) => s.st !== "offline" && s.st !== "stale");
  const bad = snap.joints.filter((s) => s.st === "fault" || s.st === "stale");
  const expected = state.cfg.joints.filter((j) => j.active !== false && j.kind === "arm").length;
  const nodeNote = (name) => (ros.connected ? (nodes[name] ? "node running" : "node not found") : "");
  const busBad = can.present && (/BUS-OFF|ERROR-PASSIVE/.test(can.state || "") || can.operstate === "down");

  const steps = [
    ["Target source", "teleop or policy → /arm/<side>/joint_targets",
      live.length ? "ok" : "idle",
      live.length ? live.map(([side, s]) => `${side} ${fmt(s.hz, 0)} Hz`).join(" · ") : "no targets"],
    ["joint_command", `clamp, rate limit, watchdog → motorCMD ${nodeNote("joint_command_node")}`,
      cmdHz > 0 ? "ok" : ros.connected && nodes.joint_command_node === false ? "idle" : "idle",
      cmdHz > 0 ? `${fmt(cmdHz, 0)} cmd/s` : "no commands"],
    ["can_node", `ROS ↔ CAN bridge → motorFeedback ${nodeNote("can_node")}`,
      fbHz > 0 ? "ok" : ros.connected && nodes.can_node === false ? "fault" : "idle",
      fbHz > 0 ? `${fmt(fbHz, 0)} fb/s` : "no feedback"],
    ["CAN bus", can.present ? `${can.iface} ${can.state || can.operstate || ""}${can.bitrate ? ` · ${can.bitrate / 1000} kbit/s` : ""}`
      : `${can.iface || state.cfg.canInterface} not on this machine`,
      !can.present ? "idle" : busBad ? "fault" : (can.rx_errors || can.tx_errors) ? "warn" : "ok",
      can.present ? `${fmt(can.rxHz, 0)} in · ${fmt(can.txHz, 0)} out` : "–"],
    ["Drives", `${expected} active arm joints configured`,
      bad.length ? "fault" : online.length ? (snap.joints.some((s) => s.st === "warn") ? "warn" : "ok") : "idle",
      `${online.length} online${bad.length ? ` · ${bad.length} faulted` : ""}`],
  ];
  const icons = { ok: "●", warn: "▲", fault: "✖", idle: "○" };
  const list = $("pipeline");
  list.textContent = "";
  for (const [name, desc, st, value] of steps) {
    const li = el("li"); li.dataset.st = st;
    li.appendChild(el("span", "pi", icons[st]));
    const mid = el("div"); mid.append(el("div", "pn", name), el("div", "pd", desc));
    li.append(mid, el("span", "pv", value));
    list.appendChild(li);
  }

  const temps = snap.joints.filter((s) => s.temp !== null)
    .map((s) => ({ s, j: state.byId.get(s.id) }))
    .sort((a, b) => ((a.j.max_temp || 80) - a.s.temp) - ((b.j.max_temp || 80) - b.s.temp)).slice(0, 7);
  const box = $("thermal");
  box.textContent = "";
  if (!temps.length) box.appendChild(el("span", "muted", "No temperature readings yet."));
  for (const { s, j } of temps) {
    const top = j.max_temp || 80;
    const meter = el("div", "meter"); const fill = el("span");
    fill.style.width = clamp(((s.temp - 20) / (top - 20)) * 100, 2, 100) + "%";
    meter.dataset.lv = !j.max_temp ? "" : s.temp >= top ? "fault"
      : s.temp >= top - state.cfg.thresholds.warnTempMarginC ? "warn" : "";
    meter.appendChild(fill);
    box.append(el("span", "tn", jointLabel(j)), meter,
      el("span", "tv", `${fmt(s.temp, 0)} °C${j.max_temp ? ` · ${fmt(top - s.temp, 0)} to limit` : ""}`));
  }
}

/* ------------------------------------------------------------------ events */
function renderEvents() {
  const only = $("ev-important").checked;
  const list = $("events");
  list.textContent = "";
  const shown = state.events.filter((e) => !only || e.sev !== "info").slice(-150).reverse();
  if (!shown.length) list.appendChild(el("li", "empty", "Nothing logged yet."));
  for (const e of shown) {
    const li = el("li"); li.dataset.sev = e.sev;
    const time = el("time", "", new Date(e.t * 1000).toLocaleTimeString([], { hour12: false }));
    const icon = el("span", "ei", e.sev === "fault" ? "✖" : e.sev === "warn" ? "▲" : "·");
    const text = el("div");
    text.append(el("span", "es", e.src + " "), el("span", "em", e.msg));
    li.append(time, icon, text);
    list.appendChild(li);
  }
}

/* ------------------------------------------------------------------ canvas helpers */
function prep(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight || Number(canvas.getAttribute("height"));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
  return { ctx, w, h };
}
function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * mag;
}
function lowerBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < x) lo = mid + 1; else hi = mid; }
  return lo;
}

/* ------------------------------------------------------------------ time-series chart */
const charts = {};
function chartFor(id) {
  if (!charts[id]) {
    const canvas = $(id);
    const chart = { canvas, hover: null, last: null };
    canvas.addEventListener("mousemove", (ev) => {
      const r = canvas.getBoundingClientRect();
      chart.hover = { x: ev.clientX - r.left, cx: ev.clientX, cy: ev.clientY };
      renderDetail();
    });
    canvas.addEventListener("mouseleave", () => { chart.hover = null; $("tooltip").hidden = true; renderDetail(); });
    charts[id] = chart;
  }
  return charts[id];
}

/* opts: {t, series:[{v, color, label}], limits:[{y, label}], digits, zero, tEnd, range, alwaysLimits} */
function drawChart(id, opts) {
  const chart = chartFor(id);
  const { ctx, w, h } = prep(chart.canvas);
  const pad = { l: 44, r: 10, t: 8, b: 20 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const t1 = opts.tEnd, t0 = t1 - opts.range;
  const i0 = Math.max(0, lowerBound(opts.t, t0) - 1), i1 = opts.t.length;

  let lo = Infinity, hi = -Infinity;
  for (const s of opts.series) {
    for (let i = i0; i < i1; i++) {
      const v = s.v[i];
      if (v !== null && v !== undefined) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
  }
  const empty = !isFinite(lo);
  if (empty) { lo = -1; hi = 1; }
  if (opts.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  const offscale = [];
  for (const lim of opts.limits || []) {
    const peak = Math.max(Math.abs(lo), Math.abs(hi));
    if (opts.alwaysLimits || peak >= 0.5 * Math.abs(lim.y)) { lo = Math.min(lo, lim.y); hi = Math.max(hi, lim.y); }
    else if (lim.y > 0) offscale.push(lim.label);
  }
  if (hi - lo < (opts.minSpan || 1)) { const mid = (hi + lo) / 2, half = (opts.minSpan || 1) / 2; lo = mid - half; hi = mid + half; }
  const padY = (hi - lo) * 0.08; lo -= padY; hi += padY;
  const X = (t) => pad.l + ((t - t0) / (t1 - t0)) * pw;
  const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * ph;

  // grid + y ticks
  ctx.lineWidth = 1; ctx.textBaseline = "middle"; ctx.textAlign = "right";
  const step = niceStep(hi - lo, 4);
  const digits = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    const y = Math.round(Y(v)) + 0.5;
    ctx.strokeStyle = Math.abs(v) < 1e-9 && opts.zero ? css("--axis") : css("--grid");
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillStyle = css("--muted");
    ctx.fillText(v.toFixed(digits), pad.l - 6, y);
  }
  // x ticks: seconds before now
  ctx.textBaseline = "top";
  for (let k = 0; k <= 4; k++) {
    const age = (opts.range * (4 - k)) / 4;
    const x = pad.l + (pw * k) / 4;
    ctx.textAlign = k === 0 ? "left" : k === 4 ? "right" : "center";
    ctx.fillStyle = css("--muted");
    ctx.fillText(k === 4 ? "now" : age >= 120 ? `−${age / 60} min` : `−${age} s`, x, h - pad.b + 5);
  }
  // limit lines
  ctx.textBaseline = "bottom"; ctx.textAlign = "right";
  for (const lim of opts.limits || []) {
    if (lim.y < lo || lim.y > hi) continue;
    const y = Math.round(Y(lim.y)) + 0.5;
    ctx.strokeStyle = css("--crit"); ctx.globalAlpha = 0.75;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.globalAlpha = 1;
    if (lim.y >= 0) { ctx.fillStyle = css("--ink-2"); ctx.fillText(lim.label, w - pad.r - 2, y - 2); }
  }
  if (offscale.length) {
    ctx.textBaseline = "top"; ctx.textAlign = "right"; ctx.fillStyle = css("--muted");
    ctx.fillText(`${offscale.join(", ")} (off scale)`, w - pad.r - 2, pad.t);
  }
  // series
  ctx.save();
  ctx.beginPath(); ctx.rect(pad.l, pad.t - 2, pw, ph + 4); ctx.clip();
  const stride = Math.max(1, Math.floor((i1 - i0) / (pw * 2)));
  for (const s of [...opts.series].reverse()) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.beginPath();
    let pen = false, prevT = null;
    for (let i = i0; i < i1; i += stride) {
      const v = s.v[i], t = opts.t[i];
      if (v === null || v === undefined || (prevT !== null && t - prevT > 1.5 * stride + 1)) pen = false;
      if (v === null || v === undefined) { prevT = t; continue; }
      if (pen) ctx.lineTo(X(t), Y(v)); else ctx.moveTo(X(t), Y(v));
      pen = true; prevT = t;
    }
    ctx.stroke();
  }
  ctx.restore();
  if (empty) {
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillStyle = css("--muted");
    ctx.fillText(opts.emptyText || "No data in this window", pad.l + pw / 2, pad.t + ph / 2);
  }
  // hover crosshair + tooltip
  if (chart.hover && !empty && chart.hover.x >= pad.l && chart.hover.x <= w - pad.r) {
    const tHover = t0 + ((chart.hover.x - pad.l) / pw) * (t1 - t0);
    let i = clamp(lowerBound(opts.t, tHover), i0, i1 - 1);
    if (i > i0 && Math.abs(opts.t[i - 1] - tHover) < Math.abs(opts.t[i] - tHover)) i--;
    const x = Math.round(X(opts.t[i])) + 0.5;
    ctx.strokeStyle = css("--axis"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
    const tip = $("tooltip");
    tip.textContent = "";
    tip.appendChild(el("b", "", new Date(opts.t[i] * 1000).toLocaleTimeString([], { hour12: false })));
    for (const s of opts.series) {
      const v = s.v[i];
      if (v === null || v === undefined) continue;
      ctx.fillStyle = s.color; ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(opts.t[i]), Y(v), 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      const row = el("div", "tt-row"); const key = el("i"); key.style.background = s.color;
      row.append(key, el("span", "", s.label), el("span", "", `${v.toFixed(opts.digits)} ${opts.unit}`));
      tip.appendChild(row);
    }
    tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = Math.min(window.innerWidth - tw - 8, chart.hover.cx + 14) + "px";
    tip.style.top = chart.hover.cy + 14 + "px";
  }
}

function renderDetail() {
  const snap = state.snap;
  if (!snap || state.selected === null) return;
  const j = state.byId.get(state.selected);
  const s = snap.joints.find((x) => x.id === state.selected);
  const h = histFor(state.selected);
  if (!j || !s) return;
  $("detail-title").textContent = jointLabel(j);
  const [icon, label] = STATUS[s.st] || STATUS.offline;
  $("detail-sub").textContent = `motor ${j.id}${j.model ? ` · ${j.model}` : ""} · ${icon} ${label}${s.why && s.st !== "idle" ? `: ${s.why}` : ""}`;

  const stats = $("detail-stats");
  stats.textContent = "";
  const stat = (name, value, unit) => {
    const d = el("div"); d.appendChild(el("div", "ds-l", name));
    const v = el("div", "ds-v", value); if (unit) v.appendChild(el("small", "", " " + unit));
    d.appendChild(v); stats.appendChild(d);
  };
  stat("Measured", fmt(s.pos), "deg");
  stat("Range", j.lower !== null ? `${fmt(j.lower, 0)} to ${fmt(j.upper, 0)}` : "none", j.lower !== null ? "deg" : "");
  stat("Gains kp / kd", s.kp !== null ? `${gain(s.kp)} / ${fmt(s.kd, 2)}` : j.cfg_kp !== null ? `${gain(j.cfg_kp)} / ${fmt(j.cfg_kd, 2)}` : "–", s.kp !== null ? "live" : j.cfg_kp !== null ? "config" : "");
  stat("Torque rated / peak", j.max_torque !== null ? `${fmt(j.max_torque, 1)} / ${j.peak_torque !== null ? fmt(j.peak_torque, 0) : "–"}` : "–", "N·m");
  stat("Session peak torque", fmt(s.pkTau, 2), "N·m");
  stat("Session peak temperature", fmt(s.pkTemp, 0), j.max_temp ? `°C of ${j.max_temp}` : "°C");
  stat("Session peak tracking error", fmt(s.pkErr, 1), j.max_track_err ? `deg of ${j.max_track_err}` : "deg");
  stat("Velocity limit", j.velocity_max !== null ? fmt(j.velocity_max, 0) : "–", "deg/s");
  stat("Drive status word", s.code === null ? "–" : `${s.code} · ${s.codeName}`, "");

  const tEnd = state.paused ? state.pausedAt : snap.t;
  const base = { t: h.t, tEnd, range: state.range };
  const err = h.pos.map((p, i) => (p !== null && h.sp[i] !== null ? p - h.sp[i] : null));
  drawChart("ch-pos", { ...base, unit: "deg", digits: 2, minSpan: 4,
    series: [{ v: h.pos, color: css("--s1"), label: "measured" }, { v: h.sp, color: css("--s2"), label: "setpoint sent" },
      { v: h.req, color: css("--s3"), label: "requested" }],
    limits: j.lower !== null ? [{ y: j.upper, label: `upper limit ${fmt(j.upper, 0)}` }, { y: j.lower, label: "" }] : [],
    emptyText: "No feedback from this joint yet" });
  drawChart("ch-err", { ...base, unit: "deg", digits: 2, zero: true, minSpan: 2,
    series: [{ v: err, color: css("--s1"), label: "error" }],
    limits: j.max_track_err ? [{ y: j.max_track_err, label: `fault at ±${j.max_track_err}` }, { y: -j.max_track_err, label: "" }] : [],
    emptyText: "Not driven: no setpoint to compare against" });
  drawChart("ch-tau", { ...base, unit: "N·m", digits: 2, zero: true, minSpan: 1,
    series: [{ v: h.tau, color: css("--s1"), label: "measured" }, { v: h.ff, color: css("--s2"), label: "feed-forward" }],
    limits: j.max_torque ? [{ y: j.max_torque, label: `rated ±${j.max_torque}` }, { y: -j.max_torque, label: "" }] : [] });
  drawChart("ch-temp", { ...base, unit: "°C", digits: 0, minSpan: 10, alwaysLimits: true,
    series: [{ v: h.temp, color: css("--s1"), label: "drive" }],
    limits: j.max_temp ? [{ y: j.max_temp, label: `limit ${j.max_temp} °C` }] : [] });
}

/* ------------------------------------------------------------------ kinematic view */
function rot(axis, deg) {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a), t = 1 - c;
  const n = Math.hypot(...axis) || 1; const [x, y, z] = axis.map((v) => v / n);
  return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
    [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
    [t * x * z - s * y, t * y * z + s * x, t * z * z + c]];
}
const mulMM = (a, b) => a.map((r) => [0, 1, 2].map((k) => r[0] * b[0][k] + r[1] * b[1][k] + r[2] * b[2][k]));
const mulMV = (m, v) => m.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

/* Forward kinematics of one arm: the points base, joint 1..6, tool tip (every URDF rpy is 0). */
function armPoints(chain, angles) {
  let R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], p = [0, 0, 0];
  const pts = [];
  chain.forEach((link, i) => {
    p = add(p, mulMV(R, link.origin));
    pts.push(p);
    R = mulMM(R, rot(link.axis, angles[i] || 0));
  });
  pts.push(add(p, mulMV(R, state.cfg.tool)));
  return pts;
}

function renderSkeleton(snap) {
  const canvas = $("skeleton");
  const { ctx, w, h } = prep(canvas);
  const { yaw, pitch } = state.view;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const scale = Math.min(w / 1.2, h / 1.12);
  const project = (p) => {
    const xr = p[0] * cy - p[1] * sy, yr = p[0] * sy + p[1] * cy;
    return [w / 2 + yr * scale, h * 0.27 - (p[2] * cp - xr * sp) * scale];
  };
  const line = (pts, color, width, dash) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.setLineDash(dash || []);
    ctx.beginPath(); pts.forEach((p, i) => { const [x, y] = project(p); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
    ctx.stroke(); ctx.setLineDash([]);
  };
  const live = new Map(snap.joints.map((s) => [s.id, s]));
  state.hits = [];

  // torso: shoulder bar, spine, head
  const shoulders = ["right", "left"].map((side) => (state.cfg.chain[side] ? state.cfg.chain[side][0].origin : [0, 0, 0]));
  const mid = [(shoulders[0][0] + shoulders[1][0]) / 2, 0, (shoulders[0][2] + shoulders[1][2]) / 2];
  line([shoulders[0], shoulders[1]], css("--axis"), 6);
  line([mid, add(mid, [0, 0, -0.42])], css("--axis"), 6);
  const [hx, hy] = project(add(mid, [0, 0, 0.13]));
  ctx.strokeStyle = css("--axis"); ctx.lineWidth = 4; ctx.beginPath(); ctx.arc(hx, hy, 0.055 * scale, 0, 2 * Math.PI); ctx.stroke();

  for (const side of Object.keys(state.cfg.chain)) {
    const joints = state.cfg.joints.filter((j) => j.side === side && j.kind === "arm");
    const slots = joints.filter((j) => j.slot !== null).sort((a, b) => a.slot - b.slot);
    if (!slots.length) continue;
    const meas = [], cmd = []; let anyCmd = false, anyPos = false;
    for (const j of slots) {
      const s = live.get(j.id) || {};
      meas[j.slot] = s.pos !== null && s.pos !== undefined ? s.pos : 0;
      cmd[j.slot] = s.sp !== null && s.sp !== undefined ? s.sp : meas[j.slot];
      anyCmd = anyCmd || (s.sp !== null && s.sp !== undefined);
      anyPos = anyPos || (s.pos !== null && s.pos !== undefined);
    }
    if (anyCmd) line(armPoints(state.cfg.chain[side], cmd), css("--s2"), 2, [5, 5]);
    const pts = armPoints(state.cfg.chain[side], meas);
    line(pts, anyPos ? css("--s1") : css("--axis"), 4);
    const gripper = joints.find((j) => j.name.startsWith("gripper"));
    const marks = slots.map((j) => [j, pts[j.slot]]);
    if (gripper) marks.push([gripper, pts[pts.length - 1]]);
    for (const [j, p] of marks) {
      const s = live.get(j.id) || { st: "offline" };
      const [x, y] = project(p);
      ctx.fillStyle = statusColor(s.st); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      state.hits.push({ x, y, j, s });
    }
    const [lx, ly] = project(add(state.cfg.chain[side][0].origin, [0, side === "left" ? 0.1 : -0.1, 0.1]));
    ctx.fillStyle = css("--ink-2"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(side === "left" ? "LEFT" : "RIGHT", lx, ly);
  }
}

function initSkeletonInput() {
  const canvas = $("skeleton");
  let drag = null;
  canvas.addEventListener("pointerdown", (ev) => { drag = { x: ev.clientX, y: ev.clientY, ...state.view }; canvas.setPointerCapture(ev.pointerId); });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("pointermove", (ev) => {
    if (drag) {
      state.view.yaw = drag.yaw + (ev.clientX - drag.x) * 0.01;
      state.view.pitch = clamp(drag.pitch + (ev.clientY - drag.y) * 0.01, -1.4, 1.4);
      $("tooltip").hidden = true;
      if (state.snap) renderSkeleton(state.snap);
      return;
    }
    const r = canvas.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    const hit = state.hits.find((p) => Math.hypot(p.x - x, p.y - y) < 11);
    const tip = $("tooltip");
    if (!hit) { tip.hidden = true; canvas.style.cursor = "grab"; return; }
    canvas.style.cursor = "pointer";
    const [icon, label] = STATUS[hit.s.st] || STATUS.offline;
    tip.textContent = "";
    tip.appendChild(el("b", "", jointLabel(hit.j)));
    tip.appendChild(el("div", "", `${icon} ${label} · ${fmt(hit.s.pos)} deg · ${fmt(hit.s.tau, 2)} N·m · ${fmt(hit.s.temp, 0)} °C`));
    tip.hidden = false;
    tip.style.left = ev.clientX + 14 + "px"; tip.style.top = ev.clientY + 14 + "px";
  });
  canvas.addEventListener("pointerleave", () => { $("tooltip").hidden = true; });
  canvas.addEventListener("click", (ev) => {
    const r = canvas.getBoundingClientRect();
    const hit = state.hits.find((p) => Math.hypot(p.x - (ev.clientX - r.left), p.y - (ev.clientY - r.top)) < 11);
    if (hit) select(hit.j.id);
  });
}

/* ------------------------------------------------------------------ system */
function kv(id, rows) {
  const dl = $(id);
  dl.textContent = "";
  for (const [k, v] of rows) { dl.append(el("dt", "", k), el("dd", "", v)); }
}
function renderSystem(snap) {
  const host = (snap.system || {}).host || {}, can = (snap.system || {}).can || {};
  const gpu = host.gpu;
  kv("sys-host", [
    ["Name", host.name || "–"],
    ["CPU", host.cpu !== null && host.cpu !== undefined ? `${fmt(host.cpu, 0)} % of ${host.cores} cores` : "–"],
    ["Load 1 / 5 / 15 min", host.load ? host.load.map((v) => v.toFixed(2)).join(" / ") : "–"],
    ["Memory", host.mem ? `${host.mem.used} of ${host.mem.total} GB` : "–"],
    ["CPU temperature", host.temp ? `${host.temp} °C` : "no sensor exposed"],
    ["GPU", gpu ? `${gpu.name} · ${fmt(gpu.util, 0)} % · ${gpu.memUsed} of ${gpu.memTotal} GB · ${fmt(gpu.temp, 0)} °C` : "none found"],
    ["Disk free", host.disk ? `${host.disk.free} of ${host.disk.total} GB (${host.disk.path})` : "–"],
  ]);
  kv("sys-can", can.present ? [
    ["Interface", `${can.iface} · ${can.operstate || "?"}`],
    ["Controller state", can.state || "not reported"],
    ["Bitrate", can.bitrate ? `${can.bitrate / 1000} kbit/s` : "–"],
    ["Frames in / out", `${fmt(can.rxHz, 0)} / ${fmt(can.txHz, 0)} per second`],
    ["Total in / out", `${can.rx_packets} / ${can.tx_packets}`],
    ["Errors rx / tx", `${can.rx_errors} / ${can.tx_errors}${can.rx_dropped ? ` · ${can.rx_dropped} dropped` : ""}`],
    ["Error counters rx / tx", can.berrRx !== undefined ? `${can.berrRx} / ${can.berrTx}` : "–"],
    ["Bus-off / restarts", can.busOff !== undefined ? `${can.busOff} / ${can.restarts}` : "–"],
  ] : [
    ["Interface", `${can.iface || state.cfg.canInterface} is not on this machine`],
    ["Why", "Run the dashboard on the robot PC (the interfacing container shares its network) to see bus counters."],
  ]);

  const ros = snap.ros || {};
  const box = $("sys-ros");
  box.textContent = "";
  if (!ros.connected) {
    box.appendChild(el("p", "muted", snap.demo ? "Demo mode: not connected to ROS." : "Waiting for the ROS node to start."));
    return;
  }
  const table = el("table", "ros-table");
  const head = el("tr"); ["Topic", "Source", "Publishers", "Rate"].forEach((t, i) => head.appendChild(el("th", i > 1 ? "num" : "", t)));
  table.appendChild(head);
  for (const [topic, info] of Object.entries(ros.topics || {})) {
    const tr = el("tr");
    tr.append(el("td", "ros-topic", topic), el("td", "", info.source), el("td", "num", String(info.pubs)), el("td", "num", `${fmt(info.hz, 0)} Hz`));
    table.appendChild(tr);
  }
  box.appendChild(table);
  const nodes = Object.entries(ros.nodes || {}).map(([n, up]) => `${up ? "●" : "○"} ${n} ${up ? "running" : "not found"}`).join("   ");
  box.appendChild(el("p", "muted", nodes));
}

/* ------------------------------------------------------------------ header + loop */
function renderHeader(snap) {
  const o = snap.overall;
  const [icon, label] = OVERALL[o.state] || OVERALL.offline;
  $("overall").dataset.state = o.state;
  $("overall-icon").textContent = icon;
  $("overall-label").textContent = label.toUpperCase();
  $("overall-text").textContent = o.text;
  $("clock").textContent = new Date(snap.t * 1000).toLocaleTimeString([], { hour12: false });
  const up = snap.uptime;
  $("uptime").textContent = `dashboard up ${up >= 3600 ? `${Math.floor(up / 3600)} h ${Math.floor((up % 3600) / 60)} min` : up >= 60 ? `${Math.floor(up / 60)} min` : `${Math.floor(up)} s`}`;
}

function render() {
  const snap = state.snap;
  if (!snap) return;
  renderHeader(snap);
  if (state.paused) return;
  renderTiles(snap);
  renderTable(snap);
  renderPipeline(snap);
  renderSkeleton(snap);
  renderDetail();
  renderSystem(snap);
}

function frame() {
  const age = performance.now() - state.lastMsg;
  const link = $("link");
  const st = !state.lastMsg || age > 3000 ? "down" : age > 600 ? "late" : "up";
  link.dataset.state = st;
  $("link-text").textContent = st === "up" ? "Live" : st === "late" ? "Lagging" : "No link to server";
  if (st === "down" && state.lastMsg) {
    $("overall").dataset.state = "offline";
    $("overall-icon").textContent = "○";
    $("overall-label").textContent = "DASHBOARD LINK LOST";
    $("overall-text").textContent = "The page is not receiving data: everything below is stale.";
  }
  if (state.dirty) { state.dirty = false; render(); }
  requestAnimationFrame(frame);
}

function onSnapshot(snap) {
  state.snap = snap;
  state.lastMsg = performance.now();
  for (const s of snap.joints) {
    if (s.pos !== null) pushHist(s.id, snap.t, [s.pos, s.sp, s.req, s.tau, s.ff, s.temp]);
  }
  if (snap.events) {
    for (const e of snap.events) if (e.seq > state.lastSeq) { state.events.push(e); state.lastSeq = e.seq; }
    if (state.events.length > 400) state.events.splice(0, state.events.length - 400);
    renderEvents();
  }
  state.dirty = true;
}

function exportCsv() {
  const lines = ["time_iso,t_unix,motor_id,limb,joint,pos_deg,setpoint_deg,requested_deg,error_deg,torque_nm,feedforward_nm,temp_c"];
  const cell = (v) => (v === null || v === undefined ? "" : v);
  for (const j of state.cfg.joints) {
    const h = state.hist.get(j.id);
    if (!h) continue;
    for (let i = 0; i < h.t.length; i++) {
      const err = h.pos[i] !== null && h.sp[i] !== null ? (h.pos[i] - h.sp[i]).toFixed(3) : "";
      lines.push([new Date(h.t[i] * 1000).toISOString(), h.t[i], j.id, j.limb, j.name, cell(h.pos[i]), cell(h.sp[i]),
        cell(h.req[i]), err, cell(h.tau[i]), cell(h.ff[i]), cell(h.temp[i])].join(","));
    }
  }
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
  const a = el("a");
  a.href = URL.createObjectURL(blob);
  a.download = `pioneer-telemetry-${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function start() {
  let theme = "dark";
  try { theme = localStorage.getItem("pioneer-telemetry-theme") || "dark"; } catch (e) { /* private mode */ }
  setTheme(theme);

  state.cfg = await (await fetch("/api/config")).json();
  for (const j of state.cfg.joints) state.byId.set(j.id, j);
  buildStatic();
  buildTable();
  initSkeletonInput();
  renderEvents();

  try {
    const rows = await (await fetch(`/api/history?s=${state.cfg.historySeconds}`)).json();
    for (const [t, joints] of rows) for (const id of Object.keys(joints)) pushHist(Number(id), t, joints[id]);
  } catch (e) { /* charts fill from the live stream instead */ }

  const firstDriven = state.cfg.joints.find((j) => j.active !== false && j.control_type === 0) || state.cfg.joints[0];
  if (firstDriven) select(firstDriven.id);

  const source = new EventSource("/api/stream");
  source.onopen = () => { state.events = []; state.lastSeq = 0; };
  source.onmessage = (msg) => onSnapshot(JSON.parse(msg.data));

  $("btn-theme").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light"));
  $("btn-csv").addEventListener("click", exportCsv);
  $("btn-pause").addEventListener("click", () => {
    state.paused = !state.paused;
    state.pausedAt = state.snap ? state.snap.t : 0;
    $("btn-pause").textContent = state.paused ? "Resume" : "Pause";
    $("btn-pause").classList.toggle("on", state.paused);
    state.dirty = true;
  });
  $("ev-important").addEventListener("change", renderEvents);
  $("range").addEventListener("click", (ev) => {
    const s = ev.target.dataset && ev.target.dataset.s;
    if (!s) return;
    state.range = Number(s);
    $("range").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === ev.target));
    state.dirty = true;
    if (state.paused) renderDetail();
  });
  window.addEventListener("resize", () => { state.dirty = true; });
  requestAnimationFrame(frame);
}

start().catch((err) => {
  $("overall-label").textContent = "DASHBOARD FAILED TO START";
  $("overall-text").textContent = String(err);
});
