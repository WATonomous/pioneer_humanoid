// Rollout dashboard page: dataset + tab state, episode player (cameras kept in sync), value and joint charts.
"use strict";

const VALUE_RANGE = [-1, 0];  // RECAP normalised return; the value chart always shows at least this range
const SYNC_TOLERANCE_S = 0.1;  // re-seek a follower camera when it drifts further than this from the master
const AV1_PROBE = 'video/mp4; codecs="av01.0.04M.08"';

const state = { datasets: [], ds: 0, tab: "sft", meta: null, episode: null, frame: 0, playing: false, videos: [] };
const table = new EpisodeTable({ onSelect: ep => loadEpisode(ep) });
const valueChart = new LineChart($("value-chart"), { fps: 1, yInclude: VALUE_RANGE, onSeek: seek, onHover: f => valueReadout(f) });
const jointChart = new LineChart($("joint-chart"), { fps: 1, onSeek: seek, onHover: f => jointReadout(f) });

const getJSON = url => fetch(url).then(r => (r.ok ? r.json() : r.json().then(e => Promise.reject(new Error(e.error)))));
const shortJoint = name => name.replace(/^left_|^right_/, "");

// ---------- dataset / tab ----------
async function init() {
  state.datasets = await getJSON("/api/datasets");
  $("dataset").innerHTML = state.datasets.map((d, i) => `<option value="${i}">${d.name} (${d.num_episodes})</option>`).join("");
  const hash = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
  state.tab = hash.tab || (state.datasets[+hash.ds || 0].has_recap ? "rl" : "sft");
  await loadDataset(+hash.ds || 0, hash.ep != null ? +hash.ep : null);
}

async function loadDataset(ds, ep = null) {
  state.ds = ds;
  state.meta = state.datasets[ds];
  $("dataset").value = ds;
  valueChart.setFps(state.meta.fps);
  jointChart.setFps(state.meta.fps);
  buildValueControls();
  buildJointControls();
  const episodes = await getJSON(`/api/episodes?ds=${ds}`);
  table.load(episodes, state.meta.values.map(v => v.name), state.meta.extra_columns, state.tab);
  setTab(state.tab);
  const first = ep ?? table.view[0]?.episode_index;
  if (first != null) table.select(first);
}

function setTab(tab) {
  state.tab = tab;
  document.body.classList.toggle("rl", tab === "rl");
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  table.setTab(tab);
  writeHash();
}

function writeHash() {
  const ep = state.episode ? `&ep=${state.episode.episode_index}` : "";
  history.replaceState(null, "", `#ds=${state.ds}&tab=${state.tab}${ep}`);
}

function buildValueControls() {
  const values = state.meta.values;
  $("value-checks").innerHTML = values.length ? values.map((v, i) =>
    `<label><input type="checkbox" value="${v.name}" checked><span class="swatch" style="background:${seriesColor(i)}"></span>${v.name}</label>`,
  ).join("") : `<span class="muted">no value functions (add recap/labels_r*.parquet or --values NAME=PATH)</span>`;
  const labelled = values.filter(v => v.has_positive);
  $("positive-source").innerHTML = labelled.map(v => `<option>${v.name}</option>`).join("");
  $("positive-source").disabled = $("show-positive").disabled = labelled.length === 0;
}

function buildJointControls() {
  const names = state.meta.joint_names["observation.state"] || state.meta.joint_names.action || [];
  $("joint-checks").innerHTML = names.map((n, i) =>
    `<label><input type="checkbox" value="${i}" checked><span class="swatch" style="background:${seriesColor(i)}"></span>${shortJoint(n)}</label>`,
  ).join("");
}

// ---------- episode ----------
async function loadEpisode(ep) {
  pause();
  state.wanted = ep;
  const data = await getJSON(`/api/episode?ds=${state.ds}&ep=${ep}`);
  if (state.wanted !== ep) return;  // a newer selection was made while this one loaded (fast ↑/↓)
  state.episode = data;
  state.frame = 0;
  $("scrub").max = data.length - 1;
  renderTitle(table.row(ep));
  renderVideos(data);
  renderTaskStrip(data);
  renderValueChart();
  renderJointChart();
  showFrame(0);
  writeHash();
}

function renderTitle(row) {
  const parts = [`<b>Episode ${row.episode_index}</b>`, row.success ? "<span class=ok>✓ success</span>" : "<span class=fail>✗ failure</span>",
    row.source, row.round != null ? `round ${row.round}` : null, row.seed != null ? `seed ${row.seed}` : null,
    row.policy || null, row.advantage && row.advantage !== "none" ? `advantage ${row.advantage}` : null,
    row.parent_episode >= 0 ? `parent ep ${row.parent_episode}` : null, `${row.seconds.toFixed(1)} s · ${row.length} frames`];
  $("episode-title").innerHTML = parts.filter(Boolean).join(" · ");
}

function renderVideos(data) {
  const useClips = state.meta.codecs.includes("av1") && document.createElement("video").canPlayType(AV1_PROBE) === "";
  state.videos = [];
  $("videos").innerHTML = "";
  for (const cam of state.meta.cameras) {
    const seg = data.videos[cam], fig = document.createElement("figure");
    fig.innerHTML = `<figcaption>${cam.replace("observation.images.", "")}</figcaption>`;
    $("videos").appendChild(fig);
    if (!seg.available) {
      fig.insertAdjacentHTML("beforeend", `<div class="missing">video file not found:<br>${seg.path}</div>`);
      continue;
    }
    const video = document.createElement("video");
    video.muted = true;
    video.preload = "auto";
    video.playsInline = true;
    const entry = { video, offset: seg.from, clip: `/clip/${state.ds}/${data.episode_index}/${cam}.mp4` };
    if (useClips) useClip(entry);
    else video.src = `/video/${state.ds}/${seg.path}`;
    video.addEventListener("error", () => { if (!entry.usingClip) useClip(entry); }, { once: true });
    video.addEventListener("loadedmetadata", () => seekVideo(entry, state.frame));
    fig.appendChild(video);
    state.videos.push(entry);
  }
}

// Fallback for browsers that cannot decode the dataset's codec: the server cuts and re-encodes one episode.
function useClip(entry) {
  entry.usingClip = true;
  entry.offset = 0;
  entry.video.src = entry.clip;
}

function renderTaskStrip(data) {
  const strip = $("task-strip"), tasks = data.task_index;
  strip.innerHTML = "";
  if (!tasks) return;
  for (const run of runs(tasks)) {
    const div = document.createElement("div");
    div.style.flex = `${run.end - run.start}`;
    div.style.background = seriesColor(run.value);
    div.title = div.textContent = state.meta.tasks[run.value] ?? `task ${run.value}`;
    div.addEventListener("click", () => seek(run.start));
    strip.appendChild(div);
  }
}

function runs(values) {
  const out = [];
  values.forEach((v, i) => {
    if (i === 0 || v !== values[i - 1]) out.push({ value: v, start: i, end: i + 1 });
    else out[out.length - 1].end = i + 1;
  });
  return out;
}

// ---------- charts ----------
function checkedValues() { return [...$("value-checks").querySelectorAll("input:checked")].map(i => i.value); }
function checkedJoints() { return [...$("joint-checks").querySelectorAll("input:checked")].map(i => +i.value); }

function renderValueChart() {
  const ep = state.episode;
  if (!ep) return;
  const names = state.meta.values.map(v => v.name), checked = new Set(checkedValues());
  const series = names.map((n, i) => ({ name: n, color: seriesColor(i), values: ep.values[n] })).filter(s => checked.has(s.name));
  const marks = $("show-progress").checked && ep.progress
    ? runs(ep.progress).slice(1).map(r => ({ frame: r.start, label: `step ${r.value}` })) : [];
  const posName = $("positive-source").value, pos = ep.positive[posName];
  const bands = $("show-positive").checked && pos ? [{ mask: pos.map(p => p === 1), color: cssVar("--band") }] : [];
  valueChart.setData({ n: ep.length, series, marks, bands });
}

function renderJointChart() {
  const ep = state.episode;
  if (!ep) return;
  const series = [];
  for (const j of checkedJoints()) {
    if ($("show-state").checked && ep["observation.state"]) series.push({ color: seriesColor(j), values: ep["observation.state"][j] });
    if ($("show-action").checked && ep.action) series.push({ color: seriesColor(j), values: ep.action[j], dash: true });
  }
  jointChart.setData({ n: ep.length, series });
}

const fmt = v => (v == null ? "–" : v.toFixed(3));

function valueReadout(hover) {
  const ep = state.episode, f = hover ?? state.frame;
  if (!ep) return;
  const names = state.meta.values.map(v => v.name), checked = new Set(checkedValues());
  const items = names.map((n, i) => (checked.has(n)
    ? `<span class="swatch" style="background:${seriesColor(i)}"></span>${n} ${fmt(ep.values[n][f])}` : null)).filter(Boolean);
  if (ep.progress) items.push(`step ${ep.progress[f]}`);
  const pos = ep.positive[$("positive-source").value];
  if (pos) items.push(pos[f] === 1 ? "positive" : pos[f] === 0 ? "negative" : "unlabelled");
  $("value-readout").innerHTML = `${(f / state.meta.fps).toFixed(2)} s &nbsp; ${items.join(" &nbsp; ")}`;
}

function jointReadout(hover) {
  const ep = state.episode, f = hover ?? state.frame, names = state.meta.joint_names["observation.state"] || [];
  if (!ep || !ep["observation.state"]) return;
  const items = checkedJoints().map(j => `<span class="swatch" style="background:${seriesColor(j)}"></span>${shortJoint(names[j] || j)} `
    + `${fmt(ep["observation.state"][j][f])}${ep.action ? ` → ${fmt(ep.action[j][f])}` : ""}`);
  $("joint-readout").innerHTML = `${(f / state.meta.fps).toFixed(2)} s &nbsp; ${items.join(" &nbsp; ")}`;
}

// ---------- playback ----------
// The first camera is the clock while playing; without video a wall-clock timer advances the frame.
function master() { return state.videos[0]; }

function frameOf(entry) {
  return Math.max(0, Math.floor((entry.video.currentTime - entry.offset) * state.meta.fps + 1e-3));
}

function seekVideo(entry, frame) {
  entry.video.currentTime = entry.offset + (frame + 0.5) / state.meta.fps;  // mid-frame avoids landing on the previous frame
}

function seek(frame) {
  if (!state.episode) return;
  frame = Math.max(0, Math.min(state.episode.length - 1, frame));
  state.videos.forEach(e => seekVideo(e, frame));
  state.wallStart = performance.now() - (frame / state.meta.fps) * 1000 / rate();
  showFrame(frame);
}

function showFrame(frame) {
  state.frame = frame;
  const n = state.episode.length, fps = state.meta.fps;
  $("scrub").value = frame;
  $("clock").textContent = `${(frame / fps).toFixed(2)} / ${(n / fps).toFixed(2)} s · ${frame}`;
  const task = state.episode.task_index?.[frame];
  const sub = state.episode.subtask_index?.[frame];
  $("instruction").innerHTML = task == null ? "" : `“${state.meta.tasks[task] ?? task}”`
    + (sub != null && sub >= 0 && sub !== task ? ` <span class="muted">subtask ${sub}</span>` : "");
  valueChart.setCursor(frame);
  jointChart.setCursor(frame);
  if (valueChart.hover == null) valueReadout(null);
  if (jointChart.hover == null) jointReadout(null);
}

const rate = () => Number($("rate").value);

function play() {
  if (!state.episode) return;
  if (state.frame >= state.episode.length - 1) seek(0);
  state.playing = true;
  $("play").textContent = "❚❚";
  state.wallStart = performance.now() - (state.frame / state.meta.fps) * 1000 / rate();
  state.videos.forEach(e => { e.video.playbackRate = rate(); e.video.play().catch(() => {}); });
  requestAnimationFrame(tick);
}

function pause() {
  state.playing = false;
  $("play").textContent = "▶";
  state.videos.forEach(e => e.video.pause());
}

function tick() {
  if (!state.playing) return;
  const m = master(), last = state.episode.length - 1;
  const frame = m ? frameOf(m) : Math.floor((performance.now() - state.wallStart) * rate() / 1000 * state.meta.fps);
  if (frame >= last) {
    pause();
    seek(last);
    return;
  }
  for (const e of state.videos.slice(1)) {
    const want = m.video.currentTime - m.offset + e.offset;
    if (Math.abs(e.video.currentTime - want) > SYNC_TOLERANCE_S) e.video.currentTime = want;
  }
  if (frame !== state.frame) showFrame(frame);
  requestAnimationFrame(tick);
}

// Hand-edited URLs (…#ep=700): writeHash uses replaceState, which does not fire hashchange, so this only sees the user's edits.
async function onHashChange() {
  const hash = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
  const ds = hash.ds != null ? +hash.ds : state.ds, ep = hash.ep != null ? +hash.ep : null;
  if (hash.tab && hash.tab !== state.tab) setTab(hash.tab);
  if (ds !== state.ds) await loadDataset(ds, ep);
  else if (ep != null && ep !== state.episode?.episode_index) table.select(ep);
}

// ---------- wiring ----------
window.addEventListener("hashchange", onHashChange);
$("dataset").addEventListener("change", e => loadDataset(+e.target.value));
document.querySelectorAll("#tabs button").forEach(b => b.addEventListener("click", () => setTab(b.dataset.tab)));
$("play").addEventListener("click", () => (state.playing ? pause() : play()));
$("scrub").addEventListener("input", e => { pause(); seek(+e.target.value); });
$("rate").addEventListener("change", () => { state.videos.forEach(e => { e.video.playbackRate = rate(); }); if (state.playing) play(); });
["value-checks", "show-progress", "show-positive", "positive-source"].forEach(id => $(id).addEventListener("change", renderValueChart));
["joint-checks", "show-state", "show-action"].forEach(id => $(id).addEventListener("change", renderJointChart));
DARK.addEventListener("change", () => {
  // recolour swatches in place so the user's checkbox choices survive
  for (const id of ["value-checks", "joint-checks"]) $(id).querySelectorAll(".swatch").forEach((el, i) => { el.style.background = seriesColor(i); });
  renderValueChart();
  renderJointChart();
});

document.addEventListener("keydown", e => {
  if (e.target.matches?.("input:not([type=range]):not([type=checkbox]):not([type=radio]), select")) return;
  const fps = state.meta?.fps || 25;
  const actions = {
    " ": () => (state.playing ? pause() : play()),
    ArrowLeft: () => { pause(); seek(state.frame - (e.shiftKey ? fps : 1)); },
    ArrowRight: () => { pause(); seek(state.frame + (e.shiftKey ? fps : 1)); },
    ArrowUp: () => table.step(-1),
    ArrowDown: () => table.step(1),
  };
  if (actions[e.key]) { e.preventDefault(); actions[e.key](); }
});

init().catch(err => { $("episode-title").textContent = `failed to load: ${err.message}`; });
