// Episode browser: filters, sort, and a virtualised list (only the visible rows are in the DOM, so
// 10k+ episodes scroll as fast as 100).
"use strict";

const $ = id => document.getElementById(id);
const fmtValue = v => (v == null ? "" : v.toFixed(2));
const fmtExtra = v => (v == null ? "" : typeof v === "boolean" ? (v ? "✓" : "") : Number.isInteger(v) ? v : v.toFixed(3));

// Column sets per tab: [key, header, grid width, render(row)].
// Subset columns (--episodes) come right after the episode id: they are why the subset was made.
function columnsFor(tab, valueNames, extras) {
  const ep = ["episode_index", "ep", "44px", r => r.episode_index];
  const secs = ["seconds", "s", "44px", r => r.seconds.toFixed(1)];
  const extra = extras.map(c => (c === "subset_rank"
    ? [c, "#", "32px", r => r[c], "row order in the --episodes file"]
    : [c, c, "60px", r => fmtExtra(r[c]), `${c} (from the --episodes file)`]));
  if (tab === "sft") {
    return [ep, secs, ...extra, ["source", "source", "70px", r => r.source], ["tasks", "tasks", "minmax(200px, 1fr)", r => r.tasks]];
  }
  const ok = ["success", "", "16px", r => (r.success ? "<b class=ok>✓</b>" : "<b class=fail>✗</b>")];
  const values = valueNames.map(n => [`value_end:${n}`, n, "56px", r => fmtValue(r[`value_end:${n}`]), `value of ${n} at the last frame`]);
  return [ok, ep, secs, ...extra, ["source", "source", "60px", r => r.source], ["round", "rnd", "28px", r => r.round ?? ""],
    ["seed", "seed", "40px", r => r.seed ?? ""], ...values, ["policy", "policy", "minmax(80px, 1fr)", r => r.policy ?? ""]];
}

class EpisodeTable {
  constructor({ onSelect }) {
    this.onSelect = onSelect;
    this.rows = [];
    this.view = [];
    this.selected = null;
    this.sort = { key: "episode_index", dir: 1 };
    this.scroll = $("table-scroll");
    this.body = $("table-body");
    this.scroll.addEventListener("scroll", () => this.renderRows());
    new ResizeObserver(() => this.renderRows()).observe(this.scroll);
    $("filters").addEventListener("input", () => this.refresh());
    // pointerdown, not click: a re-render between mousedown and mouseup (scroll, refresh) would swallow the click
    this.body.addEventListener("pointerdown", e => {
      const row = e.target.closest(".table-row");
      if (row && e.button === 0) this.select(Number(row.dataset.ep));
    });
  }

  // table: column-oriented {col: [...]} from /api/episodes
  load(table, valueNames, extras, tab) {
    const cols = Object.keys(table);
    this.rows = table.episode_index.map((_, i) => Object.fromEntries(cols.map(c => [c, table[c][i]])));
    this.playableCount = this.rows.filter(r => r.playable).length;
    this.valueNames = valueNames;
    this.extras = extras;
    this.sort = { key: extras.includes("subset_rank") ? "subset_rank" : "episode_index", dir: 1 };
    this.buildFilterOptions();
    this.setTab(tab);
  }

  setTab(tab) {
    this.tab = tab;
    this.columns = columnsFor(tab, this.valueNames, this.extras);
    document.documentElement.style.setProperty("--cols", this.columns.map(c => c[2]).join(" "));
    $("table-head").innerHTML = this.columns.map(([key, label, , , tip]) => {
      const arrow = key === this.sort.key ? (this.sort.dir > 0 ? " ▲" : " ▼") : "";
      return `<span data-key="${key}" title="${tip || "sort by " + (label || key)}">${label}${arrow}</span>`;
    }).join("");
    $("table-head").querySelectorAll("span").forEach(el => el.addEventListener("click", () => this.sortBy(el.dataset.key)));
    this.refresh();
  }

  buildFilterOptions() {
    const uniq = key => [...new Set(this.rows.map(r => r[key]).filter(v => v != null))].sort((a, b) => (a > b) - (a < b));
    const chips = (id, values) => {
      $(id).innerHTML = values.map(v => `<label><input type="checkbox" value="${v}" checked> ${v}</label>`).join("");
    };
    chips("f-source", uniq("source"));
    chips("f-round", uniq("round"));
    $("f-policy").innerHTML = `<option value="">all</option>` + uniq("policy").filter(Boolean).map(p => `<option>${p}</option>`).join("");
  }

  filters() {
    const num = id => ($(id).value === "" ? null : Number($(id).value));
    const checked = id => new Set([...$(id).querySelectorAll("input:checked")].map(i => i.value));
    return {
      search: $("f-search").value.trim().toLowerCase(),
      showAll: $("f-show-all").checked,
      lenMin: num("f-len-min"), lenMax: num("f-len-max"),
      rl: this.tab === "rl",
      outcome: document.querySelector("input[name=outcome]:checked").value,
      sources: checked("f-source"), rounds: checked("f-round"),
      policy: $("f-policy").value,
      seedMin: num("f-seed-min"), seedMax: num("f-seed-max"),
    };
  }

  matches(r, f) {
    if (!f.showAll && !r.playable) return false;
    if (f.search && !`${r.episode_index} ${r.tasks} ${r.policy ?? ""}`.toLowerCase().includes(f.search)) return false;
    if ((f.lenMin != null && r.seconds < f.lenMin) || (f.lenMax != null && r.seconds > f.lenMax)) return false;
    if (!f.rl) return true;
    if ((f.outcome === "success" && !r.success) || (f.outcome === "failure" && r.success)) return false;
    if (!f.sources.has(String(r.source))) return false;
    if (r.round != null && !f.rounds.has(String(r.round))) return false;
    if (f.policy && r.policy !== f.policy) return false;
    if (f.seedMin != null && !(r.seed >= f.seedMin)) return false;
    if (f.seedMax != null && !(r.seed <= f.seedMax)) return false;
    return true;
  }

  sortBy(key) {
    this.sort = { key, dir: this.sort.key === key ? -this.sort.dir : 1 };
    this.setTab(this.tab);
  }

  refresh() {
    const f = this.filters(), { key, dir } = this.sort;
    this.view = this.rows.filter(r => this.matches(r, f));
    this.view.sort((a, b) => {
      const x = a[key], y = b[key];
      if (x == null || y == null) return (x == null) - (y == null);
      return dir * ((x > y) - (x < y));
    });
    const pool = f.showAll ? this.rows.length : this.playableCount, hidden = this.rows.length - this.playableCount;
    $("count").textContent = `${this.view.length.toLocaleString()} of ${pool.toLocaleString()} episodes`
      + (!f.showAll && hidden ? ` · ${hidden.toLocaleString()} without video hidden` : "");
    this.body.style.height = `${this.view.length * this.rowHeight()}px`;
    this.renderRows();
  }

  rowHeight() { return parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--row-h")); }

  renderRows() {
    const h = this.rowHeight(), first = Math.floor(this.scroll.scrollTop / h);
    const last = Math.min(this.view.length, first + Math.ceil(this.scroll.clientHeight / h) + 2);
    let html = "";
    for (let i = first; i < last; i++) {
      const r = this.view[i], sel = (r.episode_index === this.selected ? " selected" : "") + (r.playable ? "" : " novideo");
      const cells = this.columns.map(([key, , , render]) => {
        const cls = ["seconds", "round", "seed", ...this.extras].includes(key) || key.startsWith("value_end") ? " class=num" : "";
        return `<span${cls} title="${key === "policy" || key === "tasks" ? r[key] ?? "" : ""}">${render(r)}</span>`;
      }).join("");
      html += `<div class="table-row${sel}" data-ep="${r.episode_index}" style="top:${i * h}px">${cells}</div>`;
    }
    this.body.innerHTML = html;
  }

  select(ep) {
    this.selected = ep;
    this.renderRows();
    this.onSelect(ep);
  }

  step(delta) {
    const i = this.view.findIndex(r => r.episode_index === this.selected) + delta;
    if (i < 0 || i >= this.view.length) return;
    const h = this.rowHeight();
    if (i * h < this.scroll.scrollTop || (i + 2) * h > this.scroll.scrollTop + this.scroll.clientHeight) {  // +1 row: sticky header
      this.scroll.scrollTop = i * h - this.scroll.clientHeight / 2;
    }
    this.select(this.view[i].episode_index);
  }

  row(ep) { return this.rows.find(r => r.episode_index === ep); }
}
