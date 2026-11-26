// Minimal canvas line chart for per-frame series with a moving time cursor.
// The static plot (axes, bands, marks, lines) is rendered once into an offscreen canvas; each animation
// frame only blits it and draws the cursor, so playback stays cheap with several 2k-frame series.
"use strict";

// Validated categorical palette (dataviz skill reference instance), fixed order, light / dark.
const SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
const SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"];
const DARK = window.matchMedia("(prefers-color-scheme: dark)");
function seriesColor(i) { return (DARK.matches ? SERIES_DARK : SERIES_LIGHT)[i % 8]; }
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

const PAD = { left: 46, right: 10, top: 10, bottom: 20 };  // px around the plot area
const X_STEPS_S = [0.5, 1, 2, 5, 10, 15, 30, 60, 120];      // candidate x tick spacings, seconds

function niceStep(span, target) {
  const raw = span / target, mag = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
}

class LineChart {
  constructor(canvas, { fps, yInclude = null, onSeek = () => {}, onHover = () => {} }) {
    Object.assign(this, { canvas, fps, yInclude, onSeek, onHover });
    this.ctx = canvas.getContext("2d");
    this.base = document.createElement("canvas");
    this.data = { n: 0, series: [], bands: [], marks: [] };
    this.cursor = 0;
    this.hover = null;
    new ResizeObserver(() => this.render()).observe(canvas);
    DARK.addEventListener("change", () => this.render());
    this.bindMouse();
  }

  bindMouse() {
    let dragging = false;
    const frameAt = e => {
      const r = this.canvas.getBoundingClientRect();
      const t = (e.clientX - r.left - PAD.left) / (r.width - PAD.left - PAD.right);
      return Math.max(0, Math.min(this.data.n - 1, Math.round(t * (this.data.n - 1))));
    };
    this.canvas.addEventListener("mousedown", e => { dragging = true; this.onSeek(frameAt(e)); });
    window.addEventListener("mouseup", () => { dragging = false; });
    this.canvas.addEventListener("mousemove", e => {
      this.hover = frameAt(e);
      if (dragging) this.onSeek(this.hover);
      this.onHover(this.hover);
      this.drawCursor();
    });
    this.canvas.addEventListener("mouseleave", () => { this.hover = null; this.onHover(null); this.drawCursor(); });
  }

  // series: [{name, color, values, dash}], bands: [{mask, color}], marks: [{frame, label}]
  setData(data) { this.data = { bands: [], marks: [], ...data }; this.render(); }

  setFps(fps) { this.fps = fps; }

  setCursor(frame) {
    if (frame === this.cursor) return;
    this.cursor = frame;
    this.drawCursor();
  }

  yRange() {
    let [lo, hi] = this.yInclude || [Infinity, -Infinity];
    for (const s of this.data.series) for (const v of s.values) if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (!isFinite(lo)) return [0, 1];
    const pad = (hi - lo) * 0.05 || 0.5;
    return [lo - pad, hi + pad];
  }

  geometry() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight, [y0, y1] = this.yRange();
    const pw = w - PAD.left - PAD.right, ph = h - PAD.top - PAD.bottom, n = Math.max(this.data.n - 1, 1);
    return {
      w, h, y0, y1,
      x: f => PAD.left + (f / n) * pw,
      y: v => PAD.top + (1 - (v - y0) / (y1 - y0)) * ph,
    };
  }

  render() {
    const dpr = window.devicePixelRatio || 1, g = this.geometry();
    for (const c of [this.canvas, this.base]) { c.width = Math.max(1, g.w * dpr); c.height = Math.max(1, g.h * dpr); }
    const ctx = this.base.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, g.w, g.h);
    if (this.data.n > 0) {
      this.drawBands(ctx, g);
      this.drawAxes(ctx, g);
      this.drawMarks(ctx, g);
      this.drawSeries(ctx, g);
    }
    this.drawCursor();
  }

  drawAxes(ctx, g) {
    ctx.font = "11px -apple-system, sans-serif";
    ctx.fillStyle = cssVar("--muted");
    ctx.lineWidth = 1;
    const yStep = niceStep(g.y1 - g.y0, 4);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let v = Math.ceil(g.y0 / yStep) * yStep; v <= g.y1 + 1e-9; v += yStep) {
      ctx.strokeStyle = cssVar("--grid");
      line(ctx, PAD.left, g.y(v), g.w - PAD.right, g.y(v));
      ctx.fillText(+v.toFixed(3), PAD.left - 6, g.y(v));
    }
    const seconds = (this.data.n - 1) / this.fps;
    const xStep = X_STEPS_S.find(s => seconds / s <= 10) || 300;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let s = 0; s <= seconds + 1e-9; s += xStep) ctx.fillText(`${+s.toFixed(1)}s`, g.x(s * this.fps), g.h - PAD.bottom + 5);
    ctx.strokeStyle = cssVar("--axis");
    line(ctx, PAD.left, g.h - PAD.bottom, g.w - PAD.right, g.h - PAD.bottom);
  }

  drawBands(ctx, g) {
    for (const band of this.data.bands) {
      ctx.fillStyle = band.color;
      let start = -1;
      for (let f = 0; f <= band.mask.length; f++) {
        const on = f < band.mask.length && band.mask[f];
        if (on && start < 0) start = f;
        if (!on && start >= 0) { ctx.fillRect(g.x(start), PAD.top, g.x(f) - g.x(start), g.h - PAD.top - PAD.bottom); start = -1; }
      }
    }
  }

  drawMarks(ctx, g) {
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = cssVar("--muted");
    ctx.fillStyle = cssVar("--muted");
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    for (const m of this.data.marks) {
      line(ctx, g.x(m.frame), PAD.top, g.x(m.frame), g.h - PAD.bottom);
      if (m.label != null) ctx.fillText(m.label, g.x(m.frame) + 2, PAD.top);
    }
    ctx.setLineDash([]);
  }

  drawSeries(ctx, g) {
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    for (const s of this.data.series) {
      ctx.strokeStyle = s.color;
      ctx.setLineDash(s.dash ? [5, 3] : []);
      ctx.beginPath();
      let pen = false;
      s.values.forEach((v, f) => {
        if (v == null) { pen = false; return; }
        pen ? ctx.lineTo(g.x(f), g.y(v)) : ctx.moveTo(g.x(f), g.y(v));
        pen = true;
      });
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  drawCursor() {
    const dpr = window.devicePixelRatio || 1, ctx = this.ctx, g = this.geometry();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.drawImage(this.base, 0, 0);
    if (this.data.n === 0) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.hover != null) {
      ctx.strokeStyle = cssVar("--axis");
      ctx.lineWidth = 1;
      line(ctx, g.x(this.hover), PAD.top, g.x(this.hover), g.h - PAD.bottom);
    }
    ctx.strokeStyle = cssVar("--cursor");
    ctx.lineWidth = 1.5;
    line(ctx, g.x(this.cursor), PAD.top, g.x(this.cursor), g.h - PAD.bottom);
  }
}

function line(ctx, x0, y0, x1, y1) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}
