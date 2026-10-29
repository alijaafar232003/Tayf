// Tayf dashboard. Plain JS + canvas, no build step.
"use strict";

const $ = (id) => document.getElementById(id);
const api = async (path, opts = {}) => {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
};

// ---------- formatting ----------
function fmtHz(hz, digits = 3) {
  const a = Math.abs(hz);
  if (a >= 1e9) return (hz / 1e9).toFixed(digits) + " GHz";
  if (a >= 1e6) return (hz / 1e6).toFixed(digits) + " MHz";
  if (a >= 1e3) return (hz / 1e3).toFixed(digits) + " kHz";
  return hz.toFixed(0) + " Hz";
}
const fmtRate = (fs) => (fs / 1e6).toFixed(3) + " MS/s";
const fmtDur = (s) => s >= 60 ? `${Math.floor(s / 60)} min ${(s % 60).toFixed(0)} s` : s.toFixed(2) + " s";
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------- colormap (viridis-like) ----------
const LUT = (() => {
  const stops = [[0, 68, 1, 84], [0.25, 59, 82, 139], [0.5, 33, 145, 140], [0.75, 94, 201, 98], [1, 253, 231, 37]];
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, r0, g0, b0] = stops[k], [t1, r1, g1, b1] = stops[k + 1];
    const u = (t - t0) / (t1 - t0);
    lut[3 * i] = r0 + u * (r1 - r0); lut[3 * i + 1] = g0 + u * (g1 - g0); lut[3 * i + 2] = b0 + u * (b1 - b0);
  }
  return lut;
})();

// ---------- state ----------
const S = {
  status: null,         // latest session status from the server
  frame: null,          // latest spectrum frame {offset, f0, df, rbw, power}
  auto: { lo: -100, hi: -20, init: false },
  overview: null,       // {canvas, meta}
  ws: null,
  antennas: [],         // antenna registry, for range warnings
  drag: null,           // live drag-to-tune / scan selection state
  presets: [],          // band presets (Lebanon, ITU Region 1)
  hold: null,           // peak-hold trace {key, power}
  sel: null,            // scan: selected frequency {hz} or range {lo, hi}
  lastScan: null,       // scan settings to return to after "Open live here"
};
const M = { left: 52, right: 12, top: 8, bottom: 22 };   // plot margins (CSS px)

// ---------- canvas helpers ----------
function fitCanvas(cv) {
  const r = window.devicePixelRatio || 1;
  const w = Math.round(cv.clientWidth * r), h = Math.round(cv.clientHeight * r);
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; return true; }
  return false;
}
function plotBox(cv) {
  const r = window.devicePixelRatio || 1;
  return { r, x: M.left * r, y: M.top * r, w: cv.width - (M.left + M.right) * r, h: cv.height - (M.top + M.bottom) * r };
}
function dbRange() {
  const lo = parseFloat($("db-min").value), hi = parseFloat($("db-max").value);
  return [isNaN(lo) ? S.auto.lo : lo, isNaN(hi) ? S.auto.hi : hi];
}
function spanOf(frame) {
  const n = frame.power.length;
  return [frame.f0 - frame.df / 2, frame.f0 + frame.df * (n - 0.5)];
}
const loOf = (frame) => frame.f0 + frame.df * frame.power.length / 2;   // center bin = LO
// The frequency window on screen. While a retune is pending (dragging, wheel, click)
// it is already the new window, so the view follows the mouse without waiting
// for the radio; the old spectrum slides along until new data arrives.
function viewSpan() {
  const [lo, hi] = spanOf(S.frame);
  if (S.pending == null) return [lo, hi];
  const half = (hi - lo) / 2;
  return [S.pending.lo - half, S.pending.lo + half];
}
// Resample a frame into `cols` pixel columns covering [vLo, vHi], max-hold so narrow
// signals stay visible. Columns with no data (outside the frame) are NaN.
function resample(frame, vLo, vHi, cols) {
  const out = new Float32Array(cols), n = frame.power.length, p = frame.power;
  const binOf = (hz) => Math.floor((hz - frame.f0) / frame.df + 0.5);
  for (let c = 0; c < cols; c++) {
    let a = binOf(vLo + (vHi - vLo) * c / cols), b = binOf(vLo + (vHi - vLo) * (c + 1) / cols);
    if (b <= a) b = a + 1;
    if (b <= 0 || a >= n) { out[c] = NaN; continue; }
    a = Math.max(0, a); b = Math.min(n, b);
    let m = -Infinity;
    for (let i = a; i < b; i++) if (p[i] > m) m = p[i];
    out[c] = m === -Infinity ? NaN : m;          // only empty bins (NaN) here
  }
  return out;
}
function median(arr) {
  const s = Float32Array.from(arr).filter((v) => !isNaN(v)).sort();
  return s[s.length >> 1];
}

// ---------- spectrum ----------
function drawSpectrum() {
  const cv = $("cv-spectrum"); fitCanvas(cv);
  const ctx = cv.getContext("2d"), b = plotBox(cv), f = S.frame;
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!f) return;
  const [lo, hi] = dbRange(), [fLo, fHi] = viewSpan();
  const yOf = (db) => b.y + b.h * (1 - (db - lo) / (hi - lo));
  const xOf = (hz) => b.x + b.w * (hz - fLo) / (fHi - fLo);
  const css = getComputedStyle(document.documentElement);
  ctx.font = `${11 * b.r}px system-ui`;

  // grid + labels
  ctx.strokeStyle = css.getPropertyValue("--grid"); ctx.fillStyle = css.getPropertyValue("--muted");
  ctx.lineWidth = 1;
  const dbStep = (hi - lo) > 60 ? 20 : 10;
  for (let db = Math.ceil(lo / dbStep) * dbStep; db <= hi; db += dbStep) {
    const y = yOf(db);
    ctx.beginPath(); ctx.moveTo(b.x, y); ctx.lineTo(b.x + b.w, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(db + "", b.x - 6 * b.r, y + 4 * b.r);
  }
  ctx.save(); ctx.translate(12 * b.r, b.y + b.h / 2); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center"; ctx.fillText(S.status?.mode === "scan" ? "dB / bin" : "dBFS / bin", 0, 0); ctx.restore();
  for (let i = 0; i <= 8; i++) {
    const hz = fLo + (fHi - fLo) * i / 8, x = xOf(hz);
    ctx.beginPath(); ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h); ctx.stroke();
    ctx.textAlign = i === 0 ? "left" : i === 8 ? "right" : "center";
    ctx.fillText((hz / 1e6).toFixed(3), x, b.y + b.h + 15 * b.r);
  }

  // annotations (truth for synthetic scenes, labels in SigMF files)
  if ($("chk-truth").checked && S.status?.annotations) {
    for (const a of S.status.annotations) {
      const lo2 = a["core:freq_lower_edge"], hi2 = a["core:freq_upper_edge"];
      if (lo2 == null || hi2 == null) continue;
      const x0 = xOf(lo2), x1 = Math.max(xOf(hi2), x0 + 2 * b.r);
      ctx.fillStyle = css.getPropertyValue("--truth"); ctx.fillRect(x0, b.y, x1 - x0, b.h);
      ctx.fillStyle = css.getPropertyValue("--truth-line"); ctx.textAlign = "center";
      ctx.fillText(a["core:label"] || "", (x0 + x1) / 2, b.y + 12 * b.r);
    }
  }

  ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
  if ($("chk-bands").checked) drawBands(ctx, b, xOf, fLo, fHi, css);
  // Scan: the selected frequency or range
  if (S.status?.mode === "scan" && S.sel) {
    ctx.fillStyle = css.getPropertyValue("--sel"); ctx.strokeStyle = css.getPropertyValue("--tuned");
    if (S.sel.hz != null) {
      ctx.lineWidth = 1.5 * b.r; ctx.setLineDash([4 * b.r, 3 * b.r]);
      const x = xOf(S.sel.hz); ctx.beginPath(); ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h); ctx.stroke();
      ctx.setLineDash([]);
    } else {
      const x0 = xOf(Math.min(S.sel.lo, S.sel.hi)), x1 = xOf(Math.max(S.sel.lo, S.sel.hi));
      ctx.fillRect(x0, b.y, x1 - x0, b.h);
    }
  }
  // peak hold (drawn under the live trace)
  const cols = Math.max(1, Math.floor(b.w));
  if ($("chk-hold").checked && S.hold) {
    const h = resample({ ...f, power: S.hold.power }, fLo, fHi, cols);
    ctx.strokeStyle = css.getPropertyValue("--hold"); ctx.lineWidth = 1 * b.r;
    ctx.beginPath();
    let pen2 = false;
    for (let c = 0; c < cols; c++) {
      if (isNaN(h[c])) { pen2 = false; continue; }
      const x = b.x + c * b.w / cols, y = yOf(h[c]);
      pen2 ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      pen2 = true;
    }
    ctx.stroke();
  }
  // trace
  const p = resample(f, fLo, fHi, cols);
  ctx.strokeStyle = css.getPropertyValue("--trace"); ctx.lineWidth = 1.2 * b.r;
  ctx.beginPath();
  let pen = false;
  for (let c = 0; c < cols; c++) {
    if (isNaN(p[c])) { pen = false; continue; }          // no data here yet
    const x = b.x + c * b.w / cols, y = yOf(p[c]);
    pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    pen = true;
  }
  ctx.stroke();
  if (S.cursorHz != null) {
    ctx.strokeStyle = css.getPropertyValue("--accent"); ctx.lineWidth = 1;
    const x = xOf(S.cursorHz); ctx.beginPath(); ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h); ctx.stroke();
  }
  // Live: the frequency you tuned to (the LO sits offset_hz away from it)
  if (S.status?.mode === "live") {
    const tuned = S.pending ? S.pending.tuned : S.status.tuned_hz;
    ctx.strokeStyle = css.getPropertyValue("--tuned"); ctx.lineWidth = 1.5 * b.r; ctx.setLineDash([4 * b.r, 3 * b.r]);
    const x = xOf(tuned); ctx.beginPath(); ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h); ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

// Band presets as labelled strips along the top of the spectrum.
function drawBands(ctx, b, xOf, fLo, fHi, css) {
  for (const band of S.presets) {
    if (band.id === "all" || band.hi_hz < fLo || band.lo_hz > fHi) continue;
    const x0 = Math.max(b.x, xOf(band.lo_hz)), x1 = Math.min(b.x + b.w, xOf(band.hi_hz));
    ctx.fillStyle = css.getPropertyValue("--band"); ctx.fillRect(x0, b.y, Math.max(x1 - x0, 2 * b.r), b.h);
    ctx.fillStyle = css.getPropertyValue("--band-line"); ctx.fillRect(x0, b.y, Math.max(x1 - x0, 2 * b.r), 3 * b.r);
    const label = band.name, w = ctx.measureText(label).width;
    if (x1 - x0 > w + 6 * b.r || (x1 - x0 > 30 * b.r)) {
      ctx.textAlign = "left";
      ctx.fillText(x1 - x0 > w + 6 * b.r ? label : band.id, x0 + 3 * b.r, b.y + 15 * b.r);
    }
  }
}

// ---------- waterfall ----------
// The waterfall image covers the frequency window [WF.lo, WF.hi]. After a retune
// the image slides sideways so every past row stays at its true frequency
// (like SDR++), instead of being cleared.
const WF = { canvas: document.createElement("canvas"), lo: null, hi: null };
function resetWaterfall() {
  const cv = $("cv-waterfall"); fitCanvas(cv);
  const b = plotBox(cv);
  WF.canvas.width = Math.max(1, Math.floor(b.w));
  WF.canvas.height = Math.max(1, Math.floor(cv.height - b.y - 4 * b.r));
  WF.lo = WF.hi = null;
  const c = WF.canvas.getContext("2d"); c.fillStyle = "#000"; c.fillRect(0, 0, WF.canvas.width, WF.canvas.height);
  drawWaterfall();
}
function alignWaterfall(lo, hi) {
  if (WF.lo == null) { WF.lo = lo; WF.hi = hi; return; }
  if (Math.abs((hi - lo) - (WF.hi - WF.lo)) > 1) {        // span changed (sample rate): start over
    resetWaterfall(); WF.lo = lo; WF.hi = hi; return;
  }
  const w = WF.canvas.width, h = WF.canvas.height, c = WF.canvas.getContext("2d");
  const hzPerPx = (hi - lo) / w;
  const dx = Math.round((WF.lo - lo) / hzPerPx);          // whole pixels the old image moves
  if (dx === 0) return;
  c.drawImage(WF.canvas, 0, 0, w, h, dx, 0, w, h);
  c.fillStyle = "#000";
  dx > 0 ? c.fillRect(0, 0, dx, h) : c.fillRect(w + dx, 0, -dx, h);
  // The image moved by a whole number of pixels; keep its window exact so new rows line up.
  WF.lo -= dx * hzPerPx; WF.hi = WF.lo + (hi - lo);
}
function pushWaterfallRow(frame) {
  const [fLo, fHi] = spanOf(frame);
  alignWaterfall(fLo, fHi);
  const w = WF.canvas.width, h = WF.canvas.height, c = WF.canvas.getContext("2d");
  c.drawImage(WF.canvas, 0, 0, w, h - 1, 0, 1, w, h - 1);     // scroll down by one row
  const [lo, hi] = dbRange(), p = resample(frame, WF.lo, WF.hi, w);
  const img = c.createImageData(w, 1);
  for (let i = 0; i < w; i++) {
    const k = Math.max(0, Math.min(255, Math.round(255 * (p[i] - lo) / (hi - lo))));
    img.data[4 * i] = LUT[3 * k]; img.data[4 * i + 1] = LUT[3 * k + 1]; img.data[4 * i + 2] = LUT[3 * k + 2]; img.data[4 * i + 3] = 255;
  }
  c.putImageData(img, 0, 0);
}
function drawWaterfall() {
  const cv = $("cv-waterfall");
  if (fitCanvas(cv)) return resetWaterfall();
  const ctx = cv.getContext("2d"), b = plotBox(cv);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.imageSmoothingEnabled = false;
  // Place the image at its own frequencies inside the current view (shifts while tuning).
  let x = b.x;
  if (S.frame && WF.lo != null) {
    const [vLo, vHi] = viewSpan();
    x = b.x + (WF.lo - vLo) / (vHi - vLo) * b.w;
  }
  ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, WF.canvas.height); ctx.clip();
  ctx.fillStyle = "#000"; ctx.fillRect(b.x, b.y, b.w, WF.canvas.height);
  ctx.drawImage(WF.canvas, x, b.y, b.w, WF.canvas.height);
  ctx.restore();
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--muted");
  ctx.font = `${11 * b.r}px system-ui`; ctx.textAlign = "right";
  ctx.fillText("now", b.x - 6 * b.r, b.y + 10 * b.r);
  ctx.fillText("older", b.x - 6 * b.r, b.y + WF.canvas.height - 4 * b.r);
}

// ---------- overview / timeline ----------
async function loadOverview() {
  const o = await api("/api/session/overview?rows=400&nfft=1024");
  const bytes = Uint8Array.from(atob(o.data_b64), (c) => c.charCodeAt(0));
  const img = document.createElement("canvas");
  img.width = o.rows; img.height = o.cols;          // x = time, y = frequency
  const c = img.getContext("2d"), d = c.createImageData(o.rows, o.cols);
  for (let r = 0; r < o.rows; r++) {
    for (let col = 0; col < o.cols; col++) {
      const k = bytes[r * o.cols + col], p = 4 * ((o.cols - 1 - col) * o.rows + r);
      d.data[p] = LUT[3 * k]; d.data[p + 1] = LUT[3 * k + 1]; d.data[p + 2] = LUT[3 * k + 2]; d.data[p + 3] = 255;
    }
  }
  c.putImageData(d, 0, 0);
  S.overview = { canvas: img, meta: o };
  drawOverview();
}
function drawOverview() {
  const cv = $("cv-overview"); fitCanvas(cv);
  const ctx = cv.getContext("2d"), r = window.devicePixelRatio || 1;
  const b = { x: M.left * r, y: 4 * r, w: cv.width - (M.left + M.right) * r, h: cv.height - 20 * r };
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!S.overview || !S.status?.active) return;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(S.overview.canvas, b.x, b.y, b.w, b.h);
  const dur = S.status.duration_s, pos = S.frame ? S.frame.offset : S.status.position_s;
  const x = b.x + b.w * Math.min(1, pos / dur);
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5 * r;
  ctx.beginPath(); ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h); ctx.stroke();
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--muted");
  ctx.font = `${11 * r}px system-ui`;
  for (let i = 0; i <= 5; i++) {
    ctx.textAlign = i === 0 ? "left" : i === 5 ? "right" : "center";
    ctx.fillText((dur * i / 5).toFixed(1) + " s", b.x + b.w * i / 5, b.y + b.h + 14 * r);
  }
  S.overviewBox = b;
}

// ---------- frames from the server ----------
function onFrame(buf) {
  const h = new Float64Array(buf, 0, 4);
  const frame = { offset: h[0], f0: h[1], df: h[2], rbw: h[3], power: new Float32Array(buf, 32) };
  S.frame = frame;
  updateHold(frame);
  // The retune we asked for has arrived: stop showing the predicted view.
  if (S.pending && !S.drag && !S.pending.unsent && Math.abs(loOf(frame) - S.pending.lo) < 2 * frame.df) S.pending = null;
  if (S.pending && performance.now() - S.pending.t > 2000 && !S.drag) S.pending = null;   // give up waiting
  // Auto dB range: floor (median) - 10 dB up to peak + 5 dB, smoothed so it doesn't jump.
  const floor = median(frame.power);
  let peak = -Infinity; for (const v of frame.power) if (v > peak) peak = v;
  const lo = floor - 10, hi = Math.max(peak + 5, floor + 30);
  if (!S.auto.init) { S.auto = { lo, hi, init: true }; }
  else { S.auto.lo += 0.05 * (lo - S.auto.lo); S.auto.hi += 0.05 * (hi - S.auto.hi); }

  $("ro-time").textContent = frame.offset.toFixed(3) + " s";
  $("ro-rbw").textContent = fmtHz(frame.rbw, 1);
  $("ro-floor").textContent = `${floor.toFixed(1)} dBFS/bin  (${(floor - 10 * Math.log10(frame.rbw)).toFixed(1)} dBFS/Hz)`;
  if (S.status?.playing) pushWaterfallRow(frame);
  scheduleDraw();
}
// Peak hold: max per bin since the grid last changed (retune, new range, new FFT size).
function updateHold(frame) {
  const key = `${frame.f0}|${frame.df}|${frame.power.length}`;
  if (!S.hold || S.hold.key !== key) { S.hold = { key, power: Float32Array.from(frame.power) }; return; }
  const h = S.hold.power, p = frame.power;
  for (let i = 0; i < p.length; i++) if (!(h[i] >= p[i])) h[i] = isNaN(p[i]) ? h[i] : p[i];
}
let drawPending = false;
function scheduleDraw() {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(() => { drawPending = false; drawSpectrum(); drawWaterfall(); drawOverview(); });
}

function connectWs() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/spectrum`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") { const m = JSON.parse(ev.data); if (m.type === "status") setStatus(m); }
    else onFrame(ev.data);
  };
  ws.onclose = () => setTimeout(connectWs, 1000);
  S.ws = ws;
}

function setStatus(st) {
  const prevId = S.status?.session_id;
  S.status = st;
  const badge = $("session-badge");
  badge.textContent = st.active ? `Session ${st.session_id}: ${st.mode}` : "No session";
  badge.classList.toggle("on", !!st.active);
  if (!st.active) return;
  const live = st.mode === "live", scan = st.mode === "scan";
  $("analysis").classList.toggle("mode-live", live);
  $("analysis").classList.toggle("mode-scan", scan);
  $("analysis").classList.toggle("mode-recording", st.mode === "recording");
  for (const id of ["cv-spectrum", "cv-waterfall"]) $(id).classList.toggle("tunable", live);
  $("btn-play").textContent = live || scan ? (st.playing ? "Freeze" : "Run") : (st.playing ? "Pause" : "Play");
  $("sel-nfft").value = String(st.nfft);
  $("chk-dc").checked = st.dc_remove;
  showHealth(st);
  $("an-title").textContent = st.name;
  if (scan) {
    showScan(st);
  } else if (live) {
    $("btn-live-back").classList.toggle("hidden", !S.lastScan);
    if (document.activeElement !== $("in-freq") && !S.pending) $("in-freq").value = (st.tuned_hz / 1e6).toFixed(4);
    $("sel-rate").value = String(st.sample_rate_hz);
    $("chk-amp").checked = st.gains.amp;
    $("sel-lna").value = String(st.gains.lna_db);
    $("rng-vga").value = st.gains.vga_db; $("vga-val").textContent = st.gains.vga_db;
    $("chk-offset").checked = st.offset_tuning;
    $("sel-antenna").value = st.antenna_id || "";
    $("ro-tuned").textContent = fmtHz(st.tuned_hz, 4);
    $("ro-lo").textContent = st.offset_hz ? `${fmtHz(st.center_freq_hz, 4)} (offset +${fmtHz(st.offset_hz, 0)})` : fmtHz(st.center_freq_hz, 4);
    $("ro-ovf").textContent = st.overflows;
    $("ro-ovf").classList.toggle("warn", st.overflows > 0);
    $("an-info").textContent = `${fmtRate(st.sample_rate_hz)} · total gain ${st.gains.total_db} dB`;
    $("ant-warn").textContent = antennaWarning(st.antenna_id, st.tuned_hz);
    showRecording(st);
  } else {
    $("sel-speed").value = String(st.speed);
    if (st.session_id !== prevId) {
      const s = st.settings;
      const gains = s.lna_gain_db == null ? "gains unknown" : `amp ${s.amp_on ? "on" : "off"}, LNA ${s.lna_gain_db} dB, VGA ${s.vga_gain_db} dB`;
      $("an-info").textContent = `${fmtHz(st.center_freq_hz)} · ${fmtRate(st.sample_rate_hz)} · ${fmtDur(st.duration_s)} · ${gains}` +
        (st.comparable ? "" : " · not comparable with history");
    }
  }
}

function showScan(st) {
  const busy = (id) => document.activeElement === $(id);
  if (!busy("scan-lo")) $("scan-lo").value = +(st.f_lo_hz / 1e6).toFixed(3);
  if (!busy("scan-hi")) $("scan-hi").value = +(st.f_hi_hz / 1e6).toFixed(3);
  $("scan-preset").value = st.preset_id || "";
  $("scan-bin").value = String(st.bin_hz);
  $("scan-amp").checked = st.gains.amp;
  $("scan-lna").value = String(st.gains.lna_db);
  $("scan-vga").value = String(st.gains.vga_db);
  $("scan-ant").value = st.antenna_id || "";
  $("ro-range").textContent = `${fmtHz(st.f_lo_hz, 1)} – ${fmtHz(st.f_hi_hz, 1)}`;
  $("ro-sweeps").textContent = `${st.sweeps} (${st.sweep_rate.toFixed(1)} per s)` + (st.error ? ` · stopped: ${st.error}` : "");
  $("ro-sweeps").classList.toggle("warn", !!st.error);
  $("an-info").textContent = `bin ${fmtHz(st.bin_hz, 0)} · total gain ${st.gains.total_db} dB`;
  $("scan-warn").textContent = scanAntennaWarning(st.antenna_id, st.f_lo_hz, st.f_hi_hz);
  $("scan-preset-info").innerHTML = presetNote(S.presets.find((b) => b.id === st.preset_id));
}
function scanAntennaWarning(antennaId, lo, hi) {
  const a = S.antennas.find((x) => x.id === antennaId);
  if (!a) return "No antenna selected";
  if (lo >= a.freq_min_hz && hi <= a.freq_max_hz) return "";
  return `${a.name} is rated ${fmtHz(a.freq_min_hz, 1)} – ${fmtHz(a.freq_max_hz, 1)}: part of this range is outside it`;
}
function presetNote(band) {
  if (!band) return "";
  const tag = band.verified ? '<span class="tag seen">seen here</span>' : '<span class="tag expected">expected (Region 1 table)</span>';
  return `${tag}${esc(band.expect)}${band.seen ? ` <span class="muted">· ${esc(band.seen)}</span>` : ""}` +
    (band.hint ? ` <span class="muted">· Scan hint: ${esc(band.hint)}</span>` : "");
}

function showHealth(st) {
  const h = st.health || {};
  if (h.level_dbfs == null) return;
  const lvl = $("ro-level"), clip = $("ro-clip");
  const hint = { ok: "", low: " (low: raise the gain)", clipping: " (clipping: lower the gain)" }[h.level_hint];
  lvl.textContent = `${h.level_dbfs.toFixed(1)} dBFS${h.level_hint === "low" ? hint : ""}`;
  clip.textContent = `${h.clip_pct.toFixed(2)} %${h.level_hint === "clipping" ? hint : ""}`;
  lvl.classList.toggle("warn", h.level_hint === "low");
  clip.classList.toggle("warn", h.level_hint === "clipping");
}

function showRecording(st) {
  const rec = st.recording, btn = $("btn-rec"), info = $("rec-info");
  btn.classList.toggle("on", !!rec);
  info.classList.toggle("on", !!rec);
  if (rec) {
    btn.innerHTML = "&#9632; Stop";
    info.textContent = `REC ${rec.seconds.toFixed(1)} s · ${rec.mbytes.toFixed(1)} MB · ` +
      `${rec.segments} segment${rec.segments > 1 ? "s" : ""}` +
      (rec.dropped_samples ? ` · ${rec.dropped_samples} samples lost (disk too slow)` : "") +
      " · changing the sample rate stops the recording";
  } else {
    btn.innerHTML = "&#9679; Record";
    const s = st.last_saved;
    info.textContent = s ? `Saved ${s.name} (${s.seconds} s, ${s.mbytes} MB): it is in the recordings list on the home page.` : "";
  }
}

function antennaWarning(antennaId, hz) {
  const a = S.antennas.find((x) => x.id === antennaId);
  if (!a) return "No antenna selected";
  if (hz < a.freq_min_hz || hz > a.freq_max_hz) return `${a.name} covers ${fmtHz(a.freq_min_hz, 1)} – ${fmtHz(a.freq_max_hz, 1)}: outside its range`;
  return "";
}

// ---------- interactions ----------
async function control(body) {
  try { setStatus(await api("/api/session/control", { method: "POST", body })); }
  catch (e) { alert(e.message); }
}
$("btn-play").onclick = () => control({ playing: !S.status?.playing });
$("sel-speed").onchange = (e) => control({ speed: parseFloat(e.target.value) });
$("sel-nfft").onchange = (e) => control({ nfft: parseInt(e.target.value) });
$("chk-dc").onchange = (e) => control({ dc_remove: e.target.checked });
$("btn-rec").onclick = async () => {
  try { setStatus(await api("/api/session/record", { method: "POST", body: { on: !S.status?.recording } })); }
  catch (e) { alert(e.message); }
};

// ---------- live tuning ----------
// Every way of tuning (drag, click, wheel, typing) first moves the view to where
// the radio is going (S.pending), then sends the retune. The view follows the mouse
// at once; retunes are rate-limited to what the HackRF can do (~60 ms each).
const RETUNE_MS = 100;
let lastTuneSent = 0, tuneTimer = null;
const clampHz = (hz) => Math.min(6e9 - 2e7, Math.max(1e6, hz));
function setPending(tuned) {
  tuned = clampHz(tuned);
  const offset = S.status.offset_tuning ? S.status.offset_hz : 0;
  S.pending = { tuned, lo: tuned + offset, t: performance.now(), unsent: true };
  $("in-freq").value = (tuned / 1e6).toFixed(4);
  scheduleDraw();
}
function sendPending(now = false) {
  clearTimeout(tuneTimer);
  const wait = now ? 0 : Math.max(0, RETUNE_MS - (performance.now() - lastTuneSent));
  tuneTimer = setTimeout(() => {
    if (!S.pending) return;
    lastTuneSent = performance.now();
    S.pending.unsent = false;
    control({ center_hz: Math.round(S.pending.tuned) });
  }, wait);
}
const tuneTo = (hz) => { setPending(hz); sendPending(true); };
const currentTuned = () => (S.pending ? S.pending.tuned : S.status.tuned_hz);
$("in-freq").onchange = (e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) tuneTo(v * 1e6); };
$("sel-rate").onchange = (e) => control({ sample_rate_hz: parseFloat(e.target.value) });
$("chk-amp").onchange = (e) => control({ amp: e.target.checked });
$("sel-lna").onchange = (e) => control({ lna_db: parseInt(e.target.value) });
$("rng-vga").oninput = (e) => { $("vga-val").textContent = e.target.value; };
$("rng-vga").onchange = (e) => control({ vga_db: parseInt(e.target.value) });
$("chk-offset").onchange = (e) => control({ offset_tuning: e.target.checked });
$("sel-antenna").onchange = (e) => control({ antenna_id: e.target.value });

// Mouse on the spectrum or waterfall (live only):
//  - drag sideways: the band moves with the mouse, like dragging a map
//  - click: tune to the clicked frequency (the dashed line jumps there)
//  - wheel: one step per notch (Shift = 10x finer); quick notches add up
function hzAtMouse(cv, e) {
  const b = plotBox(cv), x = (e.clientX - cv.getBoundingClientRect().left) * b.r;
  const [vLo, vHi] = viewSpan();
  return vLo + (vHi - vLo) * (x - b.x) / b.w;
}
for (const id of ["cv-spectrum", "cv-waterfall"]) {
  const cv = $(id);
  cv.addEventListener("mousedown", (e) => {
    if (S.status?.mode === "scan" && S.frame && e.button === 0) {
      S.drag = { cv, scan: true, x0: e.clientX, hz0: hzAtMouse(cv, e), moved: false };
      e.preventDefault();
      return;
    }
    if (S.status?.mode !== "live" || !S.frame || e.button !== 0) return;
    const [vLo, vHi] = viewSpan();
    S.drag = { cv, x0: e.clientX, start: currentTuned(), moved: false,
               hzPerPx: (vHi - vLo) / plotBox(cv).w * (window.devicePixelRatio || 1) };
    cv.classList.add("dragging");
    e.preventDefault();
  });
  cv.addEventListener("wheel", (e) => {
    if (S.status?.mode !== "live" || !S.frame) return;
    e.preventDefault();
    const step = parseFloat($("sel-step").value) / (e.shiftKey ? 10 : 1);
    const base = Math.round(currentTuned() / step) * step;
    setPending(base + (e.deltaY < 0 ? step : -step));
    sendPending();
  }, { passive: false });
}
window.addEventListener("mousemove", (e) => {
  if (!S.drag) return;
  const dx = e.clientX - S.drag.x0;
  if (Math.abs(dx) > 3) S.drag.moved = true;
  if (!S.drag.moved) return;
  if (S.drag.scan) { setSelection({ lo: S.drag.hz0, hi: hzAtMouse(S.drag.cv, e) }); return; }
  setPending(S.drag.start - dx * S.drag.hzPerPx);
  sendPending();
});
window.addEventListener("mouseup", (e) => {
  if (!S.drag) return;
  const d = S.drag;
  if (d.scan) {
    S.drag = null;
    if (!d.moved) setSelection({ hz: snapToPeak(hzAtMouse(d.cv, e)) });
    return;
  }
  d.cv.classList.remove("dragging");
  if (!d.moved) setPending(hzAtMouse(d.cv, e));         // a click: tune to that frequency
  S.drag = null;
  sendPending(true);
});
for (const id of ["db-min", "db-max", "chk-truth"]) $(id).oninput = scheduleDraw;
$("cv-overview").onclick = (e) => {
  if (!S.overviewBox || !S.status) return;
  const r = window.devicePixelRatio || 1, cv = e.currentTarget;
  const x = (e.clientX - cv.getBoundingClientRect().left) * r;
  const t = Math.max(0, Math.min(1, (x - S.overviewBox.x) / S.overviewBox.w)) * S.status.duration_s;
  control({ seek_s: t });
};
for (const id of ["cv-spectrum", "cv-waterfall"]) {
  const cv = $(id);
  cv.onmousemove = (e) => {
    if (!S.frame || S.drag) return;
    const hz = hzAtMouse(cv, e), i = Math.round((hz - S.frame.f0) / S.frame.df);
    S.cursorHz = hz;
    const p = S.frame.power[i];
    $("ro-cursor").textContent = fmtHz(hz, 4) + (p === undefined ? "" : `, ${p.toFixed(1)} dBFS`);
    scheduleDraw();
  };
  cv.onmouseleave = () => { S.cursorHz = null; scheduleDraw(); };
}
window.addEventListener("resize", () => { resetWaterfall(); scheduleDraw(); });

// ---------- scan: select, zoom, open live ----------
// A click picks the strongest bin within a few pixels, so you land on the signal.
function snapToPeak(hz) {
  const f = S.frame, [vLo, vHi] = viewSpan();
  const reach = 4 * (vHi - vLo) / $("cv-spectrum").clientWidth;
  const a = Math.max(0, Math.floor((hz - reach - f.f0) / f.df));
  const b = Math.min(f.power.length - 1, Math.ceil((hz + reach - f.f0) / f.df));
  let best = null;
  for (let i = a; i <= b; i++) if (!isNaN(f.power[i]) && (best == null || f.power[i] > f.power[best])) best = i;
  return best == null ? hz : f.f0 + best * f.df;
}
function setSelection(sel) {
  S.sel = sel;
  const point = sel?.hz != null, range = !!sel && !point;
  $("scan-actions").classList.toggle("hidden", !point);
  $("btn-zoom").classList.toggle("hidden", !range);
  if (point) {
    const i = Math.round((sel.hz - S.frame.f0) / S.frame.df), p = S.frame.power[i];
    $("scan-sel").textContent = `Selected ${fmtHz(sel.hz, 4)}` + (p == null || isNaN(p) ? "" : ` (${p.toFixed(1)} dB)`);
  } else if (range) {
    const lo = Math.min(sel.lo, sel.hi), hi = Math.max(sel.lo, sel.hi);
    $("scan-sel").textContent = `Range ${fmtHz(lo, 3)} – ${fmtHz(hi, 3)} (${fmtHz(hi - lo, 1)})`;
  } else {
    $("scan-sel").textContent = "Click a signal to select it, or drag across a range to zoom in.";
  }
  scheduleDraw();
}
async function scanControl(body) {
  setSelection(null);
  $("scan-sel").textContent = "Restarting the sweep…";
  try { setStatus(await api("/api/session/control", { method: "POST", body })); setSelection(null); }
  catch (e) { $("scan-sel").textContent = e.message; }
}
$("btn-zoom").onclick = () => {
  const lo = Math.min(S.sel.lo, S.sel.hi), hi = Math.max(S.sel.lo, S.sel.hi);
  // A narrower range can afford finer bins: aim for 1000-4000 bins.
  const bins = S.binWidths.filter((w) => (hi - lo) / w <= 4000);
  const bin = bins.find((w) => (hi - lo) / w >= 1000) ?? bins[bins.length - 1] ?? S.binWidths[0];
  scanControl({ f_lo_hz: Math.floor(lo / 1e5) * 1e5, f_hi_hz: Math.ceil(hi / 1e5) * 1e5, bin_hz: bin });
};
$("btn-go-live").onclick = async () => {
  const st = S.status;
  const back = { radio: st.radio, f_lo_hz: st.f_lo_hz, f_hi_hz: st.f_hi_hz, bin_hz: st.bin_hz, amp: st.gains.amp,
                 lna_db: st.gains.lna_db, vga_db: st.gains.vga_db, antenna_id: st.antenna_id, preset_id: st.preset_id };
  try {
    await openLive({ radio: st.radio, center_hz: Math.round(S.sel.hz / 1e3) * 1e3,
                     sample_rate_hz: parseFloat($("scan-live-rate").value), amp: st.gains.amp,
                     lna_db: st.gains.lna_db, vga_db: st.gains.vga_db, antenna_id: st.antenna_id });
    S.lastScan = back;
    $("btn-live-back").classList.remove("hidden");
  } catch (e) { $("scan-sel").textContent = e.message; }
};
$("btn-live-back").onclick = async () => {
  try { await openScan(S.lastScan); } catch (e) { alert(e.message); }
};
$("scan-preset").onchange = (e) => {
  const b = S.presets.find((x) => x.id === e.target.value);
  if (b) scanControl({ f_lo_hz: b.scan_lo_hz, f_hi_hz: b.scan_hi_hz, bin_hz: b.bin_hz, preset_id: b.id,
                       lna_db: b.lna_db, vga_db: b.vga_db, antenna_id: b.antenna_id });
};
$("scan-lo").onchange = $("scan-hi").onchange = () => {
  const lo = parseFloat($("scan-lo").value) * 1e6, hi = parseFloat($("scan-hi").value) * 1e6;
  if (lo < hi) scanControl({ f_lo_hz: lo, f_hi_hz: hi });
};
$("scan-bin").onchange = (e) => scanControl({ bin_hz: parseFloat(e.target.value) });
$("scan-amp").onchange = (e) => scanControl({ amp: e.target.checked });
$("scan-lna").onchange = (e) => scanControl({ lna_db: parseInt(e.target.value) });
$("scan-vga").onchange = (e) => scanControl({ vga_db: parseInt(e.target.value) });
$("scan-ant").onchange = (e) => control({ antenna_id: e.target.value });
$("chk-hold").onchange = () => { S.hold = null; scheduleDraw(); };
$("chk-bands").onchange = scheduleDraw;

// ---------- views ----------
function showView(name) {
  $("home").classList.toggle("hidden", name !== "home");
  $("analysis").classList.toggle("hidden", name !== "analysis");
  if (name === "analysis") { S.auto.init = false; resetWaterfall(); scheduleDraw(); }
}
async function openLive(body) {
  setStatus(await api("/api/session/live", { method: "POST", body }));
  S.frame = null; S.overview = null; S.pending = null; S.hold = null; S.lastScan = null;
  showView("analysis");
}
async function openScan(body) {
  setStatus(await api("/api/session/scan", { method: "POST", body }));
  S.frame = null; S.overview = null; S.pending = null; S.hold = null; S.lastScan = null;
  showView("analysis");
  setSelection(null);
}
async function loadOptions() {
  const o = await api("/api/radio/options");
  const opts = (vals, fmt, sel) => vals.map((v) => `<option value="${v}"${v === sel ? " selected" : ""}>${fmt(v)}</option>`).join("");
  for (const el of document.querySelectorAll(".sel-rate")) el.innerHTML = opts(o.sample_rates, fmtRate, 2e6);
  for (const el of document.querySelectorAll(".sel-lna")) el.innerHTML = opts(o.lna_steps, String, 16);
  for (const el of document.querySelectorAll(".sel-vga")) el.innerHTML = opts(o.vga_steps, String, 20);
  for (const el of document.querySelectorAll(".sel-bin")) el.innerHTML = opts(o.bin_widths, (w) => fmtHz(w, 0), 25e3);
  $("scan-live-rate").value = "10000000";
  S.binWidths = o.bin_widths;
  S.presets = await api("/api/presets");
  const popts = S.presets.map((b) => `<option value="${esc(b.id)}">${esc(b.name)}${b.verified ? "" : " (to verify)"}</option>`).join("");
  for (const el of document.querySelectorAll(".sel-preset")) el.innerHTML = '<option value="">custom range</option>' + popts;
}
// Scan form on the home page: a preset fills in the range, bin, gains and antenna.
function applyPresetToForm(id) {
  const f = $("scan-form"), b = S.presets.find((x) => x.id === id);
  $("scan-preset-note").innerHTML = presetNote(b);
  if (!b) return;
  f.f_lo_mhz.value = b.scan_lo_hz / 1e6; f.f_hi_mhz.value = b.scan_hi_hz / 1e6;
  f.bin_hz.value = String(b.bin_hz); f.lna_db.value = String(b.lna_db); f.vga_db.value = String(b.vga_db);
  f.amp.value = "0"; f.antenna_id.value = b.antenna_id;
}
$("scan-form").preset.onchange = (e) => applyPresetToForm(e.target.value);
for (const k of ["f_lo_mhz", "f_hi_mhz"]) $("scan-form")[k].addEventListener("input", () => { $("scan-form").preset.value = ""; applyPresetToForm(""); });
$("scan-form").onsubmit = async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  $("scan-msg").textContent = "Starting hackrf_sweep…";
  try {
    await openScan({
      radio: f.get("radio"), f_lo_hz: parseFloat(f.get("f_lo_mhz")) * 1e6, f_hi_hz: parseFloat(f.get("f_hi_mhz")) * 1e6,
      bin_hz: parseFloat(f.get("bin_hz")), amp: f.get("amp") === "1",
      lna_db: parseInt(f.get("lna_db")), vga_db: parseInt(f.get("vga_db")),
      antenna_id: f.get("antenna_id") || null, preset_id: f.get("preset") || null,
    });
    $("scan-msg").textContent = "";
  } catch (err) { $("scan-msg").textContent = err.message; }
};
async function checkDevice() {
  const d = await api("/api/devices");
  const el = $("dev-status");
  if (d.in_use) el.textContent = `${d.radio} is in use by the current session.`;
  else if (d.hackrf.length) el.innerHTML = `<span class="ok">HackRF found</span> (serial ${esc(d.hackrf[0].serial || "?")}).`;
  else el.innerHTML = `<span class="warn">No HackRF found.</span> Plug it in and reload, or use the simulated radio.`;
  if (!d.hackrf.length && !d.in_use) for (const f of ["live-form", "scan-form"]) $(f).radio.value = "sim";
}
async function adviseAntenna() {
  const mhz = parseFloat(document.querySelector("#live-form [name=center_mhz]").value);
  if (isNaN(mhz)) return;
  const fits = await api(`/api/antennas?freq_hz=${mhz * 1e6}`);
  $("ant-advice").textContent = fits.length
    ? `Antennas covering ${mhz} MHz: ${fits.map((a) => a.name).join(", ")}.`
    : `None of your antennas is rated for ${mhz} MHz.`;
  const sel = $("live-antenna");
  if (fits.length && !fits.some((a) => a.id === sel.value)) sel.value = fits[0].id;
}
document.querySelector("#live-form [name=center_mhz]").addEventListener("change", adviseAntenna);
$("live-form").onsubmit = async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  $("live-msg").textContent = "Opening the radio…";
  try {
    await openLive({
      radio: f.get("radio"), center_hz: parseFloat(f.get("center_mhz")) * 1e6,
      sample_rate_hz: parseFloat(f.get("sample_rate_hz")), amp: f.get("amp") === "1",
      lna_db: parseInt(f.get("lna_db")), vga_db: parseInt(f.get("vga_db")),
      antenna_id: f.get("antenna_id") || null,
    });
    $("live-msg").textContent = "";
  } catch (err) { $("live-msg").textContent = err.message; }
};
// ---------- antenna drawings ----------
// Small illustrations so you can tell which antenna is which. Not to scale.
const ANT_INFO = {
  whip: { mount: "Magnetic base with coax lead", expect: "Works from FM to UHF; best near its design band" },
  telescopic: { mount: "SMA, straight on the HackRF", expect: "Pull it out to a quarter wave for the band you want" },
  mag: { mount: "Magnetic base with coax lead", expect: "Real gain likely 2–5 dBi, not 12" },
  "mag-tall": { mount: "Large magnetic base with coax lead", expect: "Real gain a few dBi: 35 dBi is not possible at this size" },
  duck: { mount: "SMA, tilt-and-swivel joint", expect: "Resonant at 2.4 and 5 GHz; real gain ~2–5 dBi" },
  generic: { mount: "–", expect: "–" },
};
function antennaKind(a) {
  const id = (a.id + " " + a.name).toLowerCase();
  if (id.includes("35dbi") || id.includes("35 dbi")) return "mag-tall";
  if (id.includes("mag-") || id.includes("magnetic") && !id.includes("whip")) return "mag";
  if (id.includes("whip")) return "whip";
  if (id.includes("telescop")) return "telescopic";
  if (id.includes("duck")) return "duck";
  return "generic";
}
function antennaSvg(a) {
  const metal = "#c9d1d9", rubber = "#3b4149", edge = "#8b94a1", gold = "#d4a84b", base = "#4a5260";
  const magBase = (rx, y) => `<ellipse cx="30" cy="${y + 3}" rx="${rx}" ry="4" fill="${base}" stroke="${edge}"/>
    <rect x="${30 - rx}" y="${y - 2}" width="${2 * rx}" height="5" fill="${base}"/>
    <ellipse cx="30" cy="${y - 2}" rx="${rx}" ry="4" fill="#5d6675" stroke="${edge}"/>
    <path d="M${30 + rx - 2} ${y + 3} q 8 4 ${16 - rx} 6" stroke="${edge}" stroke-width="1.6" fill="none"/>`;
  const sma = (y) => `<rect x="26" y="${y}" width="8" height="7" rx="1" fill="${gold}" stroke="#a07d2c"/>
    <rect x="27" y="${y + 7}" width="6" height="4" fill="${gold}" opacity="0.7"/>`;
  const shapes = {
    whip: `<line x1="30" y1="5" x2="30" y2="62" stroke="${metal}" stroke-width="1.4"/>
      <circle cx="30" cy="5" r="1.6" fill="${metal}"/>
      <path d="M27 62 h6 M27 60 h6 M27 58 h6 M27 56 h6" stroke="${edge}" stroke-width="1"/>
      ${magBase(11, 69)}`,
    telescopic: `<rect x="28.9" y="4" width="2.2" height="16" fill="${metal}"/>
      <rect x="28.4" y="20" width="3.2" height="14" fill="${metal}" stroke="${edge}" stroke-width="0.4"/>
      <rect x="27.9" y="34" width="4.2" height="14" fill="${metal}" stroke="${edge}" stroke-width="0.4"/>
      <rect x="27.3" y="48" width="5.4" height="14" fill="${metal}" stroke="${edge}" stroke-width="0.4"/>
      <circle cx="30" cy="4" r="2" fill="${metal}"/>
      ${sma(63)}`,
    mag: `<rect x="28" y="6" width="4" height="56" rx="2" fill="${rubber}" stroke="${edge}" stroke-width="0.6"/>
      <rect x="26.5" y="30" width="7" height="10" rx="2" fill="${rubber}" stroke="${edge}" stroke-width="0.6"/>
      <path d="M27 32 h6 M27 35 h6 M27 38 h6" stroke="${edge}" stroke-width="0.6"/>
      ${magBase(12, 69)}`,
    "mag-tall": `<rect x="26.5" y="3" width="7" height="60" rx="3" fill="${rubber}" stroke="${edge}" stroke-width="0.6"/>
      <rect x="25" y="56" width="10" height="7" rx="1.5" fill="#2d3238" stroke="${edge}" stroke-width="0.6"/>
      ${magBase(15, 69)}`,
    duck: `<g transform="rotate(-12 30 60)">
        <path d="M27 14 q3 -6 6 0 l1.6 38 h-9.2 z" fill="${rubber}" stroke="${edge}" stroke-width="0.6"/>
        <path d="M26.5 44 h7 M26.3 47 h7.4" stroke="${edge}" stroke-width="0.5"/>
      </g>
      <circle cx="30" cy="59" r="3.2" fill="#2d3238" stroke="${edge}" stroke-width="0.6"/>
      ${sma(63)}`,
    generic: `<line x1="30" y1="8" x2="30" y2="62" stroke="${metal}" stroke-width="2"/>${sma(63)}`,
  };
  return `<svg viewBox="0 0 60 80" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">${shapes[antennaKind(a)]}</svg>`;
}
// Coverage bar: the antenna's rated range on a log frequency axis, with the band presets marked.
function coverageSvg(a) {
  const W = 360, x0 = 8, x1 = W - 8, lg = (hz) => Math.log10(hz);
  const xOf = (hz) => x0 + (x1 - x0) * (lg(hz) - lg(1e6)) / (lg(6e9) - lg(1e6));
  const ticks = [[1e6, "1M"], [10e6, "10M"], [100e6, "100M"], [1e9, "1G"], [6e9, "6G"]]
    .map(([hz, t]) => `<line x1="${xOf(hz)}" y1="30" x2="${xOf(hz)}" y2="34" stroke="#8b94a1"/>
      <text x="${xOf(hz)}" y="45" fill="#8b94a1" font-size="10" text-anchor="middle">${t}</text>`).join("");
  const bands = S.presets.filter((b) => b.id !== "all").map((b) => {
    const inside = b.lo_hz >= a.freq_min_hz && b.hi_hz <= a.freq_max_hz;
    return `<rect x="${xOf(b.lo_hz)}" y="10" width="${Math.max(2, xOf(b.hi_hz) - xOf(b.lo_hz))}" height="8"
      fill="${inside ? "#7fe0a0" : "#4b525c"}"><title>${esc(b.name)}</title></rect>`;
  }).join("");
  return `<svg viewBox="0 0 ${W} 58" xmlns="http://www.w3.org/2000/svg">
    <rect x="${x0}" y="20" width="${x1 - x0}" height="10" rx="2" fill="#262c35"/>
    <rect x="${xOf(a.freq_min_hz)}" y="20" width="${xOf(a.freq_max_hz) - xOf(a.freq_min_hz)}" height="10" rx="2" fill="#4fb3ff"/>
    ${bands}${ticks}
    <text x="${x0}" y="56" fill="#8b94a1" font-size="9.5">blue: rated range · green: presets it covers · grey: presets it does not</text>
  </svg>`;
}
function showAntennaCard(a, ev) {
  const card = $("ant-card"), info = ANT_INFO[antennaKind(a)];
  const mid = Math.sqrt(a.freq_min_hz * a.freq_max_hz), qw = 75 / (mid / 1e6);
  card.innerHTML = `<div class="pic">${antennaSvg(a)}</div>
    <h3>${esc(a.name)}</h3>
    <dl>
      <dt>Rated range</dt><dd>${fmtHz(a.freq_min_hz, 0)} – ${fmtHz(a.freq_max_hz, 1)}</dd>
      <dt>Claimed gain</dt><dd>${a.gain_dbi == null ? "–" : a.gain_dbi + " dBi"}</dd>
      <dt>Expect</dt><dd>${esc(info.expect)}</dd>
      <dt>Mount</dt><dd>${esc(info.mount)}</dd>
      <dt>Quarter wave</dt><dd>${qw >= 1 ? qw.toFixed(2) + " m" : (qw * 100).toFixed(1) + " cm"} at ${fmtHz(mid, 0)} (mid-range)</dd>
      <dt>Use for</dt><dd>${esc(a.uses || "–")}</dd>
    </dl>
    <div class="cover">${coverageSvg(a)}</div>`;
  card.classList.remove("hidden");
  const w = card.offsetWidth, h = card.offsetHeight;
  let x = ev.clientX + 18, y = ev.clientY + 12;
  if (x + w > window.innerWidth - 8) x = ev.clientX - w - 18;
  if (y + h > window.innerHeight - 8) y = window.innerHeight - h - 8;
  card.style.left = x + "px"; card.style.top = y + "px";
}
$("ant-table").addEventListener("mousemove", (e) => {
  const row = e.target.closest("tr[data-ant]");
  const a = row && S.antennas.find((x) => x.id === row.dataset.ant);
  if (a) showAntennaCard(a, e); else $("ant-card").classList.add("hidden");
});
$("ant-table").addEventListener("mouseleave", () => $("ant-card").classList.add("hidden"));

async function openRecording(id) {
  setStatus(await api("/api/session", { method: "POST", body: { recording_id: id } }));
  S.frame = null; S.overview = null; S.pending = null; S.hold = null;
  showView("analysis");
  loadOverview().catch((e) => console.error(e));
}
$("btn-back").onclick = $("home-link").onclick = async (e) => {
  e.preventDefault();
  if (S.status?.active) setStatus(await api("/api/session", { method: "DELETE" }));
  showView("home"); loadHome();
};

async function loadHome() {
  const [recs, ants, st] = await Promise.all([api("/api/recordings"), api("/api/antennas"), api("/api/status")]);
  $("rec-empty").classList.toggle("hidden", recs.length > 0);
  $("rec-table").querySelector("tbody").innerHTML = recs.map((r) => `
    <tr><td>${esc(r.name)}<div class="muted small">${esc(r.description)}</div></td>
    <td>${fmtHz(r.center_freq_hz)}</td><td>${fmtRate(r.sample_rate_hz)}</td><td>${fmtDur(r.duration_s)}</td>
    <td>${esc(r.datatype)}</td><td>${r.comparable ? "comparable" : '<span class="muted">not comparable</span>'}</td>
    <td><button data-open="${r.id}">Analyze</button></td></tr>`).join("");
  $("ant-table").querySelector("tbody").innerHTML = ants.map((a) => `
    <tr data-ant="${esc(a.id)}"><td class="ant-thumb">${antennaSvg(a)}</td><td>${esc(a.name)}</td><td>${fmtHz(a.freq_min_hz, 1)} – ${fmtHz(a.freq_max_hz, 1)}</td>
    <td>${a.gain_dbi == null ? "–" : a.gain_dbi + " dBi"}</td><td>${esc(a.uses)}</td><td class="muted">${esc(a.notes)}</td></tr>`).join("");
  S.antennas = ants;
  const antOpts = ants.map((a) => `<option value="${esc(a.id)}">${esc(a.name)}</option>`).join("");
  $("import-antenna").innerHTML = '<option value="">unknown</option>' + antOpts;
  $("sel-antenna").innerHTML = '<option value="">none selected</option>' + antOpts;
  $("live-antenna").innerHTML = '<option value="">none selected</option>' + antOpts;
  $("scan-antenna").innerHTML = '<option value="">none selected</option>' + antOpts;
  $("scan-ant").innerHTML = '<option value="">none selected</option>' + antOpts;
  if (!$("scan-form").f_lo_mhz.value) { $("scan-form").preset.value = "fm"; applyPresetToForm("fm"); }
  adviseAntenna();
  checkDevice().catch(() => { $("dev-status").textContent = "Could not check for a HackRF."; });
  $("sess-table").querySelector("tbody").innerHTML = st.recent_sessions.map((s) => `
    <tr><td>${esc(s.started_at.slice(0, 19).replace("T", " "))}</td><td>${esc(s.mode)}</td>
    <td>${esc(s.recording_name || "")}</td><td>${s.ended_at ? esc(s.ended_at.slice(11, 19)) : "active"}</td></tr>`).join("");
  setStatus(st.session);
}
$("rec-table").onclick = (e) => {
  const id = e.target.dataset?.open;
  if (id) openRecording(parseInt(id)).catch((err) => alert(err.message));
};
$("btn-synth").onclick = async (e) => {
  e.target.disabled = true; e.target.textContent = "Generating…";
  try { await api("/api/recordings/synthetic", { method: "POST" }); await loadHome(); }
  catch (err) { alert(err.message); }
  finally { e.target.disabled = false; e.target.textContent = "Make synthetic test scene"; }
};
$("import-form").onsubmit = async (e) => {
  e.preventDefault();
  const f = new FormData(e.target), num = (k) => f.get(k) === "" ? null : parseFloat(f.get(k));
  const body = {
    path: f.get("path"), datatype: f.get("datatype"), description: f.get("description"),
    center_freq_hz: num("fc_mhz") == null ? null : num("fc_mhz") * 1e6,
    sample_rate_hz: num("fs_msps") == null ? null : num("fs_msps") * 1e6,
    antenna_id: f.get("antenna_id") || null,
    amp_on: f.get("amp") === "" ? null : f.get("amp") === "1",
    lna_gain_db: num("lna"), vga_gain_db: num("vga"),
  };
  try { await api("/api/recordings/import", { method: "POST", body }); $("import-msg").textContent = "Imported."; loadHome(); }
  catch (err) { $("import-msg").textContent = err.message; }
};

// ---------- start ----------
(async () => {
  connectWs();
  await loadOptions();
  await loadHome();
  if (S.status?.active) {
    showView("analysis");
    if (S.status.mode === "recording") loadOverview().catch(() => {});
  }
})();
