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
  drag: null,           // live drag-to-tune state
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
    out[c] = m;
  }
  return out;
}
function median(arr) {
  const s = Float32Array.from(arr).sort();
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
  ctx.textAlign = "center"; ctx.fillText("dBFS / bin", 0, 0); ctx.restore();
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

  // trace
  const cols = Math.max(1, Math.floor(b.w));
  const p = resample(f, fLo, fHi, cols);
  ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
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
  const live = st.mode === "live";
  $("analysis").classList.toggle("mode-live", live);
  $("analysis").classList.toggle("mode-recording", !live);
  for (const id of ["cv-spectrum", "cv-waterfall"]) $(id).classList.toggle("tunable", live);
  $("btn-play").textContent = live ? (st.playing ? "Freeze" : "Run") : (st.playing ? "Pause" : "Play");
  $("sel-nfft").value = String(st.nfft);
  $("chk-dc").checked = st.dc_remove;
  showHealth(st);
  $("an-title").textContent = st.name;
  if (live) {
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
  setPending(S.drag.start - dx * S.drag.hzPerPx);
  sendPending();
});
window.addEventListener("mouseup", (e) => {
  if (!S.drag) return;
  const d = S.drag;
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

// ---------- views ----------
function showView(name) {
  $("home").classList.toggle("hidden", name !== "home");
  $("analysis").classList.toggle("hidden", name !== "analysis");
  if (name === "analysis") { S.auto.init = false; resetWaterfall(); scheduleDraw(); }
}
async function openLive(body) {
  setStatus(await api("/api/session/live", { method: "POST", body }));
  S.frame = null; S.overview = null; S.pending = null;
  showView("analysis");
}
async function loadOptions() {
  const o = await api("/api/radio/options");
  const opts = (vals, fmt, sel) => vals.map((v) => `<option value="${v}"${v === sel ? " selected" : ""}>${fmt(v)}</option>`).join("");
  for (const el of document.querySelectorAll(".sel-rate")) el.innerHTML = opts(o.sample_rates, fmtRate, 2e6);
  for (const el of document.querySelectorAll(".sel-lna")) el.innerHTML = opts(o.lna_steps, String, 16);
  document.querySelector('#live-form [name=vga_db]').innerHTML = opts(o.vga_steps, String, 20);
}
async function checkDevice() {
  const d = await api("/api/devices");
  const el = $("dev-status");
  if (d.in_use) el.textContent = `${d.radio} is in use by the current session.`;
  else if (d.hackrf.length) el.innerHTML = `<span class="ok">HackRF found</span> (serial ${esc(d.hackrf[0].serial || "?")}).`;
  else el.innerHTML = `<span class="warn">No HackRF found.</span> Plug it in and reload, or use the simulated radio.`;
  if (!d.hackrf.length && !d.in_use) document.querySelector("#live-form [name=radio]").value = "sim";
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
async function openRecording(id) {
  setStatus(await api("/api/session", { method: "POST", body: { recording_id: id } }));
  S.frame = null; S.overview = null; S.pending = null;
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
    <tr><td>${esc(a.name)}</td><td>${fmtHz(a.freq_min_hz, 1)} – ${fmtHz(a.freq_max_hz, 1)}</td>
    <td>${a.gain_dbi == null ? "–" : a.gain_dbi + " dBi"}</td><td>${esc(a.uses)}</td><td class="muted">${esc(a.notes)}</td></tr>`).join("");
  S.antennas = ants;
  const antOpts = ants.map((a) => `<option value="${esc(a.id)}">${esc(a.name)}</option>`).join("");
  $("import-antenna").innerHTML = '<option value="">unknown</option>' + antOpts;
  $("sel-antenna").innerHTML = '<option value="">none selected</option>' + antOpts;
  $("live-antenna").innerHTML = '<option value="">none selected</option>' + antOpts;
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
