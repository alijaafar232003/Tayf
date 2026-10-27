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
// Max-hold reduce bins to `cols` pixels so narrow signals stay visible.
function reduceTo(power, cols) {
  const out = new Float32Array(cols), n = power.length;
  for (let c = 0; c < cols; c++) {
    const a = Math.floor(c * n / cols), b = Math.max(a + 1, Math.floor((c + 1) * n / cols));
    let m = -Infinity;
    for (let i = a; i < b; i++) if (power[i] > m) m = power[i];
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
  const [lo, hi] = dbRange(), [fLo, fHi] = spanOf(f);
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
  const p = reduceTo(f.power, cols);
  ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
  ctx.strokeStyle = css.getPropertyValue("--trace"); ctx.lineWidth = 1.2 * b.r;
  ctx.beginPath();
  for (let c = 0; c < cols; c++) {
    const x = b.x + c * b.w / cols, y = yOf(p[c]);
    c ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();
  if (S.cursorHz != null) {
    ctx.strokeStyle = css.getPropertyValue("--accent"); ctx.lineWidth = 1;
    const x = xOf(S.cursorHz); ctx.beginPath(); ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h); ctx.stroke();
  }
  ctx.restore();
}

// ---------- waterfall ----------
const WF = { canvas: document.createElement("canvas") };
function resetWaterfall() {
  const cv = $("cv-waterfall"); fitCanvas(cv);
  const b = plotBox(cv);
  WF.canvas.width = Math.max(1, Math.floor(b.w));
  WF.canvas.height = Math.max(1, Math.floor(cv.height - b.y - 4 * b.r));
  const c = WF.canvas.getContext("2d"); c.fillStyle = "#000"; c.fillRect(0, 0, WF.canvas.width, WF.canvas.height);
  drawWaterfall();
}
function pushWaterfallRow(frame) {
  const w = WF.canvas.width, h = WF.canvas.height, c = WF.canvas.getContext("2d");
  c.drawImage(WF.canvas, 0, 0, w, h - 1, 0, 1, w, h - 1);     // scroll down by one row
  const [lo, hi] = dbRange(), p = reduceTo(frame.power, w);
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
  ctx.drawImage(WF.canvas, b.x, b.y, b.w, WF.canvas.height);
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
  $("btn-play").textContent = st.playing ? "Pause" : "Play";
  $("sel-speed").value = String(st.speed);
  $("sel-nfft").value = String(st.nfft);
  if (st.session_id !== prevId) {
    $("an-title").textContent = st.name;
    const s = st.settings;
    const gains = s.lna_gain_db == null ? "gains unknown" : `amp ${s.amp_on ? "on" : "off"}, LNA ${s.lna_gain_db} dB, VGA ${s.vga_gain_db} dB`;
    $("an-info").textContent = `${fmtHz(st.center_freq_hz)} · ${fmtRate(st.sample_rate_hz)} · ${fmtDur(st.duration_s)} · ${gains}` +
      (st.comparable ? "" : " · not comparable with history");
  }
}

// ---------- interactions ----------
async function control(body) {
  try { setStatus(await api("/api/session/control", { method: "POST", body })); }
  catch (e) { alert(e.message); }
}
$("btn-play").onclick = () => control({ playing: !S.status?.playing });
$("sel-speed").onchange = (e) => control({ speed: parseFloat(e.target.value) });
$("sel-nfft").onchange = (e) => control({ nfft: parseInt(e.target.value) });
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
    if (!S.frame) return;
    const b = plotBox(cv), x = (e.clientX - cv.getBoundingClientRect().left) * b.r;
    const [fLo, fHi] = spanOf(S.frame), hz = fLo + (fHi - fLo) * (x - b.x) / b.w;
    if (hz < fLo || hz > fHi) return;
    const i = Math.round((hz - S.frame.f0) / S.frame.df);
    S.cursorHz = hz;
    $("ro-cursor").textContent = `${fmtHz(hz, 4)}, ${S.frame.power[Math.max(0, Math.min(S.frame.power.length - 1, i))].toFixed(1)} dBFS`;
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
async function openRecording(id) {
  setStatus(await api("/api/session", { method: "POST", body: { recording_id: id } }));
  S.frame = null; S.overview = null;
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
  $("import-antenna").innerHTML = '<option value="">unknown</option>' + ants.map((a) => `<option value="${esc(a.id)}">${esc(a.name)}</option>`).join("");
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
  await loadHome();
  if (S.status?.active) { showView("analysis"); loadOverview().catch(() => {}); }
})();
