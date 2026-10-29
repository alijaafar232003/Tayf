# Tayf — Project guide

**Name:** Tayf (Arabic طيف, "spectrum"). Formerly SpectraIQ; use Tayf everywhere (repo, package, UI).  
**Date:** 2026-10-27  
**Status:** Planning finished; decisions confirmed on 2026-10-27. Phase 1 starting.  
**Main reference (read first):** "Tayf — Product Vision and Architecture" — https://claude.ai/artifact/SLAmTaA6BS7jc5QjTDYoDu  
(Read it with the Claude Docs connector, not a web fetch.)

---

## 1. Who the user is and how to work

- A Wireless Communications student, building Tayf as a 3-week course project that they genuinely care about.
- Goal: a **practical product** they fully understand, not a research project (a paper would be a bonus).
- They care about **concepts and architecture** (wireless, DSP, ML), not code details.
- Working loop: **briefly introduce the idea → compare practical alternatives → recommend → discuss → implement → test.** Short explanations; no long derivations unless asked.
- They like **small, editable Jupyter notebooks** (opened in VS Code) where they can change scenarios and parameters and see results. Results are generated when run, never pre-computed.
- Each stage pairs a module with a short concept notebook.
- **Agreed:** after every step, give a short technical (not code) summary: the concept, what was built, how it fits the pipeline, how to check it.
- Never claim something works on their machine without verifying it.
- Location: **Lebanon** (ITU Region 1). Build band presets from Region 1 and Lebanese allocations, then confirm with real scans. Don't assume.

## 2. Environment

- Windows 11, Radioconda. Existing conda env: `spectraiq` (`C:\Users\USER\radioconda\envs\spectraiq`).
- HackRF One via SoapySDR. **Quirk:** `SoapySDR.Device()` works; `SoapySDR.Device(dict(driver="hackrf"))` failed earlier.
- **Quirk:** conda DLLs (SoapySDR's `HackRFSupport.dll`, and SciPy's LAPACK: `scipy.linalg` silently kills the process with 0xc06d007f) only load when `<env>\Library\bin` is on PATH (true in an activated env, false when `python.exe` is started directly: pytest, an IDE, the preview). `tayf/__init__.py` adds it on import, so import `tayf` before SciPy.
- **Quirk:** after the HackRF stream is stopped and restarted (e.g. a sample-rate change) following a pause in reading, SoapySDR can replay old buffers endlessly (hundreds of MS/s of \"samples\"). Live mode is not affected (its reader thread never pauses; verified at 2/10/20 MS/s). Measurement code (`tayf/receiver/characterize.py`) reopens the device for a rate change, drains buffered samples until reads are paced in real time, and refuses captures that arrive faster than real time.
- The HackRF is a clone (`hackrf_info`: \"not manufactured by Great Scott Gadgets\"), firmware n_250816, serial ...2a9a8347.
- `hackrf_sweep` is available from the Radioconda prompt.
- An earlier VS Code + Radioconda Jupyter kernel crashed when plotting with Matplotlib. Check notebook plotting early (Plotly or SVG output avoided the crash).
- The old SpectraIQ project in `C:\Users\USER\Desktop\Ali\Projects\Spectral` is **reference only**. Do not copy code from it. Lessons carried over: an I/Q block data model, Welch PSD, the approach for parsing `hackrf_sweep` CSV, and the detector findings below.

## 3. Hardware

| # | Item | Range | Use |
|---|---|---|---|
| — | HackRF One (no PortaPack screen) | 1 MHz–6 GHz, ≤20 MS/s, 8-bit | Receive only |
| 4 | 5 dBi magnetic whip | 40–860 MHz | FM, airband, 433 MHz, DVB-T |
| 6 | Telescopic | 40 MHz–6 GHz | Fallback; quarter-wave = 75/f(MHz) m |
| 7 | "12 dBi" magnetic | 700–2700 MHz | Cellular downlinks, 868 MHz |
| 8 | "35 dBi" magnetic | 700–2700 MHz | Same; 35 dBi claim is implausible, expect a few dBi |
| 5 | 8 dBi rubber duck | 2.4/5/5.8 GHz | Wi-Fi, BLE, ESP32 |
| 1 | 20 dB LNA | 50 MHz–6 GHz | Weak signals; overload risk; **check its power method before enabling HackRF antenna bias power** |
| — | ESP32 | — | Known BLE (GFSK) and Wi-Fi (OFDM; maybe 802.11b DBPSK/DQPSK) transmitter for real test data |

No AM medium-wave coverage, so the AM class is tested on airband. A 50 Ω SMA terminator is recommended for the characterization (not yet confirmed).

## 4. Decisions

| # | Decision | Status |
|---|---|---|
| D1 | 3 weeks; stages 1–4 are the course MVP | **Decided** |
| D2 | Browser dashboard: FastAPI backend (REST + WebSocket) + web frontend | **Decided** |
| D4 | Antennas as above | **Decided** |
| D3 | Class tiers: real-tested WBFM, AM, OFDM, GFSK/FSK, OOK/ASK, Noise (+NBFM if available); synthetic-tested BPSK, QPSK, 16-QAM, GMSK; always Unknown | **Decided** |
| D5 | Lebanon presets from ITU Region 1 + local allocations, verified by scan | **Decided** |
| D6 | User buys a 50 Ω SMA male load (DC–6 GHz). No signal generator or second SDR available. Real test transmitters: ESP32s (BLE, Wi-Fi), a car key (likely 433.92 MHz OOK/FSK; receive only, never replay), broadcast FM, airband AM. Until the terminator arrives, characterize with the antenna removed | **Decided** |
| D7 | Auto-save a light summary of every candidate; full observation + I/Q only when investigated | **Decided** |
| D8 | No comparison notebook. Detector = **FCME**; flatten the noise floor first (receiver profile) and exclude the DC bin, since FCME assumes a flat floor | **Decided** |

## 5. Architecture (summary; the doc has the diagram)

- **Three sources, one pipeline:** recording file (I/Q + SigMF), HackRF live I/Q (SoapySDR), HackRF sweep (`hackrf_sweep`, power only).
- **Session manager:** one active mode (recording, live or scan); owns the HackRF.
- **Receiver services:** gain control (amp/LNA/VGA), Receiver Check, antenna registry, receiver profile, health (clipping, dropped samples).
- **Spectrum + detector → tracker** (stable IDs, candidate table, boxes on display).
- **Investigator** (user clicks a signal): gets I/Q from the recording, the live span, or a retune; channelizes; measures edges, bandwidth, SNR and frequency offset; shows detailed views; calls the **classifier** (features + Random Forest → class, confidence, or Unknown).
- **Observation store:** SQLite (sessions, observations, signal profiles) + SigMF I/Q files.
- **History + analytics:** new / known / changed matching, occupancy, recording comparison.
- **Offline dataset + training** in notebooks: generator, public datasets, labelled recordings. Split **by recording**, not by snippet.
- Two classification layers: an ML modulation classifier, plus rule-based signal-family hypotheses later (always labelled as hypotheses).
- "Calibration" means four separate things: one-time receiver characterization (50 Ω), the in-app Receiver Check, always-on health monitoring, and future dBm calibration.

## 6. Roadmap (21 days)

1. **Foundations (days 1–3):** data model + SigMF, recording import, spectrum + waterfall, session manager, API + dashboard shell, antenna registry. Notebook: processing gain. **Gate:** a recording is analysed in the dashboard.
2. **Live + receiver (days 4–8):** HackRF live + sweep, gain controls, health, Receiver Check, Lebanon presets. Notebook: receiver characterization (50 Ω, with/without LNA). **Gate:** live view with gain and Receiver Check.
3. **Detect + investigate (days 9–14):** FCME detector (concept notebook, no comparison) + tracker, boxes + candidate table, investigator + views, observation store, history + "first seen". **Gate:** click a signal and it is measured and saved.
4. **Classify + deliver (days 15–21):** signal generator, ESP32 + FM recordings, features + Random Forest, Unknown threshold, real-vs-synthetic evaluation, report + demo. Notebook: features. **Gate:** classifier tested on real signals; demo ready.

5. **Listen + decode (final phase, user request 2026-10-28):** demodulate the investigated signal to audio in the browser (WBFM with 50 µs de-emphasis for Region 1, NBFM, AM for airband; maybe SSB); decoders by class: FM RDS (station name), OOK/FSK bit slicer for "random" signals (pulse widths, bit rate guess, raw bits; car keys receive-only, never replay), possibly ADS-B at 1090 MHz. Only decode what is lawful and not private. Audio demod reuses the Phase 3 channelizer, so plain FM/AM listening may land earlier if time allows.

**After the course:** auto-classify inside the live span, occupancy analytics, recording comparison, signal-family hypotheses, automatic investigation, anomaly detection, neural-network classifier, dBm calibration. The database schema plans for these from day 1.

## 7. Evidence from the old detector code (synthetic, 1 MHz span, FFT 2048)

- Detector ranking was driven by **threshold choice, not algorithm**. Detection cliff (whole-band SNR): Median 12 dB → −14 dB; CA-CFAR 8 dB → about −18 dB; FCME 9 dB → −22 dB; CA-CFAR 3 dB → −26 dB with under 1% false alarms.
- The "weak tone" was not weak: FFT processing gain is about 33 dB.
- Median and CA-CFAR **self-mask on wideband signals** (noise estimate −46.7 vs true −60.1 dB); FCME handles this but assumes a flat floor.

## 8. How to run

- Env: `conda activate tayf` (a clone of `spectraiq`, plus fastapi, uvicorn, scikit-learn, plotly, sigmf, pytest). The package is installed editable (`pip install -e .`).
- Dashboard: `python -m tayf serve`, then open http://127.0.0.1:8000. Data (database + recordings) lives in `data/` (git-ignored).
- Synthetic test scene: `python -m tayf synth` (or the button on the home page).
- Raw HackRF file: `python -m tayf import-raw capture.bin --fc 98e6 --fs 2e6 --antenna whip-5dbi --amp 0 --lna 16 --vga 20`
- Tests: `pytest`.
- Layout: `tayf/core` (data model, SigMF), `sources`, `dsp`, `session`, `store` (SQLite), `api` (FastAPI); `web/` (plain HTML/JS dashboard); `notebooks/`.

## 9. Progress

- 2026-10-27: repo created, decisions confirmed. Phase 1 recording path built: data model, SigMF I/O (incl. raw hackrf_transfer import), synthetic scene, Welch spectrum, SQLite store + antenna registry, session manager, FastAPI + WebSocket, dashboard (spectrum, waterfall, timeline), notebook 01. 8 tests pass; dashboard checked on the synthetic scene. Open: notebook plotting check in VS Code.
- 2026-10-27: first real FM recording (`fm.bin`, 98.0 MHz, 2 MS/s, amp off, LNA 16, VGA 20, whip) opens and plays in the dashboard (Phase 1 gate met). Findings: no clipping, but the ADC only used values −3…+3 (gain ~20–25 dB too low; floor is quantization-limited). The "peak" at 98.0 MHz was the HackRF DC spike (drops 37 dB after DC removal), not a station. Stations seen near 97.1, 98.5, 98.9 MHz. Lessons for Phase 2: tune off-center (offset tuning) and remove DC; the Receiver Check should flag a too-low ADC level as well as clipping.
- 2026-10-28: Phase 2 steps 1–3 built (and part of 4): live source (SoapySDR HackRF + a simulated HackRF calibrated to the first FM recording), reader thread + 0.5 s ring buffer, offset tuning (LO = tuned + max(100 kHz, 10 % of fs)), DC removal toggle (per-block mean, processing only; recordings stay raw), gain controls (amp / LNA 8 dB steps / VGA 2 dB steps), sample-rate change, SDR++-style tuning (drag, wheel, type), antenna range warning, health readouts (ADC level, clipping %, overflows). 14 tests pass; dashboard checked with the simulated radio.
- 2026-10-28: live mode tested on the real HackRF (whip, FM band). Opens in 0.09 s; retune → first frame ~60 ms; no overflows at 2, 10 or 20 MS/s; ~20 frames/s at 20 MS/s (after capping live reads at 262k samples). DC spike +35.5 dB above floor → +2.6 dB with removal. Gain: +26 dB (LNA 16/VGA 20 → 32/30) lifted 97.1 MHz from +5 to +21 dB above floor (low gain was quantization-limited); clipping starts around 80 dB total gain (amp off). Good FM-band setting with the whip: LNA 32, VGA 30, amp off (≈ −22 dBFS). Stations now: 97.1, 97.9, 98.9, 99.1, 99.3 MHz; spectrum verified not mirrored (stations stay put when the LO moves). The 98.5 MHz signal in `fm.bin` (2026-10-27) was absent today at the same LO and gain: likely not on air all the time. 97.9 MHz reads stronger when far from the LO: check image rejection / IQ imbalance in the characterization notebook.
- 2026-10-28: user's car key seen near 314.36 / 314.52 MHz (315 MHz band, not 433.92), bursts with two peaks of similar strength: probably 2-FSK, but the pair is nearly symmetric around the LO, so an image test (retune 100 kHz, peaks should stay put) is still pending. Tuning UI reworked SDR++-style: the view follows the mouse instantly (predicted window while a retune is pending), the waterfall slides sideways on retune and keeps its history, click-to-tune, wheel notches accumulate (Shift = 10x finer), retunes rate-limited to one per 100 ms.
- 2026-10-28: record from live built (Phase 2 step 5): Record button saves raw ci8 SigMF in `data/recordings/live_<UTC time>_<kHz>kHz`; a retune or gain change starts a new SigMF capture segment (playback reads each segment's frequency); a sample-rate change stops the recording; disk writes run in their own thread (lost samples counted if the disk falls > 2 s behind). 15 tests pass; checked in the browser with the simulated radio. On the real HackRF: 5 s at 10 MS/s with a mid-recording retune → 0 overflows, 0 lost samples, 2 correct segments, stations at the same absolute frequencies in both; dashboard record → stop → play back works (test file `live_20261028_013454_99100kHz` left in data/recordings).
- 2026-10-29: band scan built (Phase 2 step 6) with Lebanon presets: `hackrf_sweep -B` (binary) runs as its own process; segments are stitched into whole sweeps on a fixed grid; the display shows the mean of all sweeps since the last frame; peak hold and band overlays work in every mode. Click a signal (snaps to the nearby peak) → "Open live here"; drag a range → "Scan this range" (finer bins); "Back to scan" from live. Antenna table has drawings + a hover card (rated range on a log axis, which presets it covers, quarter-wave length). 20 tests pass. On the real HackRF (whip): FM scan 87-109 MHz at ~210 sweeps/s; scan → live → back to scan hands the HackRF over cleanly (0 overflows). **Gain finding:** at LNA 32 / VGA 30 the FM band overloads (floor rose 13 dB for 8 dB more gain; live at 95.3 MHz clipped 4 %), so the FM preset is now LNA 24 / VGA 30 (live level −17 dBFS, no clipping). Preset check (3 s per band, whip): seen = FM (~30 stations), GSM 900 (carriers 925-938 MHz), 2.4 GHz (busy near 2450), 5 GHz (5180-5250). Hints only: LTE 800 range ~18 dB above neighbouring floor (probably LTE blocks), steady carriers at 120.00 / 122.47 / 160.03 / 541.35 MHz (120 and 160 may be receiver spurs: check in notebook 02), peaks at 1.5 / 14.5 / 19.5 MHz (likely spurs). 433, 868, 315, ADS-B: nothing in 3 s (bursty or wrong antenna).
- 2026-10-29: notebook 02 (receiver characterization) + `tayf/receiver` (characterize.py, profile.py) built and run on the HackRF with the **antenna removed** (preview; rerun with the 50 Ω load, CONDITION = "50 ohm load"). Profile in `data/receiver/profile.json`. Results (repeatable over 3 runs): (1) Floor = three sources: ADC floor −121 dBFS/Hz (≈ 8-bit quantization theory), VGA-stage noise −144.4 dBFS/Hz at 0 dB (same at all frequencies: baseband), front-end noise −169 (100 MHz) … −178 (2450 MHz) dBFS/Hz at 0 dB. Gain rule: LNA ≥ ~25–34 dB for the front end to set sensitivity, then VGA ≥ ~16–21 dB (at LNA 32) to clear the ADC floor; noise alone clips at LNA 32/40 + VGA 62. Fit error < 1 dB. (2) Span shape: DC spike +14…18 dB → +1.3…2.1 dB after removal; 2 and 4 MS/s floor rises +2 dB at the edges (filter can't go below 1.75 MHz → noise aliases in); 8, 16, 20 MS/s drop −1.5 dB at ±0.45 fs (flat ±1 dB over 86 %); 10 and 12.5 MS/s flat over 99 %. (3) Spur map (hackrf_sweep, 1–6000 MHz, 100 kHz, 10 s): 197 peaks ≥ 6 dB, most of them hackrf_sweep artifacts. (4) Carrier check in live I/Q: **made by the HackRF at fixed frequencies: 14, 40, 80, 120, 160, 200, 2200 MHz** (18–46 dB above floor, multiples of 40 MHz); 3029 and 5162 MHz appear only at some LO settings (tuning-dependent spurs); **122.47, 160.03, 541.35 MHz vanish with no antenna → real outside signals**. Presets' hints updated. Image rejection (whip, stations 95.3 and 94.7 MHz, offsets 0.3–3 MHz, 10 MS/s, LNA 24/VGA 20; twice): no image seen anywhere; where the mirror falls on empty spectrum IRR ≥ ~34–38 dB, which is the limit a station ~30 dB above the floor can show (a stronger test signal, e.g. an ESP32 next to the antenna, could measure it). Station level is flat within ~1 dB from 0.3 to 3.7 MHz off the LO, so the 97.9 MHz "stronger far from the LO" effect is neither image nor gain ripple; likely explanation (unconfirmed): at 2 MS/s the 1.75 MHz minimum filter lets stations just outside the span alias in near the edges. 22 tests pass.
