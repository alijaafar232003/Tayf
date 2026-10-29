"""Band presets for Lebanon (ITU Region 1).

Each band starts from the Region 1 allocation table (and what is usual in the
region). `seen` records what a real scan in Lebanon confirmed; until then a
band is only "expected". Scan ranges are a little wider than the band so its
edges are visible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Band:
    id: str
    name: str
    lo_hz: float                    # the allocation
    hi_hz: float
    scan_lo_hz: float               # what the preset scans
    scan_hi_hz: float
    bin_hz: float
    antenna_id: str
    expect: str                     # what should be there
    seen: str = ""                  # confirmed by a real scan (date: what was seen); empty = not yet
    hint: str = ""                  # something a scan showed that is not yet a confirmation
    lna_db: int = 32                # starting point; only the FM band has been tuned on real signals
    vga_db: int = 30

    def to_dict(self) -> dict:
        return {**asdict(self), "verified": bool(self.seen)}


MHZ = 1e6
BANDS = [
    Band("fm", "FM broadcast", 87.5 * MHZ, 108 * MHZ, 87 * MHZ, 109 * MHZ, 25e3, "whip-5dbi",
         "WBFM stations, 100 kHz raster",
         seen="2026-10-29 scan: about 30 stations across 87.9-107.9 MHz, the strongest ~45 dB above the floor",
         lna_db=24),                 # LNA 32 overloads here: the floor rose 13 dB for 8 dB of gain
    Band("air", "Airband (AM voice)", 118 * MHZ, 137 * MHZ, 117 * MHZ, 138 * MHZ, 10e3, "whip-5dbi",
         "AM voice, 25 / 8.33 kHz channels; ATIS from Beirut airport",
         hint="2026-10-29: steady carrier at 122.47 MHz (real: gone with no antenna). "
              "120.000 MHz is the HackRF's own clock harmonic (notebook 02), not a station"),
    Band("2m", "2 m amateur + marine VHF", 144 * MHZ, 174 * MHZ, 143 * MHZ, 175 * MHZ, 10e3, "whip-5dbi",
         "NBFM: amateur 144-146, marine 156-162, land mobile",
         hint="2026-10-29: steady carrier at 160.03 MHz (real: gone with no antenna); "
              "160.000 MHz next to it is the HackRF's own clock harmonic"),
    Band("car315", "310-320 MHz (car keys seen here)", 312 * MHZ, 318 * MHZ, 310 * MHZ, 320 * MHZ, 10e3,
         "whip-5dbi", "Short bursts from key fobs (not a Region 1 SRD band, but imported cars use it)",
         seen="2026-10-28: car key bursts near 314.36 / 314.52 MHz"),
    Band("ism433", "433 MHz ISM / SRD", 433.05 * MHZ, 434.79 * MHZ, 430 * MHZ, 440 * MHZ, 10e3, "whip-5dbi",
         "OOK/FSK bursts: remotes, sensors, doorbells"),
    Band("dvbt", "UHF TV (DVB-T)", 470 * MHZ, 694 * MHZ, 470 * MHZ, 694 * MHZ, 100e3, "whip-5dbi",
         "8 MHz-wide flat OFDM blocks, if terrestrial TV is on air",
         hint="2026-10-29: a narrow carrier at 541.35 MHz (real: gone with no antenna), no flat 8 MHz blocks seen"),
    Band("lte800", "LTE 800 downlink", 791 * MHZ, 821 * MHZ, 788 * MHZ, 824 * MHZ, 100e3, "mag-12dbi",
         "LTE blocks (band 20), if the operators use it",
         hint="2026-10-29 (whip): the whole range sat ~18 dB above the 868 MHz floor at the same gain: "
              "probably wide LTE blocks filling it; check with the 700-2700 MHz antenna"),
    Band("srd868", "868 MHz SRD", 863 * MHZ, 870 * MHZ, 860 * MHZ, 872 * MHZ, 10e3, "mag-12dbi",
         "LoRa, sensors, alarms (short bursts)"),
    Band("gsm900", "GSM / UMTS / LTE 900 downlink", 925 * MHZ, 960 * MHZ, 922 * MHZ, 962 * MHZ, 50e3,
         "mag-12dbi", "200 kHz GSM carriers and 5-10 MHz UMTS/LTE blocks",
         seen="2026-10-29 (whip): a quarter of the band occupied, strongest carriers 925-938 MHz, ~22 dB"),
    Band("adsb", "ADS-B 1090 MHz", 1089 * MHZ, 1091 * MHZ, 1080 * MHZ, 1100 * MHZ, 25e3, "mag-12dbi",
         "Aircraft transponder pulses (Phase 5 decoder candidate)"),
    Band("dcs1800", "GSM / LTE 1800 downlink", 1805 * MHZ, 1880 * MHZ, 1800 * MHZ, 1885 * MHZ, 100e3,
         "mag-12dbi", "LTE band 3 blocks, GSM 1800 carriers"),
    Band("umts2100", "UMTS / LTE 2100 downlink", 2110 * MHZ, 2170 * MHZ, 2105 * MHZ, 2175 * MHZ, 100e3,
         "mag-12dbi", "5 MHz UMTS carriers, LTE band 1"),
    Band("ism24", "2.4 GHz ISM (Wi-Fi, BLE)", 2400 * MHZ, 2483.5 * MHZ, 2395 * MHZ, 2490 * MHZ, 100e3,
         "duck-8dbi", "Bursty 20 MHz Wi-Fi, 2 MHz BLE hops; your ESP32s",
         seen="2026-10-29 (whip): busy around 2450 MHz (a Wi-Fi channel 8-9 network), up to 39 dB"),
    Band("lte2600", "LTE 2600 downlink", 2620 * MHZ, 2690 * MHZ, 2615 * MHZ, 2695 * MHZ, 100e3, "mag-12dbi",
         "LTE band 7 blocks, if used"),
    Band("wifi5", "5 GHz Wi-Fi", 5150 * MHZ, 5875 * MHZ, 5150 * MHZ, 5875 * MHZ, 250e3, "duck-8dbi",
         "Bursty 20-80 MHz Wi-Fi channels",
         seen="2026-10-29 (whip): activity at 5180-5250 MHz (channels 36-48), ~27 dB"),
    Band("all", "Everything (1 MHz - 6 GHz)", 1 * MHZ, 6000 * MHZ, 1 * MHZ, 6000 * MHZ, 1e6, "telescopic",
         "A whole-range overview; about one pass per second", lna_db=24, vga_db=30,
         hint="Receiver spurs (notebook 02, nothing connected): carriers at 14, 40, 80, 120, 160, 200 and "
              "2200 MHz are made by the HackRF; many more peaks appear only in scans (hackrf_sweep artifacts)"),
]

BY_ID = {b.id: b for b in BANDS}
