-- Tayf database schema. Tables for later phases are created now (empty) so
-- adding those features later is new code, not a schema redesign.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS antennas (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    freq_min_hz  REAL NOT NULL,
    freq_max_hz  REAL NOT NULL,
    gain_dbi     REAL,               -- as claimed by the seller
    uses         TEXT,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS recordings (
    id              INTEGER PRIMARY KEY,
    meta_path       TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    center_freq_hz  REAL NOT NULL,
    sample_rate_hz  REAL NOT NULL,
    datatype        TEXT NOT NULL,
    n_samples       INTEGER NOT NULL,
    duration_s      REAL NOT NULL,
    start_time      TEXT NOT NULL,
    antenna_id      TEXT REFERENCES antennas(id),
    comparable      INTEGER NOT NULL,   -- gains and antenna known -> usable for history comparison
    description     TEXT,
    added_at        TEXT NOT NULL
);

-- One run of the app in one mode. Exactly one is active at a time.
CREATE TABLE IF NOT EXISTS sessions (
    id             INTEGER PRIMARY KEY,
    mode           TEXT NOT NULL CHECK (mode IN ('recording', 'live', 'scan')),
    started_at     TEXT NOT NULL,
    ended_at       TEXT,
    recording_id   INTEGER REFERENCES recordings(id),
    antenna_id     TEXT REFERENCES antennas(id),
    settings_json  TEXT NOT NULL,       -- ReceiverSettings at session start
    notes          TEXT
);

-- Phase 3: light summary of every tracked candidate, written at session end (D7).
CREATE TABLE IF NOT EXISTS candidate_summaries (
    id             INTEGER PRIMARY KEY,
    session_id     INTEGER NOT NULL REFERENCES sessions(id),
    track_id       INTEGER NOT NULL,
    center_freq_hz REAL NOT NULL,
    bandwidth_hz   REAL NOT NULL,
    first_seen     TEXT NOT NULL,
    last_seen      TEXT NOT NULL,
    duty_cycle     REAL,
    peak_snr_db    REAL,
    mean_snr_db    REAL
);

-- Phase 3-4: an investigated, measured (and later classified) signal.
CREATE TABLE IF NOT EXISTS signal_profiles (
    id                INTEGER PRIMARY KEY,
    center_freq_hz    REAL NOT NULL,
    bandwidth_hz      REAL NOT NULL,
    class_label       TEXT,
    first_seen        TEXT NOT NULL,
    last_seen         TEXT NOT NULL,
    n_observations    INTEGER NOT NULL DEFAULT 0,
    typical_snr_db    REAL,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id                INTEGER PRIMARY KEY,
    session_id        INTEGER NOT NULL REFERENCES sessions(id),
    profile_id        INTEGER REFERENCES signal_profiles(id),
    time              TEXT NOT NULL,
    center_freq_hz    REAL NOT NULL,
    bandwidth_hz      REAL NOT NULL,
    freq_offset_hz    REAL,
    snr_db            REAL,
    power_dbfs        REAL,
    noise_floor_dbfs  REAL,
    class_label       TEXT,
    class_confidence  REAL,
    model_version     TEXT,
    history_status    TEXT CHECK (history_status IN ('new', 'known', 'changed')),
    antenna_id        TEXT REFERENCES antennas(id),
    settings_json     TEXT NOT NULL,
    health_json       TEXT,              -- clipping %, dropped samples at capture time
    iq_meta_path      TEXT               -- SigMF snippet, when saved
);

-- Later: occupancy per band and hour.
CREATE TABLE IF NOT EXISTS occupancy (
    id            INTEGER PRIMARY KEY,
    session_id    INTEGER REFERENCES sessions(id),
    hour_start    TEXT NOT NULL,
    freq_lo_hz    REAL NOT NULL,
    freq_hi_hz    REAL NOT NULL,
    occupancy     REAL NOT NULL          -- fraction of time above the detection threshold
);

CREATE INDEX IF NOT EXISTS idx_obs_freq ON observations(center_freq_hz);
CREATE INDEX IF NOT EXISTS idx_profile_freq ON signal_profiles(center_freq_hz);
