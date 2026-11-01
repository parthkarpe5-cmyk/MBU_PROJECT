"""Feature definitions shared by offline dataset extraction and live inference.

Pure numpy; no OpenCV/MediaPipe import. Every definition here is used by BOTH
``research.extract`` (video -> cached per-frame records -> training windows) and
``dms.learned`` (live webcam -> rolling window -> prediction), so there is a single
source of truth for what "EAR", "valid", "PERCLOS" or "blink" mean.

Terminology
-----------
* observation / record : one processed frame at timestamp ``t`` (seconds).
* valid                : face detected AND the app's own ``healthy`` check passed AND all
                         measurements finite. Everything else is MISSING, never "closed"
                         and never zero.
* window               : half-open interval [t0, t1). A sample belongs to it iff t0 <= t < t1.
* sample duration      : dt_i = min(t_{i+1} - t_i, max_gap_seconds), with the last sample in a
                         window closed by t1. Gaps longer than max_gap therefore count as
                         missing time, not as bridged observations.
* coverage             : sum(dt_i for valid samples) / (t1 - t0).
* denominators         : PERCLOS-style fractions and per-minute rates use VALID time only.

These are research features computed from approximate geometric landmarks. They are weak
proxies; none of them is a validated clinical or automotive measure.
"""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math

import numpy as np

FEATURE_VERSION = "1"
FRAME_FIELDS = ("ear_l", "ear_r", "mar", "yaw", "pitch", "roll")
ROW_FIELDS = ("t", "valid") + FRAME_FIELDS

# Columns reported for every window.
WINDOW_FEATURE_NAMES = (
    "coverage", "valid_time_s",
    "perclos_fixed", "perclos_relative", "ear_open_level",
    "blink_rate_per_min", "blink_duration_mean_s", "long_closure_rate_per_min", "longest_closure_s",
    "ear_mean", "ear_std", "ear_p10", "ear_p50", "ear_velocity",
    "ear_mean_rel", "ear_p10_rel", "ear_std_rel", "ear_velocity_rel",
    "mar_mean", "mar_std", "mar_p90", "mar_open_frac", "mouth_open_events_per_min", "mar_velocity",
    "yaw_std", "pitch_std", "roll_std", "yaw_velocity", "pitch_velocity",
    "head_off_center_frac", "pitch_range_p5_p95",
)
# Tracking-quality columns are reported and used for gating, but NOT given to models:
# tracking failure is not a drowsiness measurement and would be a dataset shortcut.
QUALITY_FEATURES = ("coverage", "valid_time_s")
MODEL_FEATURE_NAMES = tuple(n for n in WINDOW_FEATURE_NAMES if n not in QUALITY_FEATURES)

SEQUENCE_CHANNELS = (
    "ear_mean", "ear_min", "closed_frac_fixed", "mar_mean", "mar_max",
    "yaw_centered", "pitch_centered", "roll_centered", "ear_velocity", "head_velocity",
)


@dataclass(frozen=True)
class FeatureConfig:
    window_seconds: float = 30.0
    stride_seconds: float = 10.0
    max_gap_seconds: float = 0.5
    min_coverage: float = 0.70           # fixed a priori, not tuned on any split
    min_valid_samples: int = 10
    ear_closed_fixed: float = 0.21       # same default as dms.core.Config.ear_threshold
    ear_closed_relative: float = 0.72    # same factor as dms.core.Calibration (eyes * .72)
    open_level_percentile: float = 90.0  # per-window estimate of "eyes open" EAR
    blink_min_seconds: float = 0.05
    blink_max_seconds: float = 0.50      # longer bounded closures are "long closures"
    mar_open: float = 0.55               # same default as dms.core.Config.mar_threshold
    mouth_open_event_seconds: float = 1.5
    yaw_limit: float = 25.0              # same defaults as dms.core.Config
    pitch_limit: float = 20.0
    bin_seconds: float = 1.0
    bin_min_coverage: float = 0.5

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown FeatureConfig keys: {sorted(unknown)}")
        return cls(**data)

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @property
    def n_bins(self):
        return int(round(self.window_seconds / self.bin_seconds))


# --------------------------------------------------------------------------- frame records
def frame_record(data):
    """Convert ``Tracker.process`` output (or None) into one observation record.

    Uses the RAW tracker values (``data['values']``): the same numbers the live monitor
    sees before session calibration offsets are applied. Missing values are NaN.
    """
    nan = float("nan")
    if data is None:
        return dict(detected=False, valid=False, **{k: nan for k in FRAME_FIELDS})
    left, right, mar, yaw, pitch = (float(v) for v in data["values"])
    roll = float(data["roll"])
    values = dict(zip(FRAME_FIELDS, (left, right, mar, yaw, pitch, roll)))
    finite = all(math.isfinite(v) for v in values.values())
    return dict(detected=True, valid=bool(data["healthy"]) and finite, **values)


def row_from_record(t, record):
    return (float(t), bool(record["valid"])) + tuple(float(record[k]) for k in FRAME_FIELDS)


def rows_to_arrays(rows):
    """Sequence of ROW_FIELDS tuples -> dict of aligned arrays."""
    if len(rows) == 0:
        return {"t": np.zeros(0), "valid": np.zeros(0, bool), **{k: np.zeros(0) for k in FRAME_FIELDS}}
    table = np.asarray(rows, dtype=np.float64)
    out = {"t": table[:, 0].copy(), "valid": table[:, 1] > 0.5}
    for i, name in enumerate(FRAME_FIELDS):
        out[name] = table[:, 2 + i].copy()
    return out


# --------------------------------------------------------------------------- window geometry
def iter_window_bounds(t_first, t_end, cfg):
    """Yield (t0, t1) on a fixed grid. Only windows lying fully inside [t_first, t_end]."""
    k = 0
    while True:
        t0 = t_first + k * cfg.stride_seconds
        t1 = t0 + cfg.window_seconds
        if t1 > t_end + 1e-9:
            return
        yield t0, t1
        k += 1


def _slice(arrays, t0, t1):
    t = arrays["t"]
    return np.searchsorted(t, t0, "left"), np.searchsorted(t, t1, "left")


def _durations(tt, t1, max_gap):
    nxt = np.append(tt[1:], t1)
    return np.clip(np.minimum(nxt - tt, max_gap), 0.0, None)


def window_coverage(arrays, t0, t1, cfg):
    """(coverage, valid_time_s, n_valid) for [t0, t1). Cheap; used for live gating."""
    lo, hi = _slice(arrays, t0, t1)
    if hi <= lo:
        return 0.0, 0.0, 0
    tt, valid = arrays["t"][lo:hi], arrays["valid"][lo:hi]
    valid_time = float(_durations(tt, t1, cfg.max_gap_seconds)[valid].sum())
    return valid_time / (t1 - t0), valid_time, int(valid.sum())


def _runs(cond, linked):
    """Maximal runs of True in ``cond`` where consecutive samples are ``linked``.
    ``linked[i]`` joins sample i and i+1. Returns (start_idx, end_idx_inclusive)."""
    prev = np.concatenate(([False], cond[:-1] & linked))
    nxt = np.concatenate((cond[1:] & linked, [False]))
    return np.flatnonzero(cond & ~prev), np.flatnonzero(cond & ~nxt)


def _mean_abs_rate(x, tt, linked):
    gap = np.diff(tt)
    ok = linked & (gap > 0)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(np.diff(x))[ok] / gap[ok]))


def _empty_features(coverage=0.0, valid_time=0.0, n_valid=0):
    out = {name: float("nan") for name in WINDOW_FEATURE_NAMES}
    out["coverage"], out["valid_time_s"], out["n_valid"] = coverage, valid_time, n_valid
    return out


def window_features(arrays, t0, t1, cfg):
    """All window-level features for [t0, t1). Missing observations are excluded, never zero-filled.

    PERCLOS-style:
      perclos_fixed     = time(valid and max(EAR_L, EAR_R) < ear_closed_fixed) / valid_time
      perclos_relative  = same with threshold ear_closed_relative * P90(max(EAR_L, EAR_R) in window)
      "closed" requires BOTH eyes below threshold (matches dms.core.Monitor).
      This is a PERCLOS-style proxy on EAR, not the pupil-coverage PERCLOS (P80) standard.
    Blinks (fixed threshold): a maximal run of closed valid samples with a valid OPEN sample on both
      sides (bounded). Duration = sum of its sample durations. blink_min <= duration <= blink_max is a
      blink; > blink_max is a long closure. Runs touching a gap/window edge are censored: excluded from
      counts and the mean blink duration, but included in PERCLOS and longest_closure_s (lower bound).
    Rates are per minute of VALID time.
    """
    lo, hi = _slice(arrays, t0, t1)
    tt = arrays["t"][lo:hi]
    n = len(tt)
    if n == 0:
        return _empty_features()
    valid = arrays["valid"][lo:hi]
    dt = _durations(tt, t1, cfg.max_gap_seconds)
    valid_time = float(dt[valid].sum())
    n_valid = int(valid.sum())
    out = _empty_features(valid_time / (t1 - t0), valid_time, n_valid)
    if n_valid < 2 or valid_time <= 0:
        return out

    el, er, mar = (arrays[k][lo:hi] for k in ("ear_l", "ear_r", "mar"))
    yaw, pitch, roll = (arrays[k][lo:hi] for k in ("yaw", "pitch", "roll"))
    both = np.maximum(el, er)          # "eyes closed" needs max(L, R) below threshold
    mean_ear = (el + er) / 2
    linked = valid[:-1] & valid[1:] & (np.diff(tt) <= cfg.max_gap_seconds)
    minutes = valid_time / 60.0

    def frac(mask):
        return float(dt[mask & valid].sum() / valid_time)

    open_level = float(np.percentile(both[valid], cfg.open_level_percentile))
    closed_fixed = valid & (both < cfg.ear_closed_fixed)
    closed_rel = valid & (both < cfg.ear_closed_relative * open_level)
    out["perclos_fixed"] = frac(closed_fixed)
    out["perclos_relative"] = frac(closed_rel)
    out["ear_open_level"] = open_level

    starts, ends = _runs(closed_fixed, linked)
    blink_durations, long_count, longest = [], 0, 0.0
    for a, b in zip(starts, ends):
        duration = float(dt[a:b + 1].sum())
        longest = max(longest, duration)
        bounded = a > 0 and linked[a - 1] and b < n - 1 and linked[b]
        if not bounded:
            continue
        if cfg.blink_min_seconds <= duration <= cfg.blink_max_seconds:
            blink_durations.append(duration)
        elif duration > cfg.blink_max_seconds:
            long_count += 1
    out["blink_rate_per_min"] = len(blink_durations) / minutes
    out["blink_duration_mean_s"] = float(np.mean(blink_durations)) if blink_durations else float("nan")
    out["long_closure_rate_per_min"] = long_count / minutes
    out["longest_closure_s"] = longest

    e = mean_ear[valid]
    out["ear_mean"], out["ear_std"] = float(e.mean()), float(e.std())
    out["ear_p10"], out["ear_p50"] = (float(v) for v in np.percentile(e, (10, 50)))
    out["ear_velocity"] = _mean_abs_rate(mean_ear, tt, linked)
    if open_level > 1e-6:
        out["ear_mean_rel"] = out["ear_mean"] / open_level
        out["ear_p10_rel"] = out["ear_p10"] / open_level
        out["ear_std_rel"] = out["ear_std"] / open_level
        out["ear_velocity_rel"] = out["ear_velocity"] / open_level

    m = mar[valid]
    out["mar_mean"], out["mar_std"] = float(m.mean()), float(m.std())
    out["mar_p90"] = float(np.percentile(m, 90))
    mouth_open = valid & (mar > cfg.mar_open)
    out["mar_open_frac"] = frac(mouth_open)
    ms, me = _runs(mouth_open, linked)
    events = sum(1 for a, b in zip(ms, me) if dt[a:b + 1].sum() >= cfg.mouth_open_event_seconds)
    out["mouth_open_events_per_min"] = events / minutes
    out["mar_velocity"] = _mean_abs_rate(mar, tt, linked)

    out["yaw_std"], out["pitch_std"], out["roll_std"] = (float(x[valid].std()) for x in (yaw, pitch, roll))
    out["yaw_velocity"] = _mean_abs_rate(yaw, tt, linked)
    out["pitch_velocity"] = _mean_abs_rate(pitch, tt, linked)
    # Raw (uncalibrated) pose, so "off center" is measured against this window's own median pose.
    off = (np.abs(yaw - np.median(yaw[valid])) > cfg.yaw_limit) | (np.abs(pitch - np.median(pitch[valid])) > cfg.pitch_limit)
    out["head_off_center_frac"] = frac(off)
    p5, p95 = np.percentile(pitch[valid], (5, 95))
    out["pitch_range_p5_p95"] = float(p95 - p5)
    return out


def feature_vector(features):
    return np.array([features[name] for name in MODEL_FEATURE_NAMES], dtype=np.float64)


def sequence_bins(arrays, t0, t1, cfg):
    """Per-bin channels for temporal models. Returns (values [T, C] with NaN where unobserved, mask [T]).

    A bin is observed (mask=1) only if its valid time covers >= bin_min_coverage of the bin.
    Unobserved bins are NaN + mask 0 so models can ignore them; they are never closed eyes or zeros.
    Pose channels are centered on the window's median pose (no calibration needed).
    """
    n_bins = cfg.n_bins
    values = np.full((n_bins, len(SEQUENCE_CHANNELS)), np.nan)
    mask = np.zeros(n_bins)
    lo, hi = _slice(arrays, t0, t1)
    if hi - lo < 2:
        return values, mask
    tt, valid = arrays["t"][lo:hi], arrays["valid"][lo:hi]
    dt = _durations(tt, t1, cfg.max_gap_seconds)
    el, er, mar = (arrays[k][lo:hi] for k in ("ear_l", "ear_r", "mar"))
    yaw, pitch, roll = (arrays[k][lo:hi] for k in ("yaw", "pitch", "roll"))
    if valid.sum() < 2:
        return values, mask
    both, mean_ear = np.maximum(el, er), (el + er) / 2
    linked = valid[:-1] & valid[1:] & (np.diff(tt) <= cfg.max_gap_seconds)
    gap = np.diff(tt)
    ok = linked & (gap > 0)
    ear_rate = np.full(len(tt), np.nan)
    head_rate = np.full(len(tt), np.nan)
    ear_rate[:-1][ok] = np.abs(np.diff(mean_ear))[ok] / gap[ok]
    head_rate[:-1][ok] = (np.abs(np.diff(yaw)) + np.abs(np.diff(pitch)))[ok] / gap[ok]
    med = [float(np.median(x[valid])) for x in (yaw, pitch, roll)]
    which = np.minimum(((tt - t0) / cfg.bin_seconds).astype(int), n_bins - 1)
    for b in range(n_bins):
        sel = (which == b) & valid
        if not sel.any():
            continue
        vt = float(dt[sel].sum())
        if vt / cfg.bin_seconds < cfg.bin_min_coverage:
            continue
        mask[b] = 1.0
        rate_e, rate_h = ear_rate[sel], head_rate[sel]
        values[b] = (
            mean_ear[sel].mean(), mean_ear[sel].min(),
            dt[sel & (both < cfg.ear_closed_fixed)].sum() / vt,
            mar[sel].mean(), mar[sel].max(),
            yaw[sel].mean() - med[0], pitch[sel].mean() - med[1], roll[sel].mean() - med[2],
            np.nanmean(rate_e) if np.isfinite(rate_e).any() else np.nan,
            np.nanmean(rate_h) if np.isfinite(rate_h).any() else np.nan,
        )
    return values, mask


# --------------------------------------------------------------------------- preprocessing
class TabularPreprocessor:
    """Median impute -> missing-indicator columns -> standardize -> clip. Fit on TRAIN rows only."""
    CLIP = 10.0

    def __init__(self, names=MODEL_FEATURE_NAMES):
        self.names = tuple(names)
        self.median = self.mean = self.scale = None
        self.indicator_columns = []

    def fit(self, X):
        X = np.where(np.isfinite(X), X, np.nan)
        if X.ndim != 2 or X.shape[1] != len(self.names) or len(X) == 0:
            raise ValueError("fit() needs a non-empty [n, n_features] matrix")
        observed = np.isfinite(X).any(axis=0)
        self.median = np.zeros(X.shape[1])
        self.median[observed] = np.nanmedian(X[:, observed], axis=0)
        self.indicator_columns = [int(i) for i in np.flatnonzero(np.isnan(X).any(axis=0))]
        filled = np.where(np.isnan(X), self.median, X)
        self.mean = filled.mean(axis=0)
        self.scale = np.maximum(filled.std(axis=0), 1e-6)
        return self

    @property
    def output_dim(self):
        return len(self.names) + len(self.indicator_columns)

    def transform(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        X = np.where(np.isfinite(X), X, np.nan)
        missing = np.isnan(X)
        Z = (np.where(missing, self.median, X) - self.mean) / self.scale
        Z = np.clip(Z, -self.CLIP, self.CLIP)
        indicators = missing[:, self.indicator_columns].astype(np.float64)
        return np.concatenate([Z, indicators], axis=1)

    def to_dict(self):
        return dict(kind="tabular", names=list(self.names), median=self.median.tolist(), mean=self.mean.tolist(),
                    scale=self.scale.tolist(), indicator_columns=self.indicator_columns, clip=self.CLIP)

    @classmethod
    def from_dict(cls, d):
        p = cls(d["names"])
        p.median, p.mean, p.scale = (np.asarray(d[k], dtype=np.float64) for k in ("median", "mean", "scale"))
        p.indicator_columns = [int(i) for i in d["indicator_columns"]]
        return p


class SequencePreprocessor:
    """Per-channel standardization over OBSERVED bins; unobserved -> 0 plus an explicit mask channel."""
    CLIP = 10.0

    def __init__(self, channels=SEQUENCE_CHANNELS):
        self.channels = tuple(channels)
        self.mean = self.scale = None

    def fit(self, values, mask):
        flat = values[mask > 0.5].reshape(-1, len(self.channels))
        if len(flat) == 0:
            raise ValueError("No observed bins to fit on")
        flat = np.where(np.isfinite(flat), flat, np.nan)
        observed = np.isfinite(flat).any(axis=0)
        self.mean = np.zeros(len(self.channels))
        self.scale = np.ones(len(self.channels))
        self.mean[observed] = np.nanmean(flat[:, observed], axis=0)
        self.scale[observed] = np.maximum(np.nanstd(flat[:, observed], axis=0), 1e-6)
        return self

    @property
    def input_dim(self):
        return len(self.channels) + 1

    def transform(self, values, mask):
        values = np.asarray(values, dtype=np.float64)
        mask = np.asarray(mask, dtype=np.float64)
        Z = np.clip((values - self.mean) / self.scale, -self.CLIP, self.CLIP)
        Z = np.where(np.isfinite(Z) & (mask[..., None] > 0.5), Z, 0.0)
        return np.concatenate([Z, mask[..., None]], axis=-1)

    def to_dict(self):
        return dict(kind="sequence", channels=list(self.channels), mean=self.mean.tolist(),
                    scale=self.scale.tolist(), clip=self.CLIP)

    @classmethod
    def from_dict(cls, d):
        p = cls(d["channels"])
        p.mean, p.scale = (np.asarray(d[k], dtype=np.float64) for k in ("mean", "scale"))
        return p
