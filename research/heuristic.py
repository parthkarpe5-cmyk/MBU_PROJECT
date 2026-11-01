"""Evaluate the EXISTING rule-based monitor (dms.core.Monitor) against weak video labels, where comparable.

The monitor emits sustained-event alerts (eyes closed >= 1.5 s, mouth open >= 1.5 s, looking away >= 2 s),
not alert/low-vigilance/drowsy classes, so there is no like-for-like 3-class prediction. What can honestly be done:
  (a) DESCRIPTIVE, no fitting: per class, how often a window contains each heuristic alert, and the mean alert
      time fraction. (b) AUROC of each heuristic measure for pairs of classes. (c) A 3-class decision from ONE
      heuristic measure using two cutpoints chosen on VALIDATION windows only -- that is "heuristic + fitted mapping",
      not the unmodified app. The heuristic is replayed on cached per-frame records with default (uncalibrated)
      thresholds; live per-session calibration has no counterpart in the dataset.
"""
from pathlib import Path

import numpy as np

from dms.core import Config, Monitor
from dms.features import FeatureConfig, _durations
from .config import dump_json
from .extract import load_cache
from .metrics import auroc, macro_f1, summarize
from .windows import CLASS_NAMES, build_windows

MEASURES = ("hx_any_alert_frac", "hx_eyes_closed_alert_frac", "hx_yawn_events_per_min", "hx_look_away_alert_frac")
PRIMARY = "hx_any_alert_frac"


def replay(arrays, config=None):
    """Run the unmodified Monitor over cached records exactly as the live worker would feed it."""
    monitor = Monitor(config or Config())
    n = len(arrays["t"])
    eyes, yawn, away = (np.zeros(n, bool) for _ in range(3))
    yawns = np.zeros(n, int)
    for i in range(n):
        values = None
        if arrays["valid"][i]:
            values = (arrays["ear_l"][i], arrays["ear_r"][i], arrays["mar"][i], arrays["yaw"][i], arrays["pitch"][i])
        alerts = monitor.update(float(arrays["t"][i]), values)["alerts"]
        eyes[i], yawn[i], away[i] = "EYES CLOSED" in alerts, "YAWN / MOUTH OPEN" in alerts, "LOOKING AWAY" in alerts
        yawns[i] = monitor.yawns
    return dict(eyes=eyes, yawn=yawn, away=away, yawns=yawns)


def window_measures(arrays, flags, t0, t1, fcfg):
    t = arrays["t"]
    lo, hi = np.searchsorted(t, [t0, t1], "left")
    tt, valid = t[lo:hi], arrays["valid"][lo:hi]
    dt = _durations(tt, t1, fcfg.max_gap_seconds)
    vt = float(dt[valid].sum())
    if vt <= 0:
        return {m: float("nan") for m in MEASURES}
    frac = lambda m: float(dt[m[lo:hi] & valid].sum() / vt)
    before = flags["yawns"][lo - 1] if lo > 0 else 0
    return {"hx_any_alert_frac": frac(flags["eyes"] | flags["yawn"] | flags["away"]),
            "hx_eyes_closed_alert_frac": frac(flags["eyes"]), "hx_look_away_alert_frac": frac(flags["away"]),
            "hx_yawn_events_per_min": float((flags["yawns"][hi - 1] - before) / (vt / 60.0))}


def measure_table(rows, cache_dir, fcfg, splits):
    wins = build_windows(rows, cache_dir, fcfg, "tabular", splits=splits)
    cache, table = {}, {m: [] for m in MEASURES}
    for m in wins["meta"]:
        if m["video_id"] not in cache:
            arrays, _ = load_cache(cache_dir, m["video_id"])
            cache[m["video_id"]] = (arrays, replay(arrays))
        arrays, flags = cache[m["video_id"]]
        out = window_measures(arrays, flags, m["t0"], m["t1"], fcfg)
        for k in MEASURES:
            table[k].append(out[k])
    return wins, {k: np.array(v) for k, v in table.items()}


def fit_cutpoints(scores, y, candidates=40):
    """Two ordered cutpoints (c1 <= c2): class 0 if s <= c1, class 1 if s <= c2, else class 2. VALIDATION data only."""
    uniq = np.unique(np.quantile(scores, np.linspace(0, 1, candidates)))
    best = (-1.0, None)
    for i, c1 in enumerate(uniq):
        for c2 in uniq[i:]:
            f1 = macro_f1(y, np.where(scores <= c1, 0, np.where(scores <= c2, 1, 2)))
            if f1 > best[0]:
                best = (f1, (float(c1), float(c2)))
    return best[1], best[0]


def evaluate_heuristic(rows, cache_dir, cfg, out_dir, log=print):
    from .config import feature_config
    fcfg = feature_config(cfg, window_seconds=float(cfg["baseline"]["window_seconds_grid"][0]))
    wins, table = measure_table(rows, cache_dir, fcfg, ("val", "test"))
    split = np.array([m["split"] for m in wins["meta"]])
    y = wins["y"]
    report = {"feature_window_seconds": fcfg.window_seconds, "heuristic_config": vars(Config()),
              "caveat": "The monitor emits sustained-event alerts, not alert/low-vigilance/drowsy classes: no like-for-like 3-class output exists. "
                        "Cutpoint results are 'heuristic measure + validation-fitted mapping', not the unmodified app.",
              "measures": {}}
    for name in MEASURES:
        s = table[name]
        ok = np.isfinite(s)
        entry = {"descriptive_by_class": {}, "auroc": {}}
        for sp in ("val", "test"):
            sel = (split == sp) & ok
            entry["descriptive_by_class"][sp] = {
                cname: {"n_windows": int(((y == i) & sel).sum()),
                        "mean": float(s[(y == i) & sel].mean()) if ((y == i) & sel).any() else None,
                        "fraction_windows_with_any_alert": float((s[(y == i) & sel] > 0).mean()) if ((y == i) & sel).any() else None}
                for i, cname in enumerate(CLASS_NAMES)}
            sel_t = sel
            for a, b in ((0, 2), (0, 1), (1, 2)):
                pair = sel_t & ((y == a) | (y == b))
                entry["auroc"].setdefault(sp, {})[f"{CLASS_NAMES[a]}_vs_{CLASS_NAMES[b]}"] = auroc(s[pair], y[pair] == b)
        val_sel, test_sel = (split == "val") & ok, (split == "test") & ok
        if val_sel.any() and len(set(y[val_sel])) == 3:
            cut, val_f1 = fit_cutpoints(s[val_sel], y[val_sel])
            entry["cutpoints_fit_on_validation"] = cut
            entry["val"] = summarize(y[val_sel], np.where(s[val_sel] <= cut[0], 0, np.where(s[val_sel] <= cut[1], 1, 2)))
            if test_sel.any():
                entry["test"] = summarize(y[test_sel], np.where(s[test_sel] <= cut[0], 0, np.where(s[test_sel] <= cut[1], 1, 2)))
        report["measures"][name] = entry
        log(f"{name}: " + (f"test macro-F1 {entry['test']['macro_f1']:.3f}" if "test" in entry else "no test windows"))
    report["primary_measure"] = PRIMARY
    report["note"] = ("PRIMARY measure was fixed in advance (the app's own ALERT state). Other measures are reported for transparency; "
                      "choosing the best of them on test would be test-set selection.")
    dump_json(Path(out_dir) / "heuristic_report.json", report)
    return report
