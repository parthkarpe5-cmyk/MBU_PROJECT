"""Feature-based baseline: multinomial logistic regression on window features.

Train / validation only. The held-out test split is never loaded here.
Selection on validation: window length and C (macro-F1). Calibration: temperature on validation logits.
Preprocessing (median impute, indicators, standardization) is fit on TRAIN windows only.
"""
from pathlib import Path

import numpy as np

from dms.features import TabularPreprocessor
from dms.learned import softmax
from .bundle import write_bundle
from .config import dump_json, environment, feature_config, set_seed, sha256_json
from .metrics import fit_temperature, macro_f1, summarize
from .windows import CLASS_NAMES, build_windows, subset, summarize_windows


def train_baseline(cfg, rows, cache_dir, run_dir, manifest_sha=None, synthetic=False, log=print):
    from sklearn.linear_model import LogisticRegression
    set_seed(cfg["seed"])
    run_dir = Path(run_dir)
    grid = []
    best = None
    for window_seconds in cfg["baseline"]["window_seconds_grid"]:
        fcfg = feature_config(cfg, window_seconds=float(window_seconds))
        wins = build_windows(rows, cache_dir, fcfg, "tabular", splits=("train", "val"), log=log)
        train, val = subset(wins, "train"), subset(wins, "val")
        if len(set(train["y"])) < 3:
            raise RuntimeError(f"Train windows contain classes {sorted(set(train['y']))}; need all 3 (window={window_seconds}s)")
        if len(val["y"]) == 0:
            raise RuntimeError("No validation windows; cannot select hyperparameters")
        pre = TabularPreprocessor().fit(train["X"])
        Xtr, Xva = pre.transform(train["X"]), pre.transform(val["X"])
        for C in cfg["baseline"]["C_grid"]:
            clf = LogisticRegression(C=float(C), max_iter=5000, class_weight="balanced", random_state=cfg["seed"]).fit(Xtr, train["y"])
            logits = clf.decision_function(Xva)
            f1 = macro_f1(val["y"], logits.argmax(1))
            grid.append(dict(window_seconds=float(window_seconds), C=float(C), val_macro_f1=f1, n_train=len(train["y"]), n_val=len(val["y"])))
            log(f"window {window_seconds}s C={C}: val macro-F1 {f1:.4f}")
            if best is None or f1 > best["f1"]:
                best = dict(f1=f1, fcfg=fcfg, pre=pre, clf=clf, logits=logits, val=val, train=train, window_seconds=float(window_seconds), C=float(C),
                            stats=wins["stats"])
    T = fit_temperature(best["logits"], best["val"]["y"])
    training = {
        "dataset": "SYNTHETIC SMOKE FIXTURE (not UTA-RLDD)" if synthetic else "UTA-RLDD", "synthetic": bool(synthetic),
        "seed": cfg["seed"], "selected": {"window_seconds": best["window_seconds"], "C": best["C"]},
        "selection_split": "validation", "preprocessing_fit_split": "train", "calibration_fit_split": "validation",
        "train": summarize_windows(best["train"]), "validation": summarize_windows(best["val"]),
        "window_build_stats": {k: v for k, v in best["stats"].items() if k != "videos_missing_cache"},
        "manifest_sha256": manifest_sha, "config_sha256": sha256_json(cfg), "test_split_used": False,
    }
    meta = write_bundle(run_dir, "logreg", {"W": best["clf"].coef_, "b": best["clf"].intercept_}, best["pre"].to_dict(),
                        best["fcfg"], best["logits"], best["val"]["y"], T, training)
    dump_json(run_dir / "selection.json", {"grid": grid, "rule": "max validation macro-F1; first max wins ties", "selected": training["selected"]})
    dump_json(run_dir / "config.json", cfg)
    dump_json(run_dir / "env.json", environment())
    vm = summarize(best["val"]["y"], best["logits"].argmax(1))
    vm["majority_class_macro_f1_reference"] = macro_f1(best["val"]["y"], np.full(len(best["val"]["y"]), np.bincount(best["val"]["y"]).argmax()))
    vm["note"] = "VALIDATION metrics (used for selection and calibration): optimistic, not a generalization estimate."
    dump_json(run_dir / "metrics_val.json", vm)
    log(f"selected window {best['window_seconds']}s C={best['C']} val macro-F1 {best['f1']:.4f}; temperature {T:.3f}")
    return meta
