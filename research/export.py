"""Verify and copy a trained run into the directory the app loads (default models/learned)."""
from pathlib import Path
import shutil

import numpy as np

from dms.features import rows_to_arrays
from dms.learned import BUNDLE_FILE, WEIGHTS_FILE, Bundle


def selftest(bundle):
    """Score a synthetic 'healthy tracking' window through the deployed path; must be finite."""
    cfg = bundle.cfg
    fps, rng = 15.0, np.random.default_rng(0)
    n = int(cfg.window_seconds * fps) + 5
    rows = [(i / fps, True, .30 + .01 * rng.standard_normal(), .30 + .01 * rng.standard_normal(), .10 + .01 * rng.standard_normal(),
             0.5 * rng.standard_normal(), 0.5 * rng.standard_normal(), 0.2 * rng.standard_normal()) for i in range(n)]
    logits, _ = bundle.logits_from_window(rows_to_arrays(rows), 0.0, cfg.window_seconds)
    if logits is None or not np.isfinite(logits).all() or not np.isfinite(bundle.probabilities(logits)).all():
        raise RuntimeError("Export self-test failed: non-finite or missing scores")


def export_bundle(run_dir, dest, allow_synthetic=False, require_heldout=False):
    run_dir, dest = Path(run_dir), Path(dest)
    bundle = Bundle(run_dir)                      # verifies schema, feature version, shapes, weights hash
    meta = bundle.meta
    if meta["training"].get("synthetic") and not allow_synthetic:
        raise RuntimeError("Refusing to export a bundle trained on the SYNTHETIC smoke fixture into an app path. Pass --allow-synthetic only for plumbing tests.")
    if require_heldout and not meta["validation_status"].get("heldout_evaluated"):
        raise RuntimeError("Held-out evaluation has not been recorded for this run; run `evaluate --split test` first")
    selftest(bundle)
    dest.mkdir(parents=True, exist_ok=True)
    for name in (BUNDLE_FILE, WEIGHTS_FILE, "metrics_test.json", "report_test.md", "metrics_val.json", "selection.json"):
        if (run_dir / name).is_file():
            shutil.copy2(run_dir / name, dest / name)
    (dest / "BUNDLE_README.txt").write_text(
        "Research model bundle for the Driver Monitor app. Loaded offline by dms/learned.py.\n"
        f"model_type={meta['model_type']} dataset={meta['training']['dataset']} synthetic={meta['training'].get('synthetic')}\n"
        f"score kind: {bundle.score_kind}\n"
        f"heldout_evaluated={meta['validation_status'].get('heldout_evaluated')}\n"
        "Window-level class scores trained on weak video-level labels. Not event detection, not an alarm, not safety validated.\n"
        "See MODEL_CARD.md in the repository root.\n", encoding="utf-8")
    return dest
