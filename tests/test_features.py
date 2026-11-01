"""Feature definitions: denominators, missing data, blinks, window boundaries, preprocessing."""
import numpy as np
import pytest
from dms.features import (FeatureConfig, MODEL_FEATURE_NAMES, SequencePreprocessor, TabularPreprocessor, iter_window_bounds,
                          rows_to_arrays, sequence_bins, window_coverage, window_features)

CFG = FeatureConfig(window_seconds=10, stride_seconds=5, min_coverage=.7)


def make(n=150, fps=15., ear=.3, valid=None, closed=()):
    t = np.arange(n) / fps
    e = np.full(n, ear)
    for a, b in closed:
        e[(t >= a) & (t < b)] = .1
    v = np.ones(n, bool) if valid is None else valid
    nan = np.where(v, 1.0, np.nan)
    return {"t": t, "valid": v, "ear_l": e * nan, "ear_r": e * nan, "mar": .1 * nan, "yaw": 0 * nan, "pitch": 0 * nan, "roll": 0 * nan}


def test_missing_is_excluded_not_closed():
    valid = np.ones(150, bool)
    valid[30:90] = False                      # 4 s of lost tracking, which are NaN
    f = window_features(make(valid=valid), 0, 10, CFG)
    assert f["coverage"] == pytest.approx(.6, abs=.02)
    assert f["perclos_fixed"] == 0.0          # missing frames are not closed eyes
    assert not np.isnan(f["ear_mean"]) and f["ear_mean"] == pytest.approx(.3)


def test_perclos_denominator_is_valid_time():
    valid = np.ones(150, bool)
    valid[75:] = False                         # second half missing
    f = window_features(make(valid=valid, closed=[(0, 2.5)]), 0, 10, CFG)
    assert f["perclos_fixed"] == pytest.approx(2.5 / 5.0, abs=.03)   # closed time / VALID time, not / 10 s


def test_blink_and_long_closure_and_censored():
    a = make(closed=[(2, 2.2), (5, 6.0)])      # 0.2 s blink, 1.0 s long closure
    f = window_features(a, 0, 10, CFG)
    assert f["blink_rate_per_min"] == pytest.approx(60 / 10, rel=.1)
    assert f["long_closure_rate_per_min"] > 0 and f["longest_closure_s"] >= .9
    edge = window_features(make(closed=[(9.5, 12)]), 0, 10, CFG)       # touches the window end: censored
    assert edge["blink_rate_per_min"] == 0 and edge["long_closure_rate_per_min"] == 0
    assert edge["perclos_fixed"] > 0                                    # but still counted in PERCLOS


def test_gap_not_bridged():
    a = make(n=150)
    keep = (a["t"] < 3) | (a["t"] > 7)
    a = {k: v[keep] for k, v in a.items()}
    f = window_features(a, 0, 10, CFG)
    assert f["coverage"] < .75                  # a 4 s hole counts as missing time (max_gap 0.5 s)


def test_window_is_half_open_and_grid_is_inside_video():
    a = make()
    f0 = window_features(a, 0, 5, CFG)
    f1 = window_features(a, 5, 10, CFG)
    assert f0["n_valid"] + f1["n_valid"] == 150                          # each sample lands in exactly one window
    assert list(iter_window_bounds(0, 10, CFG)) == [(0, 10)]
    assert list(iter_window_bounds(0, 14.9, CFG)) == [(0, 10)]
    assert list(iter_window_bounds(0, 15, CFG)) == [(0, 10), (5, 15)]
    assert list(iter_window_bounds(0, 9.99, CFG)) == []                  # never a truncated window


def test_empty_and_sparse_windows_do_not_crash():
    a = make()
    assert window_features(a, 100, 110, CFG)["coverage"] == 0
    assert window_coverage(a, 100, 110, CFG)[0] == 0
    only_invalid = make(valid=np.zeros(150, bool))
    assert window_features(only_invalid, 0, 10, CFG)["coverage"] == 0


def test_sequence_bins_mask_missing():
    valid = np.ones(150, bool)
    valid[45:90] = False                       # seconds 3-6 are missing
    values, mask = sequence_bins(make(valid=valid), 0, 10, FeatureConfig(window_seconds=10, bin_seconds=1))
    assert mask.tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    assert np.isnan(values[3]).all() and not np.isnan(values[0]).any()


def test_tabular_preprocessor_fit_on_train_only_and_missing_indicator():
    rng = np.random.default_rng(0)
    train = rng.normal(5, 2, (50, len(MODEL_FEATURE_NAMES)))
    train[:5, 3] = np.nan
    pre = TabularPreprocessor().fit(train)
    assert pre.indicator_columns == [3]
    test = rng.normal(100, 2, (4, len(MODEL_FEATURE_NAMES)))
    before = (pre.mean.copy(), pre.scale.copy())
    out = pre.transform(test)
    assert np.array_equal(before[0], pre.mean) and np.array_equal(before[1], pre.scale)   # transform never refits
    assert out.shape == (4, len(MODEL_FEATURE_NAMES) + 1) and np.abs(out).max() <= 10
    clone = TabularPreprocessor.from_dict(pre.to_dict())
    assert np.allclose(clone.transform(test), out)


def test_sequence_preprocessor_zero_fills_only_unobserved():
    values = np.random.default_rng(0).normal(size=(5, 4, 10))
    mask = np.ones((5, 4)); mask[:, 2] = 0
    values[:, 2] = np.nan
    pre = SequencePreprocessor().fit(values, mask)
    z = pre.transform(values, mask)
    assert (z[:, 2, :] == 0).all() and (z[:, 0, -1] == 1).all() and (z[:, 2, -1] == 0).all()
