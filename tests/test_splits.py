"""Participant-level split integrity and label provenance."""
import csv
import pytest
from research.config import load_config
from research.dataset import (LeakageError, ManifestError, assert_participant_disjoint, build_manifest, read_manifest, write_manifest)
from research.windows import build_windows
from research.synthetic import write_synthetic_cache
from dms.features import FeatureConfig


def test_config_rejects_fold_in_two_splits():
    with pytest.raises(ValueError):
        load_config(None, {"splits": {"train_folds": [1, 2], "val_folds": [2], "test_folds": [5]}})


def test_leakage_detected():
    rows = [dict(participant_id="P1", split="train", fold=1), dict(participant_id="P1", split="test", fold=5)]
    with pytest.raises(LeakageError):
        assert_participant_disjoint(rows)
    assert assert_participant_disjoint([dict(participant_id="P1", split="train", fold=1), dict(participant_id="P1", split="train", fold=1)])


def _tree(root, layout):
    for rel in layout:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


CFG = {"dataset": {"layout_regex": r"Fold(?P<fold>\d)/(?P<participant>\d+)/(?P<stem>\d+)\.mp4",
                   "label_source": {"type": "filename_stem", "confirmed": True, "documentation": "unit test"}}}


def test_manifest_splits_by_fold_and_never_splits_participant(tmp_path):
    files = [f"Fold{f}/{p}/{s}.mp4" for f, ps in ((1, (1, 2)), (4, (3,)), (5, (4,))) for p in ps for s in (0, 5, 10)]
    _tree(tmp_path, files)
    rows, meta = build_manifest(tmp_path, load_config(None, CFG), probe=False, log=lambda m: None)
    assert {r["participant_id"]: r["split"] for r in rows} == {"P1": "train", "P2": "train", "P3": "val", "P4": "test"}
    assert {r["label_name"] for r in rows} == {"alert", "low_vigilance", "drowsy"}
    assert meta["warnings"]                                    # 4 participants != documented 60
    path, _ = write_manifest(rows, meta, tmp_path / "out")
    assert len(read_manifest(path)) == 12


def test_participant_in_two_folds_is_refused(tmp_path):
    _tree(tmp_path, ["Fold1/7/0.mp4", "Fold5/7/5.mp4"])
    with pytest.raises(LeakageError):
        build_manifest(tmp_path, load_config(None, CFG), probe=False, log=lambda m: None)


def test_labels_are_never_guessed(tmp_path):
    _tree(tmp_path, ["Fold1/1/0.mp4"])
    for source in (None, {"type": "filename_stem", "documentation": "x"}, {"type": "filename_stem", "confirmed": True}):
        cfg = load_config(None, {"dataset": {**CFG["dataset"], "label_source": source}})
        with pytest.raises(ManifestError):
            build_manifest(tmp_path, cfg, probe=False, log=lambda m: None)
    _tree(tmp_path, ["Fold1/2/7.mp4"])                         # stem 7 is not in the documented label map
    with pytest.raises(ManifestError, match="not in documented"):
        build_manifest(tmp_path, load_config(None, CFG), probe=False, log=lambda m: None)


def test_missing_layout_regex_refused(tmp_path):
    with pytest.raises(ManifestError, match="layout_regex"):
        build_manifest(tmp_path, load_config(None, {"dataset": {"label_source": CFG["dataset"]["label_source"]}}), probe=False)


def test_csv_label_source_requires_provenance(tmp_path):
    _tree(tmp_path, ["Fold1/1/9.mp4"])   # stem 9 means nothing; the CSV is the label source
    labels = tmp_path / "labels.csv"
    labels.write_text("video_path,label\nFold1/1/9.mp4,5\n")
    cfg = {"dataset": {**CFG["dataset"], "label_source": {"type": "csv", "path": str(labels)}}}
    with pytest.raises(ManifestError, match="provenance"):
        build_manifest(tmp_path, load_config(None, cfg), probe=False)
    cfg["dataset"]["label_source"]["provenance"] = "unit test"
    rows, _ = build_manifest(tmp_path, load_config(None, cfg), probe=False, log=lambda m: None)
    assert rows[0]["label_name"] == "low_vigilance"


def test_windows_never_cross_participants_or_splits(tmp_path):
    rows = write_synthetic_cache(tmp_path, folds=5, per_fold=2, seconds=60)
    wins = build_windows(rows, tmp_path, FeatureConfig(window_seconds=20, stride_seconds=5))
    by_participant = {}
    for m in wins["meta"]:
        by_participant.setdefault(m["participant_id"], set()).add(m["split"])
    assert all(len(s) == 1 for s in by_participant.values())
    assert {m["split"] for m in wins["meta"]} == {"train", "val", "test"}
    only = build_windows(rows, tmp_path, FeatureConfig(window_seconds=20, stride_seconds=5), splits=("train", "val"))
    assert {m["split"] for m in only["meta"]} == {"train", "val"}       # test videos are not even loaded
