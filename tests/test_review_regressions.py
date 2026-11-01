"""Regression checks added during the independent Windows review."""
import json
import warnings
import numpy as np
import pytest
from dms.features import MODEL_FEATURE_NAMES, TabularPreprocessor, SequencePreprocessor
from research.config import load_config, feature_config
from research.synthetic import write_synthetic_cache
from research.windows import build_windows
from research.smoke import run_smoke
from research.evaluate import score_split


def test_smoke_preserves_existing_files(tmp_path):
    sentinel = tmp_path / 'important.txt'
    sentinel.write_text('keep me')
    with pytest.raises(FileExistsError, match='not empty'):
        run_smoke(tmp_path)
    assert sentinel.read_text() == 'keep me'


@pytest.mark.parametrize('metadata', [{'max_seconds': 60}, {'possibly_truncated': True}])
def test_training_rejects_incomplete_cache(tmp_path, metadata):
    rows = write_synthetic_cache(tmp_path, folds=1, per_fold=1, seconds=40)
    path = tmp_path / (rows[0]['video_id'] + '.json')
    original = json.loads(path.read_text())
    path.write_text(json.dumps({**original, **metadata}))
    with pytest.raises(ValueError, match='Incomplete extraction cache'):
        build_windows(rows, tmp_path, feature_config(load_config()), splits=('train',))
    with pytest.raises(ValueError, match='Incomplete extraction cache'):
        score_split(None, rows, tmp_path, 'train')


def test_all_missing_preprocessing_emits_no_runtime_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        x = np.full((4, len(MODEL_FEATURE_NAMES)), np.nan)
        tab = TabularPreprocessor().fit(x)
        assert np.isfinite(tab.transform(x)).all()
        v = np.full((2, 3, 10), np.nan)
        seq = SequencePreprocessor().fit(v, np.ones((2, 3)))
        assert np.isfinite(seq.transform(v, np.ones((2, 3)))).all()
