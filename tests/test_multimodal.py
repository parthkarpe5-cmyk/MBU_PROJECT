import io
import wave
import numpy as np
import pytest
from dms.multimodal import Fusion, GazeCalibration, fresh, gaze, visibility
from dms.sounds import wav_bytes


def eyes(offset=0):
    p = np.zeros((478, 3))
    for a, b, iris, start in ((33, 133, 468, 10), (362, 263, 473, 60)):
        p[a, :2], p[b, :2], p[iris, :2] = (start, 20), (start+30, 20), (start+15+offset*30, 20)
    return p


def test_gaze_is_independent_of_pose_and_roll():
    p = eyes(.16)
    assert gaze(p) == pytest.approx(.16)
    angle = .3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    p[:, :2] = p[:, :2] @ rotation.T + [100, 50]
    assert gaze(p) == pytest.approx(.16)
    assert gaze(p, eyes_open=False) is None
    assert gaze(p[:468]) is None
    p[468] = np.nan
    assert gaze(p) is None


def test_gaze_disagreement_and_calibration():
    p = eyes()
    p[468, 0] += 8
    assert gaze(p) is None
    c = GazeCalibration()
    assert c.correct(.2) is None
    c.begin()
    for _ in range(30): c.add(.05)
    assert c.finish(True)
    assert c.correct(.2) == pytest.approx(.15)


def test_visibility_is_conservative_and_distinct():
    good = np.tile(np.arange(100, dtype=np.uint8)[None, :, None], (100, 1, 3))
    assert visibility(np.zeros_like(good), None)[0] == 'CAMERA VIEW UNAVAILABLE'
    assert visibility(good, None)[0] == 'FACE NOT DETECTED'
    assert 'PARTIAL' in visibility(good, {'healthy': False})[0]
    assert visibility(good, {'healthy': True})[0] == 'MONITORING'


def test_fusion_continuity_missing_gaze_and_priority():
    f = Fusion(gaze_seconds=1)
    normal = {'alerts': []}
    for t in (0, .5, 1): r = f.update(t, normal, gaze_value=.2)
    assert r['alerts'] == ['EYE GAZE AWAY'] and r['sound'] == 'distraction'
    assert f.update(1.1, normal, gaze_value=None)['alerts'] == []
    for t in (1.2, 1.7, 2.2): r = f.update(t, {'alerts': ['EYES CLOSED']}, gaze_value=.2)
    assert r['sound'] == 'fatigue'
    r = f.update(2.3, normal, 'FACE NOT DETECTED')
    assert not r['alerts'] and r['sound'] is None
    for t in (2.8, 3.3, 3.8, 4.3): r = f.update(t, normal, 'FACE NOT DETECTED')
    assert r['sound'] == 'availability'
    assert f.update(6, normal, gaze_value=.2)['alerts'] == []  # gap breaks continuity
    assert f.update(5, normal, gaze_value=.2)['alerts'] == []  # clock rollback
    assert f.update(5.1, normal, paused=True)['sound'] is None


def test_expression_context_requires_new_temporal_predictions_and_never_alarms():
    f = Fusion()
    cue = dict(label='anger', score=.9, timestamp=0)
    for t in (0, .5, 1): r = f.update(t, {'alerts': []}, expression=cue)
    assert not r['expression_stable']
    r = f.update(1.1, {'alerts': []}, expression={**cue, 'timestamp': 1.1})
    assert r['expression_stable'] and 'anger' in r['summary']
    assert not r['alerts'] and r['sound'] is None
    assert fresh(cue, 2) is None
    assert fresh({**cue, 'timestamp': 4}, 2) is None
    assert fresh({**cue, 'score': float('nan')}, 0) is None
    r = f.update(1.2, {'alerts': []}, behavior=dict(label='drowsy', timestamp=1.2, score=.9))
    assert r['modalities']['behavior'] and r['sound'] is None


def test_sounds_have_three_distinct_valid_pcm_patterns():
    patterns = [wav_bytes(name) for name in ('distraction', 'fatigue', 'availability')]
    assert len(set(patterns)) == 3
    for payload in patterns:
        with wave.open(io.BytesIO(payload)) as wav:
            assert wav.getnchannels() == 1 and wav.getframerate() == 16000 and wav.getnframes() > 0


def test_ferplus_preprocessing_and_score_contract(monkeypatch, tmp_path):
    from dms.expression import ExpressionCNN
    class Net:
        def setInput(self, blob): self.blob = blob
        def forward(self): return np.array([[0, 4, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    net = Net()
    monkeypatch.setattr('cv2.dnn.readNetFromONNX', lambda _: net)
    cnn = ExpressionCNN(tmp_path/'model.onnx')
    p = np.array([[0, 0], [100, 100], [10, 10]])
    r = cnn.infer(np.full((100, 100, 3), 128, np.uint8), p, 12)
    assert net.blob.shape == (1, 1, 64, 64)
    assert float(net.blob.mean()) == 128  # FER+, not the incompatible 48px / 255 contract
    assert r['label'] == 'happiness' and r['timestamp'] == 12 and .6 < r['score'] <= 1
    net.forward = lambda: np.zeros((1, 7))
    with pytest.raises(ValueError, match='one finite logit'): cnn.infer(np.zeros((100, 100, 3), np.uint8), p, 0)


def test_real_ferplus_inference_on_static_fixture():
    from pathlib import Path
    import cv2
    from dms.expression import ExpressionCNN
    from dms.vision import Tracker
    from skimage.data import astronaut
    root = Path(__file__).resolve().parents[1]
    if not (root/'models/emotion-ferplus-8.onnx').is_file() or not (root/'models/face_landmarker.task').is_file():
        pytest.skip('Run setup_model.py and setup_expression.py for actual model inference')
    frame = cv2.cvtColor(astronaut(), cv2.COLOR_RGB2BGR)
    tracker = Tracker(root/'models/face_landmarker.task')
    try:
        data = tracker.process(frame, 1.)
        assert data is not None
        result = ExpressionCNN(root/'models/emotion-ferplus-8.onnx').infer(frame, data['points'], 1.)
        assert result['label'] in ('neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt')
        assert 0 <= result['score'] <= 1 and result['timestamp'] == 1.
    finally:
        tracker.close()
