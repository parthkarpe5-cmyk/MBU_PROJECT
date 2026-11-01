"""The camera worker must keep running heuristics when the learned model is missing or crashes."""
from argparse import Namespace
import queue
import threading
import numpy as np
import app
import dms.learned
import dms.vision


class Cam:
    released = False
    def __init__(self, *a): self.reads = 0
    def isOpened(self): return True
    def set(self, *a): pass
    def read(self):
        self.reads += 1
        return (True, np.zeros((480, 640, 3), np.uint8)) if self.reads <= 5 else (False, None)
    def release(self): Cam.released = True


class Tracker:
    closed = False
    def __init__(self, path): pass
    def process(self, frame, now):
        return dict(points=np.zeros((478, 3)), values=(.3, .3, .1, 0, 0), roll=0., axes=np.zeros((3, 2)), healthy=True, quality="fake")
    def draw(self, frame, data, toggles): return frame
    def close(self): Tracker.closed = True


def run(monkeypatch, learned_model):
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", Cam)
    monkeypatch.setattr(dms.vision, "Tracker", Tracker)
    packets = []
    out = queue.Queue(maxsize=1)
    real_put = out.put
    out.put = lambda v: (packets.append(v), real_put(v))[1]
    app.worker(Namespace(model="x", camera=0, expression_model=None, learned_model=learned_model), queue.Queue(), out, threading.Event())
    return packets


def test_missing_model_keeps_heuristics(monkeypatch, tmp_path):
    packets = run(monkeypatch, tmp_path / "missing")
    good = [p for p in packets if "error" not in p]
    assert len(good) == 5 and all(p["state"]["state"] == "MONITORING" for p in good)
    assert "Not active" in good[-1]["learned"] and "Heuristic monitoring continues" in good[-1]["learned"]
    assert Cam.released and Tracker.closed


def test_crashing_predictor_does_not_stop_monitoring(monkeypatch, tmp_path):
    def boom(self, t, record):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(dms.learned.LearnedStatePredictor, "update", boom)
    packets = run(monkeypatch, None)
    good = [p for p in packets if "error" not in p]
    assert len(good) == 5 and good[-1]["state"]["state"] == "MONITORING"
    assert "model exploded" in good[-1]["learned"] or "Not active" in good[-1]["learned"]
