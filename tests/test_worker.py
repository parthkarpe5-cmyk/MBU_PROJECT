"""Camera lifecycle checks with a fake device; never open real hardware."""
from argparse import Namespace
import queue
import threading
import app
import dms.vision


def test_camera_failure_releases_resources(monkeypatch):
    import cv2
    devices = []
    class FakeCamera:
        def __init__(self, *args):
            self.released = False
            devices.append(self)
        def isOpened(self):
            return False
        def release(self):
            self.released = True
    class FakeTracker:
        closed = False
        def __init__(self, path):
            pass
        def close(self):
            FakeTracker.closed = True
    monkeypatch.setattr(cv2, 'VideoCapture', FakeCamera)
    monkeypatch.setattr(dms.vision, 'Tracker', FakeTracker)
    output = queue.Queue(maxsize=1)
    app.worker(Namespace(model='unused', camera=0), queue.Queue(), output, threading.Event())
    assert 'Cannot open webcam' in output.get_nowait()['error']
    assert all(d.released for d in devices)
    assert FakeTracker.closed


def test_capture_failure_releases_resources(monkeypatch):
    import cv2
    class FakeCamera:
        released = False
        def __init__(self, *args):
            pass
        def isOpened(self):
            return True
        def set(self, *args):
            pass
        def read(self):
            return False, None
        def release(self):
            FakeCamera.released = True
    class FakeTracker:
        closed = False
        def __init__(self, path):
            pass
        def close(self):
            FakeTracker.closed = True
    monkeypatch.setattr(cv2, 'VideoCapture', FakeCamera)
    monkeypatch.setattr(dms.vision, 'Tracker', FakeTracker)
    output = queue.Queue(maxsize=1)
    app.worker(Namespace(model='unused', camera=0, expression_model=None), queue.Queue(), output, threading.Event())
    assert 'stopped delivering' in output.get_nowait()['error']
    assert FakeCamera.released and FakeTracker.closed
