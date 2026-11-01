"""Synthetic pose/overlay tests, not a claim of real camera accuracy."""
import cv2
import numpy as np
import pytest
from dms.vision import head_pose, Tracker


def synthetic_points(rotation):
    model = np.float64([(0, 0, 0), (0, -330, -65), (-225, 170, -135),
                        (225, 170, -135), (-150, -150, -125), (150, -150, -125)])
    camera = np.float64([[640, 0, 320], [0, 640, 240], [0, 0, 1]])
    image, _ = cv2.projectPoints(model, rotation, np.float64([0, 0, 1500]), camera, np.zeros(4))
    points = np.zeros((478, 3))
    for i, xy in zip((1, 152, 33, 263, 61, 291), image.reshape(-1, 2)):
        points[i, :2] = xy
    return points


def test_frontal_synthetic_pose():
    yaw, pitch, roll, axes = head_pose(synthetic_points(np.float64([np.pi, 0, 0])), 640, 480)
    assert abs(yaw) < 1
    assert abs(pitch) < 1
    assert np.isfinite(axes).all()


def test_rotated_synthetic_pose():
    angle = np.deg2rad(30)
    ry = np.float64([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    rx = np.diag([1., -1., -1.])
    rotation, _ = cv2.Rodrigues(ry @ rx)
    yaw, pitch, _, _ = head_pose(synthetic_points(rotation), 640, 480)
    assert abs(yaw) == pytest.approx(30, abs=1)
    assert abs(pitch) < 1


def test_real_model_blank_image_and_connections():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'models' / 'face_landmarker.task'
    if not path.exists():
        pytest.skip('Run setup_model.py for the real model smoke test')
    tracker = Tracker(path)
    try:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert tracker.process(frame, 1.) is None
        points = synthetic_points(np.float64([np.pi, 0, 0]))
        data = dict(points=points, axes=np.array([[320, 240], [330, 240], [320, 250]]))
        toggles = {name: True for name in ('Mesh', 'Landmarks', 'Point IDs', 'Iris', 'EAR / MAR lines', 'Pose axes')}
        result = tracker.draw(frame, data, toggles)
        assert result.shape == (480, 640, 3)
        assert result.any()
    finally:
        tracker.close()
