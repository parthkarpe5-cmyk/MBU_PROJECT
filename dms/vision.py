"""MediaPipe adapter and approximate monocular head pose."""
import cv2
import numpy as np
from .core import LEFT_EYE, RIGHT_EYE, MOUTH, measurements


def head_pose(points, width, height):
    ids = (1, 152, 33, 263, 61, 291)
    model = np.array([(0, 0, 0), (0, -330, -65), (-225, 170, -135),
                      (225, 170, -135), (-150, -150, -125), (150, -150, -125)], dtype=np.float64)
    image = np.array([points[i][:2] for i in ids], dtype=np.float64)
    camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1.]], dtype=np.float64)
    ok, rotation, translation = cv2.solvePnP(model, image, camera, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("Head pose could not be estimated")
    matrix, _ = cv2.Rodrigues(rotation)
    angles = cv2.RQDecomp3x3(matrix)[0]
    # Model coordinates point upward; neutral frontal rotation is around 180 degrees.
    pitch = (angles[0] + 180) % 360 - 180
    pitch = (pitch + 90) % 180 - 90
    axes, _ = cv2.projectPoints(np.float64([(100, 0, 0), (0, 100, 0), (0, 0, 100)]),
                                rotation, translation, camera, np.zeros(4))
    return float(angles[1]), float(pitch), float(angles[2]), axes.reshape(-1, 2)


class Tracker:
    def __init__(self, path):
        import mediapipe as mp
        self.mp = mp
        vision = mp.tasks.vision
        self.detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=.6, min_face_presence_confidence=.6,
            min_tracking_confidence=.6))
        self.connections = vision.FaceLandmarksConnections
        self.timestamp = -1

    def process(self, frame, now):
        self.timestamp = max(self.timestamp + 1, int(now * 1000))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.timestamp)
        if not result.face_landmarks:
            return None
        height, width = frame.shape[:2]
        points = np.array([(p.x * width, p.y * height, p.z * width) for p in result.face_landmarks[0]])
        if len(points) < 468 or not np.isfinite(points).all():
            return None
        try:
            yaw, pitch, roll, axes = head_pose(points, width, height)
        except (ValueError, cv2.error):
            # An unreliable pose must not terminate the camera or become a zero angle.
            yaw = pitch = roll = float('nan')
            axes = np.tile(points[1, :2], (3, 1))
        left, right, mar = measurements(points)
        inside = np.mean((points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height))
        size = np.ptp(points[:, 0]) / width
        light = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        healthy = inside > .95 and size > .15 and 35 < light < 235 and np.isfinite((left, right, mar, yaw, pitch)).all()
        return dict(points=points, values=(left, right, mar, yaw, pitch), roll=roll, axes=axes,
                    healthy=healthy, quality=f"In frame {inside:.0%} | face width {size:.0%} | brightness {light:.0f}")

    def draw(self, frame, data, toggles):
        points = data['points']
        def point(i):
            return tuple(np.clip(points[i, :2], -10000, 10000).astype(int))
        if toggles['Mesh']:
            for edge in self.connections.FACE_LANDMARKS_TESSELATION:
                cv2.line(frame, point(edge.start), point(edge.end), (80, 110, 80), 1, cv2.LINE_AA)
        if toggles['Landmarks']:
            for i in LEFT_EYE + RIGHT_EYE + MOUTH:
                cv2.circle(frame, point(i), 2, (0, 220, 255), -1)
                if toggles['Point IDs']:
                    cv2.putText(frame, str(i), point(i), cv2.FONT_HERSHEY_PLAIN, .7, (255, 255, 255), 1)
        if toggles['Iris'] and len(points) >= 478:
            for i in range(468, 478):
                cv2.circle(frame, point(i), 2, (255, 100, 255), -1)
        if toggles['EAR / MAR lines']:
            for a, b in ((33, 133), (160, 144), (158, 153), (362, 263), (385, 380), (387, 373), (13, 14), (78, 308)):
                cv2.line(frame, point(a), point(b), (255, 220, 0), 1, cv2.LINE_AA)
        if toggles['Pose axes']:
            for end, color, label in zip(data['axes'], ((0, 0, 255), (0, 255, 0), (255, 0, 0)), ('X', 'Y', 'Z')):
                end = tuple(np.clip(end, -10000, 10000).astype(int))
                cv2.line(frame, point(1), end, color, 2)
                cv2.putText(frame, label, end, cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        return frame

    def close(self):
        self.detector.close()
