"""Camera-independent geometry, calibration and temporal logic."""
from dataclasses import dataclass
from statistics import median
from math import dist, isfinite

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH = (78, 82, 13, 312, 308, 317, 14, 87)


def ear(points, indices):
    a, b, c, d, e, f = (points[i][:2] for i in indices)
    return (dist(b, f) + dist(c, e)) / max(2 * dist(a, d), 1e-8)


def measurements(points):
    left, right = ear(points, LEFT_EYE), ear(points, RIGHT_EYE)
    mar = dist(points[13][:2], points[14][:2]) / max(dist(points[78][:2], points[308][:2]), 1e-8)
    return left, right, mar


@dataclass
class Config:
    ear_threshold: float = .21
    mar_threshold: float = .55
    eye_seconds: float = 1.5
    yawn_seconds: float = 1.5
    pose_seconds: float = 2.0
    yaw_limit: float = 25
    pitch_limit: float = 20


class Monitor:
    def __init__(self, config=None):
        self.config = config or Config()
        self.starts = {}
        self.last = None
        self.yawns = 0
        self.yawn_active = False

    def update(self, now, values=None):
        # Missing/late observations break continuity; never bridge lost tracking.
        if self.last is not None and (now - self.last > .75 or now < self.last):
            self.starts.clear()
            self.yawn_active = False
        self.last = now
        if values is None or not all(isfinite(v) for v in values):
            self.starts.clear()
            self.yawn_active = False
            return {"alerts": [], "durations": {}, "state": "UNAVAILABLE"}
        left, right, mar, yaw, pitch = values
        c = self.config
        flags = {"EYES CLOSED": max(left, right) < c.ear_threshold,
                 "YAWN / MOUTH OPEN": mar > c.mar_threshold,
                 "LOOKING AWAY": abs(yaw) > c.yaw_limit or abs(pitch) > c.pitch_limit}
        limits = dict(zip(flags, (c.eye_seconds, c.yawn_seconds, c.pose_seconds)))
        durations, alerts = {}, []
        for name, active in flags.items():
            if active:
                self.starts.setdefault(name, now)
                durations[name] = max(0, now - self.starts[name])
                if durations[name] >= limits[name]:
                    alerts.append(name)
            else:
                self.starts.pop(name, None)
                durations[name] = 0.
        yawn = "YAWN / MOUTH OPEN" in alerts
        if yawn and not self.yawn_active:
            self.yawns += 1
        self.yawn_active = yawn
        return {"alerts": alerts, "durations": durations, "state": "ALERT" if alerts else "MONITORING"}


class Calibration:
    def __init__(self):
        self.samples = []
        self.started = None
        self.baseline = (0., 0.)

    def begin(self, now):
        self.started = now
        self.samples = []

    def add(self, now, values, config):
        if self.started is None:
            return ""
        if values is not None:
            self.samples.append(values)
        if now - self.started < 5:
            return f"Calibrating: {len(self.samples)} valid frames; face forward, eyes open, mouth closed"
        self.started = None
        if len(self.samples) < 30:
            return "Calibration failed: need 30 valid frames in 5 seconds. Retry in good light."
        eyes = median((v[0] + v[1]) / 2 for v in self.samples)
        mouth = median(v[2] for v in self.samples)
        config.ear_threshold = max(.08, min(.35, eyes * .72))
        config.mar_threshold = max(.35, min(.9, mouth + .25))
        self.baseline = tuple(median(v[i] for v in self.samples) for i in (3, 4))
        return "Calibrated: neutral head pose and eye/mouth thresholds applied"
