"""Observable cues and time-aligned fusion; expression never proves unsafe driving."""
from math import isfinite
from statistics import median
import numpy as np


def gaze(points, eyes_open=True):
    """Iris projection in each eye's own coordinate system, independent of head pose.

    Returns horizontal offsets from eye centers in eye-width units. Closed eyes,
    implausible iris geometry, or disagreement suppress the estimate. This is a
    calibrated gaze proxy, not a point of regard or validated attention measure.
    """
    if not eyes_open or len(points) < 478:
        return None
    offsets = []
    for a, b, iris in ((33, 133, 468), (362, 263, 473)):
        start, end, center = (np.asarray(points[i][:2], float) for i in (a, b, iris))
        axis = end - start
        if axis[0] < 0:
            axis = -axis
            start = end
        width = np.linalg.norm(axis)
        if not np.isfinite(width) or width < 4:
            return None
        fraction = float(np.dot(center - start, axis) / width**2)
        delta = center - start
        vertical = abs(float(axis[0]*delta[1] - axis[1]*delta[0])) / width**2
        if not isfinite(fraction) or not .05 <= fraction <= .95 or vertical > .2:
            return None
        offsets.append(fraction - .5)
    if abs(offsets[0] - offsets[1]) > .18:
        return None
    return sum(offsets) / 2


class GazeCalibration:
    def __init__(self):
        self.baseline = None
        self.samples = []

    def begin(self):
        self.samples = []

    def add(self, value):
        if value is not None and isfinite(value):
            self.samples.append(value)

    def finish(self, success):
        if success and len(self.samples) >= 30:
            self.baseline = median(self.samples)
            return True
        return False

    def correct(self, value):
        return None if value is None or self.baseline is None else value - self.baseline


def visibility(frame, data):
    """Conservative image-quality evidence, never a claim that an obstacle is identified."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    light, contrast = float(gray.mean()), float(gray.std())
    if light < 20 or light > 245 or contrast < 3:
        return 'CAMERA VIEW UNAVAILABLE', 'Image too dark, bright or uniform; obstruction is possible.'
    if data is None:
        return 'FACE NOT DETECTED', 'No reliable face; absence and full occlusion cannot be distinguished.'
    if not data['healthy']:
        return 'POSSIBLE PARTIAL OBSTRUCTION / POOR QUALITY', 'Framing or landmarks unreliable; partial obstruction is possible.'
    return 'MONITORING', 'Face geometry available; undetected occlusion is still possible.'


def fresh(cue, now, max_age=1.5, minimum=.6):
    if not cue:
        return None
    stamp, score = cue.get('timestamp'), cue.get('score')
    if not isinstance(stamp, (int, float)) or not isinstance(score, (int, float)):
        return None
    if not isfinite(stamp) or not isfinite(score) or not 0 <= now - stamp <= max_age or not minimum <= score <= 1:
        return None
    return cue


class Fusion:
    """Explicit temporal rules, with available/unknown modalities reported separately."""
    def __init__(self, gaze_seconds=2., gaze_limit=.12, lost_seconds=2.):
        self.gaze_seconds, self.gaze_limit, self.lost_seconds = gaze_seconds, gaze_limit, lost_seconds
        self.reset()

    def reset(self):
        self.last = self.gaze_start = self.lost_start = None
        self.expression_label = self.expression_start = self.expression_stamp = None

    def update(self, now, state, visibility_state='MONITORING', gaze_value=None, expression=None, behavior=None, paused=False):
        if self.last is not None and (now < self.last or now - self.last > .75):
            self.reset()
        self.last = now
        if paused:
            self.reset()
            return dict(alerts=[], sound=None, summary='Calibrating: fusion paused', gaze='Unavailable', modalities={})
        if visibility_state != 'MONITORING':
            self.gaze_start = self.expression_start = self.expression_label = self.expression_stamp = None
            if self.lost_start is None:
                self.lost_start = now
            return dict(alerts=[], sound='availability' if now-self.lost_start >= self.lost_seconds else None,
                        summary=visibility_state, gaze='Unavailable', modalities={})
        self.lost_start = None
        alerts = list(state['alerts'])
        valid_gaze = gaze_value is not None and isfinite(gaze_value)
        away = valid_gaze and abs(gaze_value) > self.gaze_limit
        if away:
            if self.gaze_start is None:
                self.gaze_start = now
            if now-self.gaze_start >= self.gaze_seconds:
                alerts.append('EYE GAZE AWAY')
        else:
            self.gaze_start = None
        exp, beh = fresh(expression, now), fresh(behavior, now, max_age=2.)
        if exp and exp.get('label') == self.expression_label:
            # A repeated cached prediction cannot establish temporal agreement.
            if exp['timestamp'] != self.expression_stamp:
                self.expression_stamp = exp['timestamp']
        else:
            self.expression_label = exp.get('label') if exp else None
            self.expression_start = exp['timestamp'] if exp else None
            self.expression_stamp = exp['timestamp'] if exp else None
        stable = bool(exp and self.expression_start is not None and self.expression_stamp-self.expression_start >= 1.)
        fatigue = any(x in alerts for x in ('EYES CLOSED', 'YAWN / MOUTH OPEN'))
        distraction = any(x in alerts for x in ('LOOKING AWAY', 'EYE GAZE AWAY'))
        summary = 'Sustained fatigue cues' if fatigue else 'Sustained distraction cues' if distraction else 'No sustained observable alerts'
        if stable:
            summary += f" · expression context: {exp['label']} (uncalibrated)"
        if beh:
            summary += f" · research model: {beh['label']} (supporting only)"
        return dict(alerts=alerts, sound='fatigue' if fatigue else 'distraction' if distraction else None,
                    summary=summary, gaze=('Uncalibrated / unavailable' if not valid_gaze else 'Away' if away else 'Forward range'),
                    modalities={'geometry': True, 'gaze': valid_gaze, 'expression': bool(exp), 'behavior': bool(beh)},
                    expression_stable=stable)
