import math
import pytest
from dms.core import Monitor, Calibration, Config, ear

OPEN = (.3, .3, .1, 0, 0)
CLOSED = (.1, .1, .1, 0, 0)


def advance(monitor, values, duration=2):
    result = None
    for i in range(int(duration * 10) + 1):
        result = monitor.update(i / 10, values)
    return result


def test_geometry_scale_invariant():
    points = [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)]
    assert ear(points, range(6)) == pytest.approx(.5)
    assert ear([(x * 50, y * 50) for x, y in points], range(6)) == pytest.approx(.5)


def test_blink_does_not_alert_and_sustained_closure_does():
    m = Monitor()
    assert not m.update(0, CLOSED)['alerts']
    assert not m.update(.2, CLOSED)['alerts']
    assert not m.update(.3, OPEN)['alerts']
    assert 'EYES CLOSED' in advance(Monitor(), CLOSED)['alerts']


def test_tracking_gap_breaks_continuity():
    m = Monitor()
    m.update(0, CLOSED)
    m.update(.5, CLOSED)
    assert not m.update(2, CLOSED)['alerts']
    assert m.update(2.1, None)['state'] == 'UNAVAILABLE'
    assert not m.update(2.2, CLOSED)['alerts']


def test_invalid_values_are_unavailable():
    assert Monitor().update(0, (.1, .1, math.nan, 0, 0))['state'] == 'UNAVAILABLE'


def test_one_eye_does_not_trigger():
    assert not advance(Monitor(), (.1, .3, .1, 0, 0))['alerts']


def test_yawn_count_once_and_pose_alert():
    m = Monitor()
    state = advance(m, (.3, .3, .8, 40, 0), 3)
    assert set(state['alerts']) == {'YAWN / MOUTH OPEN', 'LOOKING AWAY'}
    assert m.yawns == 1
    m.update(3.1, OPEN)
    for i in range(32, 53):
        m.update(i / 10, (.3, .3, .8, 0, 0))
    assert m.yawns == 2


def test_calibration_success_and_insufficient_samples():
    c, config = Calibration(), Config()
    c.begin(0)
    for i in range(51):
        result = c.add(i / 10, (.3, .3, .1, 8, -4), config)
    assert result.startswith('Calibrated')
    assert config.ear_threshold == pytest.approx(.216)
    assert c.baseline == (8, -4)
    c.begin(10)
    assert 'failed' in c.add(15, None, config)
    assert c.baseline == (8, -4)
