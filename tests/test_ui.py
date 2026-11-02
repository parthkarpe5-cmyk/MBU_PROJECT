"""Hidden Tk window smoke test with synthetic packets; no webcam access."""
from argparse import Namespace
import time
import tkinter as tk
import numpy as np
import app
from dms.core import Config


def test_dashboard_renders_and_marks_stale(monkeypatch):
    monkeypatch.setattr(app, 'worker', lambda *args: None)
    root = tk.Tk()
    root.withdraw()
    try:
        ui = app.App(root, Namespace())
        ui.thread.join(timeout=1)
        for mode, count in [('Camera only', 1), ('Dashboard only', 1), ('Split', 2)]:
            ui.view.set(mode)
            ui.set_view()
            assert len(ui.panes.panes()) == count
        ui.sound_mapping['distraction'].set('availability')
        ui.send_sound_mapping()
        assert ui.commands.get_nowait() == ('sound_mapping', {'distraction': 'availability', 'fatigue': 'fatigue', 'availability': 'availability'})
        packet = dict(frame=np.zeros((480, 640, 3), dtype=np.uint8),
                      values=(.3, .3, .1, 0, 0), data={'roll': 0, 'quality': 'Synthetic test'},
                      state={'state': 'MONITORING', 'alerts': [], 'durations': {}},
                      config=vars(Config()), fps=20, yawns=0, message='Test', expression='Disabled',
                      debug=True, updated=time.monotonic())
        packet['learned'] = 'LEARNED MODEL: fixture prediction'
        ui.output.put(packet)
        ui.poll()
        assert ui.status.cget('text') == 'MONITORING'
        assert 'Eyes: open' in ui.metrics.cget('text')
        assert 'fixture prediction' in ui.learned.cget('text')
        packet['updated'] -= 2
        ui.poll()
        assert ui.status.cget('text') == 'STALE / UNAVAILABLE'
        assert 'previous prediction invalid' in ui.learned.cget('text')
        ui.output.put({'error': 'Synthetic camera failure'})
        ui.poll()
        assert ui.status.cget('text') == 'MONITORING UNAVAILABLE'
        assert 'Synthetic camera failure' in ui.note.cget('text')
        ui.stop.set()
    finally:
        root.destroy()
