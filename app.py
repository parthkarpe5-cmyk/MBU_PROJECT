"""Run with python app.py; no network activity at runtime."""
import argparse
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parent


def learned_text(format_status, status):
    """Dashboard text for the learned model; never raises."""
    try:
        if format_status is None:
            return 'LEARNED MODEL: not active: ' + status.get('message', '')
        return format_status(status, time.monotonic())
    except Exception as exc:
        return f'LEARNED MODEL: display error {type(exc).__name__}: {exc}'


def worker(args, commands, output, stop):
    import cv2
    from dms.core import Monitor, Calibration
    from dms.vision import Tracker
    from dms.multimodal import Fusion, GazeCalibration, gaze, visibility, fresh
    cap = tracker = sounds = None
    def publish(value):
        try:
            output.get_nowait()
        except queue.Empty:
            pass
        output.put(value)
    try:
        tracker = Tracker(args.model)
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam. Close Teams/Zoom; enable Windows camera access; try --camera 1.')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        monitor, calibration = Monitor(), Calibration()
        toggles = {name: True for name in ('Landmarks', 'Iris', 'EAR / MAR lines', 'Pose axes', 'Debug')}
        toggles.update({'Mesh': False, 'Point IDs': False, 'Mirror': True, 'Sound': False, 'Expression CNN': True,
                        'Learned model': True})
        fusion, gaze_cal = Fusion(), GazeCalibration()
        sound_mapping = {}
        expression_cue = None
        cnn = None
        cnn_error = ''
        expression = 'Disabled (optional local ONNX model)'
        if getattr(args, 'expression_model', None):
            try:
                from dms.expression import ExpressionCNN
                cnn = ExpressionCNN(args.expression_model, getattr(args, 'expression_labels', None),
                                    getattr(args, 'expression_contract', 'ferplus'))
            except Exception as exc:
                cnn_error = f'CNN unavailable: {exc}'
        # Optional trained model, fully isolated: any failure here leaves heuristics running.
        learned = frame_record = format_status = None
        learned_status = {'state': 'disabled', 'message': 'Learned model code unavailable'}
        try:
            from dms.features import frame_record
            from dms.learned import LearnedStatePredictor, format_status
            learned = LearnedStatePredictor.from_path(getattr(args, 'learned_model', None),
                                                      max_age_seconds=getattr(args, 'learned_max_age', None))
        except Exception as exc:
            learned_status = {'state': 'error', 'message': f'{type(exc).__name__}: {exc}'}
        message = 'Uncalibrated: use Calibrate before interpreting results'
        previous, frame_number = time.monotonic(), 0
        while not stop.is_set():
            now = time.monotonic()
            while True:
                try:
                    command, value = commands.get_nowait()
                except queue.Empty:
                    break
                if command == 'toggles':
                    toggles = value
                    if not toggles.get('Expression CNN'):
                        expression_cue = None
                elif command == 'calibrate':
                    calibration.begin(now)
                    gaze_cal.begin()
                    fusion.reset()
                    expression_cue = None
                    monitor.starts.clear()
                elif command == 'sound_mapping':
                    sound_mapping = value
                elif command == 'preview_sound':
                    try:
                        if sounds is None:
                            from dms.sounds import SoundPlayer
                            sounds = SoundPlayer()
                        sounds.play(value, now, sound_mapping, preview=True)
                    except Exception as exc:
                        message = f'Sound unavailable: {exc}'
                elif command == 'config':
                    for name, number in value.items():
                        setattr(monitor.config, name, number)
                    monitor.starts.clear()
                    fusion.reset()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera stopped delivering frames. Reconnect it and restart the app.')
            data = tracker.process(frame, now)
            if learned is not None:
                try:
                    learned.enabled = toggles.get('Learned model', True)
                    learned_status = learned.update(now, frame_record(data))
                except Exception as exc:
                    learned, learned_status = None, {'state': 'error', 'message': f'{type(exc).__name__}: {exc}'}
            visibility_state, visibility_detail = visibility(frame, data)
            raw = data['values'] if data and data['healthy'] and visibility_state == 'MONITORING' else None
            gaze_raw = gaze(data['points'], raw is not None and min(raw[:2]) > monitor.config.ear_threshold) if data else None
            was_calibrating = calibration.started is not None
            if was_calibrating:
                gaze_cal.add(gaze_raw)
            note = calibration.add(now, raw, monitor.config)
            if note:
                message = note
                if was_calibrating and calibration.started is None:
                    success = note.startswith('Calibrated:')
                    if not gaze_cal.finish(success):
                        message += ' · Gaze calibration unavailable; retry with both eyes visible.'
            values = None
            if raw is not None and calibration.started is None:
                values = (*raw[:3], raw[3] - calibration.baseline[0], raw[4] - calibration.baseline[1])
            state = monitor.update(now, values)
            if data:
                if cnn and toggles['Expression CNN'] and frame_number % 10 == 0 and raw is not None:
                    try:
                        expression_cue = cnn.infer(frame, data['points'], now)
                    except Exception as exc:
                        expression_cue = None
                        expression = f'CNN unavailable: {exc}'
                tracker.draw(frame, data, toggles)
            if not data or not data['healthy']:
                expression = 'Unavailable: no healthy face'
            elif not cnn:
                expression = cnn_error or 'No CNN supplied (see README)'
            if not raw or not toggles['Expression CNN']:
                expression_cue = None
            current_expression = fresh(expression_cue, now)
            if current_expression:
                expression = f"{current_expression['label']} ({current_expression['score']:.0%}, uncalibrated CNN score)"
            elif cnn and raw and toggles['Expression CNN']:
                expression = 'Uncertain / stale / awaiting CNN prediction'
            behavior = None
            pred = learned_status.get('prediction') if learned_status.get('state') == 'ok' else None
            if pred:
                behavior = dict(label=pred['label'], score=pred['probs'][pred['label']], timestamp=pred['window_end'])
            fused = fusion.update(now, state, visibility_state, gaze_cal.correct(gaze_raw), current_expression, behavior,
                                  paused=was_calibrating)
            state['alerts'] = fused['alerts']
            if was_calibrating:
                state['state'] = 'CALIBRATING'
            elif visibility_state != 'MONITORING':
                state['state'] = visibility_state
            elif state['alerts']:
                state['state'] = 'ALERT'
            fps = 1 / max(time.monotonic() - previous, .001)
            previous = time.monotonic()
            if fused['sound'] and toggles['Sound']:
                try:
                    if sounds is None:
                        from dms.sounds import SoundPlayer
                        sounds = SoundPlayer()
                    sounds.play(fused['sound'], now, sound_mapping)
                except Exception as exc:
                    message = f'Sound unavailable: {exc}'
            if toggles['Mirror']:
                frame = cv2.flip(frame, 1)
            publish(dict(frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), data=data, values=values,
                         fusion=fused, visibility=visibility_detail, gaze=gaze_cal.correct(gaze_raw),
                         state=state, fps=fps, message=message, yawns=monitor.yawns,
                         config=vars(monitor.config).copy(), expression=expression if toggles['Expression CNN'] else 'Disabled',
                         debug=toggles['Debug'], updated=time.monotonic(),
                         learned=learned_text(format_status, learned_status)))
            frame_number += 1
    except Exception as exc:
        publish({'error': f'{type(exc).__name__}: {exc}'})
    finally:
        if cap is not None:
            cap.release()
        if tracker is not None:
            tracker.close()
        if sounds is not None:
            sounds.close()


class App:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.commands, self.output, self.stop = queue.Queue(), queue.Queue(maxsize=1), threading.Event()
        self.last_packet = None
        root.title('DriveGuard AI | Intelligent Driver Monitoring')
        root.geometry('1400x900')
        root.minsize(900, 650)
        style = ttk.Style()
        style.theme_use('clam')
        root.configure(bg='#edf1f6')
        style.configure('TFrame', background='#edf1f6')
        style.configure('TLabel', background='#edf1f6', foreground='#23334b', font=('Segoe UI', 10))
        style.configure('TCheckbutton', background='#edf1f6', foreground='#23334b')
        style.configure('TButton', padding=(12, 9), background='#e3eaf3', foreground='#23334b', borderwidth=1)
        style.map('TButton', background=[('active', '#d4e2f6')])
        style.configure('Card.TFrame', background='#f8fafd', relief='raised', borderwidth=2)
        style.configure('TLabelframe', background='#edf1f6', borderwidth=1)
        style.configure('TLabelframe.Label', background='#edf1f6', foreground='#356ac3')
        header = ttk.Frame(root, padding=15)
        header.pack(fill='x')
        ttk.Label(header, text='DriveGuard AI', font=('Segoe UI', 19, 'bold')).pack(anchor='w')
        ttk.Label(header, text='Webcam research prototype • No recording or cloud upload • Stationary testing only').pack(anchor='w')
        body = ttk.Frame(root, padding=12)
        body.pack(fill='both', expand=True)
        self.panes = ttk.Panedwindow(body, orient='horizontal')
        self.panes.pack(fill='both', expand=True)
        left = ttk.Frame(self.panes, style='Card.TFrame', padding=12)
        dashboard = ttk.Frame(self.panes, width=490, style='Card.TFrame', padding=8)
        self.left, self.dashboard = left, dashboard
        self.panes.add(left, weight=3)
        self.panes.add(dashboard, weight=2)
        self.view = tk.StringVar(value='Split')
        viewbar = ttk.Frame(header)
        viewbar.pack(anchor='e')
        for mode in ('Camera only', 'Dashboard only', 'Split'):
            ttk.Radiobutton(viewbar, text=mode, value=mode, variable=self.view, command=self.set_view).pack(side='left', padx=6)
        ttk.Button(viewbar, text='Fullscreen · F11', command=self.fullscreen).pack(side='left', padx=8)
        ttk.Button(viewbar, text='Stop camera', command=self.close).pack(side='left')
        root.bind('<F11>', lambda e: self.fullscreen())
        root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))
        canvas = tk.Canvas(dashboard, bg='#edf1f6', highlightthickness=0)
        scrollbar = ttk.Scrollbar(dashboard, orient='vertical', command=canvas.yview)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        right = ttk.Frame(canvas, padding=(20, 0))
        interior = canvas.create_window((0, 0), window=right, anchor='nw')
        right.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda event: canvas.itemconfigure(interior, width=event.width))
        def scroll_dashboard(event):
            if str(event.widget).startswith(str(dashboard)):
                canvas.yview_scroll(-int(event.delta / 120), 'units')
        root.bind('<MouseWheel>', scroll_dashboard)
        self.video = ttk.Label(left, text='Starting camera…', anchor='center')
        self.video.pack(fill='both', expand=True)
        ttk.Label(left, text='Cyan: EAR/MAR measurements | Yellow: eye/mouth points\nMagenta: iris | Axes: X red, Y green, Z blue (camera coordinates)').pack(anchor='w', pady=8)
        controls = ttk.Frame(left)
        controls.pack(fill='x')
        self.toggles = {}
        for i, (name, enabled) in enumerate((('Mesh', False), ('Landmarks', True), ('Iris', True),
                    ('EAR / MAR lines', True), ('Pose axes', True), ('Point IDs', False),
                    ('Mirror', True), ('Debug', True), ('Sound', False), ('Expression CNN', True),
                    ('Learned model', True))):
            var = tk.BooleanVar(value=enabled)
            self.toggles[name] = var
            ttk.Checkbutton(controls, text=name, variable=var, command=self.send_toggles).grid(row=i // 3, column=i % 3, sticky='w', padx=5, pady=3)
        ttk.Button(left, text='Calibrate — 5 seconds facing forward', command=lambda: self.commands.put(('calibrate', None))).pack(fill='x', pady=10)
        self.status = ttk.Label(right, text='INITIALIZING', wraplength=430, font=('Segoe UI', 20, 'bold'))
        self.status.pack(anchor='w', pady=8)
        self.metrics = ttk.Label(right, text='Waiting for first observation…', wraplength=430, justify='left', font=('Consolas', 11))
        self.metrics.pack(anchor='w', pady=8)
        self.fusion = ttk.Label(right, text='MULTIMODAL FUSION · Waiting', wraplength=430, foreground='#356ac3', font=('Segoe UI', 11, 'bold'))
        self.fusion.pack(anchor='w', pady=12)
        self.learned = ttk.Label(right, text='LEARNED MODEL: waiting', wraplength=430, justify='left')
        self.learned.pack(anchor='w', pady=8)
        self.note = ttk.Label(right, wraplength=430, justify='left')
        self.note.pack(anchor='w', pady=8)
        self.debug = ttk.Label(right, wraplength=430, justify='left')
        self.debug.pack(anchor='w', pady=8)
        self.fields = {}
        settings = ttk.Frame(right)
        settings.pack(anchor='w', pady=8)
        for i, (key, label, default) in enumerate((('ear_threshold', 'EAR below', .21), ('mar_threshold', 'MAR above', .55),
                    ('eye_seconds', 'Eye closure seconds', 1.5), ('yawn_seconds', 'Mouth open seconds', 1.5),
                    ('pose_seconds', 'Look away seconds', 2), ('yaw_limit', 'Yaw limit degrees', 25), ('pitch_limit', 'Pitch limit degrees', 20))):
            ttk.Label(settings, text=label).grid(row=i, column=0, sticky='w')
            var = tk.StringVar(value=str(default))
            self.fields[key] = var
            ttk.Entry(settings, textvariable=var, width=8).grid(row=i, column=1, padx=10, pady=2)
        ttk.Button(right, text='Apply thresholds', command=self.apply).pack(anchor='w')
        ttk.Label(right, text='Confidence: model scores unavailable from this API.\nHealth uses framing, face size, light, and freshness.\nMouth opening is a yawn proxy; expression is not fatigue.', wraplength=430).pack(anchor='w', pady=12)
        sounds = ttk.LabelFrame(right, text='Alert sounds · event → pattern', padding=10)
        sounds.pack(fill='x', pady=10)
        self.sound_mapping = {}
        for row, event in enumerate(('distraction', 'fatigue', 'availability')):
            ttk.Label(sounds, text=event.capitalize()).grid(row=row, column=0, sticky='w')
            choice = tk.StringVar(value=event)
            self.sound_mapping[event] = choice
            box = ttk.Combobox(sounds, textvariable=choice, values=('distraction', 'fatigue', 'availability', 'Mute'), state='readonly', width=13)
            box.grid(row=row, column=1, padx=6, pady=4)
            box.bind('<<ComboboxSelected>>', lambda e: self.send_sound_mapping())
            ttk.Button(sounds, text='Preview', command=lambda event=event: self.commands.put(('preview_sound', event))).grid(row=row, column=2)
        ttk.Button(right, text='Stop and close camera', command=self.close).pack(anchor='w')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.thread = threading.Thread(target=worker, args=(args, self.commands, self.output, self.stop), daemon=True)
        self.thread.start()
        self.poll()

    def set_view(self):
        for child in tuple(self.panes.panes()):
            self.panes.forget(child)
        if self.view.get() != 'Dashboard only':
            self.panes.add(self.left, weight=3)
        if self.view.get() != 'Camera only':
            self.panes.add(self.dashboard, weight=2)

    def fullscreen(self):
        self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen'))

    def send_sound_mapping(self):
        self.commands.put(('sound_mapping', {k: v.get() for k, v in self.sound_mapping.items()}))

    def send_toggles(self):
        self.commands.put(('toggles', {k: v.get() for k, v in self.toggles.items()}))

    def apply(self):
        try:
            values = {k: float(v.get()) for k, v in self.fields.items()}
            import math
            if not all(math.isfinite(v) and v > 0 for v in values.values()):
                raise ValueError('Use positive finite numbers')
            if values['ear_threshold'] > 1 or values['mar_threshold'] > 2 or max(values['yaw_limit'], values['pitch_limit']) > 90:
                raise ValueError('EAR <= 1, MAR <= 2, angle limits <= 90')
            self.commands.put(('config', values))
        except ValueError as exc:
            self.note.configure(text=f'Invalid settings: {exc}')

    def poll(self):
        try:
            packet = self.output.get_nowait()
        except queue.Empty:
            packet = None
        if packet:
            if 'error' in packet:
                self.status.configure(text='MONITORING UNAVAILABLE', foreground='#af3c29')
                self.metrics.configure(text='No current measurements')
                self.note.configure(text=packet['error'])
                self.learned.configure(text='LEARNED MODEL: unavailable (no camera data)')
                self.debug.configure(text='Correct the issue and restart. No alerts are available.')
                self.fusion.configure(text='MULTIMODAL FUSION · Unavailable')
                self.last_packet = None
            else:
                from PIL import Image, ImageTk
                self.last_packet = packet
                image = Image.fromarray(packet['frame'])
                image.thumbnail((max(320, self.video.winfo_width()), max(240, self.video.winfo_height())))
                self.photo = ImageTk.PhotoImage(image)
                self.video.configure(image=self.photo, text='')
                state = packet['state']
                self.status.configure(text=state['state'], foreground='#087f75' if state['state'] == 'MONITORING' else '#af3c29')
                values = packet['values']
                text = f"Processing FPS: {packet['fps']:.1f}\nYawn proxy events: {packet['yawns']}\n"
                if values:
                    l, r, mar, yaw, pitch = values
                    text += f'EAR L / R: {l:.3f} / {r:.3f}\nMAR: {mar:.3f}\nYaw: {yaw:+.1f}°  Pitch: {pitch:+.1f}°\nRoll: {packet["data"]["roll"]:+.1f}°\n'
                    c = packet['config']
                    text += 'Eyes: ' + ('closed' if max(l, r) < c['ear_threshold'] else 'open') + '\n'
                    text += 'Mouth: ' + ('open / yawn candidate' if mar > c['mar_threshold'] else 'neutral') + '\n'
                    text += 'Head: ' + ('outside forward range' if abs(yaw) > c['yaw_limit'] or abs(pitch) > c['pitch_limit'] else 'forward range') + '\n'
                else:
                    text += 'Eye / mouth / pose: unavailable\n'
                text += '\n' + ('\n'.join(state['alerts']) or 'No sustained alerts')
                fused = packet.get('fusion', {})
                text += '\nGaze: ' + fused.get('gaze', 'Unavailable')
                text += '\n\nExpression: ' + packet['expression']
                self.fusion.configure(text='MULTIMODAL FUSION\n' + fused.get('summary', 'Waiting'))
                self.metrics.configure(text=text)
                self.learned.configure(text=packet.get('learned', 'LEARNED MODEL: not available'))
                self.note.configure(text=packet['message'])
                c = packet['config']
                explanation = (packet['data']['quality'] if packet['data'] else 'No face detected')
                explanation += f"\nActive EAR < {c['ear_threshold']:.3f}; MAR > {c['mar_threshold']:.3f}\n"
                explanation += '\n'.join(f'{k}: {v:.1f}s' for k, v in state['durations'].items())
                self.debug.configure(text=explanation if packet['debug'] else '')
        if self.last_packet and time.monotonic() - self.last_packet['updated'] > 1.5:
            self.status.configure(text='STALE / UNAVAILABLE', foreground='#af3c29')
            self.fusion.configure(text='MULTIMODAL FUSION · Stale, previous cues invalid')
            self.metrics.configure(text='Waiting for fresh camera data; previous results invalid')
            self.learned.configure(text='LEARNED MODEL: previous prediction invalid (stale camera data)')
            self.debug.configure(text='No current health or alert timing measurements')
        if not self.stop.is_set():
            self.root.after(40, self.poll)

    def close(self):
        self.stop.set()
        self.status.configure(text='Stopping camera…')
        self.finish_close()

    def finish_close(self):
        if self.thread.is_alive():
            self.root.after(100, self.finish_close)
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', type=Path, default=ROOT / 'models' / 'face_landmarker.task')
    parser.add_argument('--expression-model', type=Path, default=ROOT / 'models' / 'emotion-ferplus-8.onnx')
    parser.add_argument('--expression-contract', choices=('ferplus', 'legacy48'), default='ferplus')
    parser.add_argument('--expression-labels', type=Path)
    parser.add_argument('--learned-model', type=Path, default=ROOT / 'models' / 'learned',
                        help='Directory with bundle.json + weights.npz from the research pipeline. '
                             'Missing = heuristics only. Runs fully offline.')
    parser.add_argument('--learned-max-age', type=float, default=None,
                        help='Seconds after which a learned prediction is treated as stale (default 2x stride + 1).')
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error('Face model missing. Run python setup_model.py once (requires internet).')
    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == '__main__':
    main()
