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
    cap = tracker = None
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
        toggles.update({'Mesh': False, 'Point IDs': False, 'Mirror': True, 'Sound': False, 'Expression CNN': False,
                        'Learned model': True})
        cnn = None
        cnn_error = ''
        expression = 'Disabled (optional local ONNX model)'
        if args.expression_model:
            try:
                from dms.expression import ExpressionCNN
                if not args.expression_labels:
                    raise ValueError('--expression-labels is required with --expression-model')
                cnn = ExpressionCNN(args.expression_model, args.expression_labels)
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
        previous, last_beep, frame_number = time.monotonic(), 0., 0
        while not stop.is_set():
            now = time.monotonic()
            while True:
                try:
                    command, value = commands.get_nowait()
                except queue.Empty:
                    break
                if command == 'toggles':
                    toggles = value
                elif command == 'calibrate':
                    calibration.begin(now)
                    monitor.starts.clear()
                elif command == 'config':
                    for name, number in value.items():
                        setattr(monitor.config, name, number)
                    monitor.starts.clear()
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
            raw = data['values'] if data and data['healthy'] else None
            note = calibration.add(now, raw, monitor.config)
            if note:
                message = note
            values = None
            if raw is not None and calibration.started is None:
                values = (*raw[:3], raw[3] - calibration.baseline[0], raw[4] - calibration.baseline[1])
            state = monitor.update(now, values)
            if data:
                if cnn and toggles['Expression CNN'] and frame_number % 10 == 0 and data['healthy']:
                    try:
                        expression = cnn.predict(frame, data['points'])
                    except Exception as exc:
                        expression = f'CNN unavailable: {exc}'
                tracker.draw(frame, data, toggles)
            if not data or not data['healthy']:
                expression = 'Unavailable: no healthy face'
            elif not cnn:
                expression = cnn_error or 'No CNN supplied (see README)'
            fps = 1 / max(time.monotonic() - previous, .001)
            previous = time.monotonic()
            if state['alerts'] and toggles['Sound'] and now - last_beep > 3:
                import winsound
                winsound.PlaySound('SystemExclamation', winsound.SND_ALIAS | winsound.SND_ASYNC)
                last_beep = now
            if toggles['Mirror']:
                frame = cv2.flip(frame, 1)
            publish(dict(frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), data=data, values=values,
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


class App:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.commands, self.output, self.stop = queue.Queue(), queue.Queue(maxsize=1), threading.Event()
        self.last_packet = None
        root.title('Driver Monitor | Local webcam lab')
        root.geometry('1250x820')
        root.minsize(1050, 740)
        style = ttk.Style()
        style.theme_use('clam')
        root.configure(bg='#101a29')
        style.configure('TFrame', background='#101a29')
        style.configure('TLabel', background='#101a29', foreground='#e4edf7', font=('Segoe UI', 10))
        style.configure('TCheckbutton', background='#101a29', foreground='#e4edf7')
        header = ttk.Frame(root, padding=15)
        header.pack(fill='x')
        ttk.Label(header, text='DRIVER MONITOR  /  LOCAL LAB', font=('Segoe UI', 19, 'bold')).pack(anchor='w')
        ttk.Label(header, text='Webcam research prototype • No recording or cloud upload • Stationary testing only').pack(anchor='w')
        body = ttk.Frame(root, padding=12)
        body.pack(fill='both', expand=True)
        left = ttk.Frame(body)
        left.pack(side='left', fill='both', expand=True)
        dashboard = ttk.Frame(body, width=480)
        dashboard.pack(side='right', fill='y')
        dashboard.pack_propagate(False)
        canvas = tk.Canvas(dashboard, bg='#101a29', highlightthickness=0)
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
                    ('Mirror', True), ('Debug', True), ('Sound', False), ('Expression CNN', False),
                    ('Learned model', True))):
            var = tk.BooleanVar(value=enabled)
            self.toggles[name] = var
            ttk.Checkbutton(controls, text=name, variable=var, command=self.send_toggles).grid(row=i // 3, column=i % 3, sticky='w', padx=5, pady=3)
        ttk.Button(left, text='Calibrate — 5 seconds facing forward', command=lambda: self.commands.put(('calibrate', None))).pack(fill='x', pady=10)
        self.status = ttk.Label(right, text='INITIALIZING', font=('Segoe UI', 21, 'bold'))
        self.status.pack(anchor='w', pady=8)
        self.metrics = ttk.Label(right, text='Waiting for first observation…', justify='left', font=('Consolas', 11))
        self.metrics.pack(anchor='w', pady=8)
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
        ttk.Button(right, text='Stop and close camera', command=self.close).pack(anchor='w')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.thread = threading.Thread(target=worker, args=(args, self.commands, self.output, self.stop), daemon=True)
        self.thread.start()
        self.poll()

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
                self.status.configure(text='MONITORING UNAVAILABLE', foreground='#ff907d')
                self.metrics.configure(text='No current measurements')
                self.note.configure(text=packet['error'])
                self.learned.configure(text='LEARNED MODEL: unavailable (no camera data)')
                self.debug.configure(text='Correct the issue and restart. No alerts are available.')
                self.last_packet = None
            else:
                from PIL import Image, ImageTk
                self.last_packet = packet
                image = Image.fromarray(packet['frame'])
                image.thumbnail((max(320, self.video.winfo_width()), max(240, self.video.winfo_height())))
                self.photo = ImageTk.PhotoImage(image)
                self.video.configure(image=self.photo, text='')
                state = packet['state']
                self.status.configure(text=state['state'], foreground='#80dcc5' if state['state'] == 'MONITORING' else '#ff907d')
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
                text += '\n\nExpression: ' + packet['expression']
                self.metrics.configure(text=text)
                self.learned.configure(text=packet.get('learned', 'LEARNED MODEL: not available'))
                self.note.configure(text=packet['message'])
                c = packet['config']
                explanation = (packet['data']['quality'] if packet['data'] else 'No face detected')
                explanation += f"\nActive EAR < {c['ear_threshold']:.3f}; MAR > {c['mar_threshold']:.3f}\n"
                explanation += '\n'.join(f'{k}: {v:.1f}s' for k, v in state['durations'].items())
                self.debug.configure(text=explanation if packet['debug'] else '')
        if self.last_packet and time.monotonic() - self.last_packet['updated'] > 1.5:
            self.status.configure(text='STALE / UNAVAILABLE', foreground='#ff907d')
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
    parser.add_argument('--expression-model', type=Path)
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
