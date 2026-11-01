# Driver Monitoring System — Windows webcam MVP

A local Python desktop lab with a live camera on the left and monitoring dashboard on the right. No video, images, landmark coordinates, or calibration data are saved or uploaded. Camera frames live in memory. Setup downloads packages and the official Google model; running the app needs no network connection.

## Run

From this folder in PowerShell, after setup:

```powershell
.\run.cmd
```

First-time setup (64-bit Windows, Python 3.13 with Tkinter):

```powershell
python -m venv .venv
python -m pip --python .venv install -r requirements.txt
python setup_model.py
.\run.cmd
```

If venv reports an `ensurepip` error but creates `.venv\Scripts\python.exe`, the next `python -m pip --python .venv ...` command can populate that environment using the host pip. A standard python.org install with pip and Tcl/Tk is recommended if either is missing.

Choose another camera with `.\run.cmd --camera 1`. Close other camera apps and enable **Settings → Privacy & security → Camera → Let desktop apps access your camera** if access fails. Missing model errors ask you to run `python setup_model.py`. For an offline machine, copy the model to `models/face_landmarker.task` and preinstall dependencies.

## Use

1. Test seated at a stationary laptop in even light, with one face centered and large enough in view.
2. Click **Calibrate** and keep eyes naturally open, mouth closed, head facing forward for five seconds. At least 30 healthy frames are required. Calibration sets EAR/MAR thresholds and neutral yaw/pitch; it stays in memory for this session.
3. Toggle mesh, eye/mouth landmarks, point IDs, iris points, EAR/MAR lines, head axes, mirrored preview, debug text, sound, and optional CNN. Mirroring affects the preview only; pose remains in camera coordinates. The axes' text also mirrors with the preview.
4. Adjust threshold fields and click **Apply thresholds**. The debug panel always shows active EAR/MAR thresholds, including calibration changes; entry fields represent manual settings and are not rewritten by calibration.
5. Click **Stop and close camera** to release the camera. The window waits for the camera worker to finish. A stalled camera driver may delay shutdown.

EAR = average of two vertical eye distances divided by horizontal width. Both eyes must be below the threshold. MAR = inner lip opening divided by mouth width. Head pose uses six landmarks, a generic 3D face, and an approximate camera matrix; yaw/pitch are relative to the calibrated neutral pose, roll is raw. Defaults: closure 1.5 s, mouth opening 1.5 s, head away 2 s; yaw 25°, pitch 20°.

Timers use monotonic wall time. Lost face, low framing/light quality, nonfinite readings, calibration, or gaps over 0.75 s break alert continuity. Results older than 1.5 s are marked stale. Very low FPS can therefore suppress sustained alerts rather than invent continuity. Health checks require over 95% of landmark points inside the image, face width over 15%, and mean image brightness between 35 and 235. These are transparent heuristics, not validated confidence scores. The API's configured detection/presence/tracking minima are 0.6; it does not expose per-frame confidence in this implementation.

Mouth opening is only a **yawn proxy**, and prolonged closure is only an observable cue. Talking, laughing, glare, glasses, occlusion, extreme pose, and poor light can cause false positives or missing detections. Head orientation does not establish gaze, attention, or driving ability. No gaze, phone-use, seatbelt, identity, or medical diagnosis is claimed. This is a research MVP, not a validated vehicle safety system. Do not test while driving.

## Optional pretrained expression CNN

Disabled by default; no expression weights are bundled or silently downloaded. Supply a pretrained ONNX facial-expression CNN whose documented contract is **NCHW float32 `[1,1,48,48]`, grayscale face crop scaled to `[0,1]`, output one logit per class**. Supply an ordered JSON array of class-name strings from that model's documentation. Models using different normalization, RGB, dimensions, alignment, or output probabilities require an adapter in `dms/expression.py`; arbitrary ONNX weights are not interchangeable. Verify licensing and provenance yourself.

```powershell
.\run.cmd --expression-model models\expression.onnx --expression-labels models\labels.json
```

Enable **Expression CNN** in the window. The CNN runs every tenth processed frame on the clean face crop. Its softmax score is uncalibrated and does not affect alerts. No actual pretrained expression model has been tested here.

## Verification

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Tests exercise camera-independent geometry, the research pipeline on synthetic caches, learned-model fallback and live/offline feature parity,, blink filtering, sustained closure, one-eye handling, missing data and timing gaps, nonfinite observations, yawn event counts, head-away timing, and calibration success/failure. See `VALIDATION.md` for actual checks from the build session.

Laptop acceptance checklist (requires your hardware): camera opens and closes; layout fits your display scaling; overlays follow face/eyes/mouth/irises; calibrate successfully; ordinary blinks do not alert; sustained closed eyes do; opening mouth for the configured time raises one event; calibrated head turns trigger after their delay; covering/leaving the camera produces unavailable readings; optional sound works; alternate camera index works; note FPS and false alarms with glasses and normal lighting. Confirm the camera indicator turns off after closing.

## Structure and references

`app.py`: native Tkinter interface, worker thread, bounded latest-frame queue, camera cleanup. `dms/core.py`: pure geometry, calibration, temporal alerts. `dms/vision.py`: MediaPipe Tasks and OpenCV pose/overlays. `dms/expression.py`: optional local ONNX inference. `setup_model.py`: explicit one-time model download. `requirements.txt`: compatible dependency ranges. `requirements-lock.txt`, when present, records the environment actually exercised.

Uses the current [official MediaPipe Python Face Landmarker API](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python) and [Google's version 1 model bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task). The app uses synchronous VIDEO inference on a background worker, increasing timestamps for each live camera frame. CPU inference is sufficient for the intended single-face, 640×480 MVP; achieved FPS depends on hardware.

## Learned driver-state model (research prototype, optional)

Separate from the heuristic monitor. Observed eye/mouth/head events (EAR/MAR/pose alerts) stay exactly as before; a *learned* model may additionally output a window-level class score (`alert` / `low_vigilance` / `drowsy`) trained on UTA-RLDD **video-level** labels. It is never an alarm, never changes heuristic alerts or sound, and facial expression is not used as fatigue evidence.

**No trained weights ship with this repository and none have been trained yet** (see `VALIDATION.md`). Without `models/learned/bundle.json` the app prints "Not active" in a *Learned model* panel and runs heuristics unchanged.

```powershell
.\run.cmd --learned-model models\learned        # default path; missing = heuristics only
```

The panel shows the learned class, whether the score is an **uncalibrated softmax score**, a validation-fit calibrated score, or a held-out-checked calibrated probability (only the last is called a probability), prediction age, and "Insufficient data" (no class shown) when tracking coverage in the rolling window is below the bundle's minimum (default 70% of the window, fixed a priori), when no valid face was seen for 2 s, or when the prediction is stale. Inference is numpy-only and offline; nothing is recorded or uploaded. Any load or inference error disables only the learned panel.

### Research pipeline (offline; `pip install -r requirements-research.txt`)

```powershell
copy configs\default.json configs\my_rldd.json      # then fill dataset.layout_regex and dataset.label_source (see DATASET.md)
python -m research --config configs\my_rldd.json --dataset-root D:\UTA-RLDD inspect
python -m research --config configs\my_rldd.json --dataset-root D:\UTA-RLDD manifest
python -m research --config configs\my_rldd.json verify-splits
python -m research --config configs\my_rldd.json --dataset-root D:\UTA-RLDD extract --workers 2   # resumable
python -m research --config configs\my_rldd.json heuristic                                         # existing monitor, where comparable
python -m research --config configs\my_rldd.json train-baseline --run base1                        # logistic regression on window features
python -m research --config configs\my_rldd.json train-gru --run gru1                              # needs PyTorch; resumable
python -m research --config configs\my_rldd.json evaluate --run base1 --split val
python -m research --config configs\my_rldd.json evaluate --run base1 --split test                 # deliberate, logged
python -m research --config configs\my_rldd.json export --run base1 --require-heldout              # -> models\learned
python -m research smoke --workdir smoke_out                                                       # SYNTHETIC plumbing check; directory must be new or empty
```

Outputs go to `artifacts/` (git-ignored): manifest, cached features, runs (bundle, selection, metrics, reports, logs, checkpoints). Raw videos are never copied. A Colab notebook with explicit paths and resumable checkpoints is in `notebooks/train_colab.ipynb`. Details: `DATASET.md`, `MODEL_CARD.md`.

Design guarantees: splits come only from official fold numbers and every participant sits in exactly one split (checked on manifest, windows, and in tests); training code never loads test videos; preprocessing is fit on train, hyperparameters/early stopping/temperature calibration use validation only; frame features come from the same `Tracker` and `dms/features.py` code as the live app; missing observations are excluded, never treated as closed eyes or zeros; `evaluate` scores through the deployed runtime path.

New files: `dms/features.py` (shared feature definitions), `dms/learned.py` (runtime), `research/` (pipeline), `configs/`, `notebooks/`.
