# Capstone validation — 2 November 2026

Current branch: `feat/driveguard-capstone`, audited from main `a053155`.

- Full Python regression suite: **62 passed, 1 skipped**, 53.85 seconds. The skipped test requires optional PyTorch. Includes research split/leakage safeguards, hidden Tk rendering and view switching, fake camera failure/cleanup, gaze geometry, calibration, freshness/temporal fusion, distinct PCM sounds, CNN preprocessing and actual FER+ plus MediaPipe inference on scikit-image's public-domain astronaut image.
- Actual Python model smoke: astronaut fixture produced the `happiness` label and finite softmax score. One observed inference took 235.4 ms. This is a single static fixture and a single timing observation, not accuracy or sustained-FPS evaluation.
- Browser unit suite: **13 passed**. Production build passed. Browser integration checks and screenshot review are recorded in `web/VALIDATION.md`.
- FER+ model byte count and upstream SHA-256 verified. Model weights, camera fixtures and generated assets are excluded from Git. Historical lock cleaned to avoid overlapping OpenCV distributions; `requirements-test.txt` installs the research-test dependencies without requiring optional Torch.
- A restricted-process run hit a Windows invalid native handle while constructing MediaPipe. The full suite passed when run outside that restriction. The earlier run without research dependencies failed until scikit-learn was installed. These failures were not treated as successful checks.
- A Windows GitHub Actions workflow now runs Python, build and browser fixture checks. Remote CI results will be visible on the PR; local results do not establish that CI has passed.

**No actual webcam validation was performed.** Browser media inputs were fake or canvas-backed static images; native camera tests used mocks. Physical camera access/release, glasses and occlusion scenarios, speaker audibility, sustained performance, false-positive/false-negative rates across participants and lighting remain manual acceptance work. Native Tk was smoke-tested in a hidden window; the polished browser layout was screenshot-reviewed. No deployed Vercel URL was validated.

See `CAPSTONE_AUDIT.md` for conservative visibility semantics and model limitations. Do not report safety/accident-prevention effectiveness from these tests.

---

## Historical validation records (before this branch)

# Build-session validation — 31 October 2026

Environment inspected: empty Windows workspace; Python 3.13.15 x64 and Tk 8.6. A project-local `.venv` was created. Its ensurepip step failed, so host pip's `--python .venv` support installed the dependencies. Network downloads needed elevated execution permission in this environment. The official Google version 1 Face Landmarker model was downloaded to `models/face_landmarker.task` (3,758,596 bytes).

## Actual results

- `.venv\Scripts\python.exe -m pytest -q`: **13 passed in 3.44 s**, no skipped tests.
- Real MediaPipe FaceLandmarker initialization and blank-frame inference passed. Blank input returned no detected face.
- All mesh, iris, eye/mouth point, ID, measurement-line, and pose-axis drawing code ran against synthetic coordinates.
- Synthetic frontal and 30-degree rotated face projection tests passed. These establish consistency with the generic pose model, not real-human pose accuracy.
- Mock camera-open failure and capture failure both reported an error and released camera/tracker resources.
- A hidden native Tk window rendered a synthetic healthy dashboard, then correctly displayed stale and camera-error states. No visible-layout screenshot review was performed.
- `python -m pip --python .venv check`: **No broken requirements found.**
- Source compilation and CLI `--help` checks passed.

## Follow-up live launch

The application was subsequently launched on the user's laptop. A restricted launch produced `Cannot open webcam`; relaunching with desktop access resolved it. The live preview, eye/mouth/iris overlays, pose axes, measurements, and MONITORING dashboard were visually observed. The dashboard showed approximately 16 FPS at that moment; this is a single observation, not a sustained benchmark. No image or video was saved. The app was left running for the user to calibrate and test.

## Not tested

The original build-session tests did not open a real webcam; the follow-up live launch above adds basic hardware and visible-layout verification. Calibration accuracy, eye closure/yawn/head-turn alert accuracy, varied glasses and lighting behavior, sustained real-time FPS, camera indicator shutdown, and sound playback still require the laptop acceptance checklist in README.md. No detection-accuracy or sustained FPS benchmark is claimed.

The optional CNN adapter has no bundled pretrained weights and has not been validated with a real expression model. Its model/label/preprocessing contract is documented in README.md. Facial expressions never affect monitoring alerts.

The initial bootstrap installed both opencv-python and opencv-contrib-python at the same version; that exact environment is recorded in requirements-lock.txt. The normal requirements file asks for the contrib distribution only. Dependencies were not installed globally.

---

# Research-prototype upgrade session — 1 November 2026

Environment of this session: Linux sandbox, **1 CPU core, 3.9 GB RAM, no GPU, no network, no PyTorch, no ONNX package, no Tk, no UTA-RLDD videos**. Python 3.12.3, MediaPipe 0.10.33, OpenCV 4.13, NumPy 2.4, scikit-learn 1.8. (The original build was Windows/Python 3.13.) pytest was borrowed from the project's own `.venv` into a scratch path.

## What was run, with actual results
- Full test suite: **49 passed, 1 skipped** (`tests/test_gru_torch.py`, needs PyTorch) with `tests/test_ui.py` excluded. Before changes the same environment gave 12 passed + the UI test unrunnable. `tests/test_ui.py` **could not be run** here (no Tk); its packets do not include the new `learned` key and `app.py` reads it with `.get`, but this was not executed.
- New tests cover: participant leakage (manifest, windows, config), refusal to guess labels/layout, half-open window boundaries, no truncated windows, missing data excluded from PERCLOS denominators, censored blinks, gap handling, preprocessing fit-on-train-only, live-vs-offline feature parity, real-MediaPipe extractor-vs-live-path parity on decoded frames, model fallback (missing/corrupt/hash/version/inference crash), coverage/recent-loss/stale suppression, calibration labelling, truncated-cache reuse, export guards, worker keeps heuristics when the learned model is missing or crashes.
- `python -m research smoke`: end-to-end on **30 synthetic videos** (a public-domain still with noise, 12 s, 10 fps) through real MediaPipe extraction, manifest, heuristic replay, baseline training, val/test evaluation, export, and live-predictor replay. It passed and proves the stages connect. Extraction ran at roughly 36 frames/s on this single core (3,600 frames in about 101 s, small frames; real, larger videos will be slower). **Its accuracy numbers are meaningless and are not reported.**
- Measured on this sandbox with a 6 s window at 10 fps: `predictor.update` about 0.04 ms median, 0.7 ms max; one window's features+model about 0.6 ms. A 30 s window costs more; excludes MediaPipe/camera. Not a benchmark of your laptop.

## What was NOT done (remains untrained / unverified)
1. **No model has been trained on UTA-RLDD.** The dataset was unavailable. No real accuracy, macro-F1, calibration, participant variability, or heuristic-vs-learned comparison exists. `VALIDATION`-style claims about learned-state performance would be fabricated; none are made.
2. **GRU training code (`research/temporal.py`) has never been executed** (no PyTorch). It compiles; its export refuses to write a bundle unless torch and numpy forward passes agree; `tests/test_gru_torch.py` should be run on Colab/GPU before trusting it.
3. The real dataset layout, fold naming, and label source are unverified; the manifest builder requires you to declare them.
4. The Colab notebook has no outputs and was not run.
5. Full-dataset extraction time is unmeasured. Rough extrapolation from the sandbox rate: 180 videos x 10 min x 15 fps is about 1.6 M frames, on the order of 12 hours on one core of this class; use `--workers` and a Colab/cloud CPU.
6. The updated Tk UI panel, live webcam behaviour with a real bundle, and real-camera latency/FPS impact were not observed.
7. No event-level (blink/yawn/microsleep) accuracy, no automotive or safety validation, no fairness evaluation is claimed or possible with these labels.
