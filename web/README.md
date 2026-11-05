# DriveGuard AI browser monitor

Capstone upgrade: FER+ CNN inference via local ONNX Runtime WASM, calibrated iris gaze independent of pose, time-aligned expression context, conservative visibility checks, three assignable alert patterns, and a light Neumorphic dashboard with camera/dashboard/split/fullscreen views. See [the audit](../CAPSTONE_AUDIT.md).

A Vite website that uses MediaPipe Face Landmarker in the browser. No Python server, Docker, API key, database, camera upload, or recording is required. Webcam access starts after the user clicks Start and grants browser permission. Stop, a hidden tab, page exit, or a processing error releases the camera. Thresholds, sound assignments/mute, view and split size are saved in local browser storage. Calibration pose/gaze baselines remain session-only. Reset to defaults removes this app's saved preferences and invalidates calibration; blocked storage falls back to session-only controls.

## Session summary and local reports

Stopping an active session shows duration, sustained alert event counts by type,
brief observed blink count, mouth-open/yawn-proxy count, average/minimum processing
FPS, and average synchronous processing time. Counts increase once per new alert
episode, including monitoring-unavailable/stale events, rather than per frame or
repeated sound. Blink counts cover observed 60–450 ms closures; tracking gaps and
calibration break continuity. These are descriptive counters, not validated event
labels. Failed camera startup does not create a session report.

The small live readout shows processing FPS and synchronous milliseconds per
processed frame. Summary average FPS is the number of positive inter-frame
intervals divided by their total elapsed time; minimum FPS is the lowest reciprocal
interval. With fewer than two processed frames these values are unavailable.
Processing ms covers synchronous face analysis, expression-input preparation,
overlays and UI updates. It excludes asynchronous expression inference, camera
buffering and display latency. These figures are not end-to-end latency or a
hardware benchmark.

Select JSON or CSV and click **Save report** after stopping. This generates a
browser download containing aggregate statistics and session start/end times.
Export makes no network requests and does not upload anything. No frame data,
landmarks, expression history, or video is saved. Reports stay in memory until a
new session or page reload and are never automatically persisted. Normal page
and model loading still fetches the existing application assets.

The **About / Limitations** panel describes this research prototype and its
unvalidated recognition across people. Expression context never triggers an
alert alone. Camera-denied, absent-camera and model-load errors show actionable
messages and **Retry**. An expression-model load failure keeps the camera and
geometry monitoring active; retry reloads that model without restarting the camera.

## Run locally

Use Node 24.x (the root package pins this major version for Vercel). From the repository root:

```sh
npm ci
npm run dev
```

Open the printed localhost address. Production build from the repository root:

```sh
npm run build
```

This emits **root `dist/`**. The root build command is deliberately compatible with a Vercel project configured to use the repository root. Alternatively, `npm run build --workspace web` emits **`web/dist/`** for a Vercel project whose Root Directory is `web`.

## Vercel settings — recommended

Import `parthkarpe5-cmyk/MBU_PROJECT`, select the latest main commit, and use:

| Setting | Value |
|---|---|
| Root Directory | Repository root (`./`) |
| Framework / Application Preset | Vite |
| Install Command | `npm ci` (existing `npm install` override also works) |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Node version | 24.x |
| Environment variables | None |

Root `vercel.json` supplies these commands. The earlier Vercel settings (`./`, Vite, npm run build, dist) now have a real package.json and working web build. Redeploy the **new** main commit, not the earlier Python-only commit `08e494b`. Future GitHub pushes can trigger deployment through the existing Vercel Git integration. This repository change does not itself verify that a Vercel deployment succeeded.

If you prefer Root Directory `web`, use its own `npm run build` and `dist`. With npm workspaces, Vercel needs access to the repository's root lockfile; leave source access outside the root directory enabled. The repository-root configuration above avoids that extra setting.

The model is downloaded from Google's official version 1 URL at **build time** and checked against SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`. The matching WASM files are copied from the pinned MediaPipe npm package. All runtime scripts, WASM, and model files are served from the deployment itself, rather than third-party CDNs. Build downloads and verifies the MIT-licensed FER+ model (SHA-256 `a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c`) from ONNX Model Zoo and copies matching ONNX Runtime WASM locally. Build requires network access to npm, Google storage and GitHub; a failed model download fails the build with an explicit error. Normal hosting still serves page/model assets and receives ordinary HTTP access requests; camera frames and measurements are not transmitted.

## Implemented

- Responsive webcam/dashboard, selectable cameras, mirror, mesh, landmark IDs, eye/mouth/iris points, measurement lines, pose axes, three configurable sound patterns, debug explanations.
- EAR/MAR geometry and sustained eye closure/mouth opening/head-away timers with the Python core's formulas and default thresholds. Missing tracking, nonfinite values, or gaps over 0.75 s reset continuity.
- Five-second calibration requiring 30 healthy frames. It updates eye/mouth thresholds and neutral yaw/pitch.
- Health gates: >95% of landmarks in frame, face width >15%, mean brightness between 35 and 235. Stale video is marked unavailable after 1.5 s. Thresholds 0.6 for MediaPipe detection, presence and tracking are settings, not per-frame confidence scores.
- GPU initialization with CPU fallback; processing throttled to a target 15 FPS. Inference currently runs synchronously on the browser main thread, so weak devices can still stutter. Achieved rate is hardware-dependent.

**Pose is an approximate canonical-face transform estimate**, different from the Python app's solvePnP method. Calibration offsets do not make these methods numerically interchangeable. Axes and signs refer to camera coordinates; mirroring changes the preview. Eight-class expression inference is implemented. Person identification, trained driver-state inference, event-ground-truth accuracy, and safety validation remain unavailable in the web edition. The dashboard states these model limits explicitly.

## Tests

```sh
npm test
npx playwright test --config web/playwright.config.js
```

The unit tests use Node's built-in runner. The browser tests use an installed Edge executable on Windows when available; on other machines install Playwright Chromium using `npx playwright install chromium`. Browser tests use synthetic camera input and never require a real webcam. The optional face-fixture test requires `DMS_TEST_FACE` pointing to the public-domain `astronaut.png` bundled with scikit-image; without it, that one test is explicitly skipped. Do not substitute private user imagery into published fixtures.

See `VALIDATION.md` in this folder for actual build and browser results. Real-device acceptance remains: permit camera on the deployed HTTPS origin, start, calibrate, check overlays/alerts in stationary conditions, toggle sound, stop, and confirm the browser's camera indicator turns off. A private GitHub repository does not determine Vercel deployment access; check the project's deployment protection if the link must stay private.
