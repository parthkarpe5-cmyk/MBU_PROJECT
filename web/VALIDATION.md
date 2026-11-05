# Synthesized sound redesign — 5 November 2026

- 26 Node tests passed, including exact oscillator waveforms, frequency ramps,
  pulse durations/gaps, 15 ms gain fades, node cleanup, capped master gain,
  zero volume, three-second cooldown, mute/preview rules and preference migration.
- All 13 browser tests passed in headless Edge. The new sound test schedules real
  Web Audio oscillators for all three previews, checks friendly option names,
  per-event mute and zero-volume suppression, volume persistence/reset, and no
  network requests during synthesis. Production build passed.
- Camera lifecycle/retry tests now use deterministic canvas streams because Edge's
  synthetic hardware camera intermittently disappeared on restart. Actual
  MediaPipe/FER+ loading and track-stop assertions remain in place.
- Speaker audibility and device-dependent loudness were not tested. No audio files,
  recordings, uploads or alarm-standard compliance claims were added.

---

# Session summaries, preferences and recovery — 5 November 2026

- 21 Node unit tests passed, including summary duration, event-onset counting,
  observed blinks and gaps, yawn-proxy totals, FPS/processing aggregates, empty
  sessions, idempotent stop, JSON/CSV serialization, preference persistence/reset,
  corrupt/blocked storage, and friendly camera/model errors.
- All 12 Playwright tests passed using headless Edge and simulated/canvas camera
  input. The existing 5 tests remain passing. New tests cover live timing,
  stop summary and released tracks, no automatic downloads, JSON/CSV content,
  no HTTP requests during export, no report persistence, preferences across
  reload/reset, storage denial, About text, narrow layout, denied/absent camera
  recovery, failed face-model startup without camera access, and expression-model
  Retry while preserving the active camera stream.
- Root production build passed. Python regression suite: 62 passed, 1 skipped.
- The focused report test also passed with desktop and 390px screenshots; both
  were visually inspected. Summary values, alert counts, format selector and
  download button fit without horizontal overflow.
- Reports are aggregate local downloads only. These tests do not establish
  recognition accuracy, sound audibility, physical webcam behavior, sustained
  performance across hardware or remote Vercel behavior. Live ms excludes async
  expression inference and camera/display latency. No human webcam was opened.

---

# DriveGuard AI browser validation — 2 November 2026

- **13 unit tests passed**: geometry/timers plus iris gaze, conservative visibility statuses, fusion timing and priority, freshness, CNN score decoding, and expression-only alarm suppression.
- **Production build passed** with locally served MediaPipe, FER+ and ONNX Runtime WASM assets. Both model digests were checked.
- **5 Playwright integration tests passed** using headless Edge, fake browser camera or scikit-image's public-domain astronaut image rendered to a canvas stream: initial permission-free load and mobile fit; denied-camera retry; actual MediaPipe startup and camera track cleanup/restart; actual FER+ inference and face/pose/gaze calibration; workspace views, split resizing, sound mapping/preview interaction and light theme.
- The calibrated fixture screenshot was visually inspected: mesh and measurement overlays, iris gaze, CNN label/score, six metric cards, fusion status, calibration, sound/threshold sections and light surfaces render without clipping. The fixture showed `happiness`, forward-range gaze and no sustained alerts. It is not ground-truth driver evaluation.
- Sound tests verify distinct synthesis patterns and controls, not actual speaker audibility. Fullscreen support is implemented with the browser API; physical display/window behavior remains a manual check.

**No human webcam was opened.** The tests do not establish detection accuracy, reliable obstruction identification, real-world latency/FPS, real speaker output or fitness for driving. A trained driver-behavior model remains unavailable in this browser edition. No remote deployment was checked.

---

## Historical validation (before this branch)

# Browser edition validation — 1 November 2026

Environment: Windows x64, Node 24.13.1, npm 11.13.0, Vite 8.3.5, MediaPipe Tasks Vision 1.2.0. npm lockfile is committed. npm installation reported zero vulnerabilities at the time of the build.

Actual checks:

- Node unit suite: 9 passed. Geometry, closure timing, one-eye handling, missing/nonfinite observations, large/backwards time gaps, mouth event counts, head-away timing, calibration success/failure, transform orientation, invalid landmarks, and settings bounds.
- Root production build completed successfully and emitted index.html, JS/CSS, version 1 face model, and matching WASM assets under root dist/. The model checksum matched the pinned value.
- Playwright with headless Edge and simulated camera: the final complete run had 4 passed in 22.8 seconds. Checks: initial load without requesting webcam access; narrow viewport without horizontal overflow; actionable permission-denied recovery; real MediaPipe initialization and blank-camera no-face handling; stop/restart with ended tracks; real-face geometry and five-second calibration using scikit-image's public-domain astronaut image as a canvas-backed stream. The real-face case was then strengthened to enable full mesh, point IDs, and debug; its focused rerun passed in 18.2 seconds.
- Full-page screenshot of the synthetic face session was visually inspected. It showed calibrated measurements, overlays, model availability information, controls, and settings in a readable layout.

The optional fixture was supplied using DMS_TEST_FACE and was not copied into source or published. No human webcam was opened during these browser tests. A single fixture's successful detection is not accuracy validation. Physical camera permissions/selection, sound playback, actual closure/yawn/head-away accuracy across users, mobile hardware, and sustained FPS require testing on the deployed site. No expression model or UTA-RLDD-trained model is claimed. Python training functionality is retained separately.

Vercel's own remote build and deployed URL are not verified by these local checks. Browser head-pose estimates differ from Python solvePnP; the math test validates a known synthetic rotation, not parity with real-human pose ground truth. No safety claim is made.
