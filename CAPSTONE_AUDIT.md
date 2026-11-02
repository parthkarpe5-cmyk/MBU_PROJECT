# DriveGuard AI implementation audit

Audited against repository main `a053155` and the approved abstract-review requirements. Work is isolated on `feat/driveguard-capstone`; no main update or merge is part of this change.

## Findings and implementation

| Requirement | Main before audit | This branch |
|---|---|---|
| Startup | Separate Python/browser apps; manual setup; overlapping OpenCV packages in historical lock | Checked Windows setup script, test dependency file, single OpenCV distribution in lock, browser runtime/model assets verified at build |
| CNN emotion recognition | Optional unverified 48px ONNX interface; browser unavailable | Real FER+ opset 8 adapters for Python OpenCV and browser ONNX Runtime; explicit download/integrity verification; eight labels and uncalibrated scores |
| Eye gaze | Iris drawing only | Separate roll-invariant iris projection, binocular agreement checks, open-eye gating, forward baseline from 30 calibration samples |
| Temporal fusion | Geometry timers; expressions separate | Time-aligned geometry/gaze plus fresh expression context; optional Python research behavior context; clear missing modalities |
| Missing/obstructed view | Generic unavailable | Face missing, possible partial obstruction/poor quality, unusable camera image, and stale-frame statuses; unsupported cause remains unknown |
| Sound | One system/browser beep | Three distinct generated patterns; configurable event mapping/mute and previews; fatigue priority; cooldown |
| Dashboard | Dark interfaces, fixed split | Light blue/pearl surfaces, teal healthy state, amber attention, red danger; camera/dashboard/split views, resize, fullscreen; always accessible stop control |
| Verification | Existing unit/research tests | Regression tests plus gaze/fusion/visibility/CNN/sound/view tests; real-model inference on static fixture; browser lifecycle tests use fake camera |

## Model contract and provenance

FER+ is a pretrained deep convolutional network from [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus), upstream MIT license. It uses float32 NCHW `[1,1,64,64]`, grayscale face crops in 0–255, and eight logits ordered neutral, happiness, surprise, sadness, anger, disgust, fear, contempt. Softmax scores are not calibrated confidence or evidence of a person's mental state. Cropping without a validated alignment pipeline can affect predictions.

Weights: `emotion-ferplus-8.onnx`, 35,040,571 bytes, SHA-256 `a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c`. Python setup verifies the upstream Git LFS digest and byte count; browser build pins this digest. Weights are downloaded explicitly at setup/build, never stored in Git. Browser inference uses the pinned `onnxruntime-web` WASM package served locally with the deployment. No frames are uploaded.

The existing optional research driver-state model is **not a trained behavior CNN**. No suitable behavior weights are shipped. When a Python research bundle exists, fusion shows its fresh high-score result as supporting context; absent/error/stale results remain unavailable. We did not train or invent a driver-behavior classifier.

## Fusion and availability rules

- Existing sustained EAR/MAR/head-pose rules remain the observed-cue foundation. Missing/invalid observations or gaps above 0.75 seconds break continuity.
- Gaze is horizontal iris displacement in eye-width units, independent of pose angles. Both eyes must agree within 0.18 eye widths. Closed eyes, missing irises, invalid geometry, or missing calibration suppress gaze. A calibrated offset exceeding 0.12 for two seconds produces a gaze-away cue. It estimates neither vertical gaze nor screen/world point of regard.
- Expressions need score ≥0.6 and age ≤1.5 seconds. At least one second of distinct time-stamped agreeing predictions is needed for stable expression context. A cached result alone cannot establish that agreement. Future/nonfinite results are rejected. Expression alone never sounds an alert.
- Optional Python research-model context needs score ≥0.6 and age ≤2 seconds, as well as its own runtime validity checks. Behavior alone never sounds an alert. This is transparent rule-based fusion, not a trained or clinically validated risk estimator.
- Brightness <20/>245 or image standard deviation <3 yields camera-view unavailable. A normally exposed image without reliable face detection yields face not detected. Unreliable framing/pose/lighting yields possible partial obstruction/poor quality. MediaPipe can hallucinate landmarks behind an obstacle: these heuristics do **not** identify objects, guarantee occlusion detection, or reliably distinguish absence from complete occlusion.
- Unavailable tracking suspends face-dependent alerts and invalidates expression/gaze. Availability sounds after two continuous seconds; calibration pauses fusion and sound. Stale browser frames invalidate all displayed cues after 1.5 seconds. Camera disconnection/errors release resources and display an actionable message.
- Sound events: distraction (660/880 Hz), fatigue cues (1040/780/1040 Hz), monitoring unavailable (440/330 Hz). Per-event three-second cooldown, selectable pattern or mute, explicit previews. A fatigue cue has priority when fatigue and distraction coexist. Speaker output and real-world audibility still require manual testing.

## Run and verify

Windows: `setup.cmd`, then `run.cmd`. Setup requires Python 3.10+ with pip/Tk; Python 3.13 was exercised. Alternatively install requirements and run the two setup scripts explicitly. Missing expression weights do not stop landmark monitoring. Legacy custom models require `--expression-contract legacy48` and labels. F11 toggles fullscreen; Escape exits; drag the desktop sash to resize. Browser split uses a camera-share slider. View changes never stop analysis.

Tests: install `requirements-test.txt`, run `python -m pytest -q --basetemp .test-temp-local`. GRU training tests need optional PyTorch. Browser: `npm ci`, `npm test`, `npm run build`, then `npx playwright test --config web/playwright.config.js`. Set `DMS_TEST_FACE` to a static face PNG to include the actual-model browser test. CI constructs the public-domain scikit-image astronaut fixture. No webcam is accessed by these tests.

See `VALIDATION.md` and `web/VALIDATION.md` for actual results and outstanding manual acceptance checks. This capstone implementation does not establish accuracy, false-alarm rates, accident prevention or fitness for vehicle use.
