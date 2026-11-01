# Model card: learned driver-state classifier (research prototype)

**Status: NOT TRAINED.** No weights are included. Everything below describes the code contract and intended procedure. Replace "pending" entries with your own measured results.

## Intended use
Research on webcam-based drowsiness-class estimation from facial-geometry time series, alongside the heuristic monitor. **Not** for driving, vehicle control, safety decisions, or any claim of automotive/medical validity. It does not detect events (blinks, yawns, microsleeps) and does not interpret facial expression as fatigue.

## Model types (both numpy at runtime)
- `logreg` (baseline): multinomial logistic regression on window features.
- `gru`: one GRU layer (8-32 units) over 1-second bins; unobserved bins carry state; head = linear. Trained with PyTorch, exported to `weights.npz`, export blocked unless numpy and torch outputs agree.

## Input contract (`bundle.json` is authoritative)
Rolling window (default 30 s, stride 10 s; selected values are stored in the bundle) of per-frame records from `dms.features.frame_record`: raw, uncalibrated EAR(L/R), MAR, yaw, pitch, roll plus `valid` (face found AND app health check passed AND finite). Window features (`dms/features.py`): PERCLOS-style closed-time fractions (fixed threshold 0.21 and window-relative 0.72 x P90 of EAR; both eyes must be below threshold), bounded blink counts/durations, long closures (> 0.5 s), EAR/MAR statistics and rates of change, mouth-open fraction and events (>= 1.5 s), head-pose variability/velocity, off-center fraction versus the window's median pose. Rates and fractions use **valid time** as denominator; gaps over 0.5 s count as missing; missing data is never closed eyes or zero. Tracking-quality fields (coverage) gate predictions but are not model inputs.

## Output contract
Three logits ordered `alert, low_vigilance, drowsy`; probabilities = softmax(logits / T). `T` = 1 unless a temperature was fit **on validation windows**. Score kinds shown in the app: *uncalibrated* (default), *calibrated, held-out check not recorded*, *calibrated, held-out ECE recorded*. Even the last measures agreement with weak video labels only.

## Training data and procedure
UTA-RLDD (see `DATASET.md`), fold-based participant-level splits, weak video labels. Preprocessing fit on train; window length / C / hidden size / early stopping / temperature chosen on validation; test split evaluated deliberately and every look logged. Class-balanced loss. Fixed seed (default 1337). **Pending:** actual training run.

## Evaluation (to be filled from `report_test.md`)
Window-level macro-F1, balanced accuracy, per-class precision/recall, confusion matrix, per-participant spread, participant-bootstrap CIs, video-level aggregate, calibration (ECE/NLL), latency. **Pending: no held-out results exist.**

## Known limitations
- Weak labels; no event ground truth; each participant's three classes come from three separate sessions, so lighting, camera angle, glasses, and time of day can act as shortcuts that participant-disjoint splits do not remove.
- Frame rate (< 30 fps in data; ~15 fps live here) limits blink resolution; dataset frames are compressed video, live frames are a webcam stream; feature thresholds are defaults, not per-driver calibrated; live session calibration offsets are not used by the model.
- MediaPipe landmarks degrade with glasses, occlusion, extreme pose, poor light; low-coverage windows are dropped in training and suppressed live, so the model is silent exactly where tracking is worst.
- Twelve test participants (one fold) give wide uncertainty; fold-to-fold variation is expected.
- Not evaluated for fairness across skin tone, age, gender, facial hair or eyewear.
