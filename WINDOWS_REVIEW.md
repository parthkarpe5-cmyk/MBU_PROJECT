# Independent Windows review — 1 November 2026

The supplied MBU_PROJECT_research.zip contains implemented extraction, feature processing, baseline and GRU training, evaluation, export, app integration, notebook, and tests. It is more than an inspection report. It contains no trained UTA-RLDD bundle or raw UTA-RLDD videos.

## Verification actually performed

- Initial Windows non-UI suite: 50 passed, 1 failed in 71.03 s. Both previously skipped PyTorch tests ran successfully. The failure was the undeclared scikit-image dependency used by synthetic video generation.
- After the fixes below, complete Windows suite: **56 passed in 114.51 s, no skips**. Includes original tests, PyTorch forward parity, synthetic GRU training/resume/export, real MediaPipe extraction parity, the hidden Tk dashboard test, and four review regression cases.
- Focused missing-data/cache/smoke-preservation/extraction checks: 14 passed in 34.06 s.
- The hidden dashboard test now asserts learned-panel content and its invalidation when camera data becomes stale. It uses synthetic packets and opens no webcam.

Test environment: existing Windows Python 3.13.15 project venv with its NumPy/MediaPipe/OpenCV/Pillow/pytest. The test runner appended the host Python site-packages directory to use already-installed PyTorch and scikit-learn; PyTorch CPU threads were limited to two. scikit-image and its dependencies were installed into the project venv. This was not a clean installation of requirements-research.txt. Pytest temporary files were placed inside the review folder because the initial attempt with the default temp directory encountered access failures and was interrupted.

## Fixes in this reviewed copy

1. Smoke runs refuse a nonempty output directory, preserving existing contents instead of recursively deleting the caller's chosen folder.
2. Window building and learned-model evaluation reject deliberately truncated or possibly truncated extraction caches. The existing extraction-resume check alone did not stop training from consuming them.
3. All-missing preprocessing columns are explicitly initialized to finite defaults. Statistics run only on columns with observed values; no broad RuntimeWarning suppression remains.
4. requirements-research.txt includes scikit-image, required by the synthetic video fixture.
5. GRU loss logging detaches the tensor before converting it to a scalar, removing the autograd warning.
6. Added regression tests and scoped pytest discovery, and strengthened the UI assertions.

## Remaining limits

No real UTA-RLDD training or accuracy evaluation was performed. Synthetic training tests validate software mechanics, not driver-state performance. The full 30-video smoke CLI was not rerun during this review; its component paths were exercised by tests. The Colab notebook was inspected but not executed. No live webcam session was opened for this upgrade, and no visible-layout screenshot review was performed. The earlier MVP's webcam test does not validate the learned path.

This review is not exhaustive. In particular, verify the real dataset's layout/labels, raw-video provenance and cache freshness before a long extraction. CPU checkpoint resume is exercised; exact resume on CUDA, including all GPU random state, has not been established. Held-out calibration statistics being recorded do not by themselves prove that scores are well calibrated. Follow MODEL_CARD.md's research limitations.

The original zip in Downloads and the original MVP remain intact. The reviewed code is staged separately, and the root run-research.cmd starts it using the existing project venv. Install research requirements into your chosen training environment before using the research commands normally.
