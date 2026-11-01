# UTA-RLDD: provenance, label mapping, and what is NOT verified

## Documented by the dataset authors (checked 31 Oct 2026)
Source: UTA-RLDD homepage <https://sites.google.com/view/utarldd/home> and Ghoddoosian, Galib, Athitsos, "A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection", CVPRW 2019 (<https://arxiv.org/abs/1904.07312>).

- 60 participants, one video per class (alert, low vigilance, drowsy), 180 videos, ~10 minutes each, ~30 hours.
- Class labels: alert = 0, low vigilant = 5, drowsy = 10.
- Participants were randomly divided into **five folds of 12** for cross validation; the authors' protocol is one fold as test, four for training, averaged over folds.
- Self-recorded on phones/webcams, frame rate < 30 fps, varied resolution/quality/environments; glasses in 21 and considerable facial hair in 72 of 180 videos.
- Labels are per **video**. No frame-level, blink, yawn or microsleep annotations are part of this use.

## NOT verified by this repository (the dataset was not available when it was built)
- The folder/file layout of the download, how fold folders and participant IDs are named, and whether file names encode labels. Nothing here assumes them: `inspect` reports what is on disk, and `manifest` refuses to run until you supply `dataset.layout_regex` (named groups `fold`, `participant`) and a `dataset.label_source`:
  - `{"type": "csv", "path": ..., "provenance": ...}` with columns `video_path,label` (0/5/10), or
  - `{"type": "filename_stem", "documentation": "<where documented>", "confirmed": true}` only after you verified in the dataset's own documentation that stems are the labels. Any stem outside {0,5,10} aborts the build.
- How the authors obtained the per-video labels (see the paper, section 3). Treat them as **weak, self-report-based video labels**.
- Whether participant IDs repeat across folds. If they do, the build aborts as a leakage risk unless you set `dataset.participant_id_includes_fold` after confirming that numbering restarts per fold.

## How labels reach windows
Each window inherits its video's label (weak supervision). A "drowsy" video contains many moments when the person looks awake; a window label is not evidence about that window. Consequently window metrics measure agreement with video labels, not detection of drowsiness events.

## Splits
Default: folds 1-3 train, fold 4 validation, fold 5 test (`splits` in the config). The authors' five-fold protocol can be reproduced by re-running with different fold assignments; results then vary by fold and should be reported as such.

## Licensing / privacy
Check the dataset's terms before redistributing anything derived from it. This repo git-ignores raw videos (`*.mp4` etc.), `data/`, `artifacts/` (features, runs), and `models/learned/`.
