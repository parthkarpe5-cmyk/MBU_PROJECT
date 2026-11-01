"""Dataset inspection, manifest and participant-level split enforcement.

Rules enforced here:
  * Labels are never inferred from undocumented filenames. A label source must be declared
    (CSV you supply, or a filename-stem rule with a written documentation citation and an explicit
    ``confirmed: true``), and every file must map unambiguously or the build fails.
  * Splits derive ONLY from the official fold number. Every participant must live in exactly one fold,
    and therefore exactly one split. Violations raise LeakageError.
"""
import collections
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import re

from .config import dump_json, sha256_file

TEXT_HINT = re.compile(r"readme|label|annot|license|description|fold|split|\.txt$|\.md$|\.csv$|\.json$|\.xlsx?$|\.pdf$", re.I)
MANIFEST_FIELDS = ("participant_id", "fold", "video_id", "video_path", "label", "label_name", "duration_s",
                   "fps", "width", "height", "frame_count", "size_bytes", "split")


class LeakageError(Exception):
    pass


class ManifestError(Exception):
    pass


def resolve_root(cfg, override=None):
    root = override or cfg["dataset"].get("root") or os.environ.get("RLDD_ROOT")
    if not root:
        raise ManifestError("Dataset root not set: use --dataset-root, config dataset.root, or env RLDD_ROOT")
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ManifestError(f"Dataset root is not a directory: {root}")
    return root


def iter_videos(root, extensions):
    exts = tuple(e.lower() for e in extensions)
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            if name.lower().endswith(exts) and not name.startswith("._"):
                yield (Path(dirpath) / name)


def rel_posix(path, root):
    return Path(path).relative_to(root).as_posix()


# --------------------------------------------------------------------------- inspection
def inspect_dataset(root, cfg, max_text_lines=40):
    """Describe what is ACTUALLY on disk. Assigns no labels, folds or participants."""
    root = Path(root)
    exts = collections.Counter()
    total_bytes = 0
    videos, text_files = [], []
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            path = Path(dirpath) / name
            exts[path.suffix.lower() or "(none)"] += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
            if name.lower().endswith(tuple(cfg["dataset"]["video_extensions"])) and not name.startswith("._"):
                videos.append(path)
            elif TEXT_HINT.search(name):
                text_files.append(path)
    signatures = collections.Counter(re.sub(r"\d+", "#", rel_posix(v, root)) for v in videos)
    stems = collections.Counter(v.stem for v in videos)
    by_depth = collections.defaultdict(set)
    for v in videos:
        for depth, part in enumerate(v.relative_to(root).parts[:-1]):
            by_depth[depth].add(part)
    snippets = {}
    for path in text_files[:20]:
        try:
            if path.suffix.lower() in (".txt", ".md", ".csv", ".json") and path.stat().st_size < 200_000:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[:max_text_lines]
                snippets[rel_posix(path, root)] = lines
        except OSError:
            pass
    report = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "root_name": root.name,
        "n_files_by_extension": dict(exts),
        "n_videos": len(videos),
        "total_size_gb": round(total_bytes / 1e9, 3),
        "path_signatures_top": signatures.most_common(15),
        "filename_stems_top": stems.most_common(30),
        "directory_names_by_depth": {str(d): sorted(names)[:80] for d, names in sorted(by_depth.items())},
        "example_video_paths": [rel_posix(v, root) for v in videos[:10]],
        "documentation_like_files": [rel_posix(p, root) for p in text_files[:50]],
        "documentation_snippets": snippets,
        "documented_expectation": cfg["dataset"]["expected"],
        "label_map_documented": cfg["dataset"]["label_map"],
        "label_map_provenance": cfg["dataset"]["label_map_provenance"],
        "notes": ["No labels, folds or participants were assigned by this command.",
                  "Compare path_signatures_top with the dataset's own documentation, then set dataset.layout_regex "
                  "(named groups: fold, participant) and dataset.label_source in the config."],
    }
    return report


# --------------------------------------------------------------------------- manifest
def probe_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return dict(duration_s=None, fps=None, width=None, height=None, frame_count=None)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        valid = fps and math.isfinite(fps) and 0 < fps < 1000 and count and count > 0
        return dict(duration_s=round(count / fps, 3) if valid else None, fps=round(fps, 4) if fps and fps > 0 else None,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    frame_count=int(count) if count and count > 0 else None)
    finally:
        cap.release()


def _label_resolver(cfg, root):
    source = cfg["dataset"].get("label_source")
    label_map = {str(k): v for k, v in cfg["dataset"]["label_map"].items()}
    if not source:
        raise ManifestError("dataset.label_source is not set. Labels are never inferred silently. See DATASET.md.")
    kind = source.get("type")
    if kind == "csv":
        if not source.get("provenance"):
            raise ManifestError("label_source.provenance must say where this CSV came from")
        path = Path(source["path"])
        table = {}
        with open(path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                table[row["video_path"].replace("\\", "/")] = str(row["label"]).strip()
        provenance = f"csv:{path.name} sha256={sha256_file(path)[:16]} | {source['provenance']}"

        def resolve(rel, match):
            if rel not in table:
                raise ManifestError(f"{rel}: not present in label CSV")
            return table[rel]
        return resolve, provenance, label_map
    if kind == "filename_stem":
        if not source.get("documentation"):
            raise ManifestError("label_source.documentation must cite where the dataset documents filename labels")
        if source.get("confirmed") is not True:
            raise ManifestError("Filename-stem labels need label_source.confirmed=true after YOU verified, against the "
                                "dataset's documentation, that file stems are the class labels.")

        def resolve(rel, match):
            return Path(rel).stem
        return resolve, f"filename_stem | {source['documentation']}", label_map
    raise ManifestError(f"Unknown label_source.type {kind!r}")


def assign_split(fold, splits):
    for name in ("train", "val", "test"):
        if fold in splits[f"{name}_folds"]:
            return name
    raise ManifestError(f"Fold {fold} is not assigned to train/val/test in config.splits")


def build_manifest(root, cfg, probe=True, log=print):
    pattern = cfg["dataset"].get("layout_regex")
    if not pattern:
        raise ManifestError("dataset.layout_regex is not set. Run `inspect` first and write the regex for the real layout.")
    rx = re.compile(pattern, re.I)
    if not {"fold", "participant"} <= set(rx.groupindex):
        raise ManifestError("layout_regex needs named groups `fold` and `participant`")
    resolve, provenance, label_map = _label_resolver(cfg, root)
    rows, unmatched, problems = [], [], []
    for path in iter_videos(root, cfg["dataset"]["video_extensions"]):
        rel = rel_posix(path, root)
        match = rx.fullmatch(rel)
        if not match:
            unmatched.append(rel)
            continue
        fold_digits = re.search(r"\d+", match["fold"])
        if not fold_digits:
            problems.append(f"{rel}: fold group {match['fold']!r} has no number")
            continue
        try:
            raw_label = str(resolve(rel, match)).strip()
        except ManifestError as exc:
            problems.append(str(exc))
            continue
        if raw_label not in label_map:
            problems.append(f"{rel}: label {raw_label!r} not in documented label_map {sorted(label_map)}")
            continue
        fold = int(fold_digits.group())
        rows.append(dict(participant_token=match["participant"], fold=fold, video_path=rel, label=int(raw_label),
                         label_name=label_map[raw_label], size_bytes=path.stat().st_size))
    if unmatched or problems:
        shown = (problems + [f"{u}: did not match layout_regex" for u in unmatched])[:25]
        raise ManifestError(f"{len(problems)} label/layout problems and {len(unmatched)} unmatched videos; refusing to guess.\n  "
                            + "\n  ".join(shown))
    if not rows:
        raise ManifestError("No videos found")

    folds_of = collections.defaultdict(set)
    for r in rows:
        folds_of[r["participant_token"]].add(r["fold"])
    include_fold = bool(cfg["dataset"].get("participant_id_includes_fold"))
    ambiguous = {p: sorted(f) for p, f in folds_of.items() if len(f) > 1}
    if ambiguous and not include_fold:
        raise LeakageError(f"Participant tokens appear in several folds {dict(list(ambiguous.items())[:5])}. If the dataset "
                           "documents that numbering restarts per fold, set dataset.participant_id_includes_fold=true.")
    for r in rows:
        r["participant_id"] = f"F{r['fold']}-{r['participant_token']}" if include_fold else f"P{r['participant_token']}"
        r["video_id"] = f"{r['participant_id']}__L{r['label']}"
        r["split"] = assign_split(r["fold"], cfg["splits"])
    ids = collections.Counter(r["video_id"] for r in rows)
    duplicated = [k for k, v in ids.items() if v > 1]
    if duplicated:
        raise ManifestError(f"Several videos share (participant, label): {duplicated[:5]}")
    if probe:
        for i, r in enumerate(rows):
            r.update(probe_video(root / r["video_path"]))
            if i % 20 == 0:
                log(f"probed {i + 1}/{len(rows)}")
    else:
        for r in rows:
            r.update(duration_s=None, fps=None, width=None, height=None, frame_count=None)
    rows.sort(key=lambda r: (r["fold"], r["participant_id"], r["label"]))
    assert_participant_disjoint(rows)

    expected = cfg["dataset"]["expected"]
    participants = {r["participant_id"] for r in rows}
    warnings = []
    if len(participants) != expected["participants"] or len(rows) != expected["videos"]:
        warnings.append(f"Found {len(participants)} participants / {len(rows)} videos; dataset documents "
                        f"{expected['participants']} / {expected['videos']}. Results are for THIS subset only.")
    per_participant = collections.Counter(r["participant_id"] for r in rows)
    odd = {p: n for p, n in per_participant.items() if n != 3}
    if odd:
        warnings.append(f"{len(odd)} participants do not have exactly 3 videos (documented: one per class): {dict(list(odd.items())[:5])}")
    meta = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": cfg["dataset"]["name"], "root_name": root.name,
        "label_map": label_map, "label_map_provenance": cfg["dataset"]["label_map_provenance"],
        "label_source": provenance, "layout_regex": pattern,
        "splits_config": cfg["splits"], "participant_id_includes_fold": include_fold,
        "counts": {"videos": len(rows), "participants": len(participants),
                   "by_split": dict(collections.Counter(r["split"] for r in rows)),
                   "participants_by_split": {s: len({r["participant_id"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")},
                   "by_label": dict(collections.Counter(r["label_name"] for r in rows))},
        "warnings": warnings,
        "split_policy": "Official fold number only. A participant lives in exactly one fold -> one split. "
                        "No frame- or window-level random splitting anywhere in this repository.",
    }
    return rows, meta


def write_manifest(rows, meta, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    meta = dict(meta, manifest_sha256=sha256_file(path))
    dump_json(out_dir / "manifest_meta.json", meta)
    return path, meta


def read_manifest(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            for key in ("fold", "label", "frame_count", "width", "height", "size_bytes"):
                row[key] = int(row[key]) if row.get(key) not in (None, "", "None") else None
            for key in ("duration_s", "fps"):
                row[key] = float(row[key]) if row.get(key) not in (None, "", "None") else None
            rows.append(row)
    assert_participant_disjoint(rows)
    return rows


def assert_participant_disjoint(rows, key="participant_id"):
    """Every participant in exactly one split (and one fold). Works on manifest rows or window tables."""
    splits_of, folds_of = collections.defaultdict(set), collections.defaultdict(set)
    for r in rows:
        splits_of[r[key]].add(r["split"])
        if "fold" in r and r["fold"] is not None:
            folds_of[r[key]].add(r["fold"])
    bad = {p: sorted(s) for p, s in splits_of.items() if len(s) > 1}
    bad_folds = {p: sorted(f) for p, f in folds_of.items() if len(f) > 1}
    if bad or bad_folds:
        raise LeakageError(f"Participants spanning several splits/folds: {dict(list(bad.items())[:5])} {dict(list(bad_folds.items())[:5])}")
    return True
