"""Command line: python -m research <command>. See README.md ("Research pipeline")."""
import argparse
import json
from pathlib import Path
import sys
import time

from .config import ROOT, dump_json, load_config, set_seed


def _logger(artifacts, name):
    path = Path(artifacts) / "logs" / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with open(path, "a", encoding="utf-8") as file:
            file.write(line + "\n")
    return log


def _paths(args):
    artifacts = Path(args.artifacts)
    return artifacts, artifacts / "manifest", artifacts / "runs"


def _cache_dir(cfg, artifacts, override=None):
    from .extract import extraction_key
    if override:
        return Path(override)
    face = Path(cfg["extraction"]["face_model"])
    face = face if face.is_absolute() else ROOT / face
    return Path(artifacts) / "features" / extraction_key(cfg, face)


def _manifest(manifest_dir):
    from .dataset import read_manifest
    rows = read_manifest(Path(manifest_dir) / "manifest.csv")     # also asserts participant disjointness
    meta = json.loads((Path(manifest_dir) / "manifest_meta.json").read_text(encoding="utf-8"))
    return rows, meta


def cmd_inspect(args, cfg):
    from .dataset import inspect_dataset, resolve_root
    artifacts, _, _ = _paths(args)
    log = _logger(artifacts, "inspect")
    root = resolve_root(cfg, args.dataset_root)
    report = inspect_dataset(root, cfg)
    dump_json(artifacts / "inspection" / "dataset_inspection.json", report)
    log(f"{report['n_videos']} videos, {report['total_size_gb']} GB, extensions {report['n_files_by_extension']}")
    log("path signatures (digits -> #): " + json.dumps(report["path_signatures_top"][:8]))
    log("filename stems: " + json.dumps(report["filename_stems_top"][:12]))
    log(f"documentation-like files: {report['documentation_like_files'][:10]}")
    log(f"wrote {artifacts / 'inspection' / 'dataset_inspection.json'}; assigned NO labels/folds/participants")


def cmd_manifest(args, cfg):
    from .dataset import build_manifest, resolve_root, write_manifest
    artifacts, manifest_dir, _ = _paths(args)
    log = _logger(artifacts, "manifest")
    rows, meta = build_manifest(resolve_root(cfg, args.dataset_root), cfg, probe=not args.no_probe, log=log)
    path, meta = write_manifest(rows, meta, manifest_dir)
    log(f"wrote {path}: {meta['counts']}")
    for w in meta["warnings"]:
        log("WARNING: " + w)


def cmd_verify_splits(args, cfg):
    _, manifest_dir, _ = _paths(args)
    rows, meta = _manifest(manifest_dir)
    print(f"OK: {len({r['participant_id'] for r in rows})} participants, every one inside exactly one fold/split. counts={meta['counts']['participants_by_split']}")


def cmd_extract(args, cfg):
    from .dataset import resolve_root
    from .extract import extract_all
    artifacts, manifest_dir, _ = _paths(args)
    log = _logger(artifacts, "extract")
    rows, _ = _manifest(manifest_dir)
    key, cache, failures = extract_all(rows, resolve_root(cfg, args.dataset_root), artifacts / "features", cfg, workers=args.workers,
                                       force=args.force, max_videos=args.max_videos, max_seconds=args.max_seconds, log=log)
    log(f"features cached in {cache}")
    return 1 if failures else 0


def cmd_heuristic(args, cfg):
    from .heuristic import evaluate_heuristic
    artifacts, manifest_dir, _ = _paths(args)
    log = _logger(artifacts, "heuristic")
    rows, _ = _manifest(manifest_dir)
    evaluate_heuristic(rows, _cache_dir(cfg, artifacts, args.features_dir), cfg, artifacts / "heuristic", log)


def cmd_train_baseline(args, cfg):
    from .baseline import train_baseline
    artifacts, manifest_dir, runs = _paths(args)
    log = _logger(artifacts, f"train_baseline_{args.run}")
    rows, meta = _manifest(manifest_dir)
    train_baseline(cfg, rows, _cache_dir(cfg, artifacts, args.features_dir), runs / args.run, meta.get("manifest_sha256"), synthetic=args.synthetic, log=log)


def cmd_train_gru(args, cfg):
    from .temporal import train_gru
    artifacts, manifest_dir, runs = _paths(args)
    log = _logger(artifacts, f"train_gru_{args.run}")
    rows, meta = _manifest(manifest_dir)
    train_gru(cfg, rows, _cache_dir(cfg, artifacts, args.features_dir), runs / args.run, meta.get("manifest_sha256"),
              resume=not args.no_resume, synthetic=args.synthetic, max_epochs=args.max_epochs, log=log)


def cmd_evaluate(args, cfg):
    from .evaluate import evaluate_run
    artifacts, manifest_dir, runs = _paths(args)
    log = _logger(artifacts, f"evaluate_{args.run}")
    rows, _ = _manifest(manifest_dir)
    result = evaluate_run(runs / args.run, rows, _cache_dir(cfg, artifacts, args.features_dir), cfg, args.split, log=log)
    w = result["window_level"]
    log(f"{args.split}: macro-F1 {w['macro_f1']:.4f}, balanced acc {w['balanced_accuracy']:.4f} over {result['windows']['scored']} windows "
        f"/ {result['windows']['participants']} participants -> {runs / args.run / f'report_{args.split}.md'}")


def cmd_export(args, cfg):
    from .export import export_bundle
    artifacts, _, runs = _paths(args)
    dest = export_bundle(runs / args.run, args.dest, allow_synthetic=args.allow_synthetic, require_heldout=args.require_heldout)
    print(f"exported to {dest}")


def cmd_smoke(args, cfg):
    from .smoke import run_smoke
    return run_smoke(Path(args.workdir), Path(args.config) if args.config else None)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m research", description=__doc__)
    parser.add_argument("--config", default=None, help="JSON config (merged over defaults); see configs/")
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"), help="output root (git-ignored)")
    parser.add_argument("--dataset-root", default=None, help="UTA-RLDD root (or env RLDD_ROOT). Raw videos are never copied.")
    parser.add_argument("--features-dir", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="describe files on disk; assigns nothing").set_defaults(fn=cmd_inspect)
    p = sub.add_parser("manifest", help="build manifest.csv (needs layout_regex + label_source)")
    p.add_argument("--no-probe", action="store_true")
    p.set_defaults(fn=cmd_manifest)
    sub.add_parser("verify-splits", help="assert every participant is in exactly one split").set_defaults(fn=cmd_verify_splits)
    p = sub.add_parser("extract", help="MediaPipe features per video (resumable)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    p.set_defaults(fn=cmd_extract)
    sub.add_parser("heuristic", help="evaluate the existing rule-based monitor where comparable").set_defaults(fn=cmd_heuristic)
    for name, fn in (("train-baseline", cmd_train_baseline), ("train-gru", cmd_train_gru)):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--synthetic", action="store_true", help="mark the run as trained on a synthetic fixture")
        if name == "train-gru":
            p.add_argument("--no-resume", action="store_true")
            p.add_argument("--max-epochs", type=int, default=None)
        p.set_defaults(fn=fn)
    p = sub.add_parser("evaluate", help="score a split with the deployed numpy path")
    p.add_argument("--run", required=True)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("export", help="verify and copy a run to the app's model directory")
    p.add_argument("--run", required=True)
    p.add_argument("--dest", default=str(ROOT / "models" / "learned"))
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--require-heldout", action="store_true")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("smoke", help="end-to-end plumbing test on SYNTHETIC videos (meaningless metrics)")
    p.add_argument("--workdir", required=True)
    p.set_defaults(fn=cmd_smoke)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    return args.fn(args, cfg) or 0


if __name__ == "__main__":
    sys.exit(main())
