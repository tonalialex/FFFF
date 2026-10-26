#!/usr/bin/env python
"""One-command SAR restoration + registration entry.

Default restoration loss uses the user-specified weights:
    Charbonnier pixel + 0.2 * gradient + 0.3 * LNCC
    + 1.0 * teacher (round 2 only)
Registration consumes the raw restored_sensed.png, not the brightness-corrected
_radiometric variant.

Examples
--------
    python run.py --check-env
    python run.py --input-dir data/example/demo --output-dir outputs/example
    python run.py --sensed a_passive_PGA64.jpg --reference a_active.jpg --output-dir outputs/pair01
    python run.py --sensed ... --reference ... --round1 r1.png --round2 r2.png --output-dir outputs/reg_only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FULL_PIPELINE = ROOT / "run_full_pipeline.py"
REGISTRATION = ROOT / "registration" / "run_registration.py"
WARM_START = ROOT / "models" / "simclr_initial.pt"
LOSS_CONFIG = ROOT / "configs" / "loss_selected.json"


def load_loss() -> dict:
    config = json.loads(LOSS_CONFIG.read_text(encoding="utf-8"))
    for key in ("name", "gradient", "structure", "teacher"):
        if key not in config:
            raise ValueError(f"{LOSS_CONFIG} is missing {key}")
    return config


def make_env(loss: dict, gradient: float, structure: float, teacher: float) -> dict:
    env = os.environ.copy()
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONWARNINGS": "ignore",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "SAR_RESTORATION_LOSS_CONFIG": loss["name"],
        "SAR_RESTORATION_GRADIENT_WEIGHT": str(gradient),
        "SAR_RESTORATION_STRUCTURE_WEIGHT": str(structure),
        "SAR_RESTORATION_TEACHER_WEIGHT": str(teacher),
    })
    return env


def check_environment() -> int:
    print("Python :", sys.version.replace("\n", " "))
    print("Prefix :", sys.prefix)
    try:
        import cv2
        import numpy as np
        import scipy
        import torch
        from PIL import Image
    except Exception as exc:  # pragma: no cover - diagnostic path
        print("IMPORT FAILED:", repr(exc))
        return 2
    print("torch  :", torch.__version__)
    print("torch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU    :", torch.cuda.get_device_name(0))
        print("GPU mem:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GiB")
    print("numpy  :", np.__version__)
    print("opencv :", cv2.__version__)
    print("scipy  :", scipy.__version__)
    print("pillow :", Image.__version__)
    if not torch.cuda.is_available():
        print("NOTE: CUDA is not available; pass --device cpu or fix the GPU environment.")
    return 0


def timestamped_output() -> Path:
    return ROOT / "outputs" / time.strftime("run_%Y%m%d_%H%M%S")


def write_config(args, out: Path, loss: dict, weights: dict) -> None:
    config = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "loss": {**loss, **weights},
        "restored_variant_for_registration": "raw",
        "restoration_feedback_variant": "raw",
        "python": sys.executable,
    }
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def print_metrics(csv_path: Path) -> None:
    if not csv_path.is_file():
        print("No batch_metrics.csv produced.")
        return
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("No rows in batch_metrics.csv.")
        return
    print("\n==== Registration metrics (final stage) ====")
    print(f"{'pair':24s} {'status':24s} {'Nred':>6s} {'RMSEall':>10s} {'RMSEloo':>10s} {'Pquad':>8s} {'NCC':>10s}")
    for row in rows:
        def fmt(key, spec):
            value = row.get(key) or ""
            try:
                return format(float(value), spec)
            except (TypeError, ValueError):
                return str(value)
        print(f"{row.get('pair',''):24s} {row.get('status',''):24s} "
              f"{row.get('Nred',''):>6s} {fmt('RMSEall','.4f'):>10s} "
              f"{fmt('RMSEloo','.4f'):>10s} {fmt('Pquad','.4f'):>8s} {fmt('NCC','.6f'):>10s}")
    print(f"\nFull table: {csv_path}")


def run_full(args, out: Path, loss: dict, weights: dict, env: dict) -> int:
    command = [
        sys.executable, str(FULL_PIPELINE),
        "--output-dir", str(out),
        "--device", args.device,
        "--restoration-steps", str(args.restoration_steps),
        "--restoration-seed", str(args.restoration_seed),
        "--registration-seed", str(args.registration_seed),
        "--contrastive-epochs", str(args.contrastive_epochs),
        "--workers", str(args.workers),
        "--restored-variant", "raw",
        "--continue-existing",
    ]
    if args.input_dir:
        command += ["--input-dir", str(args.input_dir.resolve())]
        if args.expected_pairs is not None:
            command += ["--expected-pairs", str(args.expected_pairs)]
    else:
        command += ["--sensed", str(args.sensed.resolve()), "--reference", str(args.reference.resolve())]
    if args.dry_run:
        print("DRY RUN:", " ".join(command))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    write_config(args, out, loss, weights)
    print("Running full restoration + registration:")
    print("  loss :", json.dumps({"name": loss["name"], **weights}, ensure_ascii=False))
    print("  data :", args.input_dir or args.sensed)
    print("  out  :", out)
    print("  logs :", out / "logs")
    started = time.time()
    code = subprocess.run(command, cwd=str(ROOT), env=env).returncode
    print(f"\nPipeline exit code: {code}  elapsed: {time.time() - started:.1f}s")
    print_metrics(out / "batch_metrics.csv")
    return code


def run_registration_only(args, out: Path, loss: dict, weights: dict, env: dict) -> int:
    if args.input_dir:
        raise SystemExit("--input-dir cannot be combined with --round1/--round2; run one pair at a time.")
    dataset = "single_pair"
    command = [
        sys.executable, str(REGISTRATION),
        "--original", str(args.sensed.resolve()),
        "--reference", str(args.reference.resolve()),
        "--round1", str(args.round1.resolve()),
        "--round2", str(args.round2.resolve()),
        "--dataset", dataset,
        "--output-root", str(out / "registration"),
        "--device", args.device,
        "--batch-size", "64",
        "--contrastive-epochs", str(args.contrastive_epochs),
        "--steps-per-epoch", "12",
        "--seed", str(args.registration_seed),
        "--matching-iterations", "4",
        "--layer-confidences", "1,1,1",
        "--min-final-matches", "3",
        "--max-final-matches", "0",
        "--relaxed-spatial",
        "--warm-start", str(WARM_START),
    ]
    if args.dry_run:
        print("DRY RUN:", " ".join(command))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    write_config(args, out, loss, weights)
    print("Running registration only (precomputed restored images):")
    print("  round1:", args.round1)
    print("  round2:", args.round2)
    print("  out   :", out)
    started = time.time()
    code = subprocess.run(command, cwd=str(ROOT / "registration"), env=env).returncode
    print(f"\nRegistration exit code: {code}  elapsed: {time.time() - started:.1f}s")
    result = out / "registration" / "registration_results" / dataset
    if code == 0 and result.is_dir():
        target = out / "results"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(result, target)
        metrics = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
        validation = json.loads((target / "validation_report.json").read_text(encoding="utf-8"))
        summary = {"pair": dataset, "status": metrics.get("status"),
                   "Nred": metrics.get("Nred"), "RMSEall": metrics.get("RMSEall"),
                   "RMSEloo": metrics.get("RMSEloo"), "Pquad": metrics.get("Pquad"), "NCC": metrics.get("NCC"),
                   "effective": validation.get("effective"), "results": str(target)}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n==== Registration metrics (final stage) ====")
        print(f"Nred={summary['Nred']}  RMSEall={summary['RMSEall']}  "
              f"RMSEloo={summary['RMSEloo']}  Pquad={summary['Pquad']}  NCC={summary['NCC']}  effective={summary['effective']}")
        print(f"Results: {target}")
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, help="Directory containing *_active and *_passive[_PGAxx] image pairs")
    parser.add_argument("--sensed", type=Path, help="Single-pair sensed/passive image")
    parser.add_argument("--reference", type=Path, help="Single-pair reference/active image")
    parser.add_argument("--output-dir", type=Path, help="Output directory; default outputs/run_<timestamp>")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0 or cpu")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 4))
    parser.add_argument("--restoration-steps", type=int, default=300)
    parser.add_argument("--restoration-seed", type=int, default=2026)
    parser.add_argument("--registration-seed", type=int, default=20260709)
    parser.add_argument("--contrastive-epochs", type=int, default=100)
    parser.add_argument("--expected-pairs", type=int)
    parser.add_argument("--round1", type=Path, help="Precomputed round-1 restored image; skip restoration")
    parser.add_argument("--round2", type=Path, help="Precomputed round-2 restored image; skip restoration")
    parser.add_argument("--gradient-weight", type=float, help="Override selected gradient weight")
    parser.add_argument("--structure-weight", type=float, help="Override selected LNCC/structure weight")
    parser.add_argument("--teacher-weight", type=float, help="Override selected teacher weight")
    parser.add_argument("--check-env", action="store_true", help="Print Python/PyTorch/OpenCV versions and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.check_env:
        return check_environment()

    if bool(args.sensed) != bool(args.reference):
        parser.error("--sensed and --reference must be provided together")
    if not args.input_dir and not args.sensed:
        parser.error("provide --input-dir or --sensed/--reference")
    if bool(args.round1) != bool(args.round2):
        parser.error("--round1 and --round2 must be provided together")
    if args.input_dir and not args.input_dir.is_dir():
        parser.error(f"--input-dir does not exist: {args.input_dir}")
    for path in (args.sensed, args.reference, args.round1, args.round2):
        if path is not None and not path.is_file():
            parser.error(f"input file does not exist: {path}")
    if args.workers < 1 or args.restoration_steps < 1 or args.contrastive_epochs < 1:
        parser.error("workers, restoration steps and contrastive epochs must be positive")

    loss = load_loss()
    weights = {
        "gradient": loss["gradient"] if args.gradient_weight is None else args.gradient_weight,
        "structure": loss["structure"] if args.structure_weight is None else args.structure_weight,
        "teacher": loss["teacher"] if args.teacher_weight is None else args.teacher_weight,
    }
    if any(value < 0 for value in weights.values()):
        parser.error("loss weights must be nonnegative")

    out = (args.output_dir or timestamped_output()).resolve()
    if out.exists() and not args.round1 and not args.dry_run:
        # The full pipeline can resume safely; registration-only will overwrite results/.
        print(f"NOTE: output directory exists, continuing: {out}")
    env = make_env(loss, weights["gradient"], weights["structure"], weights["teacher"])
    if args.round1:
        return run_registration_only(args, out, loss, weights, env)
    return run_full(args, out, loss, weights, env)


if __name__ == "__main__":
    raise SystemExit(main())
