"""Fine-tune a pretrained YOLO detector for single-class smoke detection.

Examples
--------
Baseline (reads model/train_config.yaml):
    python code/train.py

Controlled experiment, one change at a time:
    python code/train.py --set imgsz=960 name=exp_imgsz960
    python code/train.py --config model/exp_yolo26s.yaml

Resume an interrupted run (e.g. Colab disconnect):
    python code/train.py --resume runs/baseline/weights/last.pt

Promote a run's best checkpoint to model/best.pt (and save its exact config):
    python code/train.py --set name=final --promote

Each run folder gets `run_info.json` with the merged config, seed, software
versions, hardware and git commit, which is what the brief asks you to record.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SEED,
    MODEL_DIR,
    REPO_ROOT,
    environment_info,
    load_yaml,
    parse_overrides,
    resolve_repo_path,
    save_json,
    set_seed,
)


def build_config(config_path: Path, overrides: dict) -> dict:
    cfg = load_yaml(config_path)
    cfg.update(overrides)
    cfg.setdefault("seed", DEFAULT_SEED)
    cfg.setdefault("deterministic", True)
    # Make repo-relative paths absolute so the script works from any directory.
    cfg["data"] = str(resolve_repo_path(cfg.get("data", "dataset/data.yaml")))
    cfg["project"] = str(resolve_repo_path(cfg.get("project", "runs")))
    model = str(cfg.get("model", "yolo26n.pt"))
    if resolve_repo_path(model).exists():  # local checkpoint; otherwise let Ultralytics download it by name
        model = str(resolve_repo_path(model))
    cfg["model"] = model
    return cfg


def rel(p: Path) -> str:
    """Repo-relative path when possible, absolute otherwise."""
    p = Path(p).resolve()
    return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)


def promote(run_dir: Path, cfg: dict) -> None:
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"[warn] {best} not found; nothing promoted.")
        return
    MODEL_DIR.mkdir(exist_ok=True)
    cfg = {**cfg, "data": rel(cfg["data"]), "project": rel(cfg["project"])}
    shutil.copy2(best, MODEL_DIR / "best.pt")
    with open(MODEL_DIR / "best_config.yaml", "w") as f:
        f.write(f"# Exact config that produced model/best.pt (run: {rel(run_dir)})\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Promoted {best} -> {MODEL_DIR / 'best.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="Training YAML (default model/train_config.yaml)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override config keys")
    ap.add_argument("--device", help="cuda device like 0, or cpu / mps (default: auto)")
    ap.add_argument("--resume", help="Path to a last.pt to resume an interrupted run")
    ap.add_argument("--promote", action="store_true", help="Copy best.pt to model/best.pt when done")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        run_dir = Path(model.trainer.save_dir)
        print(f"Resumed run finished: {run_dir}")
        return

    cfg = build_config(Path(args.config), parse_overrides(args.set))
    if args.device is not None:
        cfg["device"] = args.device
    set_seed(int(cfg["seed"]), bool(cfg["deterministic"]))

    model_name = cfg.pop("model")
    print(f"Training {model_name} with seed {cfg['seed']} on {cfg['data']}")
    model = YOLO(model_name)
    t0 = time.time()
    model.train(**cfg)
    elapsed = time.time() - t0

    run_dir = Path(model.trainer.save_dir)
    full_cfg = {"model": model_name, **cfg}
    info = {
        "config_file": rel(Path(args.config)),
        "overrides": args.set,
        "config": full_cfg,
        "train_seconds": round(elapsed, 1),
        "environment": environment_info(),
        "command": " ".join(sys.argv),
    }
    save_json(info, run_dir / "run_info.json")
    print(f"Run folder: {run_dir}")
    print(f"Best weights: {run_dir / 'weights' / 'best.pt'}")

    if args.promote:
        promote(run_dir, full_cfg)


if __name__ == "__main__":
    main()
