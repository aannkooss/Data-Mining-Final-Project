"""Shared helpers for the wildfire smoke YOLO pipeline.

Everything that more than one script needs lives here: repo paths, seeding,
config loading, YOLO label I/O, and box matching.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
DATASET_DIR = REPO_ROOT / "dataset"
MODEL_DIR = REPO_ROOT / "model"
RUNS_DIR = REPO_ROOT / "runs"

DEFAULT_CONFIG = MODEL_DIR / "train_config.yaml"
DEFAULT_DATA = DATASET_DIR / "data.yaml"
DEFAULT_WEIGHTS = MODEL_DIR / "best.pt"
THRESHOLD_FILE = MODEL_DIR / "threshold.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "smoke"}
DEFAULT_SEED = 42


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    """Seed python, numpy and torch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional for the split script
        pass


def environment_info() -> dict:
    """Software and hardware versions, recorded with every run (brief section 4)."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["gpu"] = "Apple MPS"
    except ImportError:
        pass
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    try:
        info["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
        info["git_dirty"] = bool(dirty.strip())
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def parse_overrides(pairs: list[str]) -> dict:
    """Turn ['epochs=50', 'lr0=0.005'] into {'epochs': 50, 'lr0': 0.005}."""
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must look like key=value")
        key, value = pair.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def resolve_repo_path(p: str | Path) -> Path:
    """Paths in configs are relative to the repo root, not the current directory."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_threshold(default: float = 0.25) -> float:
    if THRESHOLD_FILE.exists():
        with open(THRESHOLD_FILE) as f:
            return float(json.load(f)["conf"])
    return default


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def list_images(source: str | Path) -> list[Path]:
    """Images from a directory (recursive), a single image, or a .txt list."""
    source = Path(source)
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if source.suffix.lower() == ".txt":
        return read_image_list(source)
    if source.suffix.lower() in IMAGE_EXTS:
        return [source]
    raise FileNotFoundError(f"No images found at {source}")


def read_image_list(txt: str | Path) -> list[Path]:
    """Read a YOLO split file; './x' lines are relative to the file's folder."""
    txt = Path(txt)
    out = []
    for line in txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        if not p.is_absolute():
            # normpath, not resolve(): resolve() would follow symlinks out of
            # dataset/images/ and we would no longer find dataset/labels/.
            p = Path(os.path.normpath(txt.parent.absolute() / line))
        out.append(p)
    return out


def split_images(data_yaml: str | Path, split: str = "val") -> list[Path]:
    """Resolve the image list for one split of a YOLO data.yaml."""
    data_yaml = Path(data_yaml)
    cfg = load_yaml(data_yaml)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = Path(os.path.normpath(data_yaml.parent.absolute() / root))
    entry = cfg[split]
    entries = entry if isinstance(entry, list) else [entry]
    images = []
    for e in entries:
        images += list_images(root / e)
    return images


def label_path_for(image: Path) -> Path:
    """YOLO convention: .../images/x.jpg -> .../labels/x.txt."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_yolo_labels(path: Path) -> np.ndarray:
    """Return an (N, 5) array of [cls, xc, yc, w, h] (normalized). Missing file = no boxes."""
    if not path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in path.read_text().splitlines():
        vals = line.split()
        if len(vals) >= 5:
            rows.append([float(v) for v in vals[:5]])
    return np.array(rows, dtype=np.float32).reshape(-1, 5)


def xywhn_to_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    xc, yc, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
    return np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], axis=1)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (N,4) and (M,4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_boxes(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5):
    """Greedy one-to-one matching, highest score first (the usual VOC/COCO rule).

    Returns (pred_is_tp, pred_iou, gt_matched) boolean/float arrays.
    """
    order = np.argsort(-scores)
    ious = box_iou(pred, gt)
    pred_tp = np.zeros(len(pred), dtype=bool)
    pred_iou = np.zeros(len(pred), dtype=np.float32)
    gt_used = np.zeros(len(gt), dtype=bool)
    for i in order:
        if len(gt) == 0:
            break
        cand = np.where(~gt_used)[0]
        if len(cand) == 0:
            break
        j = cand[np.argmax(ious[i, cand])]
        pred_iou[i] = ious[i, j]
        if ious[i, j] >= iou_thr:
            pred_tp[i] = True
            gt_used[j] = True
    return pred_tp, pred_iou, gt_used
