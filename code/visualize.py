"""Figures for quality control, error analysis, and the report.

Subcommands
-----------
labels   Draw ground-truth boxes on a random sample of images (label QC,
         including the 10% cross-check).
           python code/visualize.py labels --data dataset/data.yaml --split train --n 40

errors   From an eval run, save the most confident false positives, the
         missed smoke (false negatives), and good detections, each image
         annotated: green = ground truth, blue = correct detection,
         red = false positive, orange = missed smoke.
           python code/visualize.py errors --eval runs/eval/baseline --k 12

curves   Precision / recall / F1 against confidence threshold for one or more
         eval runs, with the chosen threshold marked.
           python code/visualize.py curves --eval runs/eval/baseline runs/eval/exp_imgsz960

compare  Bar chart + markdown table comparing experiments (for the report).
           python code/visualize.py compare --eval runs/eval/baseline runs/eval/exp_imgsz960 runs/eval/exp_yolo26s

Everything is written to --out (default: runs/figures/).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_SEED,
    RUNS_DIR,
    label_path_for,
    read_yolo_labels,
    split_images,
    xywhn_to_xyxy,
)

GREEN, BLUE, RED, ORANGE = (60, 200, 60), (230, 140, 30), (40, 40, 230), (0, 150, 255)  # BGR


def draw_box(img, box, color, text=None, thick=None):
    h, w = img.shape[:2]
    t = thick or max(2, int(round(min(h, w) / 300)))
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t)
    if text:
        scale = max(0.5, min(h, w) / 1000)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, t // 2))
        y_text = max(th + 4, y1)
        cv2.rectangle(img, (x1, y_text - th - 4), (x1 + tw + 4, y_text), color, -1)
        cv2.putText(img, text, (x1 + 2, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                    max(1, t // 2), cv2.LINE_AA)


def caption(img, text):
    """Return the image with a black title bar added above it (so boxes stay visible)."""
    h, w = img.shape[:2]
    scale = max(0.6, min(h, w) / 900)
    bar = int(40 * scale)
    img = cv2.copyMakeBorder(img, bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(img, text, (8, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def contact_sheet(paths: list[Path], out_path: Path, cols: int = 4, width: int = 480):
    if not paths:
        return
    tiles = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        s = width / im.shape[1]
        tiles.append(cv2.resize(im, (width, int(im.shape[0] * s))))
    if not tiles:
        return
    cols = min(cols, len(tiles))
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_path), np.vstack(rows))


# --------------------------------------------------------------------------- #
def cmd_labels(args):
    out = Path(args.out) / f"labels_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    images = split_images(args.data, args.split)
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(images, min(args.n, len(images))))
    saved = []
    for p in sample:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = read_yolo_labels(label_path_for(p))
        for b in xywhn_to_xyxy(gt[:, 1:], w, h) if len(gt) else []:
            draw_box(im, b, GREEN, "smoke")
        im = caption(im, f"{p.name}  |  {len(gt)} box(es)")
        dst = out / p.name
        cv2.imwrite(str(dst), im)
        saved.append(dst)
    contact_sheet(saved, out / "_contact_sheet.jpg", cols=args.cols)
    # A checklist file for the independent 10% review.
    with open(out / "_review_checklist.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "n_boxes", "reviewer", "ok (y/n)", "issue"])
        for p in sample:
            wr.writerow([p.name, len(read_yolo_labels(label_path_for(p))), "", "", ""])
    print(f"Saved {len(saved)} annotated images + contact sheet + review checklist to {out}")


def cmd_errors(args):
    ev = Path(args.eval)
    out = Path(args.out) / f"errors_{ev.name}"
    by_img = defaultdict(list)
    with open(ev / "boxes.csv") as f:
        for r in csv.DictReader(f):
            by_img[r["path"]].append(r)
    metrics = json.loads((ev / "metrics.json").read_text())
    conf = metrics["chosen_threshold"]["conf"]

    def render(path: str, title: str, dst: Path):
        im = cv2.imread(path)
        if im is None:
            return None
        h, w = im.shape[:2]
        gt = read_yolo_labels(label_path_for(Path(path)))
        for b in xywhn_to_xyxy(gt[:, 1:], w, h) if len(gt) else []:
            draw_box(im, b, GREEN)
        for r in by_img[path]:
            b = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
            if r["kind"] == "TP":
                draw_box(im, b, BLUE, f"{float(r['score']):.2f}")
            elif r["kind"] == "FP":
                draw_box(im, b, RED, f"FP {float(r['score']):.2f}")
            else:
                draw_box(im, b, ORANGE, "missed", thick=max(3, int(min(h, w) / 200)))
        im = caption(im, title)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), im)
        return dst

    fps = sorted((r for rs in by_img.values() for r in rs if r["kind"] == "FP"), key=lambda r: -float(r["score"]))
    fns = sorted((r for rs in by_img.values() for r in rs if r["kind"] == "FN"), key=lambda r: float(r["rel_area"]))
    tps = sorted((r for rs in by_img.values() for r in rs if r["kind"] == "TP"), key=lambda r: -float(r["iou"]))

    def take(rows, k):  # one entry per image so the gallery is varied
        seen, picked = set(), []
        for r in rows:
            if r["path"] not in seen:
                seen.add(r["path"])
                picked.append(r)
            if len(picked) == k:
                break
        return picked

    groups = {
        "false_positives": (take(fps, args.k), lambda r: f"FP score {float(r['score']):.2f} | {r['image']}"),
        "false_negatives": (take(fns, args.k), lambda r: f"Missed, box {100 * float(r['rel_area']):.2f}% of image | {r['image']}"),
        "true_positives": (take(tps, args.k), lambda r: f"TP IoU {float(r['iou']):.2f} | {r['image']}"),
    }
    for name, (rows, title) in groups.items():
        saved = [s for r in rows if (s := render(r["path"], title(r), out / name / r["image"]))]
        contact_sheet(saved, out / f"{name}_sheet.jpg", cols=args.cols)
        print(f"{name}: {len(saved)} image(s)")
    print(f"Threshold used: {conf}. Figures in {out}")
    print("Legend: green = ground truth, blue = correct detection, red = false positive, orange = missed smoke")


def _load_sweep(ev: Path):
    with open(ev / "threshold_sweep.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("conf", "precision", "recall", "f1")}


def cmd_curves(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ev in map(Path, args.eval):
        s = _load_sweep(ev)
        chosen = json.loads((ev / "metrics.json").read_text())["chosen_threshold"]["conf"]
        for ax, key in zip(axes, ("precision", "recall", "f1")):
            line, = ax.plot(s["conf"], s[key], label=ev.name, lw=2)
            ax.axvline(chosen, color=line.get_color(), ls=":", lw=1)
    for ax, key in zip(axes, ("Precision", "Recall", "F1")):
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel(key)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("Validation metrics vs confidence threshold (dotted = chosen threshold)")
    fig.tight_layout()
    dst = out / "threshold_curves.png"
    fig.savefig(dst, dpi=150)
    print(f"Saved {dst}")


def cmd_compare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ev in map(Path, args.eval):
        m = json.loads((ev / "metrics.json").read_text())
        c = m["chosen_threshold"]
        rows.append({"run": ev.name, "mAP50_95": m["mAP50_95"], "AP50": m["AP50"], "precision": c["precision"],
                     "recall": c["recall"], "f1": c["f1"], "conf": c["conf"],
                     "ms_per_img": m.get("speed_ms_per_image", {}).get("inference")})
    keys = ["mAP50_95", "AP50", "precision", "recall", "f1"]
    x = np.arange(len(keys))
    wbar = 0.8 / len(rows)
    fig, ax = plt.subplots(figsize=(max(7, 2 + 1.6 * len(keys)), 4.2))
    for i, r in enumerate(rows):
        vals = [r[k] for k in keys]
        bars = ax.bar(x + i * wbar - 0.4 + wbar / 2, vals, wbar, label=r["run"])
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, ["mAP@.5:.95", "AP@.5", "Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Internal validation: experiment comparison (P/R/F1 at each run's chosen threshold)")
    fig.tight_layout()
    fig.savefig(out / "experiment_comparison.png", dpi=150)

    md = ["| Run | mAP@.5:.95 | AP@.5 | Precision | Recall | F1 | Threshold | ms/img |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ms = f"{r['ms_per_img']:.1f}" if r["ms_per_img"] is not None else ""
        md.append(f"| {r['run']} | {r['mAP50_95']:.3f} | {r['AP50']:.3f} | {r['precision']:.3f} | "
                  f"{r['recall']:.3f} | {r['f1']:.3f} | {r['conf']:.2f} | {ms} |")
    (out / "experiment_comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"Saved {out / 'experiment_comparison.png'} and .md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RUNS_DIR / "figures"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("labels")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(fn=cmd_labels)

    p = sub.add_parser("errors")
    p.add_argument("--eval", required=True, help="An eval output folder, e.g. runs/eval/baseline")
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--cols", type=int, default=3)
    p.set_defaults(fn=cmd_errors)

    p = sub.add_parser("curves")
    p.add_argument("--eval", nargs="+", required=True)
    p.set_defaults(fn=cmd_curves)

    p = sub.add_parser("compare")
    p.add_argument("--eval", nargs="+", required=True)
    p.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
