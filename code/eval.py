"""Evaluate a trained smoke detector on the internal validation split.

Reports the brief's metrics (mAP@0.5:0.95, AP@0.5, precision, recall, F1),
picks a confidence threshold from validation data only, and writes the
per-image and per-box tables that visualize.py uses for error analysis.

Examples
--------
    python code/eval.py --weights runs/baseline/weights/best.pt --name baseline
    python code/eval.py --weights model/best.pt --name final --save-threshold

Outputs in runs/eval/<name>/:
  metrics.json          headline metrics + chosen threshold + speed
  threshold_sweep.csv   precision / recall / F1 at each confidence threshold
  per_image.csv         TP / FP / FN counts per validation image
  boxes.csv             every prediction and every ground-truth box, tagged
                        TP, FP or FN at the chosen threshold
  (plus the Ultralytics PR/F1 curves and confusion matrix)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_SEED,
    DEFAULT_WEIGHTS,
    RUNS_DIR,
    THRESHOLD_FILE,
    environment_info,
    label_path_for,
    match_boxes,
    read_yolo_labels,
    save_json,
    set_seed,
    split_images,
    xywhn_to_xyxy,
)

SWEEP = np.round(np.concatenate([[0.001, 0.005, 0.01, 0.02, 0.03, 0.04], np.arange(0.05, 0.951, 0.01)]), 3)


def sweep_thresholds(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> list[dict]:
    """P/R/F1 at each threshold.

    Matching is greedy by score, so the matches of high-score boxes do not
    depend on lower-score ones; one matching at the lowest conf is therefore
    valid for every higher threshold.
    """
    rows = []
    for t in SWEEP:
        keep = scores >= t
        tp = int(is_tp[keep].sum())
        fp = int(keep.sum() - tp)
        fn = n_gt - tp
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / n_gt if n_gt else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"conf": float(t), "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    return rows


def pick_threshold(rows: list[dict], min_recall: float | None) -> dict:
    if min_recall is not None:
        ok = [r for r in rows if r["recall"] >= min_recall]
        if ok:  # highest precision that still meets the recall target
            return max(ok, key=lambda r: (r["precision"], r["f1"]))
        print(f"[warn] no threshold reaches recall {min_recall}; falling back to best F1.")
    return max(rows, key=lambda r: (r["f1"], r["recall"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640, help="Use the same size the model was trained at")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--iou-match", type=float, default=0.5, help="IoU for counting a TP in the error tables")
    ap.add_argument("--min-recall", type=float, default=None,
                    help="Pick the most precise threshold with at least this recall instead of max F1")
    ap.add_argument("--name", default="eval", help="Output folder name under runs/eval/")
    ap.add_argument("--save-threshold", action="store_true",
                    help="Write the chosen threshold to model/threshold.json (predict.py uses it)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    from ultralytics import YOLO

    out = RUNS_DIR / "eval" / args.name
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    # 1) Official-style metrics from Ultralytics (COCO-style mAP, conf=0.001).
    val = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch, conf=0.001,
                    device=args.device, single_cls=True, plots=True, project=str(out.parent),
                    name=args.name, exist_ok=True, verbose=False)
    box = val.box

    # 2) Our own predictions for threshold choice and error analysis.
    images = split_images(args.data, args.split)
    all_scores, all_tp, n_gt_total = [], [], 0
    per_image, per_box_raw = [], []
    for res in model.predict([str(p) for p in images], imgsz=args.imgsz, conf=0.001, device=args.device,
                             stream=True, verbose=False):
        img = Path(res.path)
        h, w = res.orig_shape
        gt = read_yolo_labels(label_path_for(img))
        gt_xyxy = xywhn_to_xyxy(gt[:, 1:], w, h) if len(gt) else np.zeros((0, 4))
        pred = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros(0)
        is_tp, ious, gt_hit_any = match_boxes(pred, scores, gt_xyxy, args.iou_match)
        all_scores.append(scores)
        all_tp.append(is_tp)
        n_gt_total += len(gt_xyxy)
        per_box_raw.append((img, w, h, gt_xyxy, pred, scores, ious))

    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    is_tp = np.concatenate(all_tp) if all_tp else np.zeros(0, dtype=bool)
    sweep = sweep_thresholds(scores, is_tp, n_gt_total)
    chosen = pick_threshold(sweep, args.min_recall)
    conf = chosen["conf"]

    # 3) Tag every box at the chosen threshold.
    box_rows = []
    img_pos_total = img_pos_hit = img_neg_total = img_neg_alarm = 0
    for img, w, h, gt_xyxy, pred, sc, _ in per_box_raw:
        keep = sc >= conf
        p_k, s_k = pred[keep], sc[keep]
        tp_k, iou_k, gt_hit = match_boxes(p_k, s_k, gt_xyxy, args.iou_match)
        for b, s, t, i in zip(p_k, s_k, tp_k, iou_k):
            box_rows.append({"image": img.name, "path": str(img), "kind": "TP" if t else "FP", "score": float(s),
                             "iou": float(i), "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                             "rel_area": float((b[2] - b[0]) * (b[3] - b[1]) / (w * h))})
        for b, hit in zip(gt_xyxy, gt_hit):
            if not hit:
                box_rows.append({"image": img.name, "path": str(img), "kind": "FN", "score": "", "iou": "",
                                 "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                                 "rel_area": float((b[2] - b[0]) * (b[3] - b[1]) / (w * h))})
        n_tp = int(tp_k.sum())
        per_image.append({"image": img.name, "path": str(img), "gt": len(gt_xyxy), "pred": int(keep.sum()),
                          "tp": n_tp, "fp": int(keep.sum()) - n_tp, "fn": len(gt_xyxy) - n_tp,
                          "max_score": float(s_k.max()) if len(s_k) else 0.0})
        if len(gt_xyxy):
            img_pos_total += 1
            img_pos_hit += int(keep.any())
        else:
            img_neg_total += 1
            img_neg_alarm += int(keep.any())

    metrics = {
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "n_images": len(images),
        "n_gt_boxes": n_gt_total,
        "mAP50_95": float(box.map),
        "AP50": float(box.map50),
        "ultralytics_precision": float(box.mp),
        "ultralytics_recall": float(box.mr),
        "chosen_threshold": {"rule": f"min_recall={args.min_recall}" if args.min_recall else "max_f1",
                             "iou_match": args.iou_match, **chosen},
        "image_level": {
            "smoke_images_detected": f"{img_pos_hit}/{img_pos_total}",
            "smoke_image_recall": img_pos_hit / img_pos_total if img_pos_total else None,
            "clean_images_with_false_alarm": f"{img_neg_alarm}/{img_neg_total}",
            "false_alarm_rate": img_neg_alarm / img_neg_total if img_neg_total else None,
        },
        "speed_ms_per_image": {k: round(v, 2) for k, v in val.speed.items()},
        "environment": environment_info(),
    }
    save_json(metrics, out / "metrics.json")

    def write_csv(path, rows, fields):
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)

    write_csv(out / "threshold_sweep.csv", sweep, ["conf", "precision", "recall", "f1", "tp", "fp", "fn"])
    write_csv(out / "per_image.csv", per_image, ["image", "path", "gt", "pred", "tp", "fp", "fn", "max_score"])
    write_csv(out / "boxes.csv", box_rows, ["image", "path", "kind", "score", "iou", "x1", "y1", "x2", "y2", "rel_area"])

    if args.save_threshold:
        save_json({"conf": conf, "rule": metrics["chosen_threshold"]["rule"], "weights": str(args.weights),
                   "eval": str(out)}, THRESHOLD_FILE)
        print(f"Saved threshold {conf} to {THRESHOLD_FILE}")

    print(f"\nmAP@0.5:0.95 {metrics['mAP50_95']:.4f} | AP@0.5 {metrics['AP50']:.4f}")
    print(f"At conf {conf:.2f}: P {chosen['precision']:.3f}  R {chosen['recall']:.3f}  F1 {chosen['f1']:.3f}  "
          f"(TP {chosen['tp']}, FP {chosen['fp']}, FN {chosen['fn']})")
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
