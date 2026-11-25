"""Run the final smoke detector on new images.

This is the inference script the brief asks for: for every image it outputs
the image identifier, predicted boxes and class scores.

Examples
--------
    # Default: model/best.pt, threshold from model/threshold.json (or 0.25)
    python code/predict.py --source path/to/images --out predictions

    # For mAP scoring keep low-confidence boxes too
    python code/predict.py --source path/to/images --conf 0.001 --out preds_map

Outputs in --out:
  predictions.csv   one row per box: image_id, class_id, class_name, score,
                    x1, y1, x2, y2 (pixels), xc, yc, w, h (normalized 0-1),
                    image_width, image_height
  predictions.json  {image_id: [ {class_id, class_name, score, box_xyxy, box_xywhn}, ... ]}
                    every image appears, including those with no detections
  labels/<image_id>.txt  (with --save-txt) YOLO format "cls xc yc w h score"
  images/            (with --save-images) images with boxes drawn
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_NAMES, DEFAULT_SEED, DEFAULT_WEIGHTS, list_images, load_threshold, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Image file, folder (searched recursively) or .txt list")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=None,
                    help="Confidence threshold (default: model/threshold.json, else 0.25)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Use the size the model was trained at")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--save-txt", action="store_true")
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()

    set_seed(DEFAULT_SEED)
    from ultralytics import YOLO

    conf = args.conf if args.conf is not None else load_threshold()
    images = list_images(args.source)
    if not images:
        sys.exit(f"No images found in {args.source}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.save_txt:
        (out / "labels").mkdir(exist_ok=True)
    if args.save_images:
        (out / "images").mkdir(exist_ok=True)

    model = YOLO(args.weights)
    print(f"Predicting on {len(images)} image(s) with {args.weights}, conf={conf}, imgsz={args.imgsz}")

    rows, by_image = [], {}
    t0 = time.time()
    for res in model.predict([str(p) for p in images], conf=conf, iou=args.iou, imgsz=args.imgsz,
                             max_det=args.max_det, device=args.device, batch=args.batch,
                             stream=True, verbose=False):
        path = Path(res.path)
        image_id = path.stem
        h, w = res.orig_shape
        dets = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            xywhn = res.boxes.xywhn.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            for b, bn, s, c in zip(xyxy, xywhn, scores, classes):
                name = CLASS_NAMES.get(int(c), res.names.get(int(c), str(c)))
                dets.append({"class_id": int(c), "class_name": name, "score": round(float(s), 5),
                             "box_xyxy": [round(float(v), 2) for v in b],
                             "box_xywhn": [round(float(v), 6) for v in bn]})
                rows.append({"image_id": image_id, "class_id": int(c), "class_name": name,
                             "score": round(float(s), 5),
                             "x1": round(float(b[0]), 2), "y1": round(float(b[1]), 2),
                             "x2": round(float(b[2]), 2), "y2": round(float(b[3]), 2),
                             "xc": round(float(bn[0]), 6), "yc": round(float(bn[1]), 6),
                             "w": round(float(bn[2]), 6), "h": round(float(bn[3]), 6),
                             "image_width": w, "image_height": h})
        by_image[image_id] = dets
        if args.save_txt:
            lines = [f"{d['class_id']} " + " ".join(f"{v:.6f}" for v in d["box_xywhn"]) + f" {d['score']:.5f}"
                     for d in dets]
            (out / "labels" / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if args.save_images:
            res.save(filename=str(out / "images" / path.name))
    elapsed = time.time() - t0

    fields = ["image_id", "class_id", "class_name", "score", "x1", "y1", "x2", "y2",
              "xc", "yc", "w", "h", "image_width", "image_height"]
    with open(out / "predictions.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "predictions.json", "w") as f:
        json.dump(by_image, f, indent=1)

    n_with = sum(1 for d in by_image.values() if d)
    print(f"{len(rows)} boxes on {n_with}/{len(by_image)} images in {elapsed:.1f}s "
          f"({1000 * elapsed / len(by_image):.0f} ms/image incl. loading)")
    print(f"Wrote {out / 'predictions.csv'} and {out / 'predictions.json'}")


if __name__ == "__main__":
    main()
