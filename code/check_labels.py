"""Quality-control checks for the YOLO smoke labels.

Checks every image listed in dataset/selection_manifest.csv for:
  * a missing label file (unlabeled image; an image with no smoke must
    still have an EMPTY .txt file)
  * malformed lines (not 5 numbers)
  * class ids other than 0 (smoke)
  * coordinates outside [0, 1] or boxes that spill past the image edge
  * zero or tiny boxes
  * near-duplicate boxes on the same image (IoU > 0.9)
  * label files with no matching image
  * pre-ignition frames that have boxes (they should be empty)

Prints a summary plus dataset statistics for the report.

Usage (from the repo root):
    python code/check_labels.py                 # checks dataset/labels
    python code/check_labels.py --labels some/other/labels_dir
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "dataset"
TINY = 4  # pixels; boxes smaller than this on either side are flagged


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def read_boxes(path):
    boxes, errors = [], []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {n}: expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"line {n}: not numeric")
            continue
        boxes.append((cls, x, y, w, h, n))
    return boxes, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(DATASET / "labels"),
                    help="labels root containing train/ and val/ (or a flat folder)")
    args = ap.parse_args()
    root = Path(args.labels)
    rows = list(csv.DictReader(open(DATASET / "selection_manifest.csv", newline="")))

    def label_path(r):
        p = root / r["split"] / (Path(r["dataset_file"]).stem + ".txt")
        return p if p.parent.exists() else root / (Path(r["dataset_file"]).stem + ".txt")

    problems = []
    stats = Counter()
    sizes = []
    per_split = Counter()
    expected = set()
    for r in rows:
        lp = label_path(r)
        expected.add(lp.resolve())
        W, H = int(r["width"]), int(r["height"])
        if not lp.exists():
            problems.append((r["dataset_file"], "missing label file (unlabeled?)"))
            stats["missing"] += 1
            continue
        boxes, errors = read_boxes(lp)
        for e in errors:
            problems.append((r["dataset_file"], e))
        if r["role"] == "negative_pre_ignition" and boxes:
            problems.append((r["dataset_file"], "pre-ignition frame has boxes; should be empty"))
        stats["images_with_smoke" if boxes else "images_empty"] += 1
        per_split[(r["split"], "boxes")] += len(boxes)
        per_split[(r["split"], "images")] += 1
        for cls, x, y, w, h, n in boxes:
            if cls != 0:
                problems.append((r["dataset_file"], f"line {n}: class {cls}, expected 0 (smoke)"))
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                problems.append((r["dataset_file"], f"line {n}: value outside [0,1]"))
            if x - w / 2 < -1e-3 or x + w / 2 > 1 + 1e-3 or y - h / 2 < -1e-3 or y + h / 2 > 1 + 1e-3:
                problems.append((r["dataset_file"], f"line {n}: box extends past image edge"))
            if w * W < TINY or h * H < TINY:
                problems.append((r["dataset_file"], f"line {n}: tiny box ({w * W:.0f}x{h * H:.0f} px)"))
            sizes.append((w * W, h * H))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if iou(boxes[i][1:5], boxes[j][1:5]) > 0.9:
                    problems.append((r["dataset_file"], f"lines {boxes[i][5]} and {boxes[j][5]}: duplicate boxes"))

    orphans = [p for p in root.rglob("*.txt") if p.resolve() not in expected and p.name != "classes.txt"]
    for p in orphans:
        problems.append((str(p.relative_to(root)), "label file has no image in the manifest"))

    print(f"Checked {len(rows)} images in {root}")
    for split in ("train", "val"):
        print(f"  {split}: {per_split[(split, 'images')]} labeled images, {per_split[(split, 'boxes')]} boxes")
    print(f"  with smoke: {stats['images_with_smoke']}, empty: {stats['images_empty']}, missing: {stats['missing']}")
    if sizes:
        areas = sorted(w * h for w, h in sizes)
        q = lambda f: areas[min(len(areas) - 1, int(f * len(areas)))]
        small = sum(a < 32 * 32 for a in areas)
        medium = sum(32 * 32 <= a < 96 * 96 for a in areas)
        print(f"  box area px^2: median {q(0.5):.0f}, p10 {q(0.1):.0f}, p90 {q(0.9):.0f}")
        print(f"  COCO size buckets: small {small}, medium {medium}, large {len(areas) - small - medium}")
    print(f"\n{len(problems)} problem(s)")
    for f, msg in problems[:200]:
        print(f"  {f}: {msg}")
    if len(problems) > 200:
        print(f"  ... and {len(problems) - 200} more")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
