"""Compare the main labels with a teammate's independent labels of the QC sample.

The reviewer labels the images in dataset/qc_sample.txt from scratch (without
looking at the main labels) and exports YOLO .txt files into one folder, e.g.
dataset/labels_qc/. This script matches boxes between the two label sets by
IoU and reports agreement, then lists every image that needs discussion.

Usage (from the repo root):
    python code/qc_compare.py --qc dataset/labels_qc
    python code/qc_compare.py --qc dataset/labels_qc --csv dataset/qc_results.csv
"""
import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "dataset"


def load(path):
    if not path.exists():
        return None
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            out.append(tuple(map(float, p[1:])))
    return out


def pick(root, split, stem):
    """Label file inside a train/val subfolder if present, else in a flat folder."""
    nested = root / split / f"{stem}.txt"
    return nested if nested.exists() else root / f"{stem}.txt"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match(a, b):
    """Greedy one-to-one matching by IoU. Returns list of IoUs for matched pairs."""
    pairs = sorted(((iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, ious = set(), set(), []
    for v, i, j in pairs:
        if v <= 0 or i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        ious.append(v)
    return ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", required=True, help="folder with the reviewer's YOLO .txt files")
    ap.add_argument("--labels", default=str(DATASET / "labels"), help="main labels root")
    ap.add_argument("--thr", type=float, default=0.5, help="IoU needed to count as agreement")
    ap.add_argument("--csv", help="optional per-image results file")
    args = ap.parse_args()

    manifest = {r["dataset_file"]: r for r in csv.DictReader(open(DATASET / "selection_manifest.csv", newline=""))}
    sample = [l.strip() for l in open(DATASET / "qc_sample.txt") if l.strip()]
    qc_root, main_root = Path(args.qc), Path(args.labels)

    results, all_ious = [], []
    tot_main = tot_qc = tot_match = 0
    for f in sample:
        stem = Path(f).stem
        split = manifest[f]["split"]
        m = load(pick(main_root, split, stem))
        q = load(pick(qc_root, split, stem))
        if m is None or q is None:
            results.append(dict(image=f, status="missing " + ("main" if m is None else "reviewer") + " labels"))
            continue
        ious = match(m, q)
        good = [v for v in ious if v >= args.thr]
        all_ious += ious
        tot_main += len(m)
        tot_qc += len(q)
        tot_match += len(good)
        agree = len(good) == len(m) == len(q)
        results.append(dict(
            image=f, main_boxes=len(m), reviewer_boxes=len(q), matched=len(good),
            mean_iou=f"{sum(ious) / len(ious):.2f}" if ious else "",
            status="agree" if agree else "discuss",
        ))

    n_ok = sum(r["status"] == "agree" for r in results)
    print(f"QC sample: {len(sample)} images ({len(sample) / len(manifest):.1%} of {len(manifest)})")
    print(f"  images in full agreement (same box count, every box IoU >= {args.thr}): {n_ok}/{len(results)}")
    if tot_main or tot_qc:
        p = tot_match / tot_qc if tot_qc else 0
        r = tot_match / tot_main if tot_main else 0
        print(f"  boxes: main {tot_main}, reviewer {tot_qc}, matched {tot_match}")
        print(f"  reviewer-vs-main precision {p:.2f}, recall {r:.2f}, F1 {2 * p * r / (p + r) if p + r else 0:.2f}")
    if all_ious:
        s = sorted(all_ious)
        print(f"  IoU of matched boxes: mean {sum(s) / len(s):.2f}, median {s[len(s) // 2]:.2f}")
    print("\nNeeds discussion:")
    for r in results:
        if r["status"] != "agree":
            print("  ", r)
    if args.csv:
        keys = ["image", "main_boxes", "reviewer_boxes", "matched", "mean_iou", "status"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
