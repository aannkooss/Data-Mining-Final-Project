"""Build leakage-free train/val splits grouped by fire event (or camera).

Why: neighbouring frames of the same fire look almost identical. If one frame
lands in train and the next in val, validation scores are inflated (brief,
section 4: "split at sequence, camera, location, or event level").

How images are grouped, in priority order:
  1. --groups CSV with columns `image,group` (e.g. camera or event names you
     record by hand). Images listed there use that group.
  2. The instructor file names look like `<unix_time>_<offset>.jpg`, e.g.
     `1610044421_+00120.jpg` = taken 120 s after the fire's reference time.
     `unix_time - offset` is the same for every frame of a sequence, so it is
     the sequence id. Sequences whose start times fall within --event-gap
     seconds of each other are merged into one event, so two cameras watching
     the same fire never end up on both sides of the split.
  3. Anything else is its own group (prefix `file:`), and a warning is printed.

Whole groups are then assigned to val (seeded shuffle) until val holds about
--val-frac of the images, while keeping a similar share of smoke boxes.

Outputs (in --out, default dataset/):
  images/, labels/       symlinks (or copies with --copy) to your files
  train.txt, val.txt     YOLO split lists ("./images/<name>")
  split_manifest.csv     image, group, split, n_boxes
  split_summary.json     counts per split + the seed and settings used
  data.yaml              YOLO dataset config pointing at the split lists
  label_qc.csv           any label problems found (bad class, out of range,
                         zero size, duplicates); empty means all good
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASET_DIR, DEFAULT_SEED, IMAGE_EXTS, box_iou, read_yolo_labels, save_json  # noqa: E402

NAME_RE = re.compile(r"^(?P<ts>\d{9,11})_(?P<off>[+-]\d+)$")


def sequence_start(stem: str) -> int | None:
    m = NAME_RE.match(stem)
    if not m:
        return None
    return int(m.group("ts")) - int(m.group("off"))


def assign_groups(images: list[Path], manual: dict[str, str], event_gap: int) -> dict[str, str]:
    """Map image file name -> group id."""
    groups: dict[str, str] = {}
    starts: dict[str, int] = {}
    unparsed = []
    for img in images:
        if img.name in manual:
            groups[img.name] = manual[img.name]
            continue
        s = sequence_start(img.stem)
        if s is None:
            unparsed.append(img.name)
            groups[img.name] = f"file:{img.stem}"
        else:
            starts[img.name] = s

    # Merge sequence start times that are close together into one event.
    event_of_start: dict[int, str] = {}
    current, last = None, None
    for s in sorted(set(starts.values())):
        if last is None or s - last > event_gap:
            current = f"event_{s}"
        event_of_start[s] = current
        last = s
    for name, s in starts.items():
        groups[name] = event_of_start[s]

    if unparsed:
        print(
            f"[warn] {len(unparsed)} image name(s) do not match <unix_time>_<offset> and are not in --groups; "
            f"each is treated as its own group (e.g. {unparsed[:3]}). Add them to a groups CSV to avoid leakage."
        )
    return groups


def check_labels(label: Path, dup_iou: float = 0.95) -> list[str]:
    """Return a list of problems in one YOLO label file (brief section 4, quality control)."""
    if not label.exists():
        return []
    problems = []
    for n, line in enumerate(label.read_text().splitlines(), 1):
        vals = line.split()
        if not vals:
            continue
        if len(vals) != 5:
            problems.append(f"line {n}: expected 5 values, got {len(vals)}")
            continue
        try:
            c, xc, yc, w, h = (float(v) for v in vals)
        except ValueError:
            problems.append(f"line {n}: non-numeric value")
            continue
        if c != 0:
            problems.append(f"line {n}: class id {vals[0]} (only 0 = smoke is allowed)")
        if w <= 0 or h <= 0:
            problems.append(f"line {n}: zero or negative size")
        if xc - w / 2 < -1e-6 or yc - h / 2 < -1e-6 or xc + w / 2 > 1 + 1e-6 or yc + h / 2 > 1 + 1e-6:
            problems.append(f"line {n}: box extends outside the image")
    boxes = read_yolo_labels(label)
    if len(boxes) > 1:
        xyxy = np.stack(
            [boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 2] - boxes[:, 4] / 2,
             boxes[:, 1] + boxes[:, 3] / 2, boxes[:, 2] + boxes[:, 4] / 2], axis=1)
        iou = box_iou(xyxy, xyxy)
        np.fill_diagonal(iou, 0)
        for i, j in zip(*np.where(np.triu(iou) > dup_iou)):
            problems.append(f"boxes {i + 1} and {j + 1} look like duplicates (IoU {iou[i, j]:.2f})")
    return problems


def choose_val_groups(group_imgs: dict[str, list[str]], group_boxes: dict[str, int],
                      val_frac: float, seed: int, tries: int = 200) -> set[str]:
    """Pick whole groups for val so val has ~val_frac of images and of boxes.

    Tries several seeded shuffles (deterministic for a given seed) and keeps
    the one closest to the target on both counts.
    """
    names = sorted(group_imgs)
    total_imgs = sum(len(v) for v in group_imgs.values())
    total_boxes = max(1, sum(group_boxes.values()))
    rng = random.Random(seed)
    best, best_score = set(), float("inf")
    for _ in range(tries):
        order = names[:]
        rng.shuffle(order)
        chosen, n_img, n_box = set(), 0, 0
        for g in order:
            if n_img >= val_frac * total_imgs:
                break
            chosen.add(g)
            n_img += len(group_imgs[g])
            n_box += group_boxes[g]
        if len(chosen) == len(names):  # never put everything in val
            continue
        score = abs(n_img / total_imgs - val_frac) + abs(n_box / total_boxes - val_frac)
        if score < best_score:
            best, best_score = chosen, score
    return best


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="Folder with the selected images")
    ap.add_argument("--labels", required=True, help="Folder with YOLO .txt labels (same stem as images). "
                                                    "Images with no label file are treated as smoke-free negatives.")
    ap.add_argument("--out", default=str(DATASET_DIR), help="Dataset folder to write (default: dataset/)")
    ap.add_argument("--groups", help="Optional CSV with columns image,group to override grouping")
    ap.add_argument("--val-frac", type=float, default=0.2, help="Target share of images in val (default 0.2)")
    ap.add_argument("--event-gap", type=int, default=3600,
                    help="Sequences starting within this many seconds are one event (default 3600)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlinking (use for the final package)")
    ap.add_argument("--require-labels", action="store_true",
                    help="Fail if any image has no label file (use once labelling is finished)")
    args = ap.parse_args()

    img_dir, lbl_dir, out = Path(args.images), Path(args.labels), Path(args.out)
    images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"No images found in {img_dir}")
    names = [p.name for p in images]
    if len(set(names)) != len(names):
        sys.exit("Duplicate image file names found; names must be unique across subfolders.")

    manual = {}
    if args.groups:
        with open(args.groups) as f:
            manual = {row["image"]: row["group"] for row in csv.DictReader(f)}
    groups = assign_groups(images, manual, args.event_gap)

    # Labels + QC
    n_boxes, qc_rows, missing = {}, [], []
    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            missing.append(img.name)
        for problem in check_labels(lbl):
            qc_rows.append({"image": img.name, "problem": problem})
        n_boxes[img.name] = len(read_yolo_labels(lbl))
    if missing:
        msg = f"{len(missing)} image(s) have no label file and will be used as negatives (no smoke)."
        if args.require_labels:
            sys.exit("[error] " + msg)
        print("[warn] " + msg)

    group_imgs: dict[str, list[str]] = defaultdict(list)
    group_boxes: dict[str, int] = defaultdict(int)
    for name in names:
        group_imgs[groups[name]].append(name)
        group_boxes[groups[name]] += n_boxes[name]
    if len(group_imgs) < 2:
        sys.exit("Need at least two groups (events/cameras) to make a split.")

    val_groups = choose_val_groups(group_imgs, group_boxes, args.val_frac, args.seed)

    # Write dataset folder
    out.mkdir(parents=True, exist_ok=True)
    split_lists = {"train": [], "val": []}
    manifest = []
    for img in images:
        split = "val" if groups[img.name] in val_groups else "train"
        link_or_copy(img, out / "images" / img.name, args.copy)
        lbl = lbl_dir / f"{img.stem}.txt"
        dst_lbl = out / "labels" / f"{img.stem}.txt"
        if lbl.exists():
            link_or_copy(lbl, dst_lbl, copy=True)  # labels are small: always copy so they can be committed
        elif dst_lbl.exists():
            dst_lbl.unlink()
        split_lists[split].append(f"./images/{img.name}")
        manifest.append({"image": img.name, "group": groups[img.name], "split": split, "n_boxes": n_boxes[img.name]})

    for split, lines in split_lists.items():
        (out / f"{split}.txt").write_text("\n".join(lines) + "\n")

    with open(out / "split_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "group", "split", "n_boxes"])
        w.writeheader()
        w.writerows(manifest)
    with open(out / "label_qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "problem"])
        w.writeheader()
        w.writerows(qc_rows)

    data_yaml = {"train": "train.txt", "val": "val.txt", "nc": 1, "names": {0: "smoke"}}
    with open(out / "data.yaml", "w") as f:
        f.write("# Generated by code/make_splits.py. No `path:` key on purpose: Ultralytics then\n"
                "# resolves train/val relative to this file, so the dataset works from any directory.\n")
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    summary = {"seed": args.seed, "val_frac_target": args.val_frac, "event_gap_s": args.event_gap,
               "groups_csv": args.groups, "n_groups": len(group_imgs)}
    for split in ("train", "val"):
        rows = [m for m in manifest if m["split"] == split]
        summary[split] = {
            "images": len(rows),
            "positive_images": sum(1 for r in rows if r["n_boxes"] > 0),
            "negative_images": sum(1 for r in rows if r["n_boxes"] == 0),
            "boxes": sum(r["n_boxes"] for r in rows),
            "groups": len({r["group"] for r in rows}),
        }
    summary["val_groups"] = sorted(val_groups)
    save_json(summary, out / "split_summary.json")

    t, v = summary["train"], summary["val"]
    print(f"Groups: {len(group_imgs)}  (train {t['groups']}, val {v['groups']})")
    print(f"Train: {t['images']} images, {t['boxes']} boxes, {t['negative_images']} negatives")
    print(f"Val:   {v['images']} images, {v['boxes']} boxes, {v['negative_images']} negatives")
    print(f"Label QC problems: {len(qc_rows)} (see {out / 'label_qc.csv'})")
    print(f"Wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
