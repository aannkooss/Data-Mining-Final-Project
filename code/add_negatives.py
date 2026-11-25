"""Add smoke-free images (negatives) to the dataset, with empty label files.

The instructor's images are all fire sequences, so they hold almost no
smoke-free frames. This script adds frames from another source (for example
the same HPWREN cameras on days with no fire) without any labeling: each
image gets an EMPTY label file, which tells YOLO "no smoke here".

Put the downloaded frames in one subfolder per camera, named exactly like the
camera in dataset/selection_manifest.csv (the name in the image overlay):

    ~/Downloads/negatives/
        Los Pinos South/       20240815_1400.jpg  20240816_0900.jpg
        Mt. San Miguel North/  20240920_1630.jpg

Then, from the repo root:

    python code/add_negatives.py --src ~/Downloads/negatives --dry-run   # preview
    python code/add_negatives.py --src ~/Downloads/negatives

For each image the script:
  * looks up the camera's site and puts the image in that site's split
    (train or val), so the camera-site split stays leak-free
  * copies it to dataset/images/<split>/NEG_<camera>_<name>.jpg
  * writes an empty dataset/labels/<split>/NEG_<camera>_<name>.txt
  * adds a row to dataset/selection_manifest.csv (role negative_external)
    and a line to dataset/splits/<split>.txt

Running it again with the same folder adds nothing twice. A camera that is
not in the manifest is refused unless you give its site with
--site "Camera name=Site name" (a new site goes to train).

Look at every image before adding it: there must be no smoke anywhere in the
frame. Nothing is deleted.
"""
import argparse
import csv
import re
import shutil
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "dataset"
MANIFEST = DATASET / "selection_manifest.csv"
EXTS = {".jpg", ".jpeg", ".png"}


def image_size(path):
    """(width, height) from the file header, without needing Pillow."""
    data = path.read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    raise ValueError(f"could not read image size of {path}")


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with one subfolder per camera")
    ap.add_argument("--source", default="hpwren_archive",
                    help="value for the manifest's source column (default: hpwren_archive)")
    ap.add_argument("--site", action="append", default=[], metavar="CAMERA=SITE",
                    help="site of a camera that is not in the manifest yet (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be added, change nothing")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    eol = "\r\n" if b"\r\n" in MANIFEST.read_bytes()[:4096] else "\n"
    with open(MANIFEST, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)

    site_of = {r["camera"]: r["site"] for r in rows}
    split_of_site = {r["site"]: r["split"] for r in rows}
    for item in args.site:
        cam, _, site = item.partition("=")
        site_of[cam.strip()] = site.strip()
    have = {r["dataset_file"] for r in rows}

    new_rows, refused = [], []
    for cam_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        camera = cam_dir.name
        if camera not in site_of:
            refused.append(camera)
            continue
        site = site_of[camera]
        split = split_of_site.get(site, "train")
        for img in sorted(p for p in cam_dir.iterdir() if p.suffix.lower() in EXTS):
            name = f"NEG_{slug(camera)}_{slug(img.stem)}{img.suffix.lower()}"
            if name in have:
                continue
            w, h = image_size(img)
            new_rows.append(dict(
                dataset_file=name, source_file=f"{camera}/{img.name}", split=split, event="NEG",
                camera=camera, site=site, ignition_utc="", offset_s="", width=w, height=h,
                role="negative_external", qc_sample="no", source=args.source, _src=img,
            ))
            have.add(name)

    if refused:
        known = ", ".join(sorted(site_of))
        print("Skipped folders whose name is not a known camera:", *refused, sep="\n  ")
        print(f"Rename them to match a camera exactly, or pass --site \"Camera=Site\". Known cameras: {known}")

    for split in ("train", "val"):
        n = sum(r["split"] == split for r in new_rows)
        print(f"{split}: {n} new negative image(s)")
    for r in new_rows:
        print(f"  {r['split']:5}  {r['dataset_file']}   <- {r['source_file']}")
    if args.dry_run or not new_rows:
        print("Dry run, nothing changed." if args.dry_run else "Nothing to add.")
        return

    for r in new_rows:
        img_dir = DATASET / "images" / r["split"]
        lbl_dir = DATASET / "labels" / r["split"]
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r.pop("_src"), img_dir / r["dataset_file"])
        label = lbl_dir / (Path(r["dataset_file"]).stem + ".txt")
        if not label.exists():
            label.write_text("")
        with open(DATASET / "splits" / f"{r['split']}.txt", "a", newline="") as fh:
            fh.write(f"images/{r['split']}/{r['dataset_file']}\n")

    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator=eol)
        w.writeheader()
        w.writerows(rows + new_rows)
    print(f"Added {len(new_rows)} image(s) with empty labels. Manifest now has {len(rows) + len(new_rows)} rows.")
    print("Commit dataset/selection_manifest.csv and dataset/splits/, then run python code/check_labels.py")


if __name__ == "__main__":
    main()
