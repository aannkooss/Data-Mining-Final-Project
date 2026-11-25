# Code: smoke detector pipeline

Single-class (`smoke`) YOLO detector. Every script has `--help` with examples.
All scripts can be run from any directory; paths in configs are relative to the repo root.
The random seed is **42** everywhere (`model/train_config.yaml`, splits, sampling).

| Script | What it does |
|---|---|
| `prepare_dataset.py` | Copies the 578 picked images into `dataset/images/{train,val}` and writes `data.yaml` (site-level split) |
| `check_labels.py`, `qc_compare.py` | Label QC and the 10% independent-review comparison |
| `train.py` | Fine-tunes a pretrained YOLO with `model/train_config.yaml`, records seed + versions + hardware |
| `eval.py` | mAP@0.5:0.95, AP@0.5, P/R/F1, threshold choice, per-image and per-box error tables |
| `predict.py` | Final inference: image id, boxes, class scores → CSV + JSON |
| `visualize.py` | Label QC sheets, FP/FN galleries, threshold curves, experiment comparison |
| `common.py` | Shared helpers (seed, label I/O, IoU matching) |

## 0. Install

```bash
pip install -r code/requirements.txt
```

On Colab: `!git clone <repo> && cd Data-Mining-Final-Project && pip install -q -r code/requirements.txt`,
then run the same commands with a `!` in front. Use `--device 0` for the GPU.

## 1. Prepare the dataset

The images to label and the train/val split are fixed in `dataset/selection_manifest.csv`
(578 images, all 59 fires). Val holds 6 whole camera sites, so no fire or camera view appears
in both sets. Details: `dataset/data_source_inventory.md`.

```bash
python code/prepare_dataset.py --raw ~/Downloads/training   # copies images, writes dataset/data.yaml
# ...label in CVAT, export YOLO 1.1, put the .txt files in dataset/labels/{train,val}/
python code/check_labels.py                                 # missing files, bad ids, out-of-bounds, duplicates
```

Every image needs a label file; a smoke-free image gets an empty `.txt`.

Label QC sheet and a checklist for the 10% independent review:

```bash
python code/visualize.py labels --split train --n 40
```

## 2. Train the baseline

```bash
python code/train.py                     # uses model/train_config.yaml (yolo26n, 640px, 75 epochs, seed 42)
```

Results go to `runs/baseline/`. `run_info.json` there records the merged config, seed,
Python/torch/ultralytics versions, GPU and git commit (what the brief asks you to report).
If Colab disconnects: `python code/train.py --resume runs/baseline/weights/last.pt`.

## 3. Controlled experiments (change one thing at a time)

```bash
python code/train.py --set imgsz=960 name=exp_imgsz960          # input resolution
python code/train.py --set model=yolo26s.pt name=exp_yolo26s    # model size
python code/train.py --set mosaic=0.5 hsv_v=0.6 name=exp_aug    # augmentation
```

Or copy `model/train_config.yaml` to a new file, edit one setting, and pass `--config`.
Keep the split and seed fixed so the comparison is fair.

## 4. Evaluate on the internal validation split

```bash
python code/eval.py --weights runs/baseline/weights/best.pt --name baseline
python code/eval.py --weights runs/exp_imgsz960/weights/best.pt --imgsz 960 --name exp_imgsz960
```

Pass the `--imgsz` the model was trained at. Output in `runs/eval/<name>/`:
`metrics.json`, `threshold_sweep.csv`, `per_image.csv`, `boxes.csv`, Ultralytics PR/F1 curves.
The threshold is picked by max F1 on val; use `--min-recall 0.8` to prefer fewer missed fires instead.

## 5. Error analysis and report figures

```bash
python code/visualize.py errors  --eval runs/eval/baseline --k 12
python code/visualize.py curves  --eval runs/eval/baseline runs/eval/exp_imgsz960
python code/visualize.py compare --eval runs/eval/baseline runs/eval/exp_imgsz960 runs/eval/exp_yolo26s
```

`errors` saves the most confident false positives, the missed smoke (smallest first) and good detections,
with a contact sheet for each. Pick at least three FPs and three FNs for the report.
`compare` writes a bar chart and a markdown table you can paste into the report.

## 6. Package the final model

```bash
python code/train.py --config model/<chosen>.yaml --promote       # or copy runs/<best>/weights/best.pt to model/best.pt
python code/eval.py --weights model/best.pt --name final --save-threshold
```

This writes `model/best.pt`, `model/best_config.yaml` (exact config used) and `model/threshold.json`.

## 7. Inference (what the instructor runs)

```bash
python code/predict.py --source /path/to/hidden_images --out predictions
```

Writes `predictions/predictions.csv` (one row per box: `image_id, class_id, class_name, score,
x1, y1, x2, y2` in pixels, plus normalized `xc, yc, w, h`) and `predictions/predictions.json`
(every image, including those with no boxes). The threshold comes from `model/threshold.json`.
For mAP scoring use `--conf 0.001` so low-confidence boxes are kept.
Add `--save-txt` for YOLO-format label files with scores, `--save-images` for drawn boxes.

## Reproducibility notes

- Seed 42 for training (`seed`, `deterministic: true`), the split, and figure sampling.
  Two CPU runs with the same config gave identical losses and metrics in testing.
  GPU runs can still differ very slightly because some CUDA kernels are nondeterministic.
- `ultralytics` is pinned in `requirements.txt`; the exact versions used for each run are in `run_info.json`.
