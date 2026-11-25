# Wildfire Smoke Detection with YOLO

CS 458/658 group project: a single-class (`smoke`) object detector for outdoor wildfire camera images,
fine-tuned from a pretrained YOLO model.

```
code/       training, evaluation, inference, visualization, split scripts (see code/README.md)
dataset/    data.yaml, split lists, labels, annotation guide, data-source inventory
model/      train_config.yaml, final best.pt, best_config.yaml, threshold.json
```

## Quick start

```bash
pip install -r code/requirements.txt
python code/make_splits.py --images <images_dir> --labels <labels_dir>   # event-level train/val split
python code/train.py                                                     # baseline, seed 42
python code/eval.py --weights runs/baseline/weights/best.pt --name baseline
python code/predict.py --source <new_images> --out predictions           # final inference
```

Full instructions, experiment commands and output formats: [code/README.md](code/README.md).
