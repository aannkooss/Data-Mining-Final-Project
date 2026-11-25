# Code

## Dataset preparation and label QC

All commands run from the repo root and need only Python 3 (no extra packages).

| Step | Command |
| --- | --- |
| Copy the 578 picked images into the YOLO layout and write `dataset/data.yaml` | `python code/prepare_dataset.py --raw ~/Downloads/training` |
| Check labels for missing files, bad class ids, out-of-bounds or duplicate boxes, and print dataset stats | `python code/check_labels.py` |
| Compare main labels with a teammate's independent labels of the 10% QC sample | `python code/qc_compare.py --qc dataset/labels_qc --csv dataset/qc_results.csv` |

The labeling rules and workflow are in `dataset/annotation_guide.md`. Image selection and the train/val split are documented in `dataset/data_source_inventory.md`.
