# Smoke annotation guide

Version 1.0. One class: `smoke` (YOLO class id `0`). Every labeler and reviewer follows this guide. If a case is not covered, add it to the "Decisions log" at the bottom so the team stays consistent.

## 1. What counts as smoke

**Smoke** is a plume or column of combustion smoke that you can trace back to, or that clearly rises from, a ground source in the scene. It usually:

- starts narrow at the ground or a ridge line and widens as it rises or drifts with the wind,
- is grey, white, tan or brown, and often looks semi-transparent at the edges,
- changes shape from frame to frame in the same sequence while clouds and terrain stay put.

**Not smoke. Never draw a box on:**

- clouds (including low clouds sitting on ridges), fog, marine layer, haze along the horizon,
- dust from roads or vehicles, steam, sun glare, lens flare, raindrops, dirt or reflections on the dome,
- the camera's timestamp text, black fisheye corners, towers, buildings,
- general regional haze or a brownish sky layer with no visible plume shape.

## 2. How to draw the box

1. **Tight around the visible plume.** The box edges touch the outermost pixels you can confidently see as smoke. Do not pad. Do not box the whole sky or landscape.
2. **Bottom edge at the source.** Include the base of the plume where it meets the ground or ridge line. If the base is hidden behind terrain, stop the box at the ridge line.
3. **Top edge where the plume stops being identifiable**, not where it becomes invisible after squinting. If you need to zoom in past 200% and still can't decide, the faint part stays outside the box.
4. **Wind-blown plumes.** When smoke bends and drifts sideways, the box covers the base and the drifting part while it is still connected and clearly smoke. A long, thin drift can make a wide, short box. That is fine.
5. **Faint or early smoke.** Label it when it is visually identifiable in this frame. You may flip to the later frames of the same fire (the files sort by event and time) to confirm where the source is, but only box what is visible in the current frame. Very small boxes (around 10 to 20 px at full resolution) are expected in early frames.
6. **Partially occluded smoke** (behind a tower, tree or antenna): one box around the full visible extent, drawn as if the occluder were not there, so the box spans across it.
7. **Image edge.** If the plume runs off the edge of the image, stop the box at the edge. Never let a box go outside the image.

## 3. One box or several?

- **One plume, one box.** A single continuous plume gets one box even if it has gaps in density.
- **Separate boxes** when there are two or more sources whose plumes are clearly separated by clear sky or terrain along most of their height (for example two spot fires on different ridges).
- **Merging plumes**: if two columns join into one cloud of smoke, use one box for the merged mass unless the two bases are far apart and the gap between them is clear for most of the height. When unsure, use one box.
- **Detached smoke layer** (old smoke that has drifted away from the column and spread into a flat sheet): label it only when it is still clearly connected to or obviously part of the column. A flat sheet far from the column with no plume shape is not labeled.

## 4. Ambiguous cases

| Situation | Decision |
| --- | --- |
| You can't tell smoke from a small cloud or haze, even comparing frames | Leave it unlabeled and add the file name to the decisions log with a note. |
| Smoke is only visible by comparing with other frames (not in this frame alone) | Leave unlabeled. |
| Very large smoke filling a big part of the frame, edges unclear | Box the dense, clearly-smoke part; don't extend into diffuse haze. |
| Dusk, night or backlit frames where the plume is a dark shape | Label if the plume shape is clear. |
| Pre-ignition frames (file name has a negative offset, e.g. `_-01380.jpg`) | No boxes. Save an empty label file. |
| Early frames in a sequence where no smoke is visible yet | No boxes. Save an empty label file. |

An image with no smoke is still labeled: it gets an **empty** `.txt` file. A missing `.txt` file means "not labeled yet".

## 5. Labeling workflow

1. Run `python code/prepare_dataset.py --raw ~/Downloads/training` once. It copies the 578 picked images into `dataset/images/train` and `dataset/images/val`, named `E07_1644862570_+02040.jpg` (event id first) so each fire sorts together in time order.
2. Open [makesense.ai](https://www.makesense.ai) (runs in the browser, images stay on your computer). Choose **Object Detection**, load one batch of images (one or a few events at a time, about 50 images), and create the single label `smoke`.
3. Draw boxes following sections 1 to 4. Work through a fire in time order: start from the last frame, where the plume is largest, to find the source, then go back to earlier frames.
4. Export with **Actions → Export Annotations → YOLO format**, and unzip the `.txt` files into `dataset/labels/train` or `dataset/labels/val` (same split as the images). Export after every batch, because makesense.ai doesn't save your work between sessions.
5. For images with no smoke, create an empty `.txt` file with the same name (makesense.ai doesn't export empty files). `check_labels.py` lists any image without a label file.
6. Run `python code/check_labels.py` and fix every problem it reports.

CVAT, Label Studio or LabelImg work too, as long as the export is YOLO format with class `0` = `smoke`.

## 6. Quality control (at least 10% independently checked)

The QC sample is fixed in `dataset/qc_sample.txt`: **60 images (10.4% of 578)**, one random frame from each of the 59 fires plus one pre-ignition frame, picked with seed 458. The column `qc_sample` in `selection_manifest.csv` marks the same images.

1. The reviewer is a teammate who did **not** label that fire. They label the QC images from scratch, without looking at the main labels, and save the `.txt` files in `dataset/labels_qc/`.
2. Run `python code/qc_compare.py --qc dataset/labels_qc --csv dataset/qc_results.csv`. It matches boxes at IoU ≥ 0.5 and reports agreement, box-level precision and recall, and mean IoU.
3. Labeler and reviewer go through every image marked `discuss`, agree on the right answer, fix the main labels, and record any new rule in the decisions log below.
4. If more than about 1 in 5 QC images needs a fix, re-check the rest of the labels for the same mistake before training.
5. Put the agreement numbers and the main disagreement types in the report's quality-control section.

## 7. Decisions log

Add a line whenever the team agrees on a case the guide didn't cover.

| Date | Image or situation | Decision | Who |
| --- | --- | --- | --- |
| | | | |
