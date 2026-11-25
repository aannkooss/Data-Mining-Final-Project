# Data source inventory

## Instructor-provided images

- **Source:** the course Nevadabox folder (`https://nevada.box.com/s/dl5pfb9dxyt6rt3xlzkkayyqwl6b9igu`), downloaded as one flat folder of 1,664 JPEG images.
- **Origin:** HPWREN fixed mountaintop cameras in Southern California. The overlay text in each image names the camera, e.g. "Otay Mountain West". File names are `<unix capture time>_<seconds from ignition>.jpg`, for example `1610045981_+01680.jpg` is 28 minutes after ignition.
- **Content:** 59 fire events from January 2021 to May 2025, seen by 44 camera views on 25 mountaintop sites. Each event is a sequence of frames about 1 minute apart covering roughly the first 30 minutes after ignition. One event (E24, Los Pinos South) also includes 40 frames from the 40 minutes before ignition.
- **Resolution:** 3072×2048 for most cameras, 2048×1536 for a few (High Point East 2021, Mt. Laguna South, Mt. Laguna North).

## Selection

`dataset/selection_manifest.csv` lists the 578 images picked for labeling and why. How they were chosen:

1. **Grouped into fire events** by ignition time (capture time minus offset). Frames whose ignition times are within 15 minutes of each other are one event (E01 to E59). The camera name in each frame's overlay was read so every image has a camera and a site.
2. **Spaced out in time** to avoid near-duplicate neighbours: from each event, about one frame in three, evenly spaced from the first to the last frame, always keeping the earliest frame (small, early smoke) and the last frame. Small events (8 frames or fewer) keep at least 6 frames.
3. **No-smoke frames:** 10 of the 40 pre-ignition frames from E24, evenly spaced, to give the model some true negatives.

| | Images |
| --- | --- |
| Fire-sequence frames | 568 |
| Pre-ignition frames (no smoke) | 10 |
| **Total picked** | **578** of 1,664 |

The 1,086 frames that weren't picked are mostly near-duplicates of a picked neighbour one minute away. They can be added later as a dataset-expansion experiment.

## Train and internal validation split

Split by **camera site**, not by random image, so frames of the same fire or the same view never appear in both sets. Sites that share events (for example Los Pinos and White Star cameras watching the same fire) are kept together.

| Split | Sites | Events | Images |
| --- | --- | --- | --- |
| train | 19 sites | 45 | 457 |
| val | Mt. San Miguel, Mt. Wilson, Cuyamaca Peak, Boucher Hill, Santiago Peak, Upper Talega | 14 | 121 |

The validation sites were chosen to cover both San Diego County and the Los Angeles / Orange County cameras, so validation measures generalization to unseen views. Files: `dataset/splits/train.txt`, `dataset/splits/val.txt`.

## Team-collected images

None yet.

## External images

None yet. Any image added later must be listed here with its source URL, license or terms of use, number of images and which split it went into.

| Source | URL | License / terms | Images | Split |
| --- | --- | --- | --- | --- |
| | | | | |
