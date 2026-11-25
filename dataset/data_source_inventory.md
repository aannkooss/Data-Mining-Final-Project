# Data Source Inventory

Every image in our dataset comes from one of three source types, as required by the brief (§3): **instructor-provided**, **team-collected**, or **external**. This file lists each source, how many images we took from it, the licence or permission that allows us to use it, and how it is split. The report's dataset section is written from this file.

**Hidden set rule:** we have not viewed, labeled, tuned against, or otherwise used the instructor's reserved evaluation set. No image here was obtained from it.

---

## 1. Summary

| Source ID | Type | Description | Raw images available | Images selected | Positives | Negatives | Licence / permission |
|---|---|---|---|---|---|---|---|
| `INSTR` | Instructor-provided | Raw unlabeled images from the course Nevadabox | 1,664 | 145 | _TBD_ (≤ 142) | _TBD_ (≥ 3) | Provided by the instructor for this course |
| `TEAM-*` | Team-collected | _none yet_ | — | — | — | — | — |
| `EXT-HPWREN-NEG` | External (planned) | Smoke-free frames from the same HPWREN cameras on days with no fire, added with `code/add_negatives.py` | — | 0 so far (target 10–15) | 0 | 0 so far | _check and record HPWREN's terms of use before adding_ |
| **Total** | | | | 145 | _TBD_ | _TBD_ | |

Fill in "Positives" and "Negatives" from `python code/check_labels.py` once labeling is done (142 fire-sequence frames and 3 pre-ignition frames were picked; the 3 pre-ignition frames turned out to show smoke, so negatives are only frames with an empty label file), so the numbers here match the split files exactly.

**The instructor set has almost no smoke-free frames.** Every sequence is a fire, and all 40 pre-ignition frames belong to E24, whose earliest frame (−2400 s) already shows a plume on the horizon. So negatives have to come from another source (§4).

---

## 2. Instructor-provided data (`INSTR`)

- **Where:** Nevadabox, https://nevada.box.com/s/dl5pfb9dxyt6rt3xlzkkayyqwl6b9igu
- **Downloaded:** _[date]_ by _[name]_, into a local `training/` folder.
- **Contents (checked 2026-11-25):**
  - 1,664 JPEG images, about 725 MB, all in one flat folder, no labels.
  - Resolution: 1,614 images at 3072×2048 and 50 at 2048×1536.
  - Filenames follow the pattern `<unix_timestamp>_<offset>.jpg`, e.g. `1723496059_+01861.jpg`.
  - Timestamps run from January 2021 to May 2025.
- **Filename convention (our reading, to confirm with the instructor):** the pattern matches the one used by the HPWREN Fire Ignition images Library (FIgLib), where the first number is the frame's capture time and the signed offset is seconds relative to the fire's logged start. Under that reading:
  - `timestamp − offset` gives a constant **sequence ID** for all frames of one fire-camera sequence. The 1,664 images fall into **71 sequence IDs**, with 3 to 64 frames each (most have about 10–32).
  - 40 frames have a negative offset (before the logged start, from −2400 s). A negative offset does **not** mean the frame is smoke-free: the offset is relative to the fire's logged start, and smoke can be visible earlier. All three we picked (from E24) were checked by eye and show smoke, so they are labeled as positives and listed in `dataset/preignition_smoke_ok.txt`.
  - Sequence IDs that are within a few minutes of each other may be the same fire seen from different cameras. If we treat them as separate sequences, a fire could leak across splits. See §5.
  - The overlay text in each frame names the HPWREN camera (e.g. "Otay Mountain West"). Reading it gives **44 camera views on 25 mountaintop sites**; every picked image's camera and site are in `dataset/selection_manifest.csv`.
- **Selection** (applied once, saved in `dataset/selection_manifest.csv`; `code/prepare_dataset.py` copies exactly these images):

`dataset/selection_manifest.csv` lists the 145 images picked for labeling and why. How they were chosen:

1. **Grouped into fire events** by ignition time (capture time minus offset). Frames whose ignition times are within 15 minutes of each other are one event (E01 to E59). The camera name in each frame's overlay was read so every image has a camera and a site.
2. **Spaced out in time** to avoid near-duplicate neighbours: from each event, 2 or 3 frames evenly spaced in time: the earliest frame (small, early smoke), the last frame, and for events with 10 or more frames also the middle one. Two multi-camera events (E42, E47) get one extra frame so every one of the 44 camera views appears at least once. Every fire is covered.
   (Earlier passes picked 578 and then 265 images; the selection was trimmed to keep labeling manageable. Every image here was also in those passes.)
3. **Pre-ignition frames:** 3 of the 40 pre-ignition frames from E24, evenly spaced, picked as intended negatives. On inspection all three show smoke, so they ended up as positives (see §2).

| | Images |
| --- | --- |
| Fire-sequence frames | 142 |
| Pre-ignition frames (all 3 show smoke) | 3 |
| **Total picked** | **145** of 1,664 |

The 1,519 frames that weren't picked are mostly near-duplicates of a picked neighbour from the same fire. They can be added later as a dataset-expansion experiment.

---

## 3. Team-collected data (`TEAM-*`)

None yet. If we add any (e.g. our own photos), give each batch a row here:

| Source ID | Collected by | Date | Location / camera | # images | Notes |
|---|---|---|---|---|---|
| | | | | | |

Team members own the copyright of photos they take, so these are usable for the course. Strip location metadata if it identifies a private property.

---

## 4. External data (`EXT-*`)

None yet. Every external source must be documented and legally usable for an academic course project (brief §3). Before adding any external images, fill in a row here **first** and check the licence.

| Source ID | Name | URL | Licence | Citation | # images used | How selected | Labels reused or relabeled? |
|---|---|---|---|---|---|---|---|
| `EXT-HPWREN-NEG` | HPWREN camera archive, same cameras as `INSTR` | _[archive page used]_ | _[terms of use, checked on (date)]_ | High Performance Wireless Research and Education Network (HPWREN), UC San Diego | _[n]_ | Days with no fire in our set (not within a day of any fire date in the manifest), chosen to show clouds, haze, fog, glare or dust; each frame checked by eye for smoke | None needed: empty label files |

**Adding the HPWREN negatives.** Download the frames into one subfolder per camera, named exactly like the camera in `selection_manifest.csv` (e.g. `~/Downloads/negatives/Los Pinos South/`). Then run `python code/add_negatives.py --src ~/Downloads/negatives --dry-run` to preview and the same without `--dry-run` to add them. The script puts each frame in its camera site's split (so no camera ends up in both train and val), copies it to `dataset/images/<split>/NEG_<camera>_<name>.jpg`, writes an empty label file, and adds rows to the manifest (`role` = `negative_external`, `event` = `NEG`) and the split files. Aim for roughly 10 in train and 4 or 5 in val, so the val set can also measure false alarms.

Guidance:

- Prefer sources with an explicit licence that allows research or educational use (e.g. CC BY, CC BY-NC, or a dataset licence that says so). Note any attribution requirement and follow it in the report.
- If an external dataset comes with labels, **relabel or at least review them against our annotation guide**. Other datasets use different box rules and will add inconsistency otherwise.
- Check that external images don't overlap with the instructor's set (same cameras and fires are possible, e.g. if both draw on HPWREN). Duplicates across splits would be leakage. A perceptual-hash check against all `INSTR` images catches most of these.
- Labeled datasets from other teams in this course are **not** allowed (brief §14).

---

## 5. Splits

The brief asks us to split by sequence, camera, location, or event whenever that information exists, not by random frame. We split by **camera site** (the strictest of those): every image from a site goes to the same side, and sites that watched the same fire (for example Los Pinos and White Star) are kept together. So no fire, no camera view and no mountaintop appears in both train and val. Fires were first grouped into 59 events by ignition time (frames whose `timestamp − offset` is within 15 minutes are one event), which matches the 59 groups found with the sequence-ID rule.

| Split | File | Sites | Events | # images | # positives | # negatives | # boxes |
|---|---|---|---|---|---|---|---|
| Train | `dataset/splits/train.txt` | 19 | 45 | 111 | _TBD_ | _TBD_ | _TBD_ |
| Internal val | `dataset/splits/val.txt` | 6: Mt. San Miguel, Mt. Wilson, Cuyamaca Peak, Boucher Hill, Santiago Peak, Upper Talega | 14 | 34 | _TBD_ | _TBD_ | _TBD_ |

- Ratio 77/23 by image count. The val sites were chosen by hand (no random seed) to cover both San Diego County and the Los Angeles / Orange County cameras, so val measures generalization to unseen views.
- The 10% QC sample (`dataset/qc_sample.txt`, 15 images) was drawn with seed 458: one random frame from each of 14 randomly chosen fires plus one pre-ignition frame.
- Team-collected and external images go into **train only**, unless we have a specific reason (e.g. a cross-source generalization experiment), so the internal val set stays comparable to the instructor's data.

---

## 6. Excluded images

Images that we looked at and decided not to use, with the reason. This includes `ambiguous` images the team could not agree on (annotation guide §6).

| Image | Source | Reason |
|---|---|---|
| | | |

---

## 7. Changelog

| Date | Change | By |
|---|---|---|
| 2026-11-25 | First draft; profiled the 1,664 instructor images. | |
| 2026-11-25 | Picked 578 images, read camera names, fixed the site-level train/val split and the 10% QC sample. | |
| 2026-11-25 | Trimmed the selection to 265 images (4 to 6 per fire) and redrew the 10% QC sample (27 images). | |
| 2026-11-25 | Trimmed the selection to 145 images (2 or 3 per fire, every camera view kept) and redrew the 10% QC sample (15 images). | |
| 2026-11-25 | The 3 picked pre-ignition frames show smoke; kept as positives and listed in `dataset/preignition_smoke_ok.txt`. | |
| 2026-11-25 | Found the instructor set has no smoke-free frames; planned `EXT-HPWREN-NEG` negatives and added `code/add_negatives.py`. | |
