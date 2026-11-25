# Data Source Inventory

Every image in our dataset comes from one of three source types, as required by the brief (§3): **instructor-provided**, **team-collected**, or **external**. This file lists each source, how many images we took from it, the licence or permission that allows us to use it, and how it is split. The report's dataset section is written from this file.

**Hidden set rule:** we have not viewed, labeled, tuned against, or otherwise used the instructor's reserved evaluation set. No image here was obtained from it.

---

## 1. Summary

| Source ID | Type | Description | Raw images available | Images selected | Positives | Negatives | Licence / permission |
|---|---|---|---|---|---|---|---|
| `INSTR` | Instructor-provided | Raw unlabeled images from the course Nevadabox | 1,664 | _TBD_ | _TBD_ | _TBD_ | Provided by the instructor for this course |
| `TEAM-*` | Team-collected | _none yet_ | — | — | — | — | — |
| `EXT-*` | External | _none yet_ | — | — | — | — | — |
| **Total** | | | | _TBD_ | _TBD_ | _TBD_ | |

Fill in "Images selected", "Positives" and "Negatives" from the dataset stats script once labeling is done, so the numbers here match the split files exactly.

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
  - 40 frames have a negative offset (before the logged start, from −2400 s). These are likely smoke-free and are our first candidate negatives, but each must still be checked by eye (annotation guide §3).
  - Sequence IDs that are within a few minutes of each other may be the same fire seen from different cameras. If we treat them as separate sequences, a fire could leak across splits. See §5.
- **Selection:** _[describe how we picked images from the 1,664: e.g. all frames, or every Nth frame per sequence, plus all pre-ignition frames as negatives. Record the rule and the script that applies it.]_

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
| | | | | | | | |

Guidance:

- Prefer sources with an explicit licence that allows research or educational use (e.g. CC BY, CC BY-NC, or a dataset licence that says so). Note any attribution requirement and follow it in the report.
- If an external dataset comes with labels, **relabel or at least review them against our annotation guide**. Other datasets use different box rules and will add inconsistency otherwise.
- Check that external images don't overlap with the instructor's set (same cameras and fires are possible, e.g. if both draw on HPWREN). Duplicates across splits would be leakage. A perceptual-hash check against all `INSTR` images catches most of these.
- Labeled datasets from other teams in this course are **not** allowed (brief §14).

---

## 5. Splits

The brief asks us to split by sequence, camera, location, or event whenever that information exists, not by random frame. For `INSTR`, we split by **sequence ID** (`timestamp − offset` from the filename), and we put sequence IDs that start within 10 minutes of each other into the same group, so a single fire seen from more than one camera stays on one side of the split. With that 10-minute rule the 71 sequence IDs collapse to **59 groups** (a 5-minute rule gives 60, 30 minutes gives 58, so the grouping is not sensitive to the exact window).

| Split | File | # sequence groups | # images | # positives | # negatives | # boxes |
|---|---|---|---|---|---|---|
| Train | `dataset/splits/train.txt` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Internal val | `dataset/splits/val.txt` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

- Target ratio: about 80/20 by image count, with a fixed random seed recorded here: `seed = ___`.
- Try to balance the val set across times of day, seasons, and image resolutions so it reflects the variety of the hidden set as well as we can guess.
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
