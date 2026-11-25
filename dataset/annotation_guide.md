# Annotation Guide: Wildfire Smoke Detection

**Version:** 0.1 (draft) · **Class list:** `0: smoke` · **Label format:** YOLO bounding box (`class x_center y_center width height`, normalized 0–1)

This guide is the team's operational definition of smoke and the rules every annotator follows. If you hit a case this guide does not cover, do not guess: tag the image `ambiguous` (see §6), and we add a rule here so the next person handles it the same way. Every change to this guide gets a line in the changelog at the bottom.

---

## 1. What counts as smoke

**Smoke** is a visible plume of combustion product rising from, or drifting away from, a fire on the ground. Label it when **all three** of these hold:

1. **It has a source or a rising shape.** It originates at (or points back to) a spot on the terrain, or it has the column/plume/wisp shape of something rising and being bent by wind. Clouds sit in the sky with no ground connection; smoke is anchored.
2. **It differs from the background in a way you can point at.** A grey, white, brown, or bluish region with softer edges and different texture than sky, terrain, or haze around it.
3. **You would still call it smoke after looking at the neighbouring frames.** Frames in our data come from fixed cameras in time sequences (see the data source inventory). If you are unsure on one frame, check the frames before and after: smoke grows, drifts, and changes shape between frames; clouds and haze move slowly and uniformly, and terrain features don't move at all.

### Do NOT label as smoke

| Look-alike | How to tell it apart |
|---|---|
| **Clouds** (incl. low clouds on ridgelines) | Detached from the ground, flat or puffy bases, sharp upper edges, present across many frames before any fire. |
| **Fog / marine layer** | Fills valleys or lies flat along the horizon; broad, uniform, no point source. |
| **Haze / smog / general atmospheric blur** | Covers the whole scene or a whole band of it evenly; no boundary you could box. |
| **Dust** (vehicles, wind) | Brown/tan, hugs the ground, low and short-lived, often follows a road. |
| **Steam / industrial plumes** | Bright white, from a fixed structure (stack, plant); present on many days. If you know it's a facility, don't label it. |
| **Glare, sun flare, lens artifacts** | Tied to the sun position or the lens; same place in frame regardless of scene; rings, streaks, or circles. |
| **Water droplets, dirt, or spider webs on the lens** | In focus differently from the scene, don't move with the wind. |
| **Bright sky, snow, light rock, burn scars** | Static between frames; no plume shape. |
| **Timestamp / camera overlay text** | Never label. |

When it is a look-alike, leave the image **unlabeled** (no boxes = negative image). Look-alike negatives are valuable training data; keep them in the dataset.

---

## 2. Box rules

1. **One class only:** every box is class `0` (`smoke`). No other class IDs anywhere in the label files.
2. **Box the visible plume, not the sky.** The box is the tightest rectangle that contains all smoke pixels you can identify at 100% zoom. Do not pad. Do not extend the box to the image edge, the horizon, or the whole sky.
3. **Include the base.** If the base of the plume (where it meets terrain) is visible, the box must include it. The base is the most informative part for early detection.
4. **Faint tails:** include the drifting part of the plume only as far as you can still clearly tell it apart from the background. Where it fades into haze, stop the box at the point where you would no longer be confident if that part were shown alone. Don't chase a thin tail across half the image.
5. **Partially occluded plumes** (behind a ridge, tree, or structure): box only the **visible** smoke. Do not guess where the hidden part is. If the plume is split by an occluder into two visible pieces that obviously belong to the same column, use **one** box covering both pieces.
6. **Multiple separate plumes:** one box per plume. Two columns count as separate plumes when their bases are distinct and there is clear background between them along most of their height.
7. **Disconnected puffs of the same plume** (wind tearing a puff off the top): include the puff in the main box if it is close to the column and plainly the same smoke. If it has drifted far enough that one box would be mostly empty sky, give it its own box only if it is still clearly smoke; otherwise leave it out.
8. **Merged plumes:** if two plumes have merged into one mass and you can't draw a clean boundary between them, use one box for the merged mass.
9. **Huge/diffuse smoke** (large fire, smoke covering a big part of the frame): box the dense, clearly-smoke region, including the source. Do not box thin smoke that has spread into general haze across the horizon.
10. **Small, distant smoke:** label it. Our images are mostly 3072×2048, so a tiny wisp is often still many pixels wide. Zoom to 100–200% to draw it. There is no minimum box size, but any box under 10 px on its longest side gets the `tiny` tag (§6) so we can study it in error analysis.
11. **Boxes stay inside the image.** Clip at the image border; never go past it. In YOLO format all four values must be in [0, 1] and width/height must be > 0.
12. **No duplicates.** One plume = one box. Before you submit an image, check that you haven't drawn two overlapping boxes for the same smoke.

### Boundary consistency (the rule that matters most for mAP@0.5:0.95)

The hidden-set ranking metric averages over IoU 0.50–0.95, so loose or inconsistent edges cost us directly. Everyone uses the same edge rule:

- **Top / sides:** stop where the smoke is no longer distinguishable from the background at 100% zoom.
- **Bottom:** the base on the terrain, or the occluding ridge line if the base is hidden.
- When in doubt between two edges, choose the **tighter** one.

---

## 3. Which images to label

- Label **every** visible plume in every selected image. A missed plume teaches the model that smoke is background; that is worse than not including the image.
- **Negative images** (no smoke) are kept with an empty label file. Aim for a mix that includes look-alikes (clouds, fog, haze, glare), not just clear-sky frames.
- The pre-ignition frames in the instructor set (filename offset is negative, e.g. `..._-00300.jpg`) are probably smoke-free. Treat them as candidate negatives, but still look at each one: sometimes smoke is already visible before the logged start time. If you see smoke, label it.
- Near-duplicate frames from the same sequence add little. Prefer frames that show the plume at different sizes and stages (first wisp, growing, large) over many identical consecutive frames.

---

## 4. Tooling and export

- **Tool:** _[team to fill in: CVAT / Roboflow / Label Studio / LabelImg / Makesense.ai]_. Whatever we use, everyone uses the same project with the single label `smoke`.
- **Export:** YOLO format. One `.txt` per image with the same base name (`1723496059_+01861.jpg` → `1723496059_+01861.txt`). Negative images get an **empty** `.txt` file, not a missing one.
- Do not rename image files. The filename encodes the sequence ID used for the train/val split (see the data source inventory).

---

## 5. Quality control

### 5.1 Self-check (every image, by the annotator)

Before marking an image done:

- [ ] Every visible plume has a box; checked neighbouring frames if unsure.
- [ ] No box on clouds, fog, haze, dust, steam, glare, or overlays.
- [ ] Each box includes the plume base when visible and follows the edge rule in §2.
- [ ] No duplicate or overlapping boxes for the same plume.
- [ ] Tags applied where needed (§6).

### 5.2 Automated checks (whole dataset, before every training run)

A script (to be added under `code/`) checks every label file and fails loudly on:

- Class ID other than `0`.
- Coordinates outside [0, 1], or width/height ≤ 0.
- Lines with the wrong number of values, or non-numeric values.
- A label file with no matching image, or an image with no label file.
- Near-duplicate boxes in the same image (IoU > 0.9).
- The same sequence ID appearing in both train and val split files (leakage check).

It also prints dataset stats for the report: images, positives/negatives, boxes per image, and box size distribution.

### 5.3 Independent second check (≥ 10% of images, required by the brief)

**Goal:** catch missing objects, false boxes, and inconsistent boundaries, and give us a number we can report for annotation consistency.

**Sampling.**
- Sample **at least 10%** of all selected images (we'll use 12% to keep a buffer if some get dropped later), rounded up.
- Sample with a fixed random seed and save the list to `dataset/qc/second_check_sample.txt` so the sample is reproducible.
- Stratify: take roughly the same share from each annotator, and make sure positives, negatives, and `ambiguous`/`tiny` tagged images are all represented.
- The reviewer must be **a different team member** from the original annotator.

**Procedure.**
1. The reviewer opens each sampled image **with the existing labels visible** and checks them against this guide.
2. For each image, the reviewer records one outcome:
   - `OK` — no change needed.
   - `MISSING` — a plume was not boxed.
   - `FALSE` — a box is on something that isn't smoke.
   - `BOUNDARY` — the box is on smoke but the edges break the §2 rules (e.g. too loose, misses the base, includes too much tail).
   - `DUPLICATE` — two boxes for one plume.
   - `OTHER` — anything else (write a note).
3. The reviewer does not edit labels directly. Disagreements are logged, then the annotator and reviewer resolve them together. If they can't agree, the third team member (or the whole team for a two-person team) decides, and the guide gets a new rule.
4. The final, corrected labels replace the originals.

**Log** each review in `dataset/qc/second_check_log.csv`:

```csv
image,annotator,reviewer,outcome,boxes_before,boxes_after,note
1723496059_+01861.jpg,A,B,BOUNDARY,1,1,box cut off plume base behind ridge
1706730306_+00900.jpg,B,A,OK,2,2,
```

**What we report.**
- Share of reviewed images marked `OK`.
- Count of each error type.
- Mean IoU between the original and corrected boxes on images where boxes were matched (to show boundary consistency).
- If any error type is above ~10% of the sample, re-check the rest of that annotator's images for that error, and say so in the report.

---

## 6. Ambiguous cases

Some images are genuinely uncertain, even after looking at neighbouring frames. Our policy:

1. **Tag** the image `ambiguous` in the annotation tool (or list it in `dataset/qc/ambiguous.txt`) with a one-line reason.
2. **Decide as a team** in a short review session. Each ambiguous image ends up as one of:
   - **Label it** — the team agrees it's smoke. Box it with the normal rules.
   - **Keep as negative** — the team agrees it's not smoke (e.g. cloud). Empty label file.
   - **Exclude** — the team can't agree. Remove the image from train/val entirely so we never train on an unsure label. Keep a record of excluded images and how many there were.
3. Any decision that sets a new precedent gets added to §1 or §2 and to the changelog.

**Other tags:**
- `tiny` — any box under 10 px on its longest side.
- `occluded` — plume partially hidden behind terrain or objects.
- `lookalike` — negative image that contains a cloud/fog/haze/glare look-alike.

These tags are not training labels. They exist so we can break down errors by case type in the error-analysis section of the report.

---

## 7. Worked examples

_[To fill in once labeling starts: add 6–10 screenshots with boxes drawn, saved under `dataset/qc/examples/`. At minimum: a clean plume, a faint early plume, an occluded plume, two separate plumes, a large diffuse plume, a cloud negative, a fog/haze negative, and one ambiguous case with the team's decision.]_

---

## Changelog

| Date | Version | Change | By |
|---|---|---|---|
| 2026-11-25 | 0.1 | First draft from the project brief. | |
