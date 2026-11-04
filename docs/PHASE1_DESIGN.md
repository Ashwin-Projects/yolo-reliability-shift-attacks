# Phase 1 design: position-dependent adversarial vulnerability under fisheye geometry

Status: design fixed before any Phase 1 code or results (branch `phase1-fisheye-position`).

## Question

Are people at the edge of a fisheye frame more vulnerable to adversarial attacks than at
the center, and does fisheye distortion add vulnerability beyond the person's smaller
apparent size?

## Design

### 1. Canvas and fisheye model
Every condition is rendered on a 640×640 canvas with an equidistant fisheye
(r_d = f · θ) centered on the canvas. f is chosen so the canvas half-diagonal corresponds
to θ = 80°: f = 320·√2 / (80° in radians) = **324.1139 px** (computed with
`corruptions.fisheye_focal(640, 640, 80)`; recorded again at run time). The source image
is placed on the perspective (undistorted) plane; canvas area that maps outside the
source image is gray (114, 114, 114). The existing fisheye math is reused (inverse map
r_u = f · tan(r_d / f) for pixels, forward map r_d = f · atan(r_u / f) for boxes, 20 points
per box edge).

### 2. Person selection
COCO val2017 (additional images downloaded as needed). An image qualifies if its largest
non-crowd person box has area ≥ 96×96 px, does not touch the image border, and is at least
2× larger (area) than any other person in the image. The first 100 qualifying images by
image id are used; that person is the **target**. Saved to `data/phase1_targets.json`.

### 3. Conditions (three per target)
- **center:** source placed so the target box center lies on the optical axis.
- **edge:** source translated so the target box center maps to normalized fisheye radius
  0.75 (r_d / canvas half-width = 240 px, θ = 42.43°), along +x (right of center). The
  plane offset is f · tan(240 / f) = 296.23 px.
- **center_shrunk:** source scaled anisotropically (separate x and y factors) so the
  target's box width and height on the canvas match the target's warped box width and
  height in the edge condition, then placed with the target center on the optical axis
  and rendered with the same pipeline.

For every condition the target's ground-truth box on the canvas is computed with the same
warp. Target box w, h and area are recorded per condition.

### 4. Eligibility
A target is kept only if, before any attack, YOLO11n detects it (person detection with
conf ≥ 0.25 and IoU ≥ 0.5 with the target GT box) in all three conditions. The clean
confidence per condition is the maximum confidence among matching detections.

### 5. Attack
White-box, CPU, YOLO11n, reusing the existing differentiable letterbox and gradient path.
Global L∞ PGD over the whole canvas, targeted at the one person:

- loss = sum of the top-20 person scores among raw (pre-NMS) predictions whose decoded
  box has IoU ≥ 0.3 with the target GT box (in letterbox coordinates); if none qualify,
  the top-20 person scores whose box center lies inside the target box;
- 20 steps, alpha = eps / 4, no random start, the lowest-loss iterate (including the start)
  is kept, projection and clipping at every step;
- **success** = after rounding to uint8 and saving as PNG, the normal inference pipeline
  finds no person detection with conf ≥ 0.25 and IoU ≥ 0.5 with the target GT box.

### 6. Minimum attack strength
For each target and condition, eps ∈ {1, 2, 3, 4, 6, 8}/255 is tried in ascending order,
stopping at the first success. eps_min is that eps. If none succeeds, eps_min is censored
as "> 8/255", stored as 16/255 with censored = True.

### 7. Analysis
- Per condition: mean and median eps_min, success rate at each eps (curve), fraction
  censored.
- Paired comparisons over targets: edge vs center, edge vs center_shrunk, center_shrunk vs
  center. For each: median paired difference with 95% bootstrap CI (10,000 resamples over
  targets, seed 0) and Wilcoxon signed-rank test.
- Checks: clean confidence per condition and whether it differs between conditions;
  box-size matching (edge / center_shrunk ratios of w and h).
- **Interpretation rule (fixed before results):**
  - edge ≈ center_shrunk → apparent size explains the effect;
  - edge more vulnerable than center_shrunk → distortion adds vulnerability;
  - edge less vulnerable than center_shrunk → distortion protects.

  Operationally, with d = eps_min(edge) − eps_min(center_shrunk) per target: "≈" if the
  95% bootstrap CI of the median of d contains 0; "more vulnerable" if the CI lies
  entirely below 0 (edge needs a smaller eps); "less vulnerable" if it lies entirely
  above 0. The Wilcoxon p-value is reported alongside. The edge vs center and
  center_shrunk vs center comparisons are read the same way.

### 8. Outputs
- `results/phase1/targets.csv`: per target and condition: box w, h, area, clean conf,
  eligible, eps_min, censored
- `results/phase1/summary.csv`
- `results/phase1/fig_success_curves.png`: success rate vs eps per condition
- `results/phase1/fig_paired.png`: paired differences
- per-condition visualizations

## Implementation details not fixed above (decided here, before results)

- **Border:** "does not touch the image border" means at least 1 px between the box and
  every image edge.
- **2× larger:** box area compared with every other person annotation in the image
  (crowd annotations included).
- **Rendering:** bilinear sampling (`cv2.remap`) for all three conditions, in continuous
  pixel coordinates with the optical axis at canvas point (320, 320). Plane coordinates
  share that origin, so the forward box warp is the existing `corruptions.warp_box`.
- **center_shrunk scale factors:** found by fixed-point iteration (scale x and y by
  target / current warped width and height) until both ratios are within 1e-6 of 1.
- **Canvas border:** if the target's warped box touches the canvas border in any
  condition, this is recorded (`touches_border`) and reported. The target is not
  excluded, because the design does not say to exclude it.
- **Attack loss edge cases:** fewer than 20 qualifying predictions → sum of all
  qualifying scores. If neither rule selects any prediction, the loss is 0 for that step
  and this is recorded (`no_target_preds`).
- **Units:** eps_min is stored in 1/255 units (1, 2, 3, 4, 6, 8, or 16 if censored).
  The success curve at eps e is the fraction of eligible targets with eps_min ≤ e.
- **Statistics:** bootstrap CI = 2.5th and 97.5th percentiles of the resampled medians;
  Wilcoxon two-sided, zero differences dropped (scipy default), and the number of
  nonzero pairs is reported. With a discrete eps grid, the median differences take grid
  values.
- **Inference:** the normal ultralytics predictor with the same settings as all earlier
  stages (imgsz 640, conf 0.001, CPU), keeping persons and filtering at conf ≥ 0.25 for
  matching. The attack uses a separate, unfused model instance (as in stage attack).
- **Caching:** one JSON per target in `results/phase1/cache/`, valid only if its
  metadata (all design parameters above, model weights SHA-256, letterbox settings,
  target annotation) matches. Saved attacked PNGs are kept only at eps_min, in
  `results/phase1/adv/` (ignored by git).
- **Smoke test:** `--limit 5` uses the first 5 targets and writes `*_smoke` outputs;
  per-target caches are shared with the full run because they do not depend on the limit.
