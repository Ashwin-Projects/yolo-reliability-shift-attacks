# Confirmatory design: position-dependent vulnerability under fisheye geometry

This plan was written after the Phase 1b pilot results were known.

Status: design fixed before any confirmatory code or run (branch `phase1-fisheye-position`).
Pilot results: [PILOT_SUMMARY.md](PILOT_SUMMARY.md).

## Question

Does fisheye distortion make a person at the edge of the frame more vulnerable to adversarial
attack than a person of the same apparent size at the center?

## Targets

- **Selection:** the same rules as the pilot (`phase1.select_targets`):
  - COCO val2017 images in ascending id order;
  - the largest non-crowd person in the image is the target;
  - the target box area is at least 96²;
  - the box is at least 1 px from every image border;
  - the target's area is at least 2× every other person's area.
- **Which targets:** qualifying targets 101 to 400 in that order (300 targets), i.e. the 300
  that follow the pilot's 100.
- **No overlap with the pilot.** A read-only check of the selection rule, run before this
  document was written, gave:
  - val2017 contains 952 qualifying targets;
  - the first 100 equal the saved pilot list (`data/phase1_targets.json`);
  - targets 101 to 400 run from image id 50811 to 250619 and share no image with the pilot.
- **Eligibility:** a target is eligible if it is detected (conf ≥ 0.25, IoU ≥ 0.5) on the clean
  canvas in all three conditions, as in the pilot. Ineligible targets are excluded and not
  replaced. The numbers selected and eligible are reported, and the analysis uses all eligible
  targets.

## Fixed from Phase 1b

The following are unchanged from Phase 1b:

- **Conditions:** `center`, `edge` and `center_shrunk`, with the same rendering:
  - 640×640 canvas, equidistant fisheye with θ_max 80° at the half-diagonal, gray 114 border;
  - edge target center at normalized radius 0.75;
  - center_shrunk scaled to the edge box's width and height.
- **Model and attack path:** YOLO11n (`yolo11n.pt`), the same differentiable letterbox and the
  same targeted loss (top-20 person scores with IoU ≥ 0.3 to the target box, with the
  box-center fallback).
- **Success criterion:** no person detection with conf ≥ 0.25 and IoU ≥ 0.5 with the target GT
  box, judged on the saved uint8 PNG through the normal inference path.
- **Measure 1:** global PGD at eps 1/255, alpha 0.25/255, 50 steps, no random start, no best
  iterate, confidence recorded after every step. It gives steps-to-suppress (censored = 51) and
  AUC (trapezoid over steps 0–50, divided by 50).
- **Measure 2:** PGD restricted to the target box mask, eps ∈ {1, 2, 4, 8, 16}/255 in ascending
  order, 20 steps, alpha = eps/4, best iterate. It gives localized eps_min (censored = 32).

## Metrics and comparisons

- **Primary metric: AUC** of the confidence-vs-step curve. It is the only one of the three that
  is continuous and does not saturate: in the pilot it had no tied pairs, while
  steps-to-suppress and localized eps_min had many.
- **Primary comparison:** edge vs center_shrunk.
- **Secondary:**
  - steps-to-suppress and localized eps_min;
  - edge vs center and center_shrunk vs center, for all three metrics.

Paired differences are first − second (edge − center_shrunk, edge − center,
center_shrunk − center). For every metric, smaller means more vulnerable.

## Analysis

For each metric and each of the three comparisons:

- **Median paired difference,** with a 95% bootstrap CI: 10,000 resamples over targets, seed 0,
  and the 2.5th and 97.5th percentiles of the resampled medians.
- **Wilcoxon signed-rank test:** two-sided, zero differences dropped, number of nonzero pairs
  reported.
- **Holm correction:** applied across the three comparisons within each metric. Raw and
  Holm-corrected p-values are both reported.
- **Pair counts:** the pairs where the first condition needs more, less or the same.
- **Per-condition descriptives:**
  - median and mean of every metric;
  - eps_min counts per level;
  - censored counts for both measures.

## Confidence control

Clean confidence is the target's matching confidence on the clean canvas, i.e. the step-0
value of Measure 1. For each comparison:

a) **Regression.** An ordinary least-squares regression of the paired AUC difference on the
   paired clean-confidence difference (both first − second, one row per eligible target). The
   intercept estimates the AUC difference at equal clean confidence. The report includes:
   - the intercept with its 95% CI (the classical OLS t-interval);
   - the slope with its 95% CI;
   - n.
b) **Matched subset.** The main analysis (all three metrics, same statistics, pair counts and
   Holm correction within metric) repeated on the pairs with |Δ clean conf| < 0.02. The subset
   size is reported for each comparison.

## Decision rule

The confirmatory run concludes **"distortion adds vulnerability"** only if all three
conditions hold:

1. The 95% bootstrap CI of the median paired AUC difference (edge − center_shrunk) lies
   entirely below 0, i.e. it excludes 0 in favour of edge being more vulnerable.
2. The Holm-corrected Wilcoxon p for that comparison is < 0.05.
3. The 95% CI of the regression intercept (control a, edge − center_shrunk) lies entirely
   below 0.

If any condition fails, the conclusion is that the confirmatory run does not support
"distortion adds vulnerability". The report then states which conditions failed. The secondary
metrics, the secondary comparisons and the matched subset are reported but do not enter the
decision.

## Device

- **One GPU type:** the entire confirmatory run (all targets, conditions and measures) runs on
  a single GPU type.
- **No mixing:** CPU and GPU results are never mixed. The pilot ran on CPU, so its results are
  not pooled with the confirmatory ones.
- **Recorded versions:** the GPU model and the torch and CUDA versions go into every
  per-target cache. A cache from a different device is rejected, not reused.

## Reporting

- **Separate reports:** pilot and confirmatory results are reported separately and never
  pooled.
- **Which run decides:** the confirmatory run determines the conclusion. The pilot is reported
  as exploratory.
- **Output location:** confirmatory outputs go in their own results folder, separate from
  `results/phase1/` and `results/phase1b/`. The pilot results are not changed.
