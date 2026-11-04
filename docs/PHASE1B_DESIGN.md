# Phase 1b design: finer measures of position-dependent vulnerability

Status: design fixed before any Phase 1b code or results (branch `phase1-fisheye-position`).

## Why

Phase 1 hit a floor: most targets were suppressed at the smallest eps (1/255), so eps_min
cannot discriminate the conditions (docs/PHASE1_DESIGN.md, "Phase 1 outcome: floor effect").
Phase 1b asks the same question with two finer measures.

## Question (unchanged)

Are people at the edge of a fisheye frame more vulnerable to adversarial attacks than at
the center, and does fisheye distortion add vulnerability beyond the person's smaller
apparent size?

## Fixed from Phase 1

- **Targets:** the same 95 eligible targets (Phase 1 caches, `eligible = True`).
- **Conditions:** the same three, center, edge and center_shrunk, with the same
  rendering, canvas, f, placements and GT boxes, read from the Phase 1 caches.
- **Model and attack path:** YOLO11n on CPU, with the same differentiable letterbox and
  gradient path, and the same targeted loss: top-20 person scores among raw predictions
  with IoU ≥ 0.3 to the target box, falling back to box centers inside the target box.
- **Success:** no person detection with conf ≥ 0.25 and IoU ≥ 0.5 with the target GT box,
  judged on the saved uint8 PNG through the normal inference path.

## Measure 1: steps-to-suppress

Global PGD at eps = 1/255, alpha = 0.25/255, up to 50 steps, no random start and no
best-iterate selection: each step updates the current iterate (projection to the L∞ ball
and clipping to [0, 1] at every step). After every step k = 1…50 the iterate is rounded to
uint8, and the target's **max matching person confidence** is recorded: the highest
confidence among person detections (conf ≥ 0.001, as in the normal inference path) with
IoU ≥ 0.5 with the target box, or 0 if there is none. Step 0 is the clean canvas.

- **steps_to_suppress:** the first k at which the matching confidence is < 0.25, which is
  exactly the success criterion.
- **AUC:** the area under the confidence-vs-step curve from k = 0 to 50 (trapezoid rule),
  divided by 50, so it lies in [0, 1]. Lower means faster or deeper suppression.

The attack always runs all 50 steps so that the AUC is defined.

## Measure 2: localized attack

The perturbation is restricted to the pixels inside the target's GT box on the canvas
(a binary mask of pixels whose centers lie inside the box). PGD runs 20 steps with
alpha = eps/4, the best iterate (lowest loss, including the start), no random start, and
projection, masking and clipping at every step. eps ∈ {1, 2, 4, 8, 16}/255 is tried in
ascending order, stopping at the first success. If none succeeds, eps_min is censored as
"> 16/255", stored as 32 with censored = True.

## Analysis (both measures; metrics: steps_to_suppress, AUC, local eps_min)

Paired comparisons over targets: edge vs center, edge vs center_shrunk, center_shrunk vs
center. For each one:

- the median paired difference (first − second) with a 95% bootstrap CI (10,000 resamples
  over targets, seed 0, 2.5th and 97.5th percentiles of the resampled medians);
- the Wilcoxon signed-rank test (two-sided, zero differences dropped, number of nonzero
  pairs reported);
- the counts of pairs where the first condition needs more, less or equal.

## Interpretation rule (fixed before results, as in Phase 1)

- edge ≈ center_shrunk → apparent size explains the effect;
- edge more vulnerable than center_shrunk → distortion adds vulnerability;
- edge less vulnerable than center_shrunk → distortion protects.

Operationally, with d = metric(edge) − metric(center_shrunk) per target: "≈" if the 95%
bootstrap CI of the median of d contains 0; "more vulnerable" if the CI lies entirely below 0;
"less vulnerable" if it lies entirely above 0. For all three metrics, smaller means more
vulnerable (fewer steps, lower confidence area, smaller eps). The Wilcoxon p-value and the
more / less / equal counts are reported alongside. The edge vs center and center_shrunk vs
center comparisons are read the same way.

## Implementation details (decided here, before results)

- **Censoring in Measure 1:** a target not suppressed within 50 steps is stored as
  steps_to_suppress = 51 with censored = True. This keeps it distinct from suppression at
  exactly step 50. It ranks above every uncensored value in the rank-based statistics.
- **Rounding per step:** the float iterate continues unrounded. Only the evaluated copy is
  rounded to uint8. PNG is lossless, so the in-memory uint8 image equals the decoded PNG.
  The image at steps_to_suppress is saved as PNG, reloaded and re-detected to confirm
  success on the saved file.
- **Mask:** after every step x = x0 + mask · clip(x − x0, −eps, eps), then clipped to
  [0, 1]. The mask is 1 for canvas pixels whose centers lie inside the GT box.
- **Caching:** one JSON per target in `results/phase1b/cache/`. Its metadata includes the
  Phase 1 cache metadata of the target and all Phase 1b parameters. The saved PNGs (Measure 1
  at the suppression step, Measure 2 at eps_min) go in `results/phase1b/adv/`, which git
  ignores.
- **Smoke test:** `--limit 5` uses the first 5 eligible targets and writes `*_smoke` outputs.
  Per-target caches are shared with the full run.
