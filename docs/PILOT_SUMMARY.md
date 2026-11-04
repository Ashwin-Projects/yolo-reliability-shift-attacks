# Pilot summary: Phase 1 and Phase 1b

Question: are people at the edge of a fisheye frame more vulnerable to adversarial attack than
at the center, and does fisheye distortion add vulnerability beyond the person's smaller
apparent size? Model: YOLO11n on CPU, person class, synthetic equidistant fisheye on COCO
val2017. Conditions: `center`, `edge` (target center at normalized radius 0.75) and
`center_shrunk` (centered, scaled to the edge box's width and height). Designs:
[PHASE1_DESIGN.md](PHASE1_DESIGN.md) and [PHASE1B_DESIGN.md](PHASE1B_DESIGN.md).

Every number below is copied from a saved CSV:

- `results/phase1/summary.csv` and `results/phase1/phase1_descriptives.csv`;
- `results/phase1b/summary.csv`, `results/phase1b/summary_addendum.csv` and
  `results/phase1b/targets.csv` (eps_min counts per level).

Paired differences are first − second. Smaller means more vulnerable for every metric.

## Targets

100 targets were selected and 95 are eligible (detected in all three conditions on the clean
canvas). The same 95 are used in Phase 1 and Phase 1b.

## Phase 1: floor effect

Phase 1 measured eps_min, the smallest global PGD eps in {1, 2, 3, 4, 6, 8}/255 that
suppressed the target. Most targets were already suppressed at 1/255.

| condition | eps_min = 1 | 2 | 3 | ≥ 4 | censored | success rate at 1/255 | mean eps_min |
|---|---|---|---|---|---|---|---|
| center | 85 | 9 | 1 | 0 | 0 | 0.895 | 1.116 |
| edge | 92 | 3 | 0 | 0 | 0 | 0.968 | 1.032 |
| center_shrunk | 83 | 12 | 0 | 0 | 0 | 0.874 | 1.126 |

| comparison | median diff | 95% CI | Wilcoxon p (nonzero pairs) | first more / less / equal | rule |
|---|---|---|---|---|---|
| edge − center | 0 | [0, 0] | 0.0455 (13) | 3 / 10 / 82 | edge ≈ center |
| edge − center_shrunk | 0 | [0, 0] | 0.00666 (11) | 1 / 10 / 84 | edge ≈ center_shrunk |
| center_shrunk − center | 0 | [0, 0] | 0.819 (16) | 9 / 7 / 79 | center_shrunk ≈ center |

With 79–84 of 95 pairs tied, the median difference is 0 and its CI is [0, 0] in every
comparison. The comparisons could not discriminate between conditions, which motivated Phase 1b.

## Phase 1b: finer measures

The two measures:

- **Measure 1:** global PGD at eps 1/255, alpha 0.25/255, 50 steps, with the target's matching
  confidence recorded after every step. It gives two metrics:
  - **steps-to-suppress:** the first step with confidence < 0.25 (51 = censored);
  - **AUC:** the area under the confidence-vs-step curve, divided by 50.
- **Measure 2:** PGD restricted to the target's box, eps in {1, 2, 4, 8, 16}/255. Its metric is
  the **localized eps_min** (32 = censored).

### Per condition (n = 95 each)

| | center | edge | center_shrunk |
|---|---|---|---|
| steps-to-suppress, median | 8 | 4 | 5 |
| steps-to-suppress, mean | 9.811 | 6.126 | 8.758 |
| AUC, median | 0.1202 | 0.0707 | 0.0867 |
| AUC, mean | 0.1522 | 0.0900 | 0.1189 |
| localized eps_min, median | 2 | 1 | 2 |
| localized eps_min, mean | 2.347 | 1.768 | 2.189 |
| localized eps_min counts at 1 / 2 / 4 / 8 / 16 | 47 / 26 / 15 / 6 / 1 | 62 / 19 / 11 / 3 / 0 | 44 / 28 / 19 / 4 / 0 |
| censored, Measure 1 | 0 | 0 | 1 |
| censored, Measure 2 | 0 | 0 | 0 |

The single censored Measure 1 value (target 27972, center_shrunk) enters the mean as 51.

### Paired comparisons (n = 95)

| metric | comparison | median diff | 95% CI | Wilcoxon p (nonzero pairs) | first more / less / equal | rule outcome |
|---|---|---|---|---|---|---|
| steps | edge − center | −2 | [−4, −1] | 9.95e-08 (82) | 15 / 67 / 13 | edge more vulnerable than center |
| steps | edge − center_shrunk | −1 | [−2, 0] | 4.35e-04 (77) | 26 / 51 / 18 | edge ≈ center_shrunk |
| steps | center_shrunk − center | −1 | [−1, 0] | 0.0558 (81) | 32 / 49 / 14 | center_shrunk ≈ center |
| AUC | edge − center | −0.0434 | [−0.0651, −0.0245] | 3.14e-09 (95) | 17 / 78 / 0 | edge more vulnerable than center |
| AUC | edge − center_shrunk | −0.0125 | [−0.0303, −0.0042] | 1.27e-03 (95) | 31 / 64 / 0 | edge more vulnerable than center_shrunk |
| AUC | center_shrunk − center | −0.0234 | [−0.0435, −0.0093] | 2.64e-04 (95) | 30 / 65 / 0 | center_shrunk more vulnerable than center |
| local eps_min | edge − center | 0 | [0, 0] | 0.0177 (43) | 13 / 30 / 52 | edge ≈ center |
| local eps_min | edge − center_shrunk | 0 | [0, 0] | 2.09e-03 (33) | 5 / 28 / 62 | edge ≈ center_shrunk |
| local eps_min | center_shrunk − center | 0 | [0, 0] | 0.661 (47) | 25 / 22 / 48 | center_shrunk ≈ center |

### Pre-registered rule, edge vs center_shrunk

The rule, as stated in PHASE1B_DESIGN.md:

- edge ≈ center_shrunk → apparent size explains the effect;
- edge more vulnerable than center_shrunk → distortion adds vulnerability;
- edge less vulnerable than center_shrunk → distortion protects.

| metric | outcome |
|---|---|
| steps-to-suppress | edge ≈ center_shrunk → apparent size explains the effect |
| AUC | edge more vulnerable than center_shrunk → distortion adds vulnerability |
| localized eps_min | edge ≈ center_shrunk → apparent size explains the effect |

### Addendum: clean-confidence control (pre-registered before the full run; secondary)

**(a) Clean-confidence paired differences.** Clean confidence was lower at the edge than in
both other conditions, and lower in center_shrunk than at center.

| comparison | median diff | 95% CI | Wilcoxon p (nonzero pairs) |
|---|---|---|---|
| edge − center | −0.0455 | [−0.0650, −0.0317] | 2.36e-10 (95) |
| edge − center_shrunk | −0.0237 | [−0.0328, −0.0146] | 4.64e-06 (95) |
| center_shrunk − center | −0.0222 | [−0.0286, −0.0167] | 6.11e-08 (95) |

**(b) Pairs with |Δ clean conf| < 0.05.** Subset sizes:

| comparison | pairs in subset |
|---|---|
| edge − center | 43 |
| edge − center_shrunk | 64 |
| center_shrunk − center | 62 |

| metric | comparison | median diff | 95% CI | Wilcoxon p (nonzero pairs) | first more / less / equal | rule on subset |
|---|---|---|---|---|---|---|
| steps | edge − center | −2 | [−4, −1] | 6.25e-04 (40) | 8 / 32 / 3 | edge more vulnerable than center |
| steps | edge − center_shrunk | −1 | [−3, 0] | 2.22e-03 (54) | 19 / 35 / 10 | edge ≈ center_shrunk |
| steps | center_shrunk − center | 0 | [−1, 0.5] | 0.340 (53) | 23 / 30 / 9 | center_shrunk ≈ center |
| AUC | edge − center | −0.0498 | [−0.0862, −0.0228] | 1.24e-05 (43) | 6 / 37 / 0 | edge more vulnerable than center |
| AUC | edge − center_shrunk | −0.0125 | [−0.0330, −0.0022] | 0.0117 (64) | 22 / 42 / 0 | edge more vulnerable than center_shrunk |
| AUC | center_shrunk − center | −0.0194 | [−0.0543, −0.0050] | 6.45e-03 (62) | 19 / 43 / 0 | center_shrunk more vulnerable than center |
| local eps_min | edge − center | 0 | [−1, 0] | 0.0318 (23) | 6 / 17 / 20 | edge ≈ center |
| local eps_min | edge − center_shrunk | 0 | [0, 0] | 3.35e-03 (23) | 3 / 20 / 41 | edge ≈ center_shrunk |
| local eps_min | center_shrunk − center | 0 | [0, 0] | 0.325 (37) | 18 / 19 / 25 | center_shrunk ≈ center |

On the subset, every rule outcome is the same as on all 95 pairs.

**(c) Spearman ρ between clean confidence and each metric**, pooled over conditions
(285 rows; descriptive only, rows not independent):

| metric | ρ |
|---|---|
| steps-to-suppress | 0.456 |
| AUC | 0.475 |
| localized eps_min | 0.151 |

## Official pilot conclusion

- **Edge vs center:** edge is more vulnerable than center on steps-to-suppress and on AUC. On
  localized eps_min the rule gives edge ≈ center.
- **Size effect (center_shrunk vs center):** center_shrunk is more vulnerable than center on AUC.
  On steps-to-suppress and localized eps_min the rule gives center_shrunk ≈ center.
- **Edge vs center_shrunk (distortion):**
  - The pre-registered rule supports "distortion adds vulnerability" on AUC only. Steps-to-suppress
    and localized eps_min give edge ≈ center_shrunk, i.e. "apparent size explains the effect".
  - By pair counts, all three metrics point in the same direction: edge needed less than
    center_shrunk in more pairs than it needed more. The counts (less vs more) were 51 vs 26 for
    steps, 64 vs 31 for AUC and 28 vs 5 for localized eps_min.
  - The effect is small. The median paired AUC difference is −0.0125 (95% CI [−0.0303, −0.0042])
    on a scale from 0 to 1.

## Limitations

- **Sample:** 95 eligible targets, all from the first 100 qualifying val2017 images.
- **Synthetic fisheye:** an equidistant remap of perspective images, not images from a fisheye
  camera.
- **One model:** YOLO11n only.
- **White-box attacks:** full gradient access through the model and letterbox.
- **Clean confidence:** it differs between conditions (addendum a). Phase 1b controls for it
  only by the |Δ| < 0.05 subset and a pooled correlation.
- **Discrete metrics:** steps-to-suppress and localized eps_min are discrete and often tied, so
  their median-based bootstrap CIs are conservative. This is most visible for localized eps_min,
  whose CI is [0, 0] even where most nonzero pairs favour one condition.
