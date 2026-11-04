# Confirmatory design: position-dependent vulnerability under fisheye geometry

This plan was written after the Phase 1b pilot results were known.

Status: design fixed before any confirmatory code or run (branch `phase1-fisheye-position`).
Pilot results: [PILOT_SUMMARY.md](PILOT_SUMMARY.md).

**Amendment (2026-11-04):** Sample size changed from 300 to 852 before any confirmatory data
was collected. The targets are now all remaining qualifying targets (101–952), with a
pre-specified runtime fallback to targets 101–400 (see "Targets").

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
- **Which targets:** all remaining qualifying targets in COCO val2017 after the first 100
  (targets 101-952), excluding ineligible ones without replacement.
- **Why not train2017:** COCO train2017 images are not used because YOLO11n was trained on
  them.
- **No overlap with the pilot.** A read-only check of the selection rule, run before this
  document was written, gave:
  - val2017 contains 952 qualifying targets;
  - the first 100 equal the saved pilot list (`data/phase1_targets.json`);
  - targets 101 to 952 run from image id 50811 to 581357 (target 400 is image 250619), and
    share no image with the pilot.
  - The full list of targets 101-952 is saved in `data/confirmatory_targets.json`, in
    selection order.
- **Eligibility:** a target is eligible if it is detected (conf ≥ 0.25, IoU ≥ 0.5) on the clean
  canvas in all three conditions, as in the pilot. Ineligible targets are excluded and not
  replaced. The numbers selected and eligible are reported, and the analysis uses all eligible
  targets.

### Runtime fallback (pre-specified)

After the Kaggle probe, the per-step GPU time is measured on 2 targets. If the estimated total
runtime for targets 101-952 exceeds 10 hours, the confirmatory run uses targets 101-400 instead.
This decision is recorded in docs/EXECUTION_MANIFEST.json before the run and cannot be changed
afterwards.

- **Probe targets:** exactly 2 eligible targets: the pilot list (`data/phase1_targets.json`) is
  tried in order until 2 targets are eligible on the probe machine. If fewer than 2 of the 100
  pilot targets are eligible, the probe aborts and writes no manifest. Both measures run on all
  three conditions, exactly as in the confirmatory run. Their results and images are discarded:
  nothing is written next to the confirmatory results.
- **Warm-up:** before any timing, one untimed forward and backward pass of the targeted loss
  runs on the GPU (on the first eligible probe target's center canvas). This keeps one-off
  start-up costs (CUDA context, kernel loading) out of the per-step times.
- **Per-step times** (measured after the warm-up; means over the 2 targets × 3 conditions):
  - t₁ = Measure 1 seconds per step (PGD step plus detection);
  - t₂ = Measure 2 seconds per PGD step;
  - L = mean number of Measure 2 PGD steps per condition (20 × the number of eps values
    tried).
- **Estimate:** N × 3 × (50 · t₁ + L · t₂) seconds, where N = 852 (targets 101-952), with t₁,
  t₂ and L measured after the warm-up. It ignores ineligible targets (which cost less) and the
  clean detections.
- **Decision:** the target range is 101-952 if the estimate is at most 10 hours, otherwise
  101-400. The probe writes the per-step times, the estimate and the chosen range into the
  manifest.
- **Binding:** `--stage confirmatory` reads the target range only from the committed manifest.

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
  Holm-corrected p-values are both reported. With m = 3 and the raw p-values sorted ascending
  as p(1) ≤ p(2) ≤ p(3), the adjusted values are p̃(1) = min(1, 3·p(1)),
  p̃(2) = min(1, max(p̃(1), 2·p(2))) and p̃(3) = min(1, max(p̃(2), p(3))). A comparison with an
  undefined p-value (no nonzero pairs) keeps an undefined adjusted p-value and counts as not
  significant; m stays 3.

### Statistics frozen: identical behaviour to the pilot code

The confirmatory analysis calls the same functions as the pilot
(`phase1_analysis.paired_diffs`, `phase1_analysis.bootstrap_median_ci`,
`phase1_analysis.paired` and `phase1b.summarize`), unchanged. They do the following:

- **Paired differences.** The eligible rows are pivoted to one row per target
  (`DataFrame.pivot`, index `image_id`, which sorts the targets by ascending image id). The
  difference first − second is taken, rows with a missing value are dropped, and the result is
  a float64 array in ascending image-id order. This order determines which targets each
  bootstrap index selects.
- **Random generator.** `numpy.random.default_rng(0)` (PCG64) is created once per metric,
  immediately before its three comparisons. The same generator is used for those comparisons in
  the fixed order: edge − center, edge − center_shrunk, center_shrunk − center. Each comparison
  consumes one draw. A comparison with no eligible pairs makes no draw.
  - The confidence-control analysis on the matched subset (b) follows the same pattern: a fresh
    `default_rng(0)` per metric, shared across that metric's three comparisons in the same order.
  - The clean-confidence differences (reported with (b)) use one `default_rng(0)` shared across
    the three comparisons.
- **Resample indices.** For n paired differences, one call
  `rng.integers(0, n, size=(10000, n))` draws all resamples at once. Each row is one resample
  of targets, with replacement.
- **CI.** `numpy.median(d[idx], axis=1)` gives the 10,000 resampled medians. The CI is
  `numpy.percentile(medians, 2.5)` and `numpy.percentile(medians, 97.5)` with numpy's default
  `method="linear"`. The point estimate is `numpy.median(d)`.
- **Wilcoxon.** The call is `scipy.stats.wilcoxon(d)` with all defaults: `zero_method="wilcox"`
  (zeros dropped), `correction=False`, `alternative="two-sided"`, `method="auto"`. It is not
  called when d has no nonzero value (statistic and p-value are then undefined), and its
  warnings are suppressed. In scipy 1.18.1, `method="auto"` resolves as follows, where n is the
  length of d including zeros:
  - n > 50: asymptotic normal approximation with tie correction and no continuity correction;
  - n ≤ 50, no ties and no zeros: exact distribution;
  - n ≤ 13, with ties or zeros: permutation test (`PermutationMethod()`; with 2^n ≤ 8,192 <
    9,999 resamples it enumerates all sign assignments and is deterministic);
  - otherwise: asymptotic as for n > 50.
- **Pinned versions:** numpy 2.5.4 and scipy 1.18.1, the versions that produced the pilot
  results. They are pinned in `requirements_gpu.txt` and recorded in the execution manifest.
- **Regression** (confidence control a): `scipy.stats.linregress(x, y)` with
  x = paired clean-confidence difference and y = paired AUC difference, both from
  `paired_diffs` (same target order). The intercept CI is
  intercept ± t₀.₉₇₅,ₙ₋₂ · `intercept_stderr`, with t from `scipy.stats.t.ppf(0.975, n − 2)`;
  the slope CI uses `stderr` in the same way.
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

The confirmatory run concludes **"distortion protects"** only if all three mirrored conditions
hold:

1. The 95% bootstrap CI of the median paired AUC difference (edge − center_shrunk) lies
   entirely above 0.
2. The Holm-corrected Wilcoxon p for that comparison is < 0.05.
3. The 95% CI of the regression intercept (control a, edge − center_shrunk) lies entirely
   above 0.

Otherwise the conclusion is **"does not support either direction"**. The report then lists
which criteria failed for each direction. The secondary metrics, the secondary comparisons and
the matched subset are reported but do not enter the decision.

## Device and execution protocol

- **Hardware:** the entire confirmatory run (all targets, conditions and measures) runs on a
  Kaggle GPU T4, using one GPU only (`cuda:0`; `CUDA_VISIBLE_DEVICES=0`). The recorded device
  name must be exactly one of the allow-list "Tesla T4" or "NVIDIA T4" (case-sensitive, whole
  string after stripping surrounding whitespace); any other name, e.g. "NVIDIA T400", "T4000" or
  "Tesla T40", is rejected.
- **Precision:** float32 throughout, for both the attack model and the predictor (no half
  precision).
- **Determinism and TF32 settings,** applied at startup before any CUDA work:
  - `torch.backends.cuda.matmul.allow_tf32 = False`
  - `torch.backends.cudnn.allow_tf32 = False`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.use_deterministic_algorithms(True)` (not warn-only), with the environment variable
    `CUBLAS_WORKSPACE_CONFIG=":4096:8"`.
- **No fallback:** if an operation fails under determinism, the run stops with the error. It
  does not fall back to a non-deterministic setting.
- **No mixing:** CPU and GPU results are never mixed. The pilot ran on CPU, so its results are
  not pooled with the confirmatory ones.

### Execution manifest

1. **Probe.** Before any confirmatory inference, a probe run on Kaggle
   (`run.py --stage probe --device cuda`) writes `EXECUTION_MANIFEST.json`. The probe's only
   attacks are the timing runs on the 2 pilot targets (see "Runtime fallback"); it runs nothing
   on confirmatory targets. The manifest records:
   - GPU name, number of visible GPUs and driver version;
   - CUDA and cuDNN versions;
   - torch, torchvision, ultralytics, numpy and scipy versions;
   - Python version and the SHA-256 of `yolo11n.pt`;
   - the dtype and all the flags and environment settings above;
   - the probe's per-step times, the runtime estimate and the chosen target range.
2. **Commit.** The manifest is copied to `docs/EXECUTION_MANIFEST.json` and committed before
   the confirmatory run.
3. **Startup check.** At startup, `--stage confirmatory` builds the same environment record
   from the live environment and compares it with the record in `docs/EXECUTION_MANIFEST.json`.
   It aborts with a message if:
   - the file is missing or any field differs;
   - the recorded environment does not follow the protocol: device not CUDA, a GPU other than
     one T4 on the allow-list above, a determinism or TF32 flag or CUDA environment setting different from the values
     above, a dtype other than float32, or numpy / scipy versions different from those pinned in
     `requirements_gpu.txt`;
   - the probe record does not contain exactly 2 timed targets, does not have `warm_up` exactly
     `true` (a missing field or `false` is rejected), or its target range does not follow from
     its estimate.
4. **Test option.** A `--manifest` option exists only for code testing. It requires `--limit`
   and is never used for the confirmatory run. A test run skips the protocol check, and:
   - every output file records "TEST ONLY" with the test manifest's path and SHA-256 (CSV
     columns or rows; text fields in the PNG figures). An output with no data rows still
     carries them in one row marked `row_type = "metadata: provenance only, no data rows"`;
   - its caches and images go in a separate namespace (`cache_test_only/`, `adv_test_only/`), so
     a production run never reuses them.
5. **Caches.** Every per-target cache stores the manifest record, so a cache made in another
   environment is not reused.

## Reporting

- **Separate reports:** pilot and confirmatory results are reported separately and never
  pooled.
- **Which run decides:** the confirmatory run determines the conclusion. The pilot is reported
  as exploratory.
- **Output location:** confirmatory outputs go in their own results folder, separate from
  `results/phase1/` and `results/phase1b/`. The pilot results are not changed.
