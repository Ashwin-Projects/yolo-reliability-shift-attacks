# Research log: detconf

Permanent record of problems, fixes and design decisions. Append new entries;
do not rewrite old ones. Each entry: date, what happened, impact on results,
fix or decision.

## Problems and fixes

### 1. torch install failed (DNS)
- **Date:** 2026-11-03
- **What happened:** Installing CPU torch from `download.pytorch.org` failed with `NameResolutionError ... getaddrinfo failed`.
- **Impact:** None on results.
- **Fix:** Installed torch from PyPI instead (Windows wheels are CPU-only): torch 2.15.0+cpu, `torch.cuda.is_available()` = False.

### 2. Metric inconsistency on crowd regions
- **Date:** 2026-11-03
- **What happened:** mAP (pycocotools) ignored detections on COCO crowd regions, but the TP/FP matching used for D-ECE excluded crowd GT entirely, so detections on crowd regions counted as false positives. Found by Codex audit.
- **Impact:** 143 clean detections moved from FP to ignored (FP 372 → 229). Clean D-ECE changed from 0.156724 to 0.208459. mAP50 and mAP50-95 unchanged to every digit.
- **Fix:** TP/FP matching now applies the COCO ignore rule: an unmatched detection whose intersection / detection area with any crowd person GT is ≥ 0.5 is ignored (neither TP nor FP, excluded from D-ECE and counts, reported as `n_ignored`).

### 3. Prediction cache accepted results from different settings
- **Date:** 2026-11-03
- **What happened:** The cache was keyed only by file name, so a run with different weights, imgsz, conf or image ids would silently reuse old predictions. Found by Codex (reproduced: an imgsz=320 request returned the 640 predictions).
- **Impact:** No saved result was affected.
- **Fix:** Each cache stores metadata (weights, imgsz, conf, device, condition, sorted image ids, corruption parameters). On load every field is compared; any mismatch is printed and predictions are recomputed.

### 4. Smoke tests overwrote the full-run row
- **Date:** 2026-11-03
- **What happened:** `--limit` runs wrote the `clean` row of `results/table.csv`, replacing the full-run row. Found by Codex.
- **Impact:** None on saved results (the full run was executed after the smoke test).
- **Fix:** Smoke runs write only to `results/table_smoke.csv` with condition `<condition>_limit<N>`.

### 5. Cached subset.json not validated
- **Date:** 2026-11-03
- **What happened:** `data/subset.json` was loaded without checking it against the selection rule.
- **Impact:** None; the saved subset matched exactly.
- **Fix:** On every run the subset is recomputed and compared (count, order, ids); a mismatch raises an error describing the difference.

### 6. n_ignored written as float
- **Date:** 2026-11-03
- **What happened:** After merging a new row into an older `table.csv` without that column, pandas stored `n_ignored` as `143.0`.
- **Impact:** Formatting only.
- **Fix:** Count columns are cast to integer when writing the table.

### 7. Fisheye angular cutoff missing from cache metadata and remap key
- **Date:** 2026-11-03
- **What happened:** `FISHEYE_MAX_THETA_DEG` affects the fisheye image but was in neither the prediction-cache metadata nor the in-memory remap-map key. Found by Codex.
- **Impact:** None on saved results; changing the cutoff would have reused stale predictions and maps.
- **Fix:** Added to the metadata of fisheye caches and to the remap key.

### 8. Noise seed missing from cache metadata
- **Date:** 2026-11-03
- **What happened:** `config.SEED` determines the Gaussian noise but was not in the cache metadata. Found by Codex.
- **Impact:** None on saved results.
- **Fix:** Added `seed` to the metadata of Gaussian-noise caches.

### 9. Existing caches migrated instead of recomputed (added)
- **Date:** 2026-11-03
- **What happened:** Fixes 7 and 8 made the existing fisheye and noise caches fail metadata validation, which would have forced inference to rerun.
- **Impact:** None. All 15 full-run conditions were recomputed from cache with no inference, and every pre-existing table column was identical to the previous table.
- **Fix:** Wrote `fisheye_max_theta_deg: 89.0` into the 4 fisheye caches and `seed: 0` into the 6 noise caches (full and smoke), the exact values used when they were generated (asserted against `config.py`). Detections were not modified.

### 10. Overlapping labels in the silent-failure figure
- **Date:** 2026-11-03
- **What happened:** In `fig_silent_failure`, labels in the dense high-recall cluster overlapped and could not be matched to their points. Two placement bugs followed: the label box was measured including its own leader line, and before `tight_layout` moved the axes.
- **Impact:** Figure readability only; no numbers affected.
- **Fix:** Greedy label placement over 8 offsets, measuring the text box without its leader after the layout is fixed, rejecting positions that overlap placed labels or markers or leave the axes; thin leader lines join each label to its point.

### 11. Silent-failure figure unreadable on the full run
- **Date:** 2026-11-03
- **What happened:** On the full 500-image run, 11 of the 15 points fell in a small region (recall 0.54–0.66, mean TP confidence 0.668–0.686). No label placement could separate them, and several labels still collided.
- **Impact:** Figure readability only; no numbers affected.
- **Fix:** `fig_silent_failure.png` now has one panel per corruption family (shared axes). Each panel joins clean → s1 → s2 (→ s3) in blue with labels; all other conditions appear as faint grey context points. Every condition is still plotted and labelled.

## Design decisions

### Image download
- **Date:** 2026-11-03
- **Decision:** Only the 500 subset images are downloaded, from their official COCO `coco_url` (identical to the files in `val2017.zip`).
- **Impact:** None on results; avoids the 778 MB archive.

### Person filtering
- **Date:** 2026-11-03
- **Decision:** The model predicts all classes; person detections (YOLO class 0) are kept afterwards, rather than restricting prediction to the person class.
- **Impact:** Defines which boxes survive NMS and `max_det`.

### Noise seeding
- **Date:** 2026-11-03
- **Decision:** Gaussian noise is seeded per image with `(SEED, image_id)`, so each image gets the same noise pattern at every severity, scaled by std.
- **Impact:** Severity levels differ only in noise magnitude, not pattern.

### Fisheye model and GT warping
- **Date:** 2026-11-03
- **Decision:** Equidistant model, f = half-diagonal / theta_max per image (theta_max 60° for s1, 80° for s2). GT boxes (crowd included) are warped as the bounding box of 20 forward-warped points per edge, clipped, and dropped below 4 px. Warped GT is a separate COCO object used for both mAP and TP/FP matching.
- **Impact:** Warped GT boxes are looser than hand-drawn boxes around the deformed person, which may lower mAP50-95 (known limitation). Dropped GT: 36 (s1) and 45 (s2) of 2090 person boxes, none crowd.

### Fisheye GT `area` field (added)
- **Date:** 2026-11-03
- **Decision:** Each warped box keeps the original segment-area / box-area ratio in its COCO `area` field.
- **Impact:** Affects only COCO's small/medium/large breakdown, not reported mAP50 or mAP50-95.

### Radial distance definition (added)
- **Date:** 2026-11-03
- **Decision:** `r_norm` in `results/dets/` is the distance of the detection box centre from the image centre divided by the half-diagonal (0 = centre, 1 = corner).
- **Impact:** Defines the radial variable for later per-position analysis.

### D-ECE binning (added)
- **Date:** 2026-11-03
- **Decision:** 10 equal-width bins on [0, 1], empty bins skipped. With the 0.25 threshold the first populated bin, labelled [0.2, 0.3), holds only detections in [0.25, 0.3).
- **Impact:** Bins below 0.2 are always empty.

### Low-threshold recall (stage reliability, part A)
- **Date:** 2026-11-03
- **Decision:** `recall_005` and `recall_010` reuse the same greedy matching and COCO crowd rule as `recall_025`, applied to the cached original-view predictions (stored at conf ≥ 0.001) with conf ≥ 0.05 / 0.10. Part A loads caches with `cache_only=True`, so it raises instead of ever running inference.
- **Impact:** Measures whether missed people are detected at low confidence. All previously existing table columns were verified unchanged after the recomputation.

### Augmented views (stage reliability, part B)
- **Date:** 2026-11-03
- **Decision:** Two views of the same corrupted image: horizontal flip (applied after the corruption; boxes mapped back with x1' = W − x2, x2' = W − x1) and imgsz = 576. Each view has its own cache, `results/preds/<condition>__<view>.json`, whose metadata adds the view (name, flip, imgsz) and uses the view's imgsz. Original-view caches carry no view field, so they stay valid. `--force` in this stage recomputes only the view predictions.
- **Impact:** 30 new inference runs on the full subset. Sanity check on the smoke run: TP detections have mean consistency 0.86–0.90 (IoU), which a wrong flip mapping could not produce.

### Box-level and image-level definitions (stage reliability)
- **Date:** 2026-11-03
- **Decision:**
  - Box-level AUROCs (TP = positive, ignored excluded) go to `results/box_level.csv`, not `table.csv`, so that rerunning baseline or shift (which rewrite whole rows) cannot leave them stale. Per-detection consistency is saved to `results/box_level_dets.csv`.
  - Image level: `n_det` counts all detections with conf ≥ 0.25 including ignored ones (signals must not use GT). When an image has no detections, `mean_conf` = 0 and `mean_consistency` = 0. Images without non-crowd person GT (possible after fisheye drops) are excluded.
  - `results/image_level.csv` holds the summary (per condition and pooled); per-image rows go to `results/image_level_per_image.csv`. Pooled = all per-image rows of the 15 conditions concatenated.
  - AUROC is reported raw (signal → miss_rate ≥ 0.5): values below 0.5 mean low values of the signal indicate misses (expected for `mean_conf` and `mean_consistency`).
  - Risk-coverage ranks with a stable sort, so tied scores keep the per-image, descending-confidence order of the detection list.
  - Smoke runs write `*_smoke` variants of every file and figure.
- **Impact:** `n_det` and `mean_conf` come from the same detections that determine `miss_rate`, so they are not independent of the target; this must be kept in mind when interpreting their correlations.

## Findings so far

- Clean YOLO11n person: mAP50 0.780, D-ECE 0.208, underconfident in every confidence bin. No bin is overconfident in any of the 15 conditions, so `signed_cal_gap` equals `d_ece` everywhere.
- Under corruption the detector mainly fails by missing people. For example, under blur s3, TP falls 1340 → 776 while mean TP confidence moves only 0.686 → 0.661 and FP falls 229 → 151. D-ECE barely changes because it only evaluates emitted boxes.
- Exception: at the most severe JPEG level (jpeg_s3), D-ECE rises from 0.208 to 0.289, mean TP confidence drops to 0.494, and recall at conf ≥ 0.25 falls to 0.059 (121 TP).
- (2026-11-03, stage reliability) Many missed people are present at low confidence: clean recall is 0.656 at conf ≥ 0.25 but 0.820 at conf ≥ 0.05; for gaussian_blur_s3, 0.380 vs 0.521; for jpeg_s3, 0.059 vs 0.150.
- (2026-11-03) Box level, TP vs FP: confidence alone (AUROC 0.82–0.86 in 14 of 15 conditions) beats view consistency alone (0.73–0.79) everywhere except jpeg_s3 (0.726 vs 0.776). Multiplying them improves on confidence in only 4 of 15 conditions (gaussian_blur_s3, gaussian_noise_s3, jpeg_s1, jpeg_s3), by at most 0.058 (jpeg_s3: 0.784 vs 0.726).
- (2026-11-03) Image level (pooled, 7498 images, 2787 with miss_rate ≥ 0.5): mean_conf has the strongest relation to miss_rate (Spearman −0.706; AUROC 0.141, i.e. 0.859 with the sign flipped), then mean_consistency (−0.678; 0.161), view_disagreement (0.245; 0.613) and n_det (−0.202; 0.332). Caveat: mean_conf and n_det use the same detections that define miss_rate.

### Codex audit: pooled Spearman significance ignores repeated images
- **Date:** 2026-11-03
- **Stage:** reliability
- **What is wrong:** The pooled Spearman p-values use an ordinary independent-observation test on 7,498 condition/image rows from only 500 unique source images (499 repeated 15 times, one repeated 13 times). Corruption variants of the same image are dependent, so these p-values are not calibrated for this sampling design. The pooled descriptive rho and AUROC values are unaffected; condition/image pairs are unique.
- **File and line:** reliability.py:116, reliability.py:119; run.py:246-247.
- **Found by:** found by Codex.
- **Impact:** Affects current saved results: the four pooled spearman_p entries in results/image_level.csv, and the same issue in pooled smoke significance. Does not invalidate the descriptive pooled rho or AUROC.
- **Minimal proposed fix (not applied):** Omit pooled p-values (NaN) until significance is computed with image-id clusters, or use an appropriate clustered inference procedure.
- **Fix (applied 2026-11-03):** `image_summary(..., p_values=False)` for the pooled rows, so pooled `spearman_p` is written as NaN (empty). Per-condition p-values are kept. The README no longer presents pooled p-values and explains why. Results recomputed from cache, no inference.
- **Impact:** `results/image_level.csv` pooled spearman_p: n_det 7.186e-70 → NaN, mean_conf 0.0 → NaN, view_disagreement 4.214e-103 → NaN, mean_consistency 0.0 → NaN. `results/image_level_smoke.csv` pooled spearman_p: 3.642e-07, 1.659e-30, 0.00785, 6.235e-17 → NaN. Every other value in both files (pooled rho and AUROC, all per-condition rows) is unchanged.

### Codex audit: augmented-view cache accepted for original-view request
- **Date:** 2026-11-03
- **Stage:** reliability
- **What is wrong:** Cache validation compares only requested metadata keys. Original-view requests omit the view field, so a stored hflip or rescale cache can pass validation as an original view when supplied at that request's cache path. Reproduced by loading clean__hflip.json with clean original-view metadata; it was accepted. Flip versus rescale is correctly rejected.
- **File and line:** infer.py:96-97, infer.py:109.
- **Found by:** found by Codex.
- **Impact:** Only future runs with a misrouted or misplaced view cache. Current generated filenames are distinct and saved results use the correct caches.
- **Minimal proposed fix (not applied):** Compare view presence/value symmetrically, treating absence as the original view, or validate complete metadata equality.
- **Fix (applied 2026-11-03):** `load_cache` now compares the union of stored and requested metadata keys, with an absent key read as None, so a key present on only one side (such as `view`) is a mismatch in either direction. Verified: `clean__hflip.json` requested as an original view is rejected ("Cache mismatch in view"), and all 90 existing caches (15 conditions × 3 views, full and smoke) still pass.
- **Impact:** None on saved results.

### Codex audit: all-zero-GT image subset crashes image summary
- **Date:** 2026-11-03
- **Stage:** reliability
- **What is wrong:** image_level skips zero-non-crowd-GT images but returns a columnless DataFrame when every image is skipped. image_summary then accesses miss_rate and raises KeyError. Reproduced using actual image 52891 after fisheye_s2 GT dropping.
- **File and line:** reliability.py:106, reliability.py:111; run.py:243.
- **Found by:** found by Codex.
- **Impact:** Only future runs where an entire evaluated condition subset has no retained non-crowd person GT. Current full and smoke runs retain eligible images and are unaffected.
- **Minimal proposed fix (not applied):** Preserve the per-image DataFrame schema for empty outputs and explicitly return zero-count/NaN summary statistics for empty conditions.
- **Fix (applied 2026-11-03):** `image_level` builds its DataFrame with a fixed column list (`IMAGE_COLUMNS`), so an all-skipped subset keeps the schema. `image_summary` returns n_images 0 and NaN rho, p and AUROC when fewer than 2 images remain. Verified with image 52891 under fisheye_s2: 0 rows, full schema, NaN summary, no KeyError.
- **Impact:** None on saved results.

### Gradient check reported PASSED with a too-lenient criterion
- **Date:** 2026-11-03
- **Stage:** attack (gradient check)
- **What went wrong:** `attack.gradcheck` only required the final loss to be below the first loss, so it printed "GRADIENT CHECK PASSED" although the loss rose between iterations on image 785 (pgd_eps4: 9.434 → 10.145 at step 1).
- **Fix:** The check now requires a strict decrease at every iteration (`ok &= dec`). Rerun: "GRADIENT CHECK FAILED".
- **Impact:** None on saved results (no attack images or predictions had been produced).

### PGD loss does not decrease monotonically (gradient check failed)
- **Date:** 2026-11-03
- **Stage:** attack (gradient check, images 139 and 785)
- **What went wrong:** The gradient is not None and the differentiable letterbox matches ultralytics exactly (448×640, max abs pixel difference 2.97e-08). But with alpha = 1/255, 10 sign steps and the summed-person-score loss, the loss oscillates on image 785: pgd_eps4 9.434, 10.145, 8.692, 6.441, 7.771, 6.747, 5.001, 5.901, 6.931, 7.417, final 3.326; pgd_eps8 9.434, 10.145, 8.692, 6.441, 7.771, 6.644, 6.514, 8.243, 5.838, 4.381, final 7.001. Person detections ≥ 0.25 on the saved uint8 image went 1 → 2 (eps4) and 1 → 1 (eps8). On image 139 the loss collapses (6.620 → 0.002) and detections go 1 → 0 for both eps, but for eps8 it is not strictly decreasing in the last iterations.
- **Fix:** None yet. Stopped before the smoke test, as instructed, pending a decision on the attack procedure.
- **Impact:** None on saved results (no attack images or predictions had been produced).

### Codex audit: attack entrypoint calls an undefined runner
- **Date:** 2026-11-03
- **Stage:** attack
- **What is wrong:** run.py advertises --stage attack but its normal dispatch calls run_attack, which is not defined or imported anywhere in the project. Calling run.main() with --stage attack reproduces NameError: name 'run_attack' is not defined. The gradient-check branch exists, but the generation/evaluation stage, clean_100 control, augmented attack views and attack_detection.csv analysis are not connected. No saved attack PNGs, attack predictions or attack visualizations currently exist.
- **File and line:** run.py:305.
- **Found by:** found by Codex.
- **Impact:** Only future runs: the normal attack stage currently crashes before generating or evaluating images. No current saved attack results exist to invalidate; earlier conditions are unaffected.
- **Fix (applied 2026-11-03):** Implemented `run_attack` in run.py and wired it to `--stage attack`. It generates or reuses the attacked PNGs and evaluates clean_100, rand_eps8, pgd_eps4 and pgd_eps8 through the normal pipeline: table rows, hflip/imgsz576 views, box and image level (upserted into the shared CSVs), attack_views.csv, attack_detection.csv, attack_summary.csv and visualizations. Smoke test `--limit 10` ran to completion (exit 0).
- **Impact:** None on saved results (no attack results existed before the fix).

### Codex audit: generated PGD image cache omits model and preprocessing settings
- **Date:** 2026-11-03
- **Stage:** attack
- **What is wrong:** generate reuses saved PNGs based only on the attack spec, ordered image ids and PNG existence. The attack model weights and preprocessing geometry/settings are absent from meta.json. Reproduced with mocked existing files: changing MODEL_WEIGHTS to a different model still reused the set without loading the new model. A prediction digest of unchanged stale PNGs would not repair this upstream cache error.
- **File and line:** attack.py:127-130, attack.py:142.
- **Found by:** found by Codex.
- **Impact:** Only future runs after attack images have been generated and attack weights or preprocessing settings change. No saved attack images currently exist.
- **Minimal proposed fix (not applied):** Include the attack weights fingerprint and effective preprocessing settings in the generated-image cache metadata and validate them before reuse.
- **Fix (applied 2026-11-03):** `meta.json` of each attacked-image set now stores the full attack spec (loss type, top-k, eps, alpha, iterations, best-iterate flag, random start; seed for rand_eps8), model weights name and SHA-256, imgsz, letterbox settings (auto, stride 32, pad value 114, bilinear resize) and image ids. The model stride is asserted to equal the configured letterbox stride. Reuse requires symmetric equality of all fields plus all PNGs and `attack_info.csv` present; any mismatch prints the field and regenerates. Prediction caches of attack conditions also carry a SHA-256 of the saved PNG bytes.
- **Impact:** None on saved results (no attack images had been saved before the fix).

### Attack method changed: top-100 loss with best iterate (was sum-over-all loss with last iterate)
- **Date:** 2026-11-03
- **Stage:** attack
- **What went wrong:** With loss = sum of person scores over all predictions, 10 sign steps and the last iterate returned, the loss oscillated between steps (image 785, pgd_eps4: 9.434, 10.145, 8.692, 6.441, 7.771, …, final 3.326) and person detections ≥ 0.25 on image 785 increased 1 → 2 under pgd_eps4.
- **Fix:** Loss = sum of the top-100 person scores; 20 iterations (alpha 1/255, eps 4/255 and 8/255, no random start); after each step (after projection and clipping) the loss is recorded and the image with the lowest loss seen, including the starting image, is returned. Gradient check criterion: for every image, best loss ≤ 50% of the starting loss and persons ≥ 0.25 do not increase. New check on images 139, 785, 872, 885, 1000: PASSED for both eps; on image 785 persons ≥ 0.25 now go 1 → 0 for pgd_eps4 and pgd_eps8.
- **Impact:** None on saved results (the original attack never produced saved images).

### Misaligned label text in the perturbation panel
- **Date:** 2026-11-03
- **Stage:** attack (visualization)
- **What went wrong:** In `results/vis/pgd_eps8/perturbation_panel.png` the white label text was drawn over a thicker black outline, which looked offset; "perturbation x10" could be read as "x100".
- **Fix:** Labels are now white text on a solid black box sized with `cv2.getTextSize` (scale 0.8, thickness 2). Only this image was regenerated (same source image 139 and the same saved pgd_eps8 PNG).
- **Impact:** None on saved results (figure only).

### First reproduce.bat check run did not start
- **Date:** 2026-11-03
- **Stage:** reproducibility check (reproduce.bat)
- **What went wrong:** Launching the script from Git Bash as `cmd //c reproduce.bat` failed after 1 s with "'reproduce.bat' is not recognized as an internal or external command"; no stage ran.
- **Fix:** Ran it from PowerShell in the project folder (`& .\reproduce.bat`). It completed all stages in 294 s with no inference or image regeneration, and all 215 checked files (100 CSVs, summary.md, prediction caches) were byte-identical. The script itself was not changed.
- **Impact:** None on saved results.
