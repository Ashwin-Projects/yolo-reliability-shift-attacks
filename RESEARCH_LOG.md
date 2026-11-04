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
- (2026-11-03) Image level (pooled, 7,498 condition-image observations from 500 source images, 2787 with miss_rate ≥ 0.5; descriptive only, as the observations are not independent): mean_conf has the strongest relation to miss_rate (Spearman −0.706; AUROC 0.141, i.e. 0.859 with the sign flipped), then mean_consistency (−0.678; 0.161), view_disagreement (0.245; 0.613) and n_det (−0.202; 0.332). Caveat: mean_conf and n_det use the same detections that define miss_rate.

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

### Codex final audit: ignored attack audit CSVs and metadata are absent from the frozen commit
- **Date:** 2026-11-03
- **Stage:** final whole-project audit / packaging
- **What is wrong:** The blanket results/adv/ ignore rule excludes six small attack_info.csv files and six meta.json files in addition to large PNGs. The working folder has 100 result CSVs, but HEAD a6e3538 contains only 94. The excluded files hold per-image start/best/last losses, best-iterate indices and attack-generation settings/weights hashes needed to independently check the attack method and mean_best_iter findings without regenerating the attacks.
- **File and line:** .gitignore:5; README.md:47 and README.md:215-216 identify the affected artifacts/claims.
- **Found by:** found by Codex.
- **Impact:** Affects current saved-result packaging and verification from the committed project; local CSV metric values are unaffected. Also affects future fresh checkouts.
- **Minimal proposed fix (not applied):** Exclude large PNGs while retaining and committing the small attack_info.csv and meta.json files for full and smoke conditions.
- **Fix (applied 2026-11-03):** `.gitignore` rule `results/adv/` replaced by `results/adv/**/*.png`. `git check-ignore`: all 330 PNGs are ignored; all 6 `attack_info.csv` and all 6 `meta.json` (full and smoke conditions) are not. These 12 files were committed.
- **Impact:** None on saved results. The committed result CSVs go from 94 to 100 (all in the working folder).

### Codex final audit: letterbox numerical-agreement claim lacks its image scope
- **Date:** 2026-11-03
- **Stage:** final whole-project audit / documentation
- **What is wrong:** The README reports a maximum letterbox pixel difference of 2.97e-08 without identifying that this was measured on images that require no resizing. Resized source images give larger float-versus-uint8 interpolation differences: image 1353 gives 0.00293836 and image 1584 gives approximately 0.0029824 on [0,1] pixels. The geometry still matches and both differences are below 1/255; the claim should not imply general near-bit-exact pixel equality.
- **File and line:** README.md:202-203.
- **Found by:** found by Codex.
- **Impact:** Affects current saved documentation used for the write-up; no CSV metrics are affected.
- **Minimal proposed fix (not applied):** State the tested image IDs/sizes and distinguish geometric agreement from the resize/rounding pixel tolerance.
- **Fix (applied 2026-11-03):** The README now says the geometry matches exactly. The maximum pixel difference is 2.97e-08 for images needing no resize (139, 640×426) and about 0.0029, below 1/255, for resized images, with the reason (cv2 uint8 resize vs float interpolation). Re-measured before editing: 139 → 0.00000003, 1353 (375×500) → 0.00293836, 1584 (612×612) → 0.00298240; geometry equal in all three.
- **Impact:** None on saved results (documentation only).

### Codex final audit: pooled Findings sample count needs qualification
- **Date:** 2026-11-03
- **Stage:** final whole-project audit / Findings wording
- **What is wrong:** The Findings bullet calls the pooled sample 7498 images without qualifying in that bullet that these are condition-image observations from 500 unique source images. There are 499 source IDs with 15 observations and one source ID with 13. The count and descriptive statistics are correct, but the write-up must not present this as 7498 independent source images; the README already explains this dependence correctly.
- **File and line:** RESEARCH_LOG.md:140.
- **Found by:** found by Codex.
- **Impact:** Affects current saved Findings wording and sample-size interpretation; no CSV metric values are affected.
- **Minimal proposed fix (not applied):** Say 7498 condition-image observations from 500 source images and retain the descriptive-only qualification.
- **Fix (applied 2026-11-03):** The Findings bullet now reads "7,498 condition-image observations from 500 source images … descriptive only, as the observations are not independent". README.md already said "7,498 rows from only 500 source images" and results/summary.md has no such count; neither needed a change.
- **Impact:** None on saved results (wording only).

### Report PDF not written by headless Edge
- **Date:** 2026-11-03
- **Stage:** report (build_report.py)
- **What went wrong:** The first build crashed with `FileNotFoundError: ...\report\report.pdf`. `msedge.exe --headless --print-to-pdf` returned exit code 0 after 0.1 s and wrote the PDF a few seconds later (seen when debugging: absent right after the call, present after 5 s). A first attempt that only added `--headless=new` and a separate `--user-data-dir` still checked too early.
- **Fix:** `build_report.py` uses `--headless=new` with a temporary profile directory and waits (up to 120 s) until the PDF exists and its size is stable.
- **Impact:** None on saved results (report build only).

### Report runtime table included unrelated README tables
- **Date:** 2026-11-03
- **Stage:** report (build_report.py)
- **What went wrong:** `readme_runtime_table` copied every markdown table line after the "Runtime on CPU" heading, so the report's reproducibility section also contained the README's shift-condition, fisheye f and attack-condition tables. The PDF ran to 5 pages.
- **Fix:** Only the first contiguous table after the heading is copied. The layout was also tightened (font 9.4 pt, Figure 2 at 50% width). The PDF is now 4 pages.
- **Impact:** None on saved results (report text only).

### Codex report verification: Signed calibration gap definition has the wrong implied sign
- **Date:** 2026-11-03
- **Stage:** report verification
- **What is wrong:** D-ECE is written with |mean confidence - precision|. Saying the signed gap is the same sum without the absolute value implies mean confidence - precision, whereas the code computes precision - mean confidence so that positive means underconfident. The CSV values follow the code; the report method definition is inconsistent.
- **File and line:** report/report.md:38-39 (also report/report.pdf, page 2)
- **Found by:** found by Codex.
- **Impact:** Affects the current saved report/documentation only; saved result CSV values are unaffected.
- **Minimal proposed fix (not applied):** State the signed expression explicitly as sum_b (n_b/N)*(precision_b - mean_confidence_b).
- **Fix (applied 2026-11-03):** The Metrics list now defines it directly: Σ (n_bin / N) · (precision − mean confidence) over the same bins; positive means underconfident. The "without the absolute value" wording is removed.
- **Impact:** None on saved results (report only).

### Codex report verification: Abstract overstates what confidence can reveal about misses
- **Date:** 2026-11-03
- **Stage:** report verification
- **What is wrong:** The abstract concludes that confidence of emitted boxes does not reveal misses. This is stronger than the reported evidence: pooled mean confidence has Spearman rho -0.705940784807534 with miss_rate. Per-detection calibration alone does not measure recall, but this does not establish that aggregate confidence is uninformative about misses.
- **File and line:** report/report.md:9; compare report/report.md:77
- **Found by:** found by Codex.
- **Impact:** Affects the current saved report/documentation only; saved result CSV values are unaffected.
- **Minimal proposed fix (not applied):** Limit the conclusion to per-detection calibration not accounting for false negatives; distinguish it from image-level confidence signals.
- **Fix (applied 2026-11-03):** Abstract now reads: "…underconfident in all 15 clean and shifted conditions; per-detection calibration metrics cannot account for the missed people, because a missed person produces no detection to evaluate." It no longer claims confidence is uninformative (F3 keeps the image-level Spearman −0.706).
- **Impact:** None on saved results (report only).

### Codex report verification: PDF figure readability and heading pagination need correction
- **Date:** 2026-11-03
- **Stage:** report verification
- **What is wrong:** Both PDF figures render, but axis/legend/direct-label text is roughly 3-4 points at the printed size (Figure 1 spans five panels in approximately 499 PDF points; Figure 2 is only approximately 250 points wide). The heading 6 Next steps is orphaned at the bottom of PDF page 3, with its list starting on page 4. The results table itself is readable.
- **File and line:** report/report.md:65-71 and report/report.md:92; report/build_report.py:286,295-296
- **Found by:** found by Codex.
- **Impact:** Affects the current saved report/documentation only; saved result CSV values are unaffected.
- **Minimal proposed fix (not applied):** Give plots enough space for readable labels and keep headings with their following content; no figure or report was changed.
- **Fix (applied 2026-11-03):** The report now uses its own copies of the figures in report/figures/, drawn at their printed size (Figure 1: 7.0 × 3.6 in, 3 + 2 panels; Figure 2: 4.2 × 2.2 in), so text prints at 8–10 pt. `fig_silent_failure` / `fig_recoverable` gained optional ncols/figsize/title parameters; their defaults reproduce results/fig_silent_failure.png and results/fig_recoverable.png byte-identically (md5 unchanged). Each h2 heading is wrapped with its following block, and each figure with its caption, in a break-inside: avoid container; figures moved next to F1 and F2. PDF: 4 pages, no stranded heading.
- **Impact:** None on saved results (report only).

### Codex report verification: Fisheye limitation makes an unsupported causal attribution
- **Date:** 2026-11-03
- **Stage:** report verification
- **What is wrong:** The report says loose warped GT mainly affects mAP50-95 and gives AP50-95/AP50 ratios 0.684 and 0.597. Those ratios are numerically correct, but no manually annotated warped-GT comparison or ablation isolates how much of the change is due to GT box looseness versus image distortion and detector localization.
- **File and line:** report/report.md:87
- **Found by:** found by Codex.
- **Impact:** Affects the current saved report/documentation only; saved result CSV values are unaffected.
- **Minimal proposed fix (not applied):** Describe box looseness as a possible contributor and the ratios as descriptive, rather than attributing the change mainly to it.
- **Fix (applied 2026-11-03):** Limitation now reads: the drop in mAP50–95 (ratio 0.684 clean, 0.597 fisheye_s2) may partly reflect looser warped ground-truth boxes, but this study does not separate that effect from distorted appearance or detector localization.
- **Impact:** None on saved results (report only).

### Codex report verification: Reference 1 title is incomplete
- **Date:** 2026-11-03
- **Stage:** report verification
- **What is wrong:** Reference 1 lists Benchmarking Robustness in Object Detection, but the cited arXiv:1907.07484 title is Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming. Authors/arXiv identifier/2019 first-submission year are correct.
- **File and line:** report/report.md:114
- **Found by:** found by Codex.
- **Impact:** Affects the current saved report/documentation only; saved result CSV values are unaffected.
- **Minimal proposed fix (not applied):** Use the full title including the subtitle.
- **Fix (applied 2026-11-03):** Reference 1 now has the full title "Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming". Reference 4 cites arXiv:2412.01782v3; reference 3 is given as International Journal of Computer Vision, vol. 133, pp. 1033–1047, 2025.
- **Impact:** None on saved results (report only).

### Codex report verification: Finding F4 heading describes removal of persons rather than detections
- **Date:** 2026-11-03
- **Stage:** report verification
- **What is wrong:** The F4 heading says PGD removes all persons. The data show suppression of person detections at confidence >=0.25 on the tested 100-image subset, not removal of people from the images or a universal result. The paragraph beneath supplies the correct detection-based denominator and scope, so the issue is the stronger headline.
- **File and line:** report/report.md:79
- **Found by:** found by Codex.
- **Impact:** Affects the current saved report/documentation only; saved result CSV values are unaffected.
- **Minimal proposed fix (not applied):** Say PGD suppresses all person detections at the tested threshold, and retain the subset and non-adaptive attack scope.
- **Fix (applied 2026-11-03):** F4 heading now reads "PGD suppresses all person detections at confidence ≥ 0.25 on the 100-image subset; random noise of equal size does not." The non-adaptive scope remains stated in F5 and the Limitations.
- **Impact:** None on saved results (report only).

### Overlapping y-axis labels in the 3 + 2 report figure
- **Date:** 2026-11-03
- **Stage:** report (Figure 1 layout)
- **What went wrong:** With the new 3 + 2 panel layout, the long y-axis label was drawn for each row; at the printed size it was clipped and overlapped between rows. The taller figure also pushed the PDF to 5 pages, with Figure 1 leaving half of page 2 empty.
- **Fix:** In grid layouts one figure-level y label (`fig.supylabel`) replaces the per-row labels; the one-row default is unchanged. Figure heights reduced (7.0 × 3.6 in, 4.2 × 2.2 in) and figures placed after F1 and F2. PDF back to 4 pages.
- **Impact:** None on saved results (report figures only; results/ figures byte-identical).

### PDF text comparison reported spurious differences
- **Date:** 2026-11-04
- **Stage:** report (updating the cited results commit)
- **What went wrong:** The first before/after comparison of the PDF text reported 174 changed lines, 170 of them blank. The "before" text had been saved with Windows newline translation, so its line endings did not match the freshly extracted text.
- **Fix:** Compared again with line endings normalised and blank lines dropped: 174 → 174 text lines, exactly 2 changed (the two commit-hash mentions, 8ad4f20 → a4e3e1d), and all 382 other numbers in the PDF identical. report.md is byte-identical to the previous version apart from the hash.
- **Impact:** None on saved results (verification step only).

## Phase 1

### Commit d3c5fb0 dispatched to an undefined run_phase1
- **Date:** 2026-11-04
- **What went wrong:** Of three edits to run.py, the one adding the `run_phase1` function was rejected by an editor pre-check while the two CLI edits (the `phase1` choice and its dispatch) were applied. Commit d3c5fb0 therefore contained `--stage phase1` calling a function that did not exist; `--help` still worked because the name is only resolved when the stage runs. The stage had not been run.
- **Fix:** Added `run_phase1` and checked that `run.run_phase1` is defined before committing the fix together with this entry.
- **Impact:** None on saved results (no Phase 1 results existed; earlier stages untouched).

### Byte comparison against main reported all earlier CSVs as different
- **Date:** 2026-11-04
- **What went wrong:** After the Phase 1 smoke test, comparing `sha256(git show main:<file>)` with `sha256(<file on disk>)` reported all 100 earlier result CSVs as different, while `git diff main` reported none. Cause: with `core.autocrlf=true` the stored blobs are line-ending-normalised, so each working-tree CSV is one byte per line longer than its blob (results/table.csv: 5159 bytes on disk vs 5139 in the blob, 20 lines). The comparison method was wrong, not the files.
- **Fix:** Compared with `git hash-object <file>` against `git rev-parse main:<file>` (same normalisation as git): 100 of 100 identical. Raw on-disk SHA-256 against the hashes recorded in the 2026-11-03 reproducibility check: 100 of 100 identical.
- **Impact:** None on saved results (verification only; no earlier result file changed).

### Phase 1 cache omits active global fisheye settings
- **Date:** 2026-11-04
- **What is wrong:** Per-target cache metadata includes PHASE1 but omits FISHEYE_MAX_THETA_DEG, which phase1.plane_grid uses, and FISHEYE_MIN_BOX, which the reused corruptions.warp_box uses. Changing either global can change canvases or GT while cached results remain accepted. Reproduced: changing the cutoff from 89 to 45 degrees returned cached target 785 unchanged; a freshly rebuilt render changed 23,798 pixels. The in-memory plane-grid cache is keyed only by f, so it also reuses an old grid after a cutoff change. Raising the global minimum box size changed a GT box to None without changing metadata.
- **File and line:** phase1.py:132-142, phase1.py:246-251, phase1.py:275-276; corruptions.py:109 (minimum-box rule).
- **Found by:** found by Codex.
- **Impact:** Only future runs after these active global settings change; current saved smoke results use the expected defaults and are unaffected. The full run can silently reuse stale geometry/results after such a change.
- **Minimal proposed fix (not applied):** Include the effective global warp settings in per-target metadata and the angular cutoff in the plane-grid key, or make the Phase 1 warp depend exclusively on explicitly recorded Phase 1 settings.
- **Fix (applied 2026-11-04, commit 6b75d3b):** `phase1.cache_meta` now includes a `global` block (`FISHEYE_MAX_THETA_DEG`, `FISHEYE_MIN_BOX`, `FISHEYE_EDGE_POINTS`, `YOLO_PERSON_CLS`, `PERSON_CAT_ID`), and the plane-grid cache key is `(f, FISHEYE_MAX_THETA_DEG)`. The 5 existing smoke caches were migrated by writing the values they were generated with (89.0, 4, 20, 0, 1; asserted equal to the current config); no image or attack was regenerated. Each migrated cache matches the new metadata, and its result block is identical to the backup.
- **Impact:** None on saved results.

### Phase 1 summary crashes when no targets pass eligibility
- **Date:** 2026-11-04
- **What is wrong:** paired() pivots eligible targets and accesses wide[a] and wide[b] before testing for an empty paired sample. With no eligible targets the pivot has no condition columns, causing KeyError: 'edge' instead of an empty/NaN summary. Reproduced by setting eligibility False on every smoke row and calling phase1_analysis.summarize. The later empty-sample guard is never reached.
- **File and line:** phase1_analysis.py:55-56, phase1_analysis.py:58-60.
- **Found by:** found by Codex.
- **Impact:** Only future runs with zero eligible targets, for example after changing detector or eligibility settings. Current saved smoke results and the unchanged-design full run (which includes the five eligible smoke targets) are unaffected.
- **Minimal proposed fix (not applied):** Check for an empty eligible pivot or missing comparison columns before subtracting the paired columns, and return zero counts with undefined statistics.
- **Fix (applied 2026-11-04):** New `paired_diffs` returns an empty array when the eligible pivot is empty or a comparison column is missing; `paired` and `fig_paired` both use it. `summarize` adds the row "no eligible targets: per-condition and paired statistics are undefined". Codex's reproduction (all smoke rows ineligible) now gives n_eligible 0, the message, n = 0 and NaN statistics in all 45 paired rows, and both figures render.
- **Impact:** None on saved results. The summary recomputed on the smoke data is identical to the saved summary_smoke.csv.

### Cache-metadata fix committed without its log entry
- **Date:** 2026-11-04
- **What went wrong:** The edit adding the fix and impact to the Codex entry "Phase 1 cache omits active global fisheye settings" was rejected by an editor pre-check, while the commit sent in the same batch went through. Commit 6b75d3b therefore contained the code fix and the migrated caches but not the log text, contrary to the rule of committing the log with its fix.
- **Fix:** The log text was added and committed immediately afterwards in cfe81fa, which names 6b75d3b.
- **Impact:** None on saved results.

## Phase 1b

### First launch of the full Phase 1b run was blocked
- **Date:** 2026-11-04
- **What went wrong:** The first attempt to start `python run.py --stage phase1b` (95 targets) was rejected by the auto-mode permission check before it started, because another call in the same batch had been rewritten by a hook. Nothing ran and no file was written.
- **Fix:** The run command was reissued on its own. It completed with exit 0 in 10172 s wall time (10155.6 s summed per-target runtime; the 5 smoke targets were reused from cache).
- **Impact:** None on saved results.


### Confirmatory statistical implementation leaves outcome-relevant choices open
- **Date:** 2026-11-04
- **Stage:** Phase 1b documentation / confirmatory pre-registration.
- **What is wrong:** The design specifies a percentile bootstrap with 10,000 resamples and seed 0, and a two-sided Wilcoxon test with zero differences dropped, but does not fix the bootstrap generator and stream/reseeding order or the Wilcoxon p-value method and continuity correction. The confirmatory decision directly uses the resulting CI and corrected p-value. Legitimate implementations can therefore differ while satisfying the text. Reproduced on the pilot primary AUC contrast: exact Wilcoxon p = 0.0011148863314869735; asymptotic without continuity correction = 0.001273462545278989; asymptotic with correction = 0.0012817358997447585. Seed-0 Generator and RandomState bootstraps also give different endpoints on pilot confidence-matched subsets. These demonstrations do not change any pilot conclusion.
- **File and line:** docs/CONFIRMATORY_DESIGN.md:69-72 (decision uses these quantities at lines 101-105).
- **Found by:** found by Codex.
- **Impact:** Only future confirmatory runs; saved pilot CSVs and their reported rule outcomes are unaffected.
- **Minimal proposed fix (not applied):** Freeze the statistical implementation/version, bootstrap generator, reset policy and comparison order, and Wilcoxon method, zero handling and continuity correction before confirmatory results exist.
- **Fix (applied 2026-11-04, committed with this text):** docs/CONFIRMATORY_DESIGN.md now has "Statistics frozen: identical behaviour to the pilot code". It describes the pilot code as it is: `default_rng(0)` (PCG64) created once per metric and shared across the three comparisons in a fixed order, one `rng.integers(0, n, size=(10000, n))` call per comparison over targets in ascending image-id order, `numpy.percentile` with the default linear method, and `scipy.stats.wilcoxon(d)` with all defaults (`zero_method="wilcox"`, `correction=False`, `alternative="two-sided"`, `method="auto"`), including how `auto` resolves in scipy 1.18.1. numpy 2.5.4 and scipy 1.18.1 are pinned. It also fixes the Holm procedure and the regression CI method. The pilot code is unchanged.
- **Impact:** None on saved results.

### Confirmatory GPU execution protocol is not fixed before results
- **Date:** 2026-11-04
- **Stage:** Phase 1b documentation / confirmatory pre-registration.
- **What is wrong:** The design requires one GPU type within a run and records device and torch/CUDA versions in caches, but does not fix the GPU type or require an execution manifest to be committed before any confirmatory inference. Precision/math settings, TF32 and deterministic execution settings are also unspecified. Recording these after execution does not close the opportunity to choose a GPU/runtime configuration after inspecting outcomes; the text does not bind which configuration defines the confirmatory run.
- **File and line:** docs/CONFIRMATORY_DESIGN.md:114-119.
- **Found by:** found by Codex.
- **Impact:** Only future confirmatory runs; no confirmatory code or results currently exist, and saved pilot results are unaffected.
- **Minimal proposed fix (not applied):** Commit the GPU/software/precision/seed/determinism configuration, or require a binding manifest before the first confirmatory inference, and specify which completed run determines the decision.
- **Fix (applied 2026-11-04, committed with this text):** docs/CONFIRMATORY_DESIGN.md now fixes the protocol: a Kaggle GPU T4 with one GPU only, float32, TF32 off, cuDNN benchmark off and deterministic on, `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=":4096:8"`, and a stop rather than a fallback when an operation fails under determinism. A probe run writes an execution manifest, which is committed as docs/EXECUTION_MANIFEST.json before any confirmatory inference; the confirmatory run aborts if the manifest is missing or the live environment differs from it.
- **Impact:** None on saved results.

### Pilot limitation overstates bootstrap coverage as conservative
- **Date:** 2026-11-04
- **Stage:** Phase 1b documentation / pilot summary.
- **What is wrong:** The summary says discreteness and ties make the median bootstrap CIs conservative. The CSVs establish many ties and degenerate [0,0] intervals, but do not establish conservative confidence coverage. Ties alone do not guarantee that property. This is an unsupported statistical assurance, not a numerical mismatch or a failure of the registered interpretation rule.
- **File and line:** docs/PILOT_SUMMARY.md:164-166.
- **Found by:** found by Codex.
- **Impact:** Affects the current saved pilot documentation only; saved numerical results are unchanged. It is not a future-run-only issue.
- **Minimal proposed fix (not applied):** Describe the observed ties and limited resolution/degenerate intervals without asserting conservative coverage.
- **Fix (applied 2026-11-04, committed with this text):** The limitation in docs/PILOT_SUMMARY.md now says that tied pairs and [0, 0] intervals reflect the limited resolution of discrete metrics, not conservative coverage. It also states that the three pilot metrics were equally weighted and that AUC was designated primary only for the confirmatory study, after the pilot results were known.
- **Impact:** None on saved results (documentation only).

## Confirmatory

### --device detection crashed when --device was absent
- **Date:** 2026-11-04
- **What went wrong:** The first version of the pre-import check in run.py that sets `CUDA_VISIBLE_DEVICES` and `CUBLAS_WORKSPACE_CONFIG` used `sys.argv.index("--device")`, which raises `ValueError` when `--device` is not given, i.e. on every default CPU run. Found by reading the code before running it, not committed.
- **Fix:** The check now scans the arguments for `--device cuda` or `--device=cuda` without assuming the flag exists, and takes the cuBLAS value from config. Committed in 38baa1d.
- **Impact:** None on saved results.

### Confirmatory runner had a syntax error
- **Date:** 2026-11-04
- **What went wrong:** Two `print(f"\n...")` lines in `run_confirmatory` were written with a literal line break inside the f-string (an escaping slip when the code was inserted by script), so `python run.py` failed with `SyntaxError: unterminated f-string literal (detected at line 611)`. A regression test run at that moment reported "identical" only because it compared stale outputs from an earlier run; nothing had executed.
- **Fix:** Replaced the line breaks with `\n` escapes; deleted the stale test outputs and reran the regression test, which regenerated the Phase 1b CSVs from the caches identically. Not committed in the broken state.
- **Impact:** None on saved results.

### Inline edit command for the target-range change was rejected by bash
- **Date:** 2026-11-04
- **What went wrong:** The first attempt to apply the target-range code changes (config, confirmatory, execution, run) as an inline Python heredoc failed before running: `/usr/bin/bash: -c: line 153: unexpected EOF while looking for matching `''`. No file was modified (`git status` was clean afterwards).
- **Fix:** The same edits were written to a one-off script in the scratchpad and run from there; it applied all edits, which were tested on CPU and committed in d64e045. The script was then deleted.
- **Impact:** None on saved results.


### Production manifest validation accepts a self-consistent but non-protocol environment
- **Date:** 2026-11-04
- **Stage:** Confirmatory setup.
- **What is wrong:** check_manifest checks equality between the saved and live environment, but never checks either against the frozen production protocol. A complete environment record from this actual CPU environment was accepted for the default docs/EXECUTION_MANIFEST.json path, including device=cpu, gpu_name=None, cudnn_deterministic=False and deterministic_algorithms=False. Matching records with wrong numpy/scipy versions or dtype are likewise accepted. The normal probe permits CPU execution and can create such a record. Thus a committed, matching manifest does not enforce the required T4/cuda:0, float32, deterministic flags and pinned statistical versions for a production run.
- **File and line:** execution.py:106-112; run.py:590-592.
- **Found by:** found by Codex.
- **Impact:** Only future confirmatory runs; no saved confirmatory results exist and earlier results are unaffected.
- **Minimal proposed fix (not applied):** Validate the default production manifest and live environment against the normative device/precision/flag/version requirements as well as against each other. Keep any deliberately relaxed CPU test profile limited to the explicitly marked test-manifest path.
- **Fix (applied 2026-11-04, committed with this text):** `execution.validate_production()` runs at confirmatory startup on the default manifest. It aborts with a list of every violation unless: the device is CUDA; there is exactly one T4; the dtype is float32; all determinism and TF32 flags and the CUDA environment variables have the values in docs/CONFIRMATORY_DESIGN.md; numpy and scipy equal the versions pinned in requirements_gpu.txt. Tested: a manifest recorded on this CPU, used on the default production path, is refused with 6 violations. Only the `--manifest` test path (which requires `--limit`) skips the check.
- **Impact:** None on saved results.

### Alternate test manifest is not recorded in outputs or distinguished in shared caches
- **Date:** 2026-11-04
- **Stage:** Confirmatory setup.
- **What is wrong:** The CLI correctly rejects --manifest without --limit, but the override path or a test-only flag is never saved in the five CSV outputs or the per-target cache metadata. Reproduced run_confirmatory(limit=1, manifest='alternate.json') with all output writes captured in memory: no output records the override. Only the environment block is passed to run_target/cache_meta, so an override test and a production run with the same environment share indistinguishable cache keys. The *_smoke CSV suffix records a limit, not the use or identity of an alternate manifest, and a later full run can silently reuse those test caches.
- **File and line:** run.py:590-608 and run.py:615-621; confirmatory.py:124-125.
- **Found by:** found by Codex.
- **Impact:** Only future test/confirmatory outputs and cache provenance; saved pilot results are unaffected.
- **Minimal proposed fix (not applied):** Persist the manifest path/hash, full manifest record and explicit test-only status in run outputs and cache metadata, and prevent production cache reuse from an alternate-manifest test unless its production provenance is explicitly validated.
- **Fix (applied 2026-11-04, committed with this text):** Every confirmatory output records the run label ("TEST ONLY" with `--manifest`, otherwise "PRODUCTION") with the manifest path and SHA-256: as provenance rows in the long CSVs, as columns in the wide CSVs, and as PNG text fields in the figures. Every cache's metadata includes it too. Test runs use their own cache and image folders (`cache_test_only/`, `adv_test_only/`) and `_test_only` file names, so a production run cannot reuse them. Tested with a CPU test run (--limit 2): the label is in all 5 CSVs, both PNGs and both caches, and no production cache folder was created.
- **Impact:** None on saved results.

### Probe accepts fewer than the required two eligible pilot targets
- **Date:** 2026-11-04
- **Stage:** Confirmatory setup / runtime fallback.
- **What is wrong:** probe_timing stops after two eligible targets when available but does not require that two were found before returning timing estimates. Reproduced with the full 100-target pilot list and mocked clean detections in which only target 785 is eligible: it returns a finite estimate from one target, and select_range accepts the resulting manifest and binds targets 101-952. The pre-registered fallback requires timing two eligible pilot targets. With zero eligible targets the same path divides by zero.
- **File and line:** confirmatory.py:104-112; confirmatory.py:75-80.
- **Found by:** found by Codex.
- **Impact:** Only future probes/runs where fewer than two pilot targets pass eligibility; no current saved results are affected.
- **Minimal proposed fix (not applied):** Abort with a clear insufficient-eligibility message unless exactly two targets were timed, and validate this required probe sample in the manifest before accepting its binding range.
- **Fix (applied 2026-11-04, committed with this text):** `probe_timing` tries the pilot list in order until exactly 2 targets are eligible; with fewer it aborts before any division and no manifest is written. `select_range` also rejects a manifest whose probe record does not hold exactly 2 timed targets. Tested with mocked detections: 0 and 1 eligible targets abort with a message, and a manifest with 1 probe target is rejected.
- **Impact:** None on saved results.

### Zero eligible confirmatory targets crash analysis before reporting exclusions
- **Date:** 2026-11-04
- **Stage:** Confirmatory setup / analysis.
- **What is wrong:** analyze filters out ineligible records, then passes phase1b.table([]), a DataFrame with no columns, into phase1b.summarize. An otherwise valid selected record with eligible=False reproduces KeyError: 'image_id'. run_confirmatory calls analyze before writing selected_smoke.csv/selected.csv, so a limited run whose selected targets are all ineligible produces no exclusion summary or undefined-statistics decision despite having finished its eligibility checks.
- **File and line:** confirmatory.py:249-253; phase1b.py:153 and phase1b.py:158; run.py:614-617.
- **Found by:** found by Codex.
- **Impact:** Only future zero-eligible runs, including limited smoke runs; saved pilot results are unaffected.
- **Minimal proposed fix (not applied):** Preserve the expected empty-table schema and handle zero eligible targets explicitly, reporting selection/exclusion counts and undefined statistics with a non-support conclusion.
- **Fix (applied 2026-11-04, committed with this text):** `phase1b.table` now always returns the 14 named columns (pilot CSVs regenerated from the caches are identical). `confirmatory.analyze` handles zero eligible targets: the summary holds the selected and eligible counts, the censoring and eps counts (all 0) and the message "no eligible targets: all statistics are undefined, so no decision criterion can be met". The control file holds the message, and the decision is "does not support either direction" with the message. The exclusion report (selected CSV) is written before the analysis. Tested with 2 ineligible records.
- **Impact:** None on saved results.

### Inspection command for the test labels failed to parse
- **Date:** 2026-11-04
- **What went wrong:** While checking the test-run outputs, the one-line Python command that reads the PNG text fields failed with `SyntaxError: unterminated string literal` (a backslash in a path split was mangled by the shell quoting). The test run itself had finished with exit 0.
- **Fix:** Reran the check with `os.path.basename` and the folder passed through an environment variable; it showed "TEST ONLY" in both PNGs and both caches.
- **Impact:** None on saved results.


### Production startup accepts a cold or pre-warm-up probe manifest
- **Date:** 2026-11-04
- **Stage:** Confirmatory setup / probe protocol validation.
- **What is wrong:** The new probe records warm_up=True after the documented untimed pass, but production startup never requires that field. select_range validates the number of timed targets, full target count and range versus the estimate only. Reproduced production check_manifest followed by select_range with a matching valid T4/float32/deterministic/pinned-version environment and otherwise consistent timing records: both warm_up=False and a missing warm_up field were accepted and selected all 852 targets. A manifest from the pre-warm-up probe implementation can therefore remain accepted after the timing protocol changed, allowing a cold-start estimate to determine the binding sample-size fallback.
- **File and line:** confirmatory.py:75-85; execution.py:145-147.
- **Found by:** found by Codex.
- **Impact:** Only future confirmatory runs that reuse a cold or pre-warm-up probe manifest; no saved confirmatory results exist and earlier results are unchanged.
- **Minimal proposed fix (not applied):** Require probe.warm_up to be True before accepting the manifest's timing estimate and target range; reject older/cold records with a request to rerun the probe.
- **Fix (applied 2026-11-04, committed with this text):** `confirmatory.select_range`, which every confirmatory run (production and test) passes before any target is touched, now requires `probe.warm_up` to be exactly `True`; a missing field, `False` or any other value aborts with a message naming the field. docs/CONFIRMATORY_DESIGN.md states the requirement. Unit tests: missing, `False` and `1` are rejected; `True` is accepted.
- **Impact:** None on saved results.

### Empty targets CSV still drops test-manifest provenance values
- **Date:** 2026-11-04
- **Stage:** Confirmatory setup / alternate-manifest output provenance.
- **What is wrong:** with_provenance assigns provenance columns to a wide DataFrame, but when the zero-eligible targets table has no rows, CSV serialization contains only column names. The TEST ONLY label, manifest path and SHA-256 values are all absent. Reproduced an end-to-end limited alternate-manifest run with one ineligible selected target and all file writes captured in memory: selected, summary, confidence-control and decision CSVs contain provenance, while targets_test_only.csv has none of the three values. This leaves the requirement that every test output record its manifest identity incomplete even though the zero-eligible analysis and cache isolation now work.
- **File and line:** confirmatory.py:317-323; run.py:620-621.
- **Found by:** found by Codex.
- **Impact:** Only future zero-eligible test outputs (and empty production target tables lack their production provenance values too); saved pilot results are unaffected.
- **Minimal proposed fix (not applied):** Handle the empty-table output explicitly so the file retains the provenance values while still representing zero target observations.
- **Fix (applied 2026-11-04, committed with this text):** `confirmatory.with_provenance` turns an empty wide table into one row that holds only the provenance (run label, manifest path, manifest SHA-256), marked `row_type = "metadata: provenance only, no data rows"`, with the table's columns kept empty. Long tables already append provenance rows, so they carry it even with no data rows. Unit tests write each output to CSV and read it back: the empty targets table, an empty selection report, and all five outputs of a zero-eligible analysis contain "TEST ONLY", the path and the SHA-256. Non-empty tables are unchanged apart from the provenance columns.
- **Impact:** None on saved results.


### Production GPU-name validation accepts T400 as T4
- **Date:** 2026-11-04
- **Stage:** Confirmatory setup / production protocol validation.
- **What is wrong:** validate_production checks whether the substring "T4" occurs anywhere in gpu_name. This accepts "NVIDIA T400" (and "T400") as the required T4 model. Reproduced with one CUDA GPU, float32, all expected flags/environment settings and the pinned numpy/scipy versions: validate_production accepted the record, and production check_manifest accepted the same saved and mocked live environment naming NVIDIA T400. Thus matching manifests on a different GPU model can still pass the fixed T4 protocol. This was a memory-only validation test, not an actual GPU run.
- **File and line:** execution.py:113.
- **Found by:** found by Codex.
- **Impact:** Only future production runs on an incorrectly accepted GPU model; no saved confirmatory results exist and all earlier results are unchanged.
- **Minimal proposed fix (not applied):** Match the complete T4 model name/token against the accepted T4 identifiers instead of an arbitrary substring, preserving legitimate naming aliases while rejecting T400 and other different models.
- **Fix (applied 2026-11-04, committed with this text):** `execution.validate_production` now requires the recorded GPU name, after stripping surrounding whitespace, to equal one entry of the explicit allow-list `T4_NAMES = {"Tesla T4", "NVIDIA T4"}` (case-sensitive, whole string); anything else, including a missing name, is rejected. docs/CONFIRMATORY_DESIGN.md states the allow-list (committed first, ccff3c4). Unit tests: "Tesla T4", "NVIDIA T4" and "  Tesla T4
" are accepted; "NVIDIA T400", "T4000", "Tesla T40", "T4", "tesla t4", "NVIDIA Tesla T4", "Tesla T4 (2)", "" and None are rejected; all 20 tests pass.
- **Impact:** None on saved results.

### kaggle_code.zip held CRLF copies instead of the committed files
- **Date:** 2026-11-04
- **What went wrong:** The zip was built with `git archive`, which applies `core.autocrlf=true` on this machine and wrote every text file with CRLF line endings. Found while checking the rebuilt zip that now includes docs/EXECUTION_MANIFEST.json: the manifest in the zip (1200 bytes, 51 CR) differed from the committed blob (1149 bytes, LF); all 22 files in the zip had CRLF. Every earlier zip, including the one used for the Kaggle probe, was built the same way, so the earlier report that the zip carries the committed files was wrong about line endings. Python, JSON and the requirements parser read CRLF and LF identically, so the code that ran and the probe's recorded values are unaffected. The confirmatory run's provenance hash of the manifest would have been taken over the CRLF copy instead of the committed, downloaded file.
- **Fix:** The zip is built with `git -c core.autocrlf=false archive --format=zip -o kaggle_code.zip HEAD -- <files>`. Checked: all 22 files are byte-identical to the committed blobs, none has CRLF, and the manifest's SHA-256 in the zip equals that of the downloaded file (2b8c7756e717db9e9db5b1354c592488da5f76164845820026919c3554c498c9).
- **Impact:** None on saved results; no confirmatory data exists yet.
