"""Build report/report.md, report.html and report.pdf from the result CSVs.

Every number in the report is read from results/*.csv, results/bins, results/fisheye,
config.py or the README runtime table; none is typed. Claims made in the text
(monotonic decrease, no overconfident bin, ...) are asserted against the data.
PDF: headless Microsoft Edge (or Chrome) prints the HTML rendering.
"""
import datetime
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import markdown
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

REPORT = ROOT / "report"
RES = ROOT / "results"
BROWSERS = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")]

TITLE = ("Does a Person Detector Know What It Misses? A Preliminary Study of Confidence "
         "Reliability under Corruptions, Fisheye Distortion and Adversarial Attacks")
SHIFT = list(config.SHIFT_CONDITIONS)
NATURAL = ["clean"] + SHIFT  # 15 conditions on 500 images
TABLE1 = ["clean", "gaussian_blur_s3", "gaussian_noise_s3", "low_light_s3", "jpeg_s3",
          "fisheye_s1", "fisheye_s2", "clean_100", "rand_eps8", "pgd_eps4", "pgd_eps8"]


def f3(x):
    return "–" if isinstance(x, float) and math.isnan(x) else f"{x:.3f}"


def params(key, family):
    return ", ".join(f"{c[key]:g}" for c in config.SHIFT_CONDITIONS.values() if c["family"] == family)


def numbers():
    t = pd.read_csv(RES / "table.csv").set_index("condition")
    box = pd.read_csv(RES / "box_level.csv").set_index("condition")
    img = pd.read_csv(RES / "image_level.csv")
    summ = pd.read_csv(RES / "attack_summary.csv").set_index("condition")
    views = pd.read_csv(RES / "attack_views.csv").set_index("condition")
    det = pd.read_csv(RES / "attack_detection.csv").set_index(["negative", "signal"])
    v = {}

    # setup
    v["n"] = int(t.at["clean", "n_images"])
    v["na"] = int(t.at["clean_100", "n_images"])
    v["n_gt"] = int(t.at["clean", "n_gt"])
    v["n_gt100"] = int(t.at["clean_100", "n_gt"])
    v["n_shift"] = len(SHIFT)
    v["blur"], v["noise"] = params("sigma", "gaussian_blur"), params("std", "gaussian_noise")
    v["light"], v["jpeg"] = params("factor", "low_light"), params("quality", "jpeg")
    v["fe"] = params("theta_max_deg", "fisheye")
    pgd = config.ATTACK_CONDITIONS["pgd_eps4"]
    v["steps"], v["topk"], v["alpha"] = pgd["steps"], pgd["top_k"], pgd["alpha_255"]
    v["rand_eps"] = config.ATTACK_CONDITIONS["rand_eps8"]["eps_255"]
    for s in ("s1", "s2"):
        fe = pd.read_csv(RES / "fisheye" / f"fisheye_{s}.csv")
        v[f"fe_drop_{s}"], v["fe_boxes"] = int(fe["n_dropped"].sum()), int(fe["n_boxes"].sum())

    # F1: recall vs precision / TP confidence, calibration sign
    sh = t.loc[SHIFT]
    v["clean_rec"], v["clean_prec"] = f3(t.at["clean", "recall_025"]), f3(t.at["clean", "precision_025"])
    v["clean_mtp"] = f3(t.at["clean", "mean_conf_tp"])
    v["r_min"], v["r_min_c"] = f3(sh["recall_025"].min()), sh["recall_025"].idxmin()
    v["r_max"], v["r_max_c"] = f3(sh["recall_025"].max()), sh["recall_025"].idxmax()
    v["p_min"], v["p_max"] = f3(sh["precision_025"].min()), f3(sh["precision_025"].max())
    assert sh["mean_conf_tp"].idxmin() == "jpeg_s3"
    rest = sh.drop(index="jpeg_s3")
    v["m_min"], v["m_max"] = f3(rest["mean_conf_tp"].min()), f3(rest["mean_conf_tp"].max())
    v["jpeg_mtp"] = f3(t.at["jpeg_s3", "mean_conf_tp"])
    gaps = t.loc[NATURAL, "signed_cal_gap"]
    assert (gaps > 0).all()
    v["gap_min"], v["gap_max"], v["n_nat"] = f3(gaps.min()), f3(gaps.max()), len(NATURAL)
    bins = pd.concat([pd.read_csv(RES / "bins" / f"{c}.csv") for c in NATURAL])
    v["n_bins"], v["n_over"] = len(bins), int((bins["precision"] < bins["mean_conf"]).sum())
    assert v["n_over"] == 0

    # F2: recoverable share
    v["rs_clean"] = f3(t.at["clean", "recoverable_share"])
    v["p005_clean"] = f3(t.at["clean", "precision_005"])
    fam = {}
    for name, c in config.SHIFT_CONDITIONS.items():
        fam.setdefault(c["family"], []).append(name)
    for names in fam.values():
        seq = t.loc[["clean"] + names, "recoverable_share"].to_numpy()
        assert all(b < a for a, b in zip(seq, seq[1:])), names
    for c in ("gaussian_blur_s3", "gaussian_noise_s3", "low_light_s3", "jpeg_s3", "fisheye_s2"):
        v[f"rs_{c}"] = f3(t.at[c, "recoverable_share"])
    v["n_fam"] = len(fam)

    # F3: confidence vs view consistency
    nb = box.loc[NATURAL]
    better = nb["auroc_conf"] > nb["auroc_cons"]
    v["n_conf_better"] = int(better.sum())
    assert list(nb.index[~better]) == ["jpeg_s3"]
    nbr = nb.drop(index="jpeg_s3")
    v["ac_min"], v["ac_max"] = f3(nbr["auroc_conf"].min()), f3(nbr["auroc_conf"].max())
    v["cc_min"], v["cc_max"] = f3(nbr["auroc_cons"].min()), f3(nbr["auroc_cons"].max())
    v["j_conf"], v["j_cons"] = f3(box.at["jpeg_s3", "auroc_conf"]), f3(box.at["jpeg_s3", "auroc_cons"])
    pooled = img[img["condition"] == "pooled"].set_index("signal")
    v["n_pooled"] = f"{int(pooled.at['mean_conf', 'n_images']):,}"
    v["rho_mc"] = f3(pooled.at["mean_conf", "spearman_rho"])
    v["rho_vd"] = f3(pooled.at["view_disagreement", "spearman_rho"])

    # F4: attack success vs random noise
    had = int(summ.at["pgd_eps4", "n_images_with_persons_before"])
    v["had"] = had
    for c in ("pgd_eps4", "pgd_eps8", "rand_eps8"):
        v[f"succ_{c}"] = round(summ.at[c, "attack_success_rate"] * had)
        v[f"after_{c}"] = f"{summ.at[c, 'mean_persons_after']:.2f}"
        v[f"map_{c}"] = f3(t.at[c, "mAP50"])
    v["before"] = f"{summ.at['pgd_eps4', 'mean_persons_before']:.2f}"
    v["map_clean_100"] = f3(t.at["clean_100", "mAP50"])

    # F5: views and attack detection
    v["c100_rec"] = f3(views.at["clean_100", "recall_025"])
    for c in ("pgd_eps4", "pgd_eps8"):
        v[f"{c}_hf"] = f3(views.at[c, "recall_025_hflip"])
        v[f"{c}_576"] = f3(views.at[c, "recall_025_imgsz576"])
    v["vd_clean"] = f3(det.at[("clean_100", "view_disagreement"), "auroc"])
    v["vd_rand"] = f3(det.at[("rand_eps8", "view_disagreement"), "auroc"])
    mc = det.at[("clean_100", "mean_conf"), "auroc"]
    v["mc_clean"], v["mc_flip"] = f3(mc), f3(1 - mc)

    # limitations
    v["ratio_clean"] = f3(t.at["clean", "mAP50_95"] / t.at["clean", "mAP50"])
    v["ratio_fe2"] = f3(t.at["fisheye_s2", "mAP50_95"] / t.at["fisheye_s2", "mAP50"])

    v["table1"] = table1(t)
    v["runtimes"] = readme_runtime_table()
    v["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
    v["date"] = datetime.date.today().isoformat()
    return v


def table1(t):
    cols = [("mAP50", "mAP50"), ("recall_025", "Recall"), ("precision_025", "Precision"),
            ("mean_conf_tp", "Mean TP conf."), ("signed_cal_gap", "Signed gap"),
            ("recoverable_share", "Recov. share")]
    lines = ["| Condition | Images | " + " | ".join(h for _, h in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for c in TABLE1:
        lines.append(f"| {c} | {int(t.at[c, 'n_images'])} | "
                     + " | ".join(f3(t.at[c, k]) for k, _ in cols) + " |")
    return "\n".join(lines)


def readme_runtime_table():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = text.split("### Runtime on CPU", 1)[1].splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("|"))
    end = next((i for i in range(start, len(lines)) if not lines[i].startswith("|")), len(lines))
    return "\n".join(lines[start:end])  # only the first table after the heading


TEMPLATE = """# {title}

**Ashwin Lakshminarasimhan**, VIT Vellore

**Preliminary results** · {date} · all numbers produced by the code and data of commit `{commit}`

## Abstract

In headcounts and surveillance, a person the detector misses is often the costly error. We ask whether a detector's confidence reveals such misses. We evaluate YOLO11n (CPU, person class only) on {n} COCO val2017 images under {n_shift} synthetic shifts, including a synthetic equidistant fisheye, and on {na} images under a white-box PGD attack with a random-noise control. Under shift, recall at confidence ≥ 0.25 falls from {clean_rec} to as low as {r_min}, while precision stays between {p_min} and {p_max}. The detector is underconfident in all {n_nat} clean and shifted conditions; per-detection calibration metrics cannot account for the missed people, because a missed person produces no detection to evaluate. The share of misses recoverable at confidence ≥ 0.05 falls from {rs_clean} to {rs_jpeg_s3}. PGD with ε = 4/255 removes all person detections in {succ_pgd_eps4}/{had} images, random noise of size 8/255 in {succ_rand_eps8}/{had}. Main limitations: one small model, few images, synthetic fisheye and a non-adaptive attacker.

## 1 Introduction

People counting, occupancy monitoring and surveillance rely on person detectors. In these uses a missed person (a false negative) is often the costly error: a headcount comes out too low, or a person in a restricted area goes unreported. A detector's confidence score is the usual signal of how far to trust its output, and calibration studies ask whether the confidence of an emitted box matches its probability of being correct. A missed person, however, produces no box and therefore no confidence.

This report asks: when the input to a person detector degrades, through common corruptions, fisheye-like lens distortion or adversarial perturbation, do its confidence scores, or simple test-time signals derived from them, indicate that people are being missed? We study this for one small detector (YOLO11n) on COCO person images. The results are preliminary and are meant to decide where a fuller study should focus.

## 2 Related work

Michaelis et al. [1] showed that object detectors lose substantial accuracy under common image corruptions; our shift conditions follow the same idea with a small set of corruptions. Calibration of detectors is commonly measured with the detection expected calibration error (D-ECE) of Küppers et al. [2], which we use, and Huseljic et al. [3] systematically evaluated uncertainty calibration in pretrained detectors. Park et al. [4] point out that per-prediction calibration does not account for missed objects and study image-level uncertainty for detection transformers; our first finding is an empirical demonstration of this point for a YOLO detector under distribution shift, not a new observation.

Adversarial attacks on person detection are well established: Thys et al. [5] hid persons from a detector with printed adversarial patches. We use projected gradient descent (PGD) [6] in the digital domain. Nesti et al. [7] detect adversarial examples through input transformations, defense perturbations and voting; our fifth finding is consistent with this known effect. Athalye et al. [8] showed that attacks optimised over a distribution of transformations (EOT) stay effective under such transformations, so an adaptive attacker would likely defeat the signals tested here. For wide-angle imagery, Zhang et al. [9] craft distortion-aware adversarial perturbations for 360° images from perspective views, and Duan et al. [10] address rotation-aware people detection in real overhead fisheye images.

## 3 Method

**Model and data.** YOLO11n, input size 640, CPU only, seed 0. The data are the first {n} COCO val2017 images (by image id) that contain at least one non-crowd person ({n_gt} non-crowd person boxes); only the person class is evaluated. Predictions are kept down to confidence 0.001 for mAP; all other metrics use detections with confidence ≥ 0.25 unless stated.

**Shift conditions ({n_shift}).** Gaussian blur (σ = {blur} px), additive Gaussian noise (std {noise} on [0, 1], seeded per image), low light (intensity × {light}), JPEG re-encoding (quality {jpeg}) and a synthetic equidistant fisheye whose image half-diagonal maps to {fe}° (s1, s2). For fisheye, ground-truth boxes are warped with the forward mapping (20 points per edge, then the bounding box, clipped); {fe_drop_s1} (s1) and {fe_drop_s2} (s2) of {fe_boxes} person boxes become smaller than 4 px and are dropped.

**Attack.** On the first {na} images ({n_gt100} non-crowd persons), a white-box PGD "vanishing" attack [6] perturbs the image at its original resolution and minimises the sum of the top-{topk} person scores over all predictions: {steps} sign-gradient steps of {alpha}/255, ε = 4/255 or 8/255, no random start, keeping the iterate with the lowest loss. The letterbox resize and padding are differentiable and reproduce ultralytics' rectangular letterbox. Attacked images are rounded to 8 bits, saved as PNG and evaluated through the normal pipeline. Control: uniform noise in ±{rand_eps}/255 (rand_eps8). Reference: the same {na} clean images (clean_100).

**Matching.** Detections are matched greedily in descending confidence to non-crowd person boxes at IoU ≥ 0.5. An unmatched detection that overlaps a crowd box by at least half of its own area is ignored (COCO crowd rule).

**Metrics.**

- mAP50, mAP50–95: pycocotools COCOeval, person class only.
- Recall at threshold t: matched non-crowd persons / all non-crowd persons, with detections of confidence ≥ t (t = 0.05, 0.10, 0.25).
- Precision at t: TP / (TP + FP); ignored detections are excluded.
- D-ECE [2]: Σ (n_bin / N) · |mean confidence − precision| over 10 equal-width confidence bins.
- Signed calibration gap: Σ (n_bin / N) · (precision − mean confidence) over the same bins; positive means underconfident.
- Recoverable share: (recall at 0.05 − recall at 0.25) / (1 − recall at 0.25), the share of persons missed at 0.25 that are found at 0.05.
- AUROC: separability of true from false positives (box level), of images that miss at least half of their persons (image level), or of attacked from non-attacked images.
- Risk-coverage: false-positive fraction among the detections kept when ranking by a score.
- View signals: consistency = mean over two views (horizontal flip, input size 576) of a detection's best IoU with a detection in that view; view disagreement = mean relative change in the number of detections across views.

**Audit.** The code was checked by an independent second agent; every issue found and its fix are recorded in RESEARCH_LOG.md.

## 4 Results

**Table 1.** Key conditions. Recall, precision and mean true-positive confidence at confidence ≥ 0.25. "Signed gap" equals D-ECE wherever defined, because no bin is overconfident. "–" = undefined (no detections at confidence ≥ 0.25). The first seven rows use {n} images, the last four the first {na}.

{table1}

**F1 Silent omission under natural shift.** Across the {n_shift} shift conditions, recall at 0.25 ranges from {r_min} ({r_min_c}) to {r_max} ({r_max_c}), compared with {clean_rec} on clean images, while precision stays between {p_min} and {p_max} (clean {clean_prec}). The mean confidence of true positives stays between {m_min} and {m_max} (clean {clean_mtp}) in all conditions except jpeg_s3 ({jpeg_mtp}) (Figure 1). The signed calibration gap is positive in all {n_nat} clean and shifted conditions ({gap_min} to {gap_max}), and none of their {n_bins} populated confidence bins is overconfident. The boxes that are emitted are at least as reliable as their confidence suggests, while a growing share of persons is not reported at all. Per-detection calibration cannot reveal this, as Park et al. [4] noted.

![Figure 1](figures/fig_silent_failure.png)

*Figure 1. Recall at confidence ≥ 0.25 against the mean confidence of true positives, one panel per corruption family (clean → s1 → s2 → s3; grey: all other conditions).*

**F2 Fewer misses are recoverable as severity grows.** On clean images, {rs_clean} of the persons missed at 0.25 are found at 0.05, at a precision of {p005_clean} (vs {clean_prec} at 0.25). In all {n_fam} families this share decreases at every severity step (Figure 2), down to {rs_gaussian_blur_s3} (blur s3), {rs_gaussian_noise_s3} (noise s3), {rs_low_light_s3} (low light s3), {rs_jpeg_s3} (JPEG s3) and {rs_fisheye_s2} (fisheye s2). Under strong shift, missed persons are increasingly absent from the low-confidence detections as well, so lowering the threshold recovers fewer of them.

![Figure 2](figures/fig_recoverable.png)

*Figure 2. Recoverable share against severity (clean = 0), one line per corruption family.*

**F3 View consistency does not beat confidence under natural shift.** For separating true from false positives, confidence reaches AUROC {ac_min} to {ac_max} and view consistency {cc_min} to {cc_max} in the {n_conf_better} of {n_nat} clean and shifted conditions where confidence is better. Consistency is higher only for jpeg_s3 ({j_cons} vs {j_conf}). At image level (pooled over {n_pooled} condition-image observations from {n} source images; descriptive only), mean confidence is more strongly associated with the miss rate (Spearman {rho_mc}) than view disagreement is ({rho_vd}).

**F4 PGD suppresses all person detections at confidence ≥ 0.25 on the {na}-image subset; random noise of equal size does not.** With ε = 4/255, all {succ_pgd_eps4} of the {had} images that had at least one person detection at 0.25 lose all of them (mean persons per image {before} → {after_pgd_eps4}); ε = 8/255 gives {succ_pgd_eps8}/{had}. Uniform noise of size 8/255 removes all persons in {succ_rand_eps8}/{had} images ({before} → {after_rand_eps8} persons per image), and mAP50 moves from {map_clean_100} to {map_rand_eps8}, compared with {map_pgd_eps4} (ε = 4/255) and {map_pgd_eps8} (ε = 8/255) under PGD. The effect comes from the optimised direction of the perturbation, not from its size.

**F5 Against this non-adaptive attacker, simple views partly undo the attack.** On PGD images, recall at 0.25 is 0 in the original view, but {pgd_eps4_hf} with a horizontal flip and {pgd_eps4_576} at input size 576 for ε = 4/255 ({pgd_eps8_hf} and {pgd_eps8_576} for ε = 8/255), against {c100_rec} on the clean images. View disagreement separates pgd_eps8 images from clean images with AUROC {vd_clean} and from random-noise images with {vd_rand}. This is consistent with the known effect used for detection by input transformations [7]. The attack was not optimised over these views, and an adaptive EOT attack [8] would likely remove the effect. Mean confidence separates attacked from clean images almost perfectly (raw AUROC {mc_clean}, {mc_flip} with the sign flipped) only because attacked images have no detections; it flags any image without confident detections in the same way.

## 5 Limitations

- One small model (YOLO11n) and only the person class; larger models or other detector families may behave differently.
- {n} images ({na} for the attack) and no confidence intervals, so small differences between conditions should not be over-interpreted.
- The fisheye is synthetic, and warped ground-truth boxes are bounding boxes of warped rectangles, looser than boxes drawn on real fisheye images. The drop in mAP50–95 (ratio mAP50–95 / mAP50: {ratio_clean} on clean images, {ratio_fe2} for fisheye_s2) may partly reflect these looser warped ground-truth boxes, but this study does not separate that effect from distorted appearance or detector localization.
- The attack is white-box, digital (not a physical patch) and not adaptive to the view transformations.
- At low thresholds many boxes are emitted (precision {p005_clean} at 0.05 on clean images), so some matches at IoU ≥ 0.5 may be chance overlaps; recall at 0.05 and the recoverable share may be overestimated for this reason.
- Pooled image-level statistics combine {n_nat} conditions of the same {n} images and are descriptive only; no p-values are reported for them.

## 6 Next steps

- Position-dependent adversarial vulnerability under fisheye geometry: place the same person at the image centre and at the edge, and measure the minimal attack strength that removes the detection.
- An adaptive EOT attack [8] against the flip and rescale signals.
- Larger YOLO models and at least one non-YOLO detector.
- Real overhead fisheye data, for example the datasets used with RAPiD [10].

## 7 Reproducibility

Code, all result CSVs and the prediction caches are in a local git repository (results frozen at commit `{commit}`). `reproduce.bat` (Windows) and `reproduce.sh` run all stages and rebuild `results/summary.md`; rerunning them from the caches reproduced all 100 result CSVs byte-identically. `RESEARCH_LOG.md` records every problem, fix and design decision. CPU runtimes as measured on the machine that produced the results (from the README):

{runtimes}

## Acknowledgements

Code for this study was written with the assistance of AI coding tools (Claude Code and OpenAI Codex). All design decisions, analyses and conclusions are the author's; every result was independently re-audited, and all problems found are documented in RESEARCH_LOG.md.

## References

[1] Michaelis et al. Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming. arXiv:1907.07484, 2019.

[2] Küppers et al. Multivariate Confidence Calibration for Object Detection. CVPR Workshops, 2020.

[3] Huseljic et al. Systematic Evaluation of Uncertainty Calibration in Pretrained Object Detectors. International Journal of Computer Vision, vol. 133, pp. 1033–1047, 2025.

[4] Park, Sobolewski, Azizan. Quantifying the Reliability of Predictions in Detection Transformers: Object-Level Calibration and Image-Level Uncertainty. arXiv:2412.01782v3, 2024.

[5] Thys et al. Fooling Automated Surveillance Cameras: Adversarial Patches to Attack Person Detection. CVPR Workshops, 2019.

[6] Madry et al. Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR, 2018.

[7] Nesti et al. Detecting Adversarial Examples by Input Transformations, Defense Perturbations, and Voting. arXiv:2101.11466, 2021.

[8] Athalye et al. Synthesizing Robust Adversarial Examples. ICML, 2018.

[9] Zhang et al. 360-Attack: Distortion-Aware Perturbations from Perspective-Views. CVPR, 2022.

[10] Duan et al. RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images. CVPR Workshops, 2020.
"""

CSS = """
@page { size: A4; margin: 14mm 15mm 14mm 15mm; }
body { font-family: Cambria, Georgia, serif; font-size: 9.4pt; line-height: 1.3; color: #111; }
h1 { font-size: 14.5pt; line-height: 1.22; margin: 0 0 4px 0; }
h2 { font-size: 11pt; margin: 9px 0 3px 0; }
p { margin: 3px 0 5px 0; text-align: justify; }
ul { margin: 2px 0 5px 0; padding-left: 18px; }
li { margin: 1px 0; }
table { border-collapse: collapse; margin: 4px 0 6px 0; font-size: 8.4pt; width: 100%; }
th, td { border-bottom: 1px solid #bbb; padding: 1.5px 5px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { border-bottom: 1.5px solid #333; }
img { display: block; margin: 4px auto 0 auto; }
img[alt="Figure 1"] { width: 7.0in; }
img[alt="Figure 2"] { width: 4.2in; }
.keep { break-inside: avoid; page-break-inside: avoid; }
em { font-size: 8.6pt; }
code { font-size: 8.6pt; }
"""


FIGSIZE = {"fig_silent_failure": (7.0, 3.6), "fig_recoverable": (4.2, 2.2)}  # inches, = printed width


def report_figures():
    """Report copies of the result figures, drawn at their printed size (CSS width = figsize
    width) so matplotlib font sizes are the printed point sizes. results/ figures are untouched."""
    import reliability

    table = pd.read_csv(RES / "table.csv")
    (REPORT / "figures").mkdir(exist_ok=True)
    reliability.fig_silent_failure(table, REPORT / "figures" / "fig_silent_failure.png", ncols=3,
                                   figsize=FIGSIZE["fig_silent_failure"], title=False)
    reliability.fig_recoverable(table, REPORT / "figures" / "fig_recoverable.png",
                                figsize=FIGSIZE["fig_recoverable"], title=False)


def keep_together(html):
    """Wrap each h2 with the block after it, and each figure with its caption, so a page
    break cannot separate them."""
    block = r"(?:<p>.*?</p>|<ul>.*?</ul>|<table>.*?</table>)"
    html = re.sub(rf"(<h2>.*?</h2>\s*{block})", r'<div class="keep">\1</div>', html, flags=re.S)
    return re.sub(r"(<p><img [^>]*></p>\s*<p><em>.*?</em></p>)", r'<div class="keep">\1</div>',
                  html, flags=re.S)


def main():
    v = numbers()
    report_figures()
    md_text = TEMPLATE.format(title=TITLE, **v)
    (REPORT / "report.md").write_text(md_text, encoding="utf-8", newline="\n")
    body = keep_together(markdown.markdown(md_text, extensions=["tables"]))
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{TITLE}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    html_path = REPORT / "report.html"
    html_path.write_text(html, encoding="utf-8")
    browser = next(b for b in BROWSERS if b.exists())
    pdf_path = REPORT / "report.pdf"
    pdf_path.unlink(missing_ok=True)
    # a separate profile keeps headless Edge from handing off to a running browser instance
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", f"--print-to-pdf={pdf_path}",
                        html_path.as_uri()], check=True, capture_output=True, timeout=120)
        # msedge.exe returns at once and the PDF is written a few seconds later
        size, deadline = -1, time.time() + 120
        while time.time() < deadline:
            if pdf_path.exists() and pdf_path.stat().st_size == size > 0:
                break
            size = pdf_path.stat().st_size if pdf_path.exists() else -1
            time.sleep(1)
        else:
            raise RuntimeError(f"{browser.name} did not write {pdf_path} within 120 s")
    n_pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf_path.read_bytes()))
    print(f"Wrote {REPORT / 'report.md'}, {html_path}, {pdf_path} ({n_pages} pages)")


if __name__ == "__main__":
    main()
