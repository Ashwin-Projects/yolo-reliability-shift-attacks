"""Phase 1b: steps-to-suppress and localized-attack measures on the Phase 1 eligible targets.

Geometry, canvases, GT boxes, the targeted loss and the success criterion are reused from
Phase 1 (phase1.py and its per-target caches). See docs/PHASE1B_DESIGN.md.
"""
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch

import attack
import config
import phase1
import phase1_analysis
from metrics import iou_xywh

B = config.PHASE1B
P = config.PHASE1
CONDS = config.PHASE1_CONDITIONS
METRICS = ["steps_to_suppress", "auc", "local_eps_min_255"]  # smaller = more vulnerable


def load_phase1_result(target, f):
    """The target's Phase 1 cache; it must still match the current Phase 1 metadata."""
    with open(config.PHASE1_DIR / "cache" / f"{target['id']}.json") as fh:
        cached = json.load(fh)
    if cached["meta"] != phase1.cache_meta(target, f):
        raise ValueError(f"Phase 1 cache of target {target['id']} is stale; rerun --stage phase1 first")
    return cached["meta"], cached["result"]


def matching_conf(dets, gt):
    """Highest confidence among person detections with IoU >= match_iou with the target, else 0."""
    if not dets:
        return 0.0
    boxes = np.array([d[:4] for d in dets], dtype=float)
    ok = iou_xywh(np.array(gt, dtype=float), boxes) >= P["match_iou"]
    return float(max((d[4] for d, m in zip(dets, ok) if m), default=0.0))


def steps_to_suppress(model, predictor, canvas_bgr, gt, adv_dir, stem):
    """Measure 1: plain PGD (eps 1/255, alpha 0.25/255), every step rounded and detected."""
    stride = attack.model_stride(model)
    x0 = attack.to_tensor(canvas_bgr)
    loss_fn, stats = phase1.targeted_loss(gt, stride, *canvas_bgr.shape[:2])
    eps, alpha = B["steps_eps_255"] / 255.0, B["steps_alpha_255"] / 255.0
    confs = [matching_conf(phase1.detect(predictor, canvas_bgr), gt)]
    first, pgd_s, det_s, saved_ok = None, 0.0, 0.0, None
    x = x0.clone()
    for k in range(1, B["steps_max"] + 1):
        t0 = time.perf_counter()
        x.requires_grad_(True)
        with torch.enable_grad():
            loss = loss_fn(model, x, stride)
            loss.backward()
        x = x.detach() - alpha * x.grad.sign()
        x = torch.min(torch.max(x, x0 - eps), x0 + eps).clamp(0.0, 1.0)
        t1 = time.perf_counter()
        img = attack.to_uint8_bgr(x)
        conf = matching_conf(phase1.detect(predictor, img), gt)
        det_s += time.perf_counter() - t1
        pgd_s += t1 - t0
        confs.append(conf)
        if first is None and conf < P["match_conf"]:
            first = k
            adv_dir.mkdir(parents=True, exist_ok=True)
            path = adv_dir / f"{stem}_step{k}.png"
            cv2.imwrite(str(path), img)
            reloaded = cv2.imread(str(path))
            saved_ok = bool(np.array_equal(reloaded, img)
                            and phase1.target_conf(phase1.detect(predictor, reloaded), gt) is None)
    auc = float(np.trapezoid(confs) / B["steps_max"])
    return {"confs": confs, "steps_to_suppress": first if first is not None else B["steps_censored"],
            "censored": first is None, "auc": auc, "saved_png_confirms": saved_ok,
            "pgd_seconds_per_step": pgd_s / B["steps_max"], "detect_seconds_per_step": det_s / B["steps_max"],
            **stats}


def box_mask(gt, shape):
    """1 for canvas pixels whose centers lie inside the GT box, shaped (1, 1, H, W)."""
    h, w = shape[:2]
    x, y, bw, bh = gt
    cx, cy = np.arange(w) + 0.5, np.arange(h) + 0.5
    m = np.outer((cy >= y) & (cy <= y + bh), (cx >= x) & (cx <= x + bw)).astype(np.float32)
    return torch.from_numpy(m)[None, None].to(attack.device())


def cache_meta(phase1_meta):
    return {"phase1": phase1_meta, "phase1b": B}


def run_target(target, f, get_models, force=False):
    """Both measures for the three conditions of one eligible target; cached per target."""
    p1_meta, p1 = load_phase1_result(target, f)
    if not p1["eligible"]:
        return None
    path = config.PHASE1B_DIR / "cache" / f"{target['id']}.json"
    meta = cache_meta(p1_meta)
    if path.exists() and not force:
        with open(path) as fh:
            cached = json.load(fh)
        if cached.get("meta") == meta:
            return cached["result"]
        print(f"Phase 1b cache for target {target['id']} is stale; recomputing")
    model, predictor = get_models()
    src = cv2.imread(str(config.IMG_DIR / target["file_name"]))
    result = {"id": target["id"], "conditions": {}}
    for cond in CONDS:
        c = p1["conditions"][cond]
        canvas = phase1.render(src, c["A"], c["s"], f)
        gt = c["gt"]
        m1 = steps_to_suppress(model, predictor, canvas, gt,
                               config.PHASE1B_DIR / "adv" / "steps" / cond, str(target["id"]))
        t0 = time.perf_counter()
        m2 = phase1.min_eps(model, predictor, canvas, gt, config.PHASE1B_DIR / "adv" / "local" / cond,
                            str(target["id"]), eps_list=B["local_eps_255"],
                            censored_value=B["local_censored_eps_255"], mask=box_mask(gt, canvas.shape))
        m2["seconds"] = time.perf_counter() - t0
        result["conditions"][cond] = {"gt": gt, "clean_conf": c["clean_conf"], "steps": m1, "local": m2}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"meta": meta, "result": result}, fh)
    return result


# ---------- analysis

def table(results):
    rows = []
    for r in results:
        for cond in CONDS:
            c = r["conditions"][cond]
            s, loc = c["steps"], c["local"]
            local_steps = len(loc["tries"]) * P["steps"]
            rows.append({
                "image_id": r["id"], "condition": cond, "eligible": True, "clean_conf": c["clean_conf"],
                "steps_to_suppress": s["steps_to_suppress"], "steps_censored": s["censored"],
                "auc": s["auc"], "saved_png_confirms": s["saved_png_confirms"],
                "local_eps_min_255": loc["eps_min_255"], "local_censored": loc["censored"],
                "local_n_eps_tried": len(loc["tries"]),
                "steps_pgd_seconds_per_step": s["pgd_seconds_per_step"],
                "steps_detect_seconds_per_step": s["detect_seconds_per_step"],
                "local_pgd_seconds_per_step": sum(t["pgd_seconds"] for t in loc["tries"]) / local_steps,
            })
    return pd.DataFrame(rows)


def summarize(df):
    """Per metric and comparison: median difference, bootstrap CI, Wilcoxon, more/less/equal."""
    rows = [("overall", "targets", "n", df["image_id"].nunique())]
    for metric in METRICS:
        for cond in CONDS:
            x = df[df["condition"] == cond][metric].astype(float)
            rows += [(metric, cond, "mean", x.mean()), (metric, cond, "median", x.median())]
        cens = {"steps_to_suppress": "steps_censored", "local_eps_min_255": "local_censored"}.get(metric)
        if cens:
            for cond in CONDS:
                rows.append((metric, cond, "frac_censored", df[df["condition"] == cond][cens].astype(float).mean()))
        rng = np.random.default_rng(B["seed"])
        for a, b in config.PHASE1_COMPARISONS:
            name = f"{a} - {b}"
            res = phase1_analysis.paired(df, metric, a, b, rng)
            d = phase1_analysis.paired_diffs(df, metric, a, b)
            rows += [(metric, name, k, v) for k, v in res.items()]
            rows += [(metric, name, f"n_{a}_more", int((d > 0).sum())),
                     (metric, name, f"n_{a}_less", int((d < 0).sum())),
                     (metric, name, "n_equal", int((d == 0).sum())),
                     (metric, name, "interpretation",
                      phase1_analysis.interpret(res["ci_low"], res["ci_high"], a, b))]
    return pd.DataFrame(rows, columns=["metric_or_section", "name", "statistic", "value"])


def perturbation_figure(target, result, f, path, gain=10):
    """For each condition: localized attacked canvas at eps_min | perturbation x gain."""
    src = cv2.imread(str(config.IMG_DIR / target["file_name"]))
    _, p1 = load_phase1_result(target, f)
    cols = []
    for cond in CONDS:
        c = p1["conditions"][cond]
        clean = phase1.render(src, c["A"], c["s"], f)
        loc = result["conditions"][cond]["local"]
        e = loc["tries"][-1]["eps_255"]
        png = config.PHASE1B_DIR / "adv" / "local" / cond / f"{target['id']}_eps{e}.png"
        adv = cv2.imread(str(png)) if png.exists() else clean
        delta = adv.astype(np.float64) - clean.astype(np.float64)
        amp = np.clip(127.5 + gain * delta, 0, 255).astype(np.uint8)
        x, y, w, h = c["gt"]
        tiles = []
        for img, label in ((adv, f"{cond}: localized attack, eps {e}/255"
                            f"{' (censored)' if loc['censored'] else ''}"),
                           (amp, f"{cond}: perturbation x{gain}")):
            tile = img.copy()
            cv2.rectangle(tile, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(tile, (0, 0), (tw + 12, th + base + 12), (0, 0, 0), thickness=-1)
            cv2.putText(tile, label, (6, 6 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            tiles.append(tile)
        cols.append(np.vstack(tiles))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack(cols))


# ---------- pre-registered addendum: clean-confidence control (docs/PHASE1B_DESIGN.md)

CONF_DIFF_MAX = 0.05  # (b): pairs with |clean-confidence difference| below this


def addendum(df):
    """(a) paired clean-confidence differences, (b) main comparisons on pairs with similar clean
    confidence, (c) pooled Spearman correlations (descriptive). Long format like summarize()."""
    from scipy.stats import spearmanr

    rows = []
    rng = np.random.default_rng(B["seed"])
    for a, b in config.PHASE1_COMPARISONS:
        res = phase1_analysis.paired(df, "clean_conf", a, b, rng)
        rows += [("a_clean_conf", f"{a} - {b}", k, v) for k, v in res.items()]
    wide = df.pivot(index="image_id", columns="condition", values="clean_conf")
    for metric in METRICS:
        rng = np.random.default_rng(B["seed"])
        for a, b in config.PHASE1_COMPARISONS:
            name = f"{a} - {b}"
            ids = wide.index[(wide[a] - wide[b]).abs() < CONF_DIFF_MAX]
            sub = df[df["image_id"].isin(ids)]
            res = phase1_analysis.paired(sub, metric, a, b, rng)
            d = phase1_analysis.paired_diffs(sub, metric, a, b)
            rows += [(f"b_{metric}", name, "n_subset", len(ids))]
            rows += [(f"b_{metric}", name, k, v) for k, v in res.items()]
            rows += [(f"b_{metric}", name, f"n_{a}_more", int((d > 0).sum())),
                     (f"b_{metric}", name, f"n_{a}_less", int((d < 0).sum())),
                     (f"b_{metric}", name, "n_equal", int((d == 0).sum())),
                     (f"b_{metric}", name, "rule_on_subset",
                      phase1_analysis.interpret(res["ci_low"], res["ci_high"], a, b))]
    for metric in METRICS:
        rho = spearmanr(df["clean_conf"], df[metric].astype(float)).statistic
        rows += [("c_spearman", metric, "rho_clean_conf", float(rho)), ("c_spearman", metric, "n_rows", len(df))]
    return pd.DataFrame(rows, columns=["section", "name", "statistic", "value"])


# ---------- figures (descriptive)

def fig_paired(df, summary, path):
    """One panel per metric: per-target paired differences and median with 95% bootstrap CI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reliability import BLUE, INK, INK_2, MUTED, _style

    fig, axes = plt.subplots(1, len(METRICS), figsize=(13.5, 4.0), facecolor="#fcfcfb")
    rng = np.random.default_rng(B["seed"])  # jitter only
    labels = {"steps_to_suppress": "Steps to suppress", "auc": "Confidence AUC",
              "local_eps_min_255": "Localized eps_min (1/255)"}
    s = summary.set_index(["metric_or_section", "name", "statistic"])["value"]
    for ax, metric in zip(axes, METRICS):
        _style(ax)
        for i, (a, b) in enumerate(config.PHASE1_COMPARISONS):
            d = phase1_analysis.paired_diffs(df, metric, a, b)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=12, color=MUTED, alpha=0.6, zorder=2)
            name = f"{a} - {b}"
            med, lo, hi = (float(s.get((metric, name, k), np.nan)) for k in ("median_diff", "ci_low", "ci_high"))
            if not np.isnan(med):
                ax.errorbar(i + 0.28, med, yerr=[[med - lo], [hi - med]], fmt="o", color=BLUE,
                            markersize=6, capsize=4, linewidth=2, zorder=3)
        ax.axhline(0, color=INK_2, linewidth=0.8, zorder=1)
        ax.set_xticks(range(3), [f"{a}\n- {b}" for a, b in config.PHASE1_COMPARISONS], fontsize=8)
        ax.set_title(f"{labels[metric]}: paired difference", loc="left", fontsize=10, color=INK)
    fig.suptitle("Per-target differences (grey); median with 95% bootstrap CI (blue). Below 0: first "
                 "condition more vulnerable", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_confidence_curves(results, path):
    """Median (and interquartile range) of the target's matching confidence per PGD step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from phase1_analysis import COLORS
    from reliability import INK, INK_2, _style

    fig, ax = plt.subplots(figsize=(6.5, 3.9), facecolor="#fcfcfb")
    _style(ax)
    for cond in CONDS:
        curves = np.array([r["conditions"][cond]["steps"]["confs"] for r in results])
        k = np.arange(curves.shape[1])
        q1, med, q3 = np.percentile(curves, [25, 50, 75], axis=0)
        ax.fill_between(k, q1, q3, color=COLORS[cond], alpha=0.15, linewidth=0)
        ax.plot(k, med, color=COLORS[cond], linewidth=2, label=f"{cond} (n = {len(curves)})")
    ax.axhline(P["match_conf"], color=INK_2, linewidth=0.8, linestyle="--")
    ax.set_xlabel("PGD step (eps 1/255, alpha 0.25/255)")
    ax.set_ylabel("Target's matching confidence")
    ax.set_title("Median and interquartile range per step; dashed: 0.25", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
