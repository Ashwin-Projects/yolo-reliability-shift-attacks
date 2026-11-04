"""Phase 1 analysis: per-target table, paired statistics, figures and visualizations."""
import math
import warnings

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import config  # noqa: E402
import phase1  # noqa: E402
from reliability import BLUE, INK, INK_2, MUTED, ORANGE, _style  # noqa: E402

P = config.PHASE1
CONDS = config.PHASE1_CONDITIONS
COLORS = dict(zip(CONDS, [BLUE, ORANGE, "#1baf7a"]))  # categorical slots 1-3, fixed order


def targets_table(results):
    """One row per target and condition."""
    rows = []
    for r in results:
        for cond in CONDS:
            c = r["conditions"][cond]
            a = c.get("attack") or {}
            tries = a.get("tries", [])
            n_steps = len(tries) * P["steps"]
            rows.append({
                "image_id": r["id"], "condition": cond,
                "box_w": c["box_w"], "box_h": c["box_h"], "box_area": c["box_area"],
                "touches_border": c["touches_border"], "clean_conf": c["clean_conf"],
                "eligible": r["eligible"],
                "eps_min_255": a.get("eps_min_255", float("nan")),
                "censored": a.get("censored", float("nan")),
                "n_eps_tried": len(tries),
                "fallback_steps": a.get("fallback_steps", float("nan")),
                "no_target_preds": a.get("no_target_preds", float("nan")),
                "pgd_seconds_per_step": (sum(t["pgd_seconds"] for t in tries) / n_steps) if n_steps else float("nan"),
            })
    return pd.DataFrame(rows)


def bootstrap_median_ci(d, rng):
    idx = rng.integers(0, len(d), size=(P["bootstrap"], len(d)))
    meds = np.median(d[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def paired_diffs(df, column, a, b):
    """Per-target a - b over eligible targets; empty when no eligible target has both."""
    wide = df[df["eligible"]].pivot(index="image_id", columns="condition", values=column)
    if wide.empty or a not in wide.columns or b not in wide.columns:
        return np.array([], dtype=float)
    return (wide[a] - wide[b]).dropna().to_numpy(dtype=float)


def paired(df, column, a, b, rng):
    """Median of (a - b) over eligible targets, bootstrap CI, Wilcoxon signed-rank."""
    d = paired_diffs(df, column, a, b)
    out = {"n": len(d), "n_nonzero": int((d != 0).sum())}
    if not len(d):
        return {**out, "median_diff": math.nan, "ci_low": math.nan, "ci_high": math.nan,
                "wilcoxon_stat": math.nan, "wilcoxon_p": math.nan}
    lo, hi = bootstrap_median_ci(d, rng)
    stat = p = math.nan
    if out["n_nonzero"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = wilcoxon(d)
        stat, p = float(res.statistic), float(res.pvalue)
    return {**out, "median_diff": float(np.median(d)), "ci_low": lo, "ci_high": hi,
            "wilcoxon_stat": stat, "wilcoxon_p": p}


def interpret(ci_low, ci_high, a, b):
    """Interpretation rule fixed in docs/PHASE1_DESIGN.md (CI of the median difference)."""
    if math.isnan(ci_low):
        return "undefined"
    if ci_low <= 0 <= ci_high:
        return f"{a} ~ {b}"
    return f"{a} more vulnerable than {b}" if ci_high < 0 else f"{a} less vulnerable than {b}"


def summarize(df):
    """Long-format summary: section, name, metric, value."""
    rows = []
    el = df[df["eligible"]]
    n_targets = df["image_id"].nunique()
    rows.append(("overall", "targets", "n_selected", n_targets))
    rows.append(("overall", "targets", "n_eligible", el["image_id"].nunique()))
    if el.empty:
        rows.append(("overall", "message", "note",
                     "no eligible targets: per-condition and paired statistics are undefined"))
    for cond in CONDS:
        x = el[el["condition"] == cond]
        e = x["eps_min_255"].astype(float)
        rows += [("condition", cond, "n", len(x)),
                 ("condition", cond, "mean_eps_min_255", e.mean()),
                 ("condition", cond, "median_eps_min_255", e.median()),
                 ("condition", cond, "frac_censored", x["censored"].astype(float).mean()),
                 ("condition", cond, "mean_clean_conf", x["clean_conf"].mean()),
                 ("condition", cond, "frac_touches_border", df[df["condition"] == cond]["touches_border"].mean())]
        for eps in P["eps_255"]:
            rows.append(("condition", cond, f"success_rate_eps{eps}", (e <= eps).mean() if len(e) else math.nan))
    for section, column in (("paired_eps_min_255", "eps_min_255"), ("paired_clean_conf", "clean_conf")):
        rng = np.random.default_rng(P["seed"])
        for a, b in config.PHASE1_COMPARISONS:
            res = paired(df, column, a, b, rng)
            name = f"{a} - {b}"
            rows += [(section, name, k, v) for k, v in res.items()]
            if column == "eps_min_255":
                rows.append((section, name, "interpretation", interpret(res["ci_low"], res["ci_high"], a, b)))
    wide = df.pivot(index="image_id", columns="condition", values=["box_w", "box_h"])
    for dim in ("box_w", "box_h"):
        ratio = wide[(dim, "edge")] / wide[(dim, "center_shrunk")]
        rows += [("size_match", f"edge / center_shrunk {dim}", k, v)
                 for k, v in (("min", ratio.min()), ("max", ratio.max()), ("mean", ratio.mean()))]
    return pd.DataFrame(rows, columns=["section", "name", "metric", "value"])


def descriptives(df, value="eps_min_255", values=None, censored_value=None):
    """Counts per value and condition, and per comparison the more / less / equal pair counts
    with the non-tied targets listed. Long format: section, name, image_id, metric, value."""
    el = df[df["eligible"]]
    values = values or P["eps_255"]
    censored_value = censored_value or P["censored_eps_255"]
    rows = []
    for cond in CONDS:
        x = el[el["condition"] == cond]
        for v in values:
            n = int(((x[value] == v) & ~x["censored"].astype(bool)).sum())
            rows.append(("counts", cond, "", f"n_{value}_{v}", n))
        rows.append(("counts", cond, "", "n_censored", int(x["censored"].astype(bool).sum())))
    for a, b in config.PHASE1_COMPARISONS:
        wide = el.pivot(index="image_id", columns="condition", values=value).astype(float)
        name = f"{a} - {b}"
        d = (wide[a] - wide[b]).dropna() if not wide.empty else pd.Series(dtype=float)
        rows += [("pairs", name, "", f"n_{a}_more", int((d > 0).sum())),
                 ("pairs", name, "", f"n_{a}_less", int((d < 0).sum())),
                 ("pairs", name, "", "n_equal", int((d == 0).sum()))]
        for img_id in d.index[d != 0]:
            rows.append(("non_tied", name, int(img_id), f"{a}|{b}",
                         f"{wide.at[img_id, a]:g}|{wide.at[img_id, b]:g}"))
    return pd.DataFrame(rows, columns=["section", "name", "image_id", "metric", "value"])


def fig_success_curves(df, path):
    el = df[df["eligible"]]
    fig, ax = plt.subplots(figsize=(6.0, 3.8), facecolor="#fcfcfb")
    _style(ax)
    for cond in CONDS:
        e = el[el["condition"] == cond]["eps_min_255"].astype(float)
        rates = [(e <= eps).mean() if len(e) else math.nan for eps in P["eps_255"]]
        ax.plot(P["eps_255"], rates, color=COLORS[cond], linewidth=2, marker="o", markersize=6,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=f"{cond} (n = {len(e)})")
    ax.set_xticks(P["eps_255"], [f"{e}/255" for e in P["eps_255"]])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Attack strength eps")
    ax.set_ylabel("Success rate (target not detected)")
    ax.set_title("Share of eligible targets suppressed at eps or below", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_paired(df, summary, path):
    fig, ax = plt.subplots(figsize=(6.0, 3.8), facecolor="#fcfcfb")
    _style(ax)
    rng = np.random.default_rng(P["seed"])  # jitter only
    s = summary[summary["section"] == "paired_eps_min_255"].set_index(["name", "metric"])["value"]
    for i, (a, b) in enumerate(config.PHASE1_COMPARISONS):
        d = paired_diffs(df, "eps_min_255", a, b)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=16, color=MUTED, alpha=0.7, zorder=2)
        name = f"{a} - {b}"
        med, lo, hi = (float(s.get((name, k), math.nan)) for k in ("median_diff", "ci_low", "ci_high"))
        if not math.isnan(med):
            ax.errorbar(i + 0.28, med, yerr=[[med - lo], [hi - med]], fmt="o", color=BLUE,
                        markersize=6, capsize=4, linewidth=2, zorder=3)
    ax.axhline(0, color=INK_2, linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(config.PHASE1_COMPARISONS)),
                  [f"{a}\n- {b}" for a, b in config.PHASE1_COMPARISONS])
    ax.set_ylabel("Paired difference in eps_min (1/255)")
    ax.set_title("Per-target differences (grey) and median with 95% bootstrap CI (blue)",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw(canvas, cond_result):
    """Canvas with the target GT box (green) and person detections >= 0.25 (red)."""
    img = canvas.copy()
    x, y, w, h = cond_result["gt"]
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    for dx, dy, dw, dh, conf in cond_result["dets_025"]:
        cv2.rectangle(img, (int(dx), int(dy)), (int(dx + dw), int(dy + dh)), (0, 0, 255), 2)
        cv2.putText(img, f"{conf:.2f}", (int(dx), max(int(dy) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return img


def render_conditions(target, result, f):
    src = cv2.imread(str(config.IMG_DIR / target["file_name"]))
    out = {}
    for cond in CONDS:
        c = result["conditions"][cond]
        out[cond] = draw(phase1.render(src, c["A"], c["s"], f), c)
    return out


def save_visualizations(targets, results, f):
    """results/phase1/vis/<condition>/<image_id>.jpg (clean canvas, GT green, detections red)."""
    for t, r in zip(targets, results):
        for cond, img in render_conditions(t, r, f).items():
            d = config.PHASE1_DIR / "vis" / cond
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{t['id']}.jpg"), img)


def grid_figure(targets, results, f, path):
    """One row per target: center | edge | center_shrunk, with labels."""
    rows = []
    for t, r in zip(targets, results):
        tiles = []
        for cond, img in render_conditions(t, r, f).items():
            c = r["conditions"][cond]
            conf = "none" if c["clean_conf"] is None else f"{c['clean_conf']:.2f}"
            label = f"{t['id']} {cond}  box {c['box_w']:.0f}x{c['box_h']:.0f}  conf {conf}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (0, 0), (tw + 12, th + base + 12), (0, 0, 0), thickness=-1)
            cv2.putText(img, label, (6, 6 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(path), np.vstack(rows))
