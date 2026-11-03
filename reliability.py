"""Stage "reliability": view consistency, box- and image-level failure signals, figures."""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import config  # noqa: E402
from metrics import count_gt, iou_xywh  # noqa: E402

IMAGE_SIGNALS = ["n_det", "mean_conf", "view_disagreement", "mean_consistency"]
IMAGE_COLUMNS = ["image_id", "n_gt", "n_matched", "miss_rate"] + IMAGE_SIGNALS

# chart colours: reference categorical slots 1-2 (light mode) and chrome
BLUE, ORANGE = "#2a78d6", "#eb6834"
FAMILY_COLORS = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4"]  # slots 1-5, fixed order
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _boxes_by_image(detections, conf_thr=config.CALIB_CONF):
    out = {}
    for d in detections:
        if d["score"] >= conf_thr:
            out.setdefault(d["image_id"], []).append(d["bbox"])
    return {k: np.array(v, dtype=float).reshape(-1, 4) for k, v in out.items()}


def safe_auroc(y, score):
    """AUROC of `score` for binary `y`; NaN when only one class is present."""
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        return float("nan")
    return float(roc_auc_score(y, score))


def box_level(labeled, view_dets):
    """Per detection (conf >= CALIB_CONF): label and consistency across views.

    consistency = mean over views of the max IoU with any view detection
    (conf >= CALIB_CONF) in the same image, 0 when the view has none.
    """
    view_boxes = {v: _boxes_by_image(dets) for v, dets in view_dets.items()}
    rows = []
    for d, label in labeled:
        box = np.array(d["bbox"], dtype=float)
        ious = []
        for boxes in view_boxes.values():
            others = boxes.get(d["image_id"])
            ious.append(float(iou_xywh(box, others).max()) if others is not None else 0.0)
        rows.append({"image_id": d["image_id"], "score": d["score"], "label": label,
                     "consistency": float(np.mean(ious))})
    return pd.DataFrame(rows, columns=["image_id", "score", "label", "consistency"])


def box_summary(box_df):
    """AUROC of confidence, consistency and their product for TP vs FP (ignored excluded)."""
    kept = box_df[box_df["label"] != "ignored"]
    y = (kept["label"] == "TP").to_numpy()
    return {
        "n_tp": int(y.sum()),
        "n_fp": int((~y).sum()),
        "auroc_conf": safe_auroc(y, kept["score"]),
        "auroc_cons": safe_auroc(y, kept["consistency"]),
        "auroc_combined": safe_auroc(y, kept["score"] * kept["consistency"]),
        "mean_cons_tp": float(kept.loc[y, "consistency"].mean()) if y.any() else float("nan"),
        "mean_cons_fp": float(kept.loc[~y, "consistency"].mean()) if (~y).any() else float("nan"),
    }


def image_level(coco_gt, images, box_df, view_dets):
    """Per image: miss rate at conf >= CALIB_CONF and GT-free signals.

    Images without non-crowd person GT (possible after fisheye drops) are skipped.
    For images with no detection, mean_conf and mean_consistency are 0.
    """
    by_image = dict(tuple(box_df.groupby("image_id")))
    view_counts = {v: {} for v in view_dets}
    for v, dets in view_dets.items():
        for d in dets:
            if d["score"] >= config.CALIB_CONF:
                view_counts[v][d["image_id"]] = view_counts[v].get(d["image_id"], 0) + 1
    rows, n_skipped = [], 0
    for im in images:
        n_gt = count_gt(coco_gt, [im["id"]])
        if n_gt == 0:
            n_skipped += 1
            continue
        boxes = by_image.get(im["id"], box_df.iloc[0:0])
        n_det = len(boxes)
        matched = int((boxes["label"] == "TP").sum())
        disagreement = np.mean([abs(view_counts[v].get(im["id"], 0) - n_det) / max(n_det, 1)
                                for v in view_dets])
        rows.append({
            "image_id": im["id"],
            "n_gt": n_gt,
            "n_matched": matched,
            "miss_rate": 1.0 - matched / n_gt,
            "n_det": n_det,
            "mean_conf": float(boxes["score"].mean()) if n_det else 0.0,
            "view_disagreement": float(disagreement),
            "mean_consistency": float(boxes["consistency"].mean()) if n_det else 0.0,
        })
    # keep the schema even when every image was skipped
    return pd.DataFrame(rows, columns=IMAGE_COLUMNS), n_skipped


def image_summary(img_df, condition, p_values=True):
    """Spearman(signal, miss_rate) and AUROC(signal -> miss_rate >= MISS_HALF) per signal.

    p_values=False leaves spearman_p as NaN (used for the pooled rows, whose
    7,498 rows repeat the same 500 source images, so independence fails).
    An empty img_df gives n_images 0 and NaN statistics.
    """
    y = (img_df["miss_rate"] >= config.MISS_HALF).to_numpy()
    rows = []
    for signal in IMAGE_SIGNALS:
        rho = p = float("nan")
        if len(img_df) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # constant input -> NaN
                rho, p = spearmanr(img_df[signal], img_df["miss_rate"])
        rows.append({"condition": condition, "signal": signal, "n_images": len(img_df),
                     "n_miss_half": int(y.sum()), "spearman_rho": float(rho),
                     "spearman_p": float(p) if p_values else float("nan"),
                     "auroc_miss_half": safe_auroc(y, img_df[signal])})
    return rows


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _place_labels(fig, ax, points):
    """Greedy direct labels: first of 8 offsets whose box overlaps no placed label or point."""
    offsets = [(6, 5), (6, -12), (-6, 5), (-6, -12), (14, 16), (-14, 16), (14, -22), (-14, -22)]
    renderer = fig.canvas.get_renderer()
    inside = ax.get_window_extent(renderer)
    taken = []
    for _, x, y in points:  # keep labels off the markers themselves
        px, py = ax.transData.transform((x, y))
        taken.append(matplotlib.transforms.Bbox([[px - 4, py - 4], [px + 4, py + 4]]))
    for text, x, y in points:
        best = None
        for dx, dy in offsets:
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                              fontsize=8, color=INK_2, ha="left" if dx > 0 else "right")
            # measured without a leader line, so the extent is the text box alone
            box = ann.get_window_extent(renderer).expanded(1.05, 1.15)
            ann.remove()
            n_overlap = sum(box.overlaps(t) for t in taken)
            if not inside.contains(box.x0, box.y0) or not inside.contains(box.x1, box.y1):
                n_overlap += 100  # label would be clipped by the axes
            if best is None or n_overlap < best[0]:
                best = (n_overlap, dx, dy, box)
            if n_overlap == 0:
                break
        _, dx, dy, box = best
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    color=INK_2, ha="left" if dx > 0 else "right",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=4))
        taken.append(box)


def fig_silent_failure(table, path, suffix=""):
    """recall_025 vs mean_conf_tp per condition, one panel per corruption family.

    Each panel joins clean to the family's severities; all other conditions are
    shown as faint context points. Shared axes keep panels comparable.
    """
    rows = table.set_index("condition")
    clean = "clean" + suffix
    keep = {clean} | {n + suffix for n in config.SHIFT_CONDITIONS}  # no attack-stage rows
    rows = rows.loc[[c for c in rows.index if c in keep]]
    families = {}
    for name, c in config.SHIFT_CONDITIONS.items():
        families.setdefault(c["family"], []).append(name + suffix)
    fig, axes = plt.subplots(1, len(families), figsize=(3.6 * len(families), 4.4),
                             sharex=True, sharey=True, facecolor="#fcfcfb")
    for ax, (family, names) in zip(axes, families.items()):
        _style(ax)
        chain = [n for n in [clean] + names if n in rows.index]
        ax.scatter(rows["recall_025"], rows["mean_conf_tp"], s=14, color=GRID, zorder=1)
        ax.plot(rows.loc[chain, "recall_025"], rows.loc[chain, "mean_conf_tp"],
                color=BLUE, linewidth=1.5, zorder=2)
        ax.scatter(rows.loc[chain, "recall_025"], rows.loc[chain, "mean_conf_tp"], s=46,
                   color=BLUE, edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        ax.set_title(family, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Recall at conf ≥ 0.25")
    axes[0].set_ylabel("Mean confidence of true positives")
    axes[0].margins(x=0.12, y=0.08)
    fig.suptitle("Recall at conf ≥ 0.25 vs mean confidence of true positives, per condition "
                 "(grey: all other conditions)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()  # fix the layout before measuring label positions
    for ax, names in zip(axes, families.values()):
        chain = [n for n in [clean] + names if n in rows.index]
        _place_labels(fig, ax, [("clean" if n == clean else n.removesuffix(suffix).rsplit("_", 1)[1],
                                 rows.at[n, "recall_025"], rows.at[n, "mean_conf_tp"])
                                for n in chain])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_recoverable(table, path, suffix=""):
    """recoverable_share vs severity (clean = 0), one line per corruption family."""
    rows = table.set_index("condition")
    clean = "clean" + suffix
    families = {}
    for name, c in config.SHIFT_CONDITIONS.items():
        families.setdefault(c["family"], []).append(name + suffix)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="#fcfcfb")
    _style(ax)
    for color, (family, names) in zip(FAMILY_COLORS, families.items()):
        chain = [n for n in [clean] + names if n in rows.index]
        sev = list(range(len(chain)))
        vals = rows.loc[chain, "recoverable_share"].to_numpy()
        ax.plot(sev, vals, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=family)
    ax.set_xticks(range(4), ["0 (clean)", "1", "2", "3"])
    ax.set_xlabel("Severity")
    ax.set_ylabel("Recoverable share")
    ax.set_title("Share of people missed at conf ≥ 0.25 that are matched at conf ≥ 0.05",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def risk_coverage(scores, is_fp):
    """Coverage and FP fraction among the top-k detections ranked by descending score."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    fp = np.asarray(is_fp, dtype=float)[order]
    k = np.arange(1, len(fp) + 1)
    return k / len(fp), np.cumsum(fp) / k


def fig_risk_coverage(box_dfs, path):
    """Box-level risk-coverage ranked by confidence vs consistency, one panel per condition."""
    fig, axes = plt.subplots(1, len(box_dfs), figsize=(4.2 * len(box_dfs), 4.2),
                             sharey=True, facecolor="#fcfcfb")
    axes = np.atleast_1d(axes)
    for ax, (name, df) in zip(axes, box_dfs.items()):
        _style(ax)
        kept = df[df["label"] != "ignored"]
        is_fp = (kept["label"] == "FP").to_numpy()
        for col, color, label in (("score", BLUE, "Confidence"), ("consistency", ORANGE, "Consistency")):
            cov, risk = risk_coverage(kept[col].to_numpy(), is_fp)
            ax.plot(cov, risk, color=color, linewidth=2, label=label)
        ax.set_title(f"{name} (n = {len(kept)})", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Coverage (fraction of detections kept)")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Risk (FP fraction among kept)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
