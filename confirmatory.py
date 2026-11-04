"""Confirmatory study (docs/CONFIRMATORY_DESIGN.md).

Qualifying targets 101-400 of the Phase 1 selection order, the Phase 1 rendering and
eligibility check, the Phase 1b measures, the pilot statistics unchanged (phase1_analysis,
phase1b.summarize / addendum), Holm correction, the clean-confidence regression, the matched
subset and the pre-registered decision rule.
"""
import json
import math

import cv2
import pandas as pd
from scipy import stats

import config
import execution
import phase1
import phase1_analysis
import phase1b

CF = config.CONFIRM
B = config.PHASE1B
CONDS = config.PHASE1_CONDITIONS
COMPARISONS = config.PHASE1_COMPARISONS
METRICS = phase1b.METRICS


# ---------- targets

def load_or_create_targets(coco):
    """Targets first..last of the selection order; must match data/confirmatory_targets.json."""
    pilot = phase1.load_or_create_targets(coco)  # validates the saved pilot list
    ranked = phase1.select_targets(coco, n=CF["last_target"])
    if len(ranked) < CF["last_target"]:
        raise ValueError(f"only {len(ranked)} qualifying targets; {CF['last_target']} needed")
    if ranked[:len(pilot)] != pilot:
        raise ValueError("selection order no longer starts with the pilot targets")
    targets = ranked[CF["first_target"] - 1:CF["last_target"]]
    overlap = {t["id"] for t in targets} & {t["id"] for t in pilot}
    if overlap:
        raise ValueError(f"confirmatory targets overlap the pilot: {sorted(overlap)}")
    path = config.CONFIRM_TARGETS_FILE
    if path.exists():
        with open(path) as fh:
            if json.load(fh) != targets:
                raise ValueError(f"{path} does not match the confirmatory selection rule")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(targets, fh, indent=1)
        print(f"Saved {len(targets)} confirmatory targets to {path}")
    return targets


# ---------- per-target pipeline with cache

def cache_meta(target, f, env):
    return {"phase1": phase1.cache_meta(target, f), "phase1b": B, "confirm": CF, "environment": env}


def run_target(target, f, get_models, env, force=False):
    """Clean detection in the three conditions; both Phase 1b measures if eligible."""
    path = config.CONFIRM_DIR / "cache" / f"{target['id']}.json"
    meta = json.loads(json.dumps(cache_meta(target, f, env)))  # tuples -> lists, as stored
    if path.exists() and not force:
        with open(path) as fh:
            cached = json.load(fh)
        if cached.get("meta") == meta:
            return cached["result"]
        print(f"Confirmatory cache for target {target['id']} is stale; recomputing")
    model, predictor = get_models()
    src = cv2.imread(str(config.IMG_DIR / target["file_name"]))
    conds, canvases = phase1.clean_conditions(predictor, src, target["target_bbox"], f)
    eligible = all(c["detected"] for c in conds.values())
    if eligible:
        for cond in CONDS:
            conds[cond]["steps"], conds[cond]["local"] = phase1b.measure(
                model, predictor, canvases[cond], conds[cond]["gt"], config.CONFIRM_DIR / "adv", cond,
                str(target["id"]))
    if config.DEVICE != "cpu":
        execution.verify_cuda_protocol()
    result = {"id": target["id"], "eligible": eligible, "conditions": conds}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"meta": meta, "result": result}, fh)
    return result


# ---------- analysis

def holm(pvals):
    """Holm-adjusted p-values; an undefined p stays undefined and m = len(pvals)."""
    m = len(pvals)
    adj, running = [math.nan] * m, 0.0
    for rank, (p, i) in enumerate(sorted((p, i) for i, p in enumerate(pvals) if not math.isnan(p))):
        running = max(running, min(1.0, (m - rank) * p))
        adj[i] = running
    return adj


def lookup(summary, section_col, section, name, statistic):
    v = summary[(summary[section_col] == section) & (summary["name"] == name)
                & (summary["statistic"] == statistic)]["value"]
    return float(v.iloc[0]) if len(v) else math.nan


def add_holm(summary, section_col, sections):
    """Append a wilcoxon_p_holm row per comparison, adjusting within each section's three comparisons."""
    rows = []
    names = [f"{a} - {b}" for a, b in COMPARISONS]
    for section in sections:
        ps = [lookup(summary, section_col, section, name, "wilcoxon_p") for name in names]
        rows += [(section, name, "wilcoxon_p_holm", p) for name, p in zip(names, holm(ps))]
    return pd.concat([summary, pd.DataFrame(rows, columns=summary.columns)], ignore_index=True)


def descriptives(df, selected):
    """Selected / eligible counts, censored counts and localized eps_min counts per level."""
    rows = [("overall", "targets", "n_selected", len(selected)),
            ("overall", "targets", "n_eligible", sum(r["eligible"] for r in selected))]
    for cond in CONDS:
        x = df[df["condition"] == cond] if len(df) else df
        rows += [("steps_to_suppress", cond, "n_censored", int(x["steps_censored"].sum()) if len(x) else 0),
                 ("local_eps_min_255", cond, "n_censored", int(x["local_censored"].sum()) if len(x) else 0)]
        rows += [("local_eps_min_255", cond, f"n_eps_{e}",
                  int((x["local_eps_min_255"] == e).sum()) if len(x) else 0) for e in B["local_eps_255"]]
    return pd.DataFrame(rows, columns=["metric_or_section", "name", "statistic", "value"])


def regression(df, a, b):
    """OLS of the paired AUC difference on the paired clean-confidence difference (a - b)."""
    y = phase1_analysis.paired_diffs(df, "auc", a, b)
    x = phase1_analysis.paired_diffs(df, "clean_conf", a, b)
    keys = ("intercept", "intercept_ci_low", "intercept_ci_high", "slope", "slope_ci_low", "slope_ci_high")
    if len(y) < 3:
        return {"n": len(y), **{k: math.nan for k in keys}}
    res = stats.linregress(x, y)
    t = stats.t.ppf(0.975, len(y) - 2)
    return {"n": len(y), "intercept": res.intercept,
            "intercept_ci_low": res.intercept - t * res.intercept_stderr,
            "intercept_ci_high": res.intercept + t * res.intercept_stderr,
            "slope": res.slope, "slope_ci_low": res.slope - t * res.stderr,
            "slope_ci_high": res.slope + t * res.stderr}


def confidence_control(df):
    """(a) regressions; (b) the matched subset |delta clean conf| < conf_diff_max, with Holm."""
    rows = [("regression_auc_on_clean_conf", f"{a} - {b}", k, v)
            for a, b in COMPARISONS for k, v in regression(df, a, b).items()]
    reg = pd.DataFrame(rows, columns=["section", "name", "statistic", "value"])
    sub = phase1b.addendum(df, conf_diff_max=CF["conf_diff_max"])
    sub = sub[sub["section"] != "c_spearman"]
    sub = add_holm(sub, "section", [f"b_{m}" for m in METRICS])
    return pd.concat([reg, sub], ignore_index=True)


def decision(summary, control):
    """Pre-registered decision rule on the primary metric and comparison."""
    a, b = CF["primary_comparison"]
    name, metric = f"{a} - {b}", CF["primary_metric"]
    lo, hi = (lookup(summary, "metric_or_section", metric, name, s) for s in ("ci_low", "ci_high"))
    p = lookup(summary, "metric_or_section", metric, name, "wilcoxon_p_holm")
    ilo, ihi = (lookup(control, "section", "regression_auc_on_clean_conf", name, s)
                for s in ("intercept_ci_low", "intercept_ci_high"))
    criteria = {  # comparisons with NaN are False, so undefined statistics never satisfy a criterion
        "distortion adds vulnerability": {"median CI entirely below 0": hi < 0,
                                          f"Holm-corrected p < {CF['alpha']}": p < CF["alpha"],
                                          "intercept CI entirely below 0": ihi < 0},
        "distortion protects": {"median CI entirely above 0": lo > 0,
                                f"Holm-corrected p < {CF['alpha']}": p < CF["alpha"],
                                "intercept CI entirely above 0": ilo > 0},
    }
    outcome = next((d for d, c in criteria.items() if all(c.values())), "does not support either direction")
    rows = [("input", name, "auc_ci_low", lo), ("input", name, "auc_ci_high", hi),
            ("input", name, "auc_wilcoxon_p_holm", p),
            ("input", name, "intercept_ci_low", ilo), ("input", name, "intercept_ci_high", ihi)]
    rows += [("criterion", direction, crit, met) for direction, c in criteria.items() for crit, met in c.items()]
    rows.append(("outcome", name, "conclusion", outcome))
    return pd.DataFrame(rows, columns=["section", "name", "statistic", "value"]), outcome


def analyze(selected):
    df = phase1b.table([r for r in selected if r["eligible"]])
    summary = add_holm(phase1b.summarize(df), "metric_or_section", METRICS)
    summary = pd.concat([summary, descriptives(df, selected)], ignore_index=True)
    control = confidence_control(df)
    dec, outcome = decision(summary, control)
    return df, summary, control, dec, outcome
