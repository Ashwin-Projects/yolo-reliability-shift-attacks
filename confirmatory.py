"""Confirmatory study (docs/CONFIRMATORY_DESIGN.md).

Qualifying targets 101-952 (or 101-400, chosen by the probe and fixed in the execution
manifest) of the Phase 1 selection order, the Phase 1 rendering and
eligibility check, the Phase 1b measures, the pilot statistics unchanged (phase1_analysis,
phase1b.summarize / addendum), Holm correction, the clean-confidence regression, the matched
subset and the pre-registered decision rule.
"""
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import stats

import attack
import config
import data
import execution
import phase1
import phase1_analysis
import phase1b

CF = config.CONFIRM
B = config.PHASE1B
P = config.PHASE1
CONDS = config.PHASE1_CONDITIONS
COMPARISONS = config.PHASE1_COMPARISONS
METRICS = phase1b.METRICS


# ---------- targets

def load_or_create_targets(coco):
    """All qualifying targets from first_target on, in selection order; must match
    data/confirmatory_targets.json. The run uses the leading part given by the manifest."""
    pilot = phase1.load_or_create_targets(coco)  # validates the saved pilot list
    ranked = phase1.select_targets(coco, n=math.inf)
    if len(ranked) < CF["fallback_last_target"]:
        raise ValueError(f"only {len(ranked)} qualifying targets; {CF['fallback_last_target']} needed")
    if ranked[:len(pilot)] != pilot:
        raise ValueError("selection order no longer starts with the pilot targets")
    targets = ranked[CF["first_target"] - 1:]
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


def full_last_target(targets):
    return CF["first_target"] - 1 + len(targets)


def choose_range(targets, estimate_hours):
    """Pre-specified fallback: all targets unless the runtime estimate exceeds the limit."""
    last = (full_last_target(targets) if estimate_hours <= CF["runtime_limit_hours"]
            else CF["fallback_last_target"])
    return [CF["first_target"], last]


def select_range(targets, manifest):
    """The targets of the manifest's range; the range must follow from the recorded estimate."""
    rng = manifest.get("target_range")
    probe = manifest.get("probe") or {}
    if rng is None or "estimate_hours" not in probe:
        raise SystemExit("The execution manifest has no target range or runtime estimate; rerun --stage probe")
    if len(probe.get("probe_target_ids", [])) != CF["probe_targets"]:
        raise SystemExit(f"The probe record must contain exactly {CF['probe_targets']} timed targets: {probe}")
    if probe.get("warm_up") is not True:
        raise SystemExit(f"The probe record must show the untimed warm-up pass (warm_up = true), found "
                         f"{probe.get('warm_up', 'no warm_up field')!r}; rerun --stage probe")
    if probe.get("n_full") != len(targets) or rng != choose_range(targets, probe["estimate_hours"]):
        raise SystemExit(f"Manifest target range {rng} does not follow from its probe record {probe}")
    return targets[:rng[1] - rng[0] + 1]


# ---------- probe: per-step timing on pilot targets (results discarded)

def warm_up(model, canvas, gt):
    """One untimed forward and backward pass of the targeted loss (CUDA context, kernels)."""
    stride = attack.model_stride(model)
    loss_fn, _ = phase1.targeted_loss(gt, stride, *canvas.shape[:2])
    x = attack.to_tensor(canvas).requires_grad_(True)
    with torch.enable_grad():
        loss_fn(model, x, stride).backward()
    if config.DEVICE != "cpu":
        torch.cuda.synchronize()


def probe_timing(pilot, f, get_models):
    """Both measures on exactly probe_targets pilot targets: the pilot list is tried in order
    until that many are eligible here, after one untimed warm-up pass; aborts if too few are
    eligible. The attacked images go to a temporary folder that is deleted; nothing is saved."""
    model, predictor = get_models()
    m1_step, m2_steps, m2_seconds, used = [], [], 0.0, []
    with tempfile.TemporaryDirectory() as tmp:
        for t in pilot:
            data.ensure_images([t])
            src = cv2.imread(str(config.IMG_DIR / t["file_name"]))
            conds, canvases = phase1.clean_conditions(predictor, src, t["target_bbox"], f)
            if not all(c["detected"] for c in conds.values()):
                print(f"probe target {t['id']}: not eligible here; trying the next pilot target")
                continue
            if not used:
                warm_up(model, canvases["center"], conds["center"]["gt"])
            for cond in CONDS:
                m1, m2 = phase1b.measure(model, predictor, canvases[cond], conds[cond]["gt"], Path(tmp),
                                         cond, str(t["id"]))
                m1_step.append(m1["pgd_seconds_per_step"] + m1["detect_seconds_per_step"])
                m2_steps.append(len(m2["tries"]) * P["steps"])
                m2_seconds += sum(tr["pgd_seconds"] for tr in m2["tries"])
            used.append(t["id"])
            print(f"probe target {t['id']}: timed")
            if len(used) == CF["probe_targets"]:
                break
    if len(used) < CF["probe_targets"]:
        raise SystemExit(f"Only {len(used)} of {len(pilot)} pilot targets are eligible here; the probe needs "
                         f"{CF['probe_targets']}. No manifest written.")
    if config.DEVICE != "cpu":
        execution.verify_cuda_protocol()
    return {"probe_target_ids": used, "warm_up": True, "m1_seconds_per_step": float(np.mean(m1_step)),
            "m2_seconds_per_step": m2_seconds / sum(m2_steps),
            "m2_mean_steps_per_condition": float(np.mean(m2_steps))}


def estimate_hours(timing, n_targets):
    """N x 3 x (50 t1 + L t2) seconds, in hours (docs/CONFIRMATORY_DESIGN.md, runtime fallback)."""
    per_condition = (B["steps_max"] * timing["m1_seconds_per_step"]
                     + timing["m2_mean_steps_per_condition"] * timing["m2_seconds_per_step"])
    return n_targets * len(CONDS) * per_condition / 3600


# ---------- per-target pipeline with cache

TEST_LABEL = "TEST ONLY"


def provenance(manifest_path, test):
    """Run label with the manifest's path and SHA-256; test runs are labelled TEST ONLY."""
    import hashlib

    return {"run_label": TEST_LABEL if test else "PRODUCTION", "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}


def namespace(prov):
    """Test runs keep caches and images apart, so a production run can never reuse them."""
    return "_test_only" if prov["run_label"] == TEST_LABEL else ""


def cache_meta(target, f, env, prov):
    return {"phase1": phase1.cache_meta(target, f), "phase1b": B, "confirm": CF, "environment": env,
            "provenance": prov}


def run_target(target, f, get_models, env, prov, force=False):
    """Clean detection in the three conditions; both Phase 1b measures if eligible."""
    path = config.CONFIRM_DIR / f"cache{namespace(prov)}" / f"{target['id']}.json"
    meta = json.loads(json.dumps(cache_meta(target, f, env, prov)))  # tuples -> lists, as stored
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
                model, predictor, canvases[cond], conds[cond]["gt"], config.CONFIRM_DIR / f"adv{namespace(prov)}", cond,
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


SUMMARY_COLUMNS = ["metric_or_section", "name", "statistic", "value"]
LONG_COLUMNS = ["section", "name", "statistic", "value"]
NO_ELIGIBLE = "no eligible targets: all statistics are undefined, so no decision criterion can be met"


def analyze(selected):
    df = phase1b.table([r for r in selected if r["eligible"]])
    if df.empty:
        summary = pd.concat([descriptives(df, selected),
                             pd.DataFrame([("overall", "message", "note", NO_ELIGIBLE)], columns=SUMMARY_COLUMNS)],
                            ignore_index=True)
        control = pd.DataFrame([("message", "all", "note", NO_ELIGIBLE)], columns=LONG_COLUMNS)
    else:
        summary = add_holm(phase1b.summarize(df), "metric_or_section", METRICS)
        summary = pd.concat([summary, descriptives(df, selected)], ignore_index=True)
        control = confidence_control(df)
    dec, outcome = decision(summary, control)
    if df.empty:
        dec = pd.concat([dec, pd.DataFrame([("message", "all", "note", NO_ELIGIBLE)], columns=LONG_COLUMNS)],
                        ignore_index=True)
    return df, summary, control, dec, outcome


def selection_report(selected):
    """Exclusion report: every selected target, its eligibility and clean confidences."""
    return pd.DataFrame([{"image_id": r["id"], "eligible": r["eligible"],
                          **{f"clean_conf_{c}": r["conditions"][c]["clean_conf"] for c in CONDS}}
                         for r in selected])


METADATA_ROW = "metadata: provenance only, no data rows"


def with_provenance(df, prov):
    """Long tables get provenance rows; wide tables get provenance columns. An empty wide table
    gets one row holding only the provenance, marked in a row_type column as metadata."""
    if "statistic" in df.columns:
        rows = [("provenance", "run", k, v) for k, v in prov.items()]
        return pd.concat([df, pd.DataFrame(rows, columns=df.columns)], ignore_index=True)
    if df.empty:
        return pd.DataFrame([{"row_type": METADATA_ROW, **{c: None for c in df.columns}, **prov}])
    return df.assign(**prov)


def label_png(path, prov):
    """Write the provenance into the PNG's text fields."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    for k, v in prov.items():
        info.add_text(k, v)
    with Image.open(path) as im:
        im.load()
        im.save(path, pnginfo=info)
