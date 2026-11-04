"""Entry point: python run.py --stage {baseline,shift,reliability} [--limit N] [--force] [--only CONDITION]"""
import argparse
import math
import re
import time
from functools import partial

import cv2
import numpy as np
import pandas as pd

import attack
import config
import corruptions
import data
import infer
import metrics
import reliability

CONDITION_ORDER = (["clean"] + list(config.SHIFT_CONDITIONS)
                   + [config.ATTACK_CLEAN] + list(config.ATTACK_CONDITIONS))


def _base_condition(name):
    return re.sub(r"_limit\d+$", "", name)


def _condition_rank(name):
    base = _base_condition(name)
    return CONDITION_ORDER.index(base) if base in CONDITION_ORDER else len(CONDITION_ORDER)


def save_visualizations(coco_gt, images, detections, out_dir, n=config.N_VIS, transform=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    for im in images[:n]:
        img = cv2.imread(str(config.IMG_DIR / im["file_name"]))
        if transform is not None:
            img = transform(img, im)
        ann_ids = coco_gt.getAnnIds(imgIds=[im["id"]], catIds=[config.PERSON_CAT_ID])
        for a in coco_gt.loadAnns(ann_ids):
            x, y, w, h = a["bbox"]
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        for d in detections:
            if d["image_id"] != im["id"] or d["score"] < config.CALIB_CONF:
                continue
            x, y, w, h = d["bbox"]
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            cv2.putText(img, f"{d['score']:.2f}", (int(x), max(int(y) - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / im["file_name"]), img)
    print(f"Saved {min(n, len(images))} visualizations to {out_dir}")


def write_row(table_file, condition, row):
    """Insert or replace the row for `condition` in `table_file` (rows kept in condition order)."""
    new = pd.DataFrame([{"condition": condition, **row}])[config.TABLE_COLUMNS]
    if table_file.exists():
        old = pd.read_csv(table_file)
        new = pd.concat([old[old["condition"] != condition], new], ignore_index=True)
        new = new.reindex(columns=config.TABLE_COLUMNS)
        # columns absent from an older table come back as float; keep counts integer
        for col in ("n_images", "n_tp", "n_fp", "n_ignored", "n_gt"):
            new[col] = new[col].astype("Int64")
    new = (new.assign(_rank=new["condition"].map(_condition_rank))
              .sort_values("_rank", kind="stable").drop(columns="_rank").reset_index(drop=True))
    table_file.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(table_file, index=False)
    return new


def upsert_rows(path, df):
    """Replace the rows of every condition in `df` within the CSV at `path`; keep the others.

    Lets stages (reliability, attack) share result files without overwriting each other.
    """
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[~old["condition"].isin(df["condition"].unique())], df], ignore_index=True)
    df = (df.assign(_rank=df["condition"].map(_condition_rank))
            .sort_values("_rank", kind="stable").drop(columns="_rank").reset_index(drop=True))
    df.to_csv(path, index=False)
    return df


def write_bins(name, bins):
    config.BINS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bins, columns=["bin", "n", "mean_conf", "precision"]).to_csv(
        config.BINS_DIR / f"{name}.csv", index=False)


def write_dets(name, labeled, images):
    """One row per detection with conf >= CALIB_CONF: label and normalized radial distance.

    r_norm is the distance of the box centre from the image centre divided by the
    half-diagonal (0 = centre, 1 = corner).
    """
    sizes = {im["id"]: (im["width"], im["height"]) for im in images}
    rows = []
    for d, label in labeled:
        w, h = sizes[d["image_id"]]
        x, y, bw, bh = d["bbox"]
        r_norm = math.hypot(x + bw / 2 - w / 2, y + bh / 2 - h / 2) / (0.5 * math.hypot(w, h))
        rows.append({"image_id": d["image_id"], "score": d["score"], "label": label,
                     "x": x, "y": y, "w": bw, "h": bh, "r_norm": r_norm})
    config.DETS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["image_id", "score", "label", "x", "y", "w", "h", "r_norm"]).to_csv(
        config.DETS_DIR / f"{name}.csv", index=False)


def evaluate_condition(condition, corruption, coco_gt, images, limit, force, n_vis, transform=None):
    """Predict (cached), score and write all outputs for one condition.

    `transform` defaults to the corruption from corruptions.apply. Returns
    (table, table_file, labeled detections, detections).
    """
    name = condition if limit is None else f"{condition}_limit{limit}"
    table_file = config.TABLE_FILE if limit is None else config.SMOKE_TABLE_FILE
    if transform is None and corruption is not None:
        transform = partial(corruptions.apply, corruption)

    t1 = time.perf_counter()
    detections = infer.get_predictions(images, config.PREDS_DIR / f"{name}.json", name,
                                       force=force, corruption=corruption, transform=transform)
    t_infer = time.perf_counter() - t1

    img_ids = [im["id"] for im in images]
    row, bins, labeled = metrics.summarize(coco_gt, detections, img_ids)
    table = write_row(table_file, name, row)
    write_bins(name, bins)
    write_dets(name, labeled, images)
    save_visualizations(coco_gt, images, detections, config.VIS_DIR / condition,
                        n=n_vis, transform=transform)

    print("\nD-ECE bins (conf >= %.2f):" % config.CALIB_CONF)
    print(pd.DataFrame(bins).to_string(index=False))
    print(f"{name}: inference/cache-load time {t_infer:.1f}s")
    return table, table_file, labeled, detections


def prepare(limit):
    infer.set_seed()
    data.ensure_annotations()
    coco_gt = data.load_coco()
    images = data.load_or_create_subset(coco_gt)
    if limit is not None:
        images = images[:limit]
    data.ensure_images(images)
    return coco_gt, images


def run_baseline(limit=None, force=False):
    t0 = time.perf_counter()
    coco_gt, images = prepare(limit)
    table, table_file, _, _ = evaluate_condition("clean", None, coco_gt, images, limit, force, config.N_VIS)
    print(f"\n{table_file}:")
    print(table.to_string(index=False))
    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f}s")


def report_fisheye(name, per_image):
    df = pd.DataFrame(per_image)
    config.FISHEYE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.FISHEYE_DIR / f"{name}.csv", index=False)
    sizes = df.groupby(["width", "height"]).agg(n_images=("image_id", "size"), f=("f", "first"))
    sizes = sizes.sort_values("n_images", ascending=False).reset_index()
    print(f"{name}: f = half_diagonal / theta_max; {len(sizes)} distinct image sizes, "
          f"f range {df['f'].min():.4f}..{df['f'].max():.4f} px. Most common sizes:")
    print(sizes.head(5).to_string(index=False))
    print(f"{name}: GT person boxes {int(df['n_boxes'].sum())}, dropped {int(df['n_dropped'].sum())} "
          f"(of which crowd {int(df['n_dropped_crowd'].sum())})")
    return int(df["n_boxes"].sum()), int(df["n_dropped"].sum()), int(df["n_dropped_crowd"].sum())


def sanity_checks(table, suffix, fisheye_drops):
    print("\nSanity check: mAP50 should decrease with severity within each family")
    rows = table.set_index("condition")
    clean = "clean" + suffix
    if clean in rows.index:
        print(f"  reference {clean}: mAP50 {rows.at[clean, 'mAP50']:.4f}")
    families = {}
    for name, c in config.SHIFT_CONDITIONS.items():
        families.setdefault(c["family"], []).append(name + suffix)
    for family, names in families.items():
        vals = [(n, float(rows.at[n, "mAP50"])) for n in names if n in rows.index]
        if len(vals) < 2:
            continue
        ok = all(b < a for (_, a), (_, b) in zip(vals, vals[1:]))
        seq = " -> ".join(f"{v:.4f}" for _, v in vals)
        print(f"  {family}: {seq}  {'OK' if ok else 'FLAG: not strictly decreasing'}")
    for name, (n_boxes, n_dropped, n_crowd) in fisheye_drops.items():
        print(f"  {name}: dropped {n_dropped} of {n_boxes} GT person boxes ({n_crowd} crowd)")


def run_shift(limit=None, force=False, only=None):
    t0 = time.perf_counter()
    coco_gt, images = prepare(limit)
    suffix = "" if limit is None else f"_limit{limit}"
    names = [only] if only else list(config.SHIFT_CONDITIONS)
    timings, fisheye_drops, table, table_file = {}, {}, None, None
    for condition in names:
        t1 = time.perf_counter()
        print(f"\n=== {condition} ===")
        corruption = config.SHIFT_CONDITIONS[condition]
        gt = coco_gt
        if corruption["family"] == "fisheye":
            gt, per_image = corruptions.fisheye_coco(coco_gt, images, corruption["theta_max_deg"])
            fisheye_drops[condition + suffix] = report_fisheye(condition + suffix, per_image)
        table, table_file, _, _ = evaluate_condition(condition, corruption, gt, images, limit, force,
                                               config.N_VIS_SHIFT)
        timings[condition] = time.perf_counter() - t1

    print(f"\n{table_file}:")
    print(table.to_string(index=False))
    sanity_checks(table, suffix, fisheye_drops)
    print("\nPer-condition runtime (s): " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items()))
    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")


def run_reliability(limit=None, force=False):
    """Part A: low-threshold recall from cached predictions (no inference).
    Part B: augmented views (cached), box- and image-level reliability signals, figures.
    --force recomputes only the view predictions.
    """
    t0 = time.perf_counter()
    coco_gt, images = prepare(limit)
    suffix = "" if limit is None else f"_limit{limit}"
    smoke = "" if limit is None else "_smoke"
    table_file = config.TABLE_FILE if limit is None else config.SMOKE_TABLE_FILE
    img_ids = [im["id"] for im in images]

    timings, box_rows, box_frames, image_frames, image_rows, box_dfs = {}, [], [], [], [], {}
    conditions = [("clean", None)] + list(config.SHIFT_CONDITIONS.items())
    for condition, corruption in conditions:
        t1 = time.perf_counter()
        name = condition + suffix
        print(f"\n=== {name} ===")
        gt = coco_gt
        if corruption is not None and corruption["family"] == "fisheye":
            gt, _ = corruptions.fisheye_coco(coco_gt, images, corruption["theta_max_deg"])
        transform = None if corruption is None else partial(corruptions.apply, corruption)

        # Part A: original view, cached predictions only
        detections = infer.get_predictions(images, config.PREDS_DIR / f"{name}.json", name,
                                           corruption=corruption, transform=transform,
                                           cache_only=True)
        row, _, labeled = metrics.summarize(gt, detections, img_ids)
        table = write_row(table_file, name, row)

        # Part B: augmented views of the same condition image
        view_dets = {
            v: infer.get_predictions(images, config.PREDS_DIR / f"{name}__{v}.json", name,
                                     force=force, corruption=corruption, transform=transform,
                                     view_name=v)
            for v in config.VIEWS
        }
        box_df = reliability.box_level(labeled, view_dets)
        box_rows.append({"condition": name, **reliability.box_summary(box_df)})
        box_frames.append(box_df.assign(condition=name))
        box_dfs[condition] = box_df
        img_df, n_skipped = reliability.image_level(gt, images, box_df, view_dets)
        if n_skipped:
            print(f"{name}: {n_skipped} images without non-crowd person GT skipped at image level")
        img_df.insert(0, "condition", name)
        image_frames.append(img_df)
        image_rows += reliability.image_summary(img_df, name)
        timings[name] = time.perf_counter() - t1

    pooled = pd.concat(image_frames, ignore_index=True)
    # pooled rows repeat each source image across conditions: rho/AUROC only, no p-values
    image_rows += reliability.image_summary(pooled, "pooled", p_values=False)
    box_summary = pd.DataFrame(box_rows)
    image_summary = pd.DataFrame(image_rows)
    upsert_rows(config.RESULTS_DIR / f"box_level{smoke}.csv", box_summary)
    upsert_rows(config.RESULTS_DIR / f"box_level_dets{smoke}.csv", pd.concat(box_frames, ignore_index=True))
    upsert_rows(config.RESULTS_DIR / f"image_level{smoke}.csv", image_summary)
    upsert_rows(config.RESULTS_DIR / f"image_level_per_image{smoke}.csv", pooled)
    reliability.fig_silent_failure(table, config.RESULTS_DIR / f"fig_silent_failure{smoke}.png", suffix)
    reliability.fig_recoverable(table, config.RESULTS_DIR / f"fig_recoverable{smoke}.png", suffix)
    reliability.fig_risk_coverage({c + suffix: box_dfs[c] for c in config.RISK_COVERAGE_CONDITIONS},
                                  config.RESULTS_DIR / f"fig_risk_coverage{smoke}.png")

    pd.set_option("display.width", 250)
    print(f"\n{table_file} (recall columns):")
    print(table[["condition", "n_gt", "recall_005", "recall_010", "recall_025", "precision_005",
                 "precision_010", "precision_025", "recoverable_share"]].to_string(index=False))
    print("\nBox level (TP vs FP, conf >= 0.25, ignored excluded):")
    print(box_summary.to_string(index=False))
    print(f"\nImage level (target: miss_rate >= {config.MISS_HALF}):")
    print(image_summary.to_string(index=False))
    print("\nPer-condition runtime (s): " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items()))
    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")


def _persons_per_image(detections, img_ids):
    counts = {i: 0 for i in img_ids}
    for d in detections:
        if d["score"] >= config.CALIB_CONF:
            counts[d["image_id"]] += 1
    return counts


def run_attack(limit=None, force=False):
    """Stage attack on the first ATTACK_N_IMAGES subset images (--limit N: the first N of those).

    Attacked images are generated (or reused when their metadata matches), saved as
    PNG, and evaluated through the normal pipeline: table rows, views, box/image level,
    attack_views, attack_detection and attack_summary. --force regenerates the images
    and recomputes all predictions of this stage.
    """
    t0 = time.perf_counter()
    n = config.ATTACK_N_IMAGES if limit is None else min(limit, config.ATTACK_N_IMAGES)
    coco_gt, images = prepare(n)
    suffix = "" if limit is None else f"_limit{limit}"
    smoke = "" if limit is None else "_smoke"
    img_ids = [im["id"] for im in images]
    n_gt = metrics.count_gt(coco_gt, img_ids)

    timings, gen_times, infos, persons, img_frames = {}, {}, {}, {}, {}
    box_rows, box_frames, image_rows, view_rows = [], [], [], []
    conditions = [(config.ATTACK_CLEAN, None)] + list(config.ATTACK_CONDITIONS.items())
    for condition, spec in conditions:
        t1 = time.perf_counter()
        name = condition + suffix
        print(f"\n=== {name} ===")
        corruption = transform = None
        if spec is not None:
            adv_dir = config.ADV_DIR / name
            infos[condition], gen_times[condition] = attack.generate(spec, images, adv_dir, force=force)
            # prediction caches are tied to the exact saved PNG bytes
            corruption = {**spec, "png_sha256": attack.digest(adv_dir, images)}
            transform = partial(attack.load_saved, adv_dir)
            if condition == config.ATTACK_DETECT_POSITIVE:
                im = images[0]
                attack.perturbation_panel(cv2.imread(str(config.IMG_DIR / im["file_name"])),
                                          cv2.imread(str(attack.png_path(adv_dir, im))),
                                          config.VIS_DIR / condition / "perturbation_panel.png")

        table, table_file, labeled, detections = evaluate_condition(
            condition, corruption, coco_gt, images, limit, force, config.N_VIS_SHIFT, transform=transform)
        persons[condition] = _persons_per_image(detections, img_ids)

        view_dets = {
            v: infer.get_predictions(images, config.PREDS_DIR / f"{name}__{v}.json", name,
                                     force=force, corruption=corruption, transform=transform, view_name=v)
            for v in config.VIEWS
        }
        box_df = reliability.box_level(labeled, view_dets)
        box_rows.append({"condition": name, **reliability.box_summary(box_df)})
        box_frames.append(box_df.assign(condition=name))
        img_df, n_skipped = reliability.image_level(coco_gt, images, box_df, view_dets)
        if n_skipped:
            print(f"{name}: {n_skipped} images without non-crowd person GT skipped at image level")
        img_df.insert(0, "condition", name)
        img_frames[condition] = img_df
        image_rows += reliability.image_summary(img_df, name)

        # does an input transform undo the attack? recall at conf >= 0.25 per view
        view_row = {"condition": name, "n_gt": n_gt,
                    "recall_025": sum(lab == "TP" for _, lab in labeled) / n_gt}
        for v, dets in view_dets.items():
            labels = metrics.label_detections(coco_gt, dets, img_ids)
            view_row[f"recall_025_{v}"] = sum(lab == "TP" for _, lab in labels) / n_gt
        view_rows.append(view_row)
        timings[name] = time.perf_counter() - t1

    upsert_rows(config.RESULTS_DIR / f"box_level{smoke}.csv", pd.DataFrame(box_rows))
    upsert_rows(config.RESULTS_DIR / f"box_level_dets{smoke}.csv", pd.concat(box_frames, ignore_index=True))
    upsert_rows(config.RESULTS_DIR / f"image_level{smoke}.csv", pd.DataFrame(image_rows))
    upsert_rows(config.RESULTS_DIR / f"image_level_per_image{smoke}.csv",
                pd.concat(img_frames.values(), ignore_index=True))
    views = pd.DataFrame(view_rows)
    views.to_csv(config.RESULTS_DIR / f"attack_views{smoke}.csv", index=False)

    # attack detection: separate attacked images (positive) from clean / random-noise images
    detection_rows = []
    pos = img_frames[config.ATTACK_DETECT_POSITIVE]
    for negative in config.ATTACK_DETECT_NEGATIVES:
        neg = img_frames[negative]
        y = np.r_[np.ones(len(pos), dtype=bool), np.zeros(len(neg), dtype=bool)]
        for signal in config.ATTACK_DETECT_SIGNALS:
            detection_rows.append({
                "positive": config.ATTACK_DETECT_POSITIVE + suffix, "negative": negative + suffix,
                "signal": signal, "n_pos": len(pos), "n_neg": len(neg),
                "auroc": reliability.safe_auroc(y, np.r_[pos[signal].to_numpy(), neg[signal].to_numpy()]),
            })
    detection = pd.DataFrame(detection_rows)
    detection.to_csv(config.RESULTS_DIR / f"attack_detection{smoke}.csv", index=False)

    # attack success on saved images: persons >= 0.25 per image, before (clean) vs after
    before = persons[config.ATTACK_CLEAN]
    had = [i for i in img_ids if before[i] >= 1]
    summary_rows = []
    for condition in config.ATTACK_CONDITIONS:
        after = persons[condition]
        info = infos[condition]
        summary_rows.append({
            "condition": condition + suffix,
            "n_images": len(img_ids),
            "n_images_with_persons_before": len(had),
            "attack_success_rate": (sum(after[i] == 0 for i in had) / len(had)) if had else float("nan"),
            "mean_persons_before": float(np.mean([before[i] for i in img_ids])),
            "mean_persons_after": float(np.mean([after[i] for i in img_ids])),
            "mean_best_iter": float(info["best_iter"].mean()) if "best_iter" in info else float("nan"),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.RESULTS_DIR / f"attack_summary{smoke}.csv", index=False)

    pd.set_option("display.width", 250)
    names = [c + suffix for c, _ in conditions]
    print(f"\n{table_file} (attack rows):")
    print(table[table["condition"].isin(names)].to_string(index=False))
    print("\nattack_summary:"); print(summary.to_string(index=False))
    print("\nattack_views:"); print(views.to_string(index=False))
    print("\nattack_detection:"); print(detection.to_string(index=False))
    print("\nbox level:"); print(pd.DataFrame(box_rows).to_string(index=False))
    print("\nimage level:"); print(pd.DataFrame(image_rows).to_string(index=False))
    print("\nImage generation (s): " + ", ".join(f"{k} {v:.1f}" for k, v in gen_times.items()))
    print("Per-condition runtime incl. generation (s): "
          + ", ".join(f"{k} {v:.1f}" for k, v in timings.items()))
    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")


def run_phase1(limit=None, force=False):
    """Phase 1 (docs/PHASE1_DESIGN.md): minimum eps to hide a target person at the fisheye
    center, at the edge, and at the center shrunk to the edge size. --limit N: first N targets."""
    import phase1
    import phase1_analysis

    t0 = time.perf_counter()
    infer.set_seed()
    data.ensure_annotations()
    targets = phase1.load_or_create_targets(data.load_coco())
    if limit is not None:
        targets = targets[:limit]
    data.ensure_images(targets)
    smoke = "" if limit is None else "_smoke"
    f = phase1.focal()
    print(f"Phase 1: {len(targets)} targets, canvas {config.PHASE1['canvas']} px, f = {f:.4f} px "
          f"(half-diagonal at {config.PHASE1['theta_max_deg']} deg)")

    get_models = phase1.model_loader()
    results, seconds = [], []
    for t in targets:
        t1 = time.perf_counter()
        results.append(phase1.run_target(t, f, get_models, force=force))
        seconds.append(time.perf_counter() - t1)
        print(f"target {t['id']}: {seconds[-1]:.1f}s")

    out = config.PHASE1_DIR
    df = phase1_analysis.targets_table(results)
    df.to_csv(out / f"targets{smoke}.csv", index=False)
    summary = phase1_analysis.summarize(df)
    summary.to_csv(out / f"summary{smoke}.csv", index=False)
    phase1_analysis.fig_success_curves(df, out / f"fig_success_curves{smoke}.png")
    phase1_analysis.fig_paired(df, summary, out / f"fig_paired{smoke}.png")
    phase1_analysis.descriptives(df).to_csv(out / f"phase1_descriptives{smoke}.csv", index=False)
    phase1_analysis.save_visualizations(targets, results, f)
    if limit is not None:
        phase1_analysis.grid_figure(targets, results, f, out / "vis_smoke.png")

    pd.set_option("display.width", 250)
    for t, r in zip(targets, results):
        c = r["conditions"]
        print(f"\ntarget {t['id']} eligible={r['eligible']} "
              f"raw-vs-NMS top-box IoU (center) = {r['raw_vs_nms_iou_center']:.4f}")
        for cond in config.PHASE1_CONDITIONS:
            x = c[cond]
            a = x.get("attack") or {}
            tries = ", ".join(f"{tr['eps_255']}:{'ok' if tr['success'] else 'no'}" for tr in a.get("tries", []))
            conf = "none" if x["clean_conf"] is None else f"{x['clean_conf']:.4f}"
            print(f"  {cond:14s} box {x['box_w']:.1f} x {x['box_h']:.1f} (area {x['box_area']:.0f}), "
                  f"border={x['touches_border']}, clean conf {conf}, eps_min "
                  f"{a.get('eps_min_255', '-')}{' (censored)' if a.get('censored') else ''} [{tries}]")
        print(f"  edge / center_shrunk: w {c['edge']['box_w'] / c['center_shrunk']['box_w']:.6f}, "
              f"h {c['edge']['box_h'] / c['center_shrunk']['box_h']:.6f}")
    print(f"\n{out / f'summary{smoke}.csv'}:")
    print(summary.to_string(index=False))
    print(f"\nPer-target runtime (s): {', '.join(f'{s:.1f}' for s in seconds)}; "
          f"total {time.perf_counter() - t0:.1f}s")


def run_phase1b(limit=None, force=False):
    """Phase 1b (docs/PHASE1B_DESIGN.md): steps-to-suppress and localized eps_min on the
    Phase 1 eligible targets. --limit N: the first N eligible targets."""
    import phase1
    import phase1b

    t0 = time.perf_counter()
    infer.set_seed()
    data.ensure_annotations()
    targets = phase1.load_or_create_targets(data.load_coco())
    data.ensure_images(targets)
    f = phase1.focal()
    eligible = [t for t in targets if phase1b.load_phase1_result(t, f)[1]["eligible"]]
    if limit is not None:
        eligible = eligible[:limit]
    smoke = "" if limit is None else "_smoke"
    print(f"Phase 1b: {len(eligible)} eligible targets, f = {f:.4f} px")

    get_models = phase1.model_loader()
    results, seconds = [], []
    for t in eligible:
        t1 = time.perf_counter()
        results.append(phase1b.run_target(t, f, get_models, force=force))
        seconds.append(time.perf_counter() - t1)
        print(f"target {t['id']}: {seconds[-1]:.1f}s")

    out = config.PHASE1B_DIR
    out.mkdir(parents=True, exist_ok=True)
    df = phase1b.table(results)
    df.to_csv(out / f"targets{smoke}.csv", index=False)
    summary = phase1b.summarize(df)
    summary.to_csv(out / f"summary{smoke}.csv", index=False)
    phase1b.addendum(df).to_csv(out / f"summary_addendum{smoke}.csv", index=False)
    phase1b.fig_paired(df, summary, out / f"fig_paired{smoke}.png")
    phase1b.fig_confidence_curves(results, out / f"fig_confidence_curves{smoke}.png")
    if limit is not None:
        phase1b.perturbation_figure(eligible[0], results[0], f, out / "vis_smoke.png")

    pd.set_option("display.width", 250)
    cols = ["image_id", "condition", "clean_conf", "steps_to_suppress", "steps_censored", "auc",
            "saved_png_confirms", "local_eps_min_255", "local_censored", "local_n_eps_tried"]
    print(df[cols].to_string(index=False))
    print(f"\n{out / f'summary{smoke}.csv'}:")
    print(summary.to_string(index=False))
    print(f"\nPer-target runtime (s): {', '.join(f'{s:.1f}' for s in seconds)}; "
          f"total {time.perf_counter() - t0:.1f}s")


GPU_STAGES = []  # stages that support --device cuda


def startup_info():
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"torch {torch.__version__}; CUDA available: {torch.cuda.is_available()}; GPU: {gpu}; "
          f"device: {config.DEVICE}; out: {config.RESULTS_DIR}; data: {config.DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="detconf experiments")
    parser.add_argument("--stage", required=True,
                        choices=["baseline", "shift", "reliability", "attack", "phase1", "phase1b"])
    parser.add_argument("--limit", type=int, default=None, help="use only the first N subset images")
    parser.add_argument("--force", action="store_true", help="recompute cached predictions")
    parser.add_argument("--only", choices=list(config.SHIFT_CONDITIONS),
                        help="shift stage: run a single condition")
    parser.add_argument("--gradcheck", type=int, metavar="N",
                        help="attack stage: only run the gradient check on the first N images")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="cuda: one GPU (cuda:0); only for --stage probe and confirmatory")
    parser.add_argument("--out", default="results", help="results folder (relative: to this folder)")
    parser.add_argument("--data", default="data", help="data folder (relative: to this folder)")
    args = parser.parse_args()
    if args.device == "cuda" and args.stage not in GPU_STAGES:
        parser.error(f"--device cuda applies only to --stage {' / '.join(GPU_STAGES)}")
    config.set_paths(args.out, args.data)
    if args.device == "cuda":
        config.DEVICE = config.CUDA_DEVICE
    startup_info()
    if args.gradcheck is not None and args.stage != "attack":
        parser.error("--gradcheck applies to --stage attack")
    if args.stage == "baseline":
        if args.only:
            parser.error("--only applies to --stage shift")
        run_baseline(limit=args.limit, force=args.force)
    elif args.stage == "shift":
        run_shift(limit=args.limit, force=args.force, only=args.only)
    elif args.stage == "reliability":
        if args.only:
            parser.error("--only is not supported for --stage reliability (pooled stats need all conditions)")
        run_reliability(limit=args.limit, force=args.force)
    elif args.stage == "attack":
        if args.only:
            parser.error("--only is not supported for --stage attack")
        if args.gradcheck is not None:
            _, images = prepare(args.gradcheck)
            t0 = time.perf_counter()
            attack.gradcheck(images)
            print(f"Gradient check runtime: {time.perf_counter() - t0:.1f}s")
        else:
            run_attack(limit=args.limit, force=args.force)
    elif args.stage == "phase1":
        if args.only:
            parser.error("--only is not supported for --stage phase1")
        run_phase1(limit=args.limit, force=args.force)
    elif args.stage == "phase1b":
        if args.only:
            parser.error("--only is not supported for --stage phase1b")
        run_phase1b(limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
