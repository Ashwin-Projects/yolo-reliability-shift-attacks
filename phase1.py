"""Phase 1: position-dependent adversarial vulnerability under fisheye geometry.

Target selection, canvas geometry and rendering, detection matching, the targeted
PGD attack with minimum-eps search, and per-target caching. See docs/PHASE1_DESIGN.md.

Plane and canvas coordinates share one origin: the optical axis at canvas point
(C, C), C = canvas / 2. A source pixel u maps to the plane point A + s * u; the plane
point maps to the canvas with the equidistant fisheye (corruptions.fisheye_forward).
"""
import json
import math
import time

import cv2
import numpy as np
import torch

import attack
import config
import corruptions
import infer
from metrics import iou_xywh

P = config.PHASE1
C = P["canvas"] / 2.0


def focal():
    """f such that the canvas half-diagonal maps to theta_max (existing fisheye math)."""
    return corruptions.fisheye_focal(P["canvas"], P["canvas"], P["theta_max_deg"])


# ---------- target selection

def select_targets(coco, n=None):
    """First n (default n_targets) val2017 images (by id) whose largest non-crowd person qualifies."""
    n = P["n_targets"] if n is None else n
    targets = []
    for img_id in sorted(coco.getImgIds()):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], catIds=[config.PERSON_CAT_ID]))
        noncrowd = [a for a in anns if not a["iscrowd"]]
        if not noncrowd:
            continue
        target = max(noncrowd, key=lambda a: a["bbox"][2] * a["bbox"][3])
        x, y, w, h = target["bbox"]
        area = w * h
        img = coco.loadImgs(img_id)[0]
        m = P["border_margin"]
        if area < P["min_box_side"] ** 2:
            continue
        if x < m or y < m or x + w > img["width"] - m or y + h > img["height"] - m:
            continue
        others = [a["bbox"][2] * a["bbox"][3] for a in anns if a["id"] != target["id"]]
        if any(area < P["size_ratio"] * o for o in others):
            continue
        targets.append({"id": img_id, "file_name": img["file_name"], "width": img["width"],
                        "height": img["height"], "coco_url": img["coco_url"],
                        "target_ann_id": target["id"], "target_bbox": target["bbox"]})
        if len(targets) == n:
            break
    return targets


def load_or_create_targets(coco):
    """Recompute the selection; it must match data/phase1_targets.json when that exists."""
    targets = select_targets(coco)
    path = config.PHASE1_TARGETS_FILE
    if path.exists():
        with open(path) as f:
            saved = json.load(f)
        if saved != targets:
            raise ValueError(f"{path} does not match the Phase 1 selection rule; delete it to regenerate")
    else:
        with open(path, "w") as f:
            json.dump(targets, f, indent=1)
        print(f"Saved {len(targets)} Phase 1 targets to {path}")
    return targets


# ---------- geometry

def box_center(bbox):
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def canvas_box(bbox, A, s, f):
    """Warp a source box to the canvas: plane box, then the forward fisheye of 20 points per edge."""
    x, y, w, h = bbox
    plane = [A[0] + s[0] * x, A[1] + s[1] * y, s[0] * w, s[1] * h]
    return corruptions.warp_box(plane, P["canvas"], P["canvas"], f)


def centered(bbox, s):
    """Offset A that puts the target box center on the optical axis at scale s."""
    cx, cy = box_center(bbox)
    return (C - s[0] * cx, C - s[1] * cy)


def edge_placement(bbox, f):
    """Scale 1, target center on the plane at f * tan(r_d / f) right of the axis."""
    r_u = f * math.tan(P["edge_radius"] * C / f)
    cx, cy = box_center(bbox)
    return (C + r_u - cx, C - cy), (1.0, 1.0)


def shrunk_placement(bbox, f, edge_box, tol=1e-6, max_iter=200):
    """Anisotropic scale so the centered target's warped w, h equal edge_box's w, h."""
    s = [1.0, 1.0]
    for _ in range(max_iter):
        b = canvas_box(bbox, centered(bbox, s), s, f)
        rx, ry = edge_box[2] / b[2], edge_box[3] / b[3]
        if abs(rx - 1) < tol and abs(ry - 1) < tol:
            return centered(bbox, s), tuple(s)
        s = [s[0] * rx, s[1] * ry]
    raise RuntimeError(f"center_shrunk scale did not converge for box {bbox}")


def placements(bbox, f):
    """(A, s) per condition; edge first because center_shrunk matches its box."""
    out = {"center": (centered(bbox, (1.0, 1.0)), (1.0, 1.0))}
    out["edge"] = edge_placement(bbox, f)
    out["center_shrunk"] = shrunk_placement(bbox, f, canvas_box(bbox, *out["edge"], f))
    return out


_PLANE = {}


def plane_grid(f):
    """Plane coordinates of every canvas pixel center (inverse equidistant map, as in
    corruptions._fisheye_maps): r_d -> theta = r_d / f -> r_u = f * tan(theta)."""
    key = (f, config.FISHEYE_MAX_THETA_DEG)
    if key not in _PLANE:
        n = P["canvas"]
        dx, dy = np.meshgrid(np.arange(n) + 0.5 - C, np.arange(n) + 0.5 - C)
        r_d = np.hypot(dx, dy)
        theta = r_d / f
        valid = theta < math.radians(config.FISHEYE_MAX_THETA_DEG)
        r_u = f * np.tan(np.where(valid, theta, 0.0))
        scale = np.divide(r_u, r_d, out=np.ones_like(r_d), where=r_d > 0)
        px = np.where(valid, C + dx * scale, np.inf)
        py = np.where(valid, C + dy * scale, np.inf)
        _PLANE[key] = (px, py)
    return _PLANE[key]


def render(src_bgr, A, s, f):
    """Canvas image: bilinear sample of the source placed at A + s * u; gray outside it."""
    px, py = plane_grid(f)
    map_x = ((px - A[0]) / s[0] - 0.5).astype(np.float32)  # source pixel-index coordinates
    map_y = ((py - A[1]) / s[1] - 0.5).astype(np.float32)
    map_x[~np.isfinite(map_x)] = -10.0
    map_y[~np.isfinite(map_y)] = -10.0
    g = P["gray"]
    return cv2.remap(src_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(g, g, g))


def touches_border(box):
    x, y, w, h = box
    n = P["canvas"]
    return x <= 0 or y <= 0 or x + w >= n or y + h >= n


# ---------- detection through the normal pipeline

def detect(predictor, img_bgr):
    """Person detections [x, y, w, h, conf] (same predictor settings as all stages)."""
    r = predictor.predict(img_bgr, imgsz=config.IMGSZ, conf=config.EVAL_CONF,
                          device=config.DEVICE, verbose=False)[0]
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    return [[float(a), float(b), float(c - a), float(d - b), float(p)]
            for (a, b, c, d), p, k in zip(xyxy, conf, cls) if k == config.YOLO_PERSON_CLS]


def target_conf(dets, gt):
    """Max confidence among detections with conf >= match_conf and IoU >= match_iou, else None."""
    cand = np.array([d for d in dets if d[4] >= P["match_conf"]], dtype=float).reshape(-1, 5)
    if not len(cand):
        return None
    ok = iou_xywh(np.array(gt, dtype=float), cand[:, :4]) >= P["match_iou"]
    return float(cand[ok, 4].max()) if ok.any() else None


# ---------- targeted attack

def targeted_loss(gt, stride, h, w):
    """Loss on raw predictions near the target, plus counters of the fallback rules."""
    new_h, new_w, top, _, left, _ = attack.letterbox_params(h, w, stride)
    rx, ry = new_w / w, new_h / h
    x, y, bw, bh = gt
    g = torch.tensor([x * rx + left, y * ry + top, (x + bw) * rx + left, (y + bh) * ry + top],
                     device=attack.device())
    stats = {"fallback_steps": 0, "no_target_preds": 0}

    def loss_fn(model, xin, stride_):
        out = attack.raw_output(model, xin, stride_)[0]  # (4 + classes, N); rows 0-3: cx, cy, w, h
        cx, cy, pw, ph = (out[i].detach() for i in range(4))
        x1, y1, x2, y2 = cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2
        inter = ((torch.minimum(x2, g[2]) - torch.maximum(x1, g[0])).clamp(min=0)
                 * (torch.minimum(y2, g[3]) - torch.maximum(y1, g[1])).clamp(min=0))
        union = pw * ph + (g[2] - g[0]) * (g[3] - g[1]) - inter
        sel = inter / union.clamp(min=1e-9) >= P["loss_iou"]
        if not bool(sel.any()):
            stats["fallback_steps"] += 1
            sel = (cx >= g[0]) & (cx <= g[2]) & (cy >= g[1]) & (cy <= g[3])
        scores = out[4 + config.YOLO_PERSON_CLS]
        if not bool(sel.any()):
            stats["no_target_preds"] += 1
            return scores.sum() * 0.0
        picked = scores[sel]
        return torch.topk(picked, k=min(P["loss_top_k"], picked.numel())).values.sum()

    return loss_fn, stats


def min_eps(model, predictor, canvas_bgr, gt, adv_dir, stem, eps_list=None, censored_value=None,
            mask=None):
    """Try eps in ascending order; stop at the first eps whose saved PNG hides the target.

    Defaults are the Phase 1 settings; Phase 1b passes its eps list, censor value and a mask."""
    eps_list = eps_list or P["eps_255"]
    censored_value = censored_value or P["censored_eps_255"]
    stride = attack.model_stride(model)
    x0 = attack.to_tensor(canvas_bgr)
    loss_fn, stats = targeted_loss(gt, stride, *canvas_bgr.shape[:2])
    adv_dir.mkdir(parents=True, exist_ok=True)
    tries, success = [], False
    for e in eps_list:
        eps = e / 255.0
        t0 = time.perf_counter()
        x, losses, best_iter = attack.pgd_vanish(model, x0, eps, eps * P["alpha_frac"], P["steps"],
                                                 stride, None, loss_fn=loss_fn, mask=mask)
        pgd_s = time.perf_counter() - t0
        path = adv_dir / f"{stem}_eps{e}.png"
        cv2.imwrite(str(path), attack.to_uint8_bgr(x))
        conf_after = target_conf(detect(predictor, cv2.imread(str(path))), gt)
        success = conf_after is None
        tries.append({"eps_255": e, "loss_start": losses[0], "loss_best": losses[best_iter],
                      "best_iter": best_iter, "success": success, "target_conf_after": conf_after,
                      "pgd_seconds": pgd_s})
        if success:
            break
        path.unlink()  # keep only the PNG at eps_min
    return {"eps_min_255": tries[-1]["eps_255"] if success else censored_value,
            "censored": not success, "tries": tries, **stats}


# ---------- per-target pipeline with cache

def global_settings():
    """Global (non-Phase-1) settings that affect rendering, GT warping or detection matching."""
    return {"fisheye_max_theta_deg": config.FISHEYE_MAX_THETA_DEG,  # render cutoff (plane_grid)
            "fisheye_min_box": config.FISHEYE_MIN_BOX,  # warp_box drops smaller boxes
            "fisheye_edge_points": config.FISHEYE_EDGE_POINTS,  # warp_box samples per edge
            "yolo_person_cls": config.YOLO_PERSON_CLS, "person_cat_id": config.PERSON_CAT_ID}


def cache_meta(target, f):
    return {"design": P, "conditions": config.PHASE1_CONDITIONS, "f": f,
            "global": global_settings(),
            "model": config.MODEL_WEIGHTS, "weights_sha256": attack.weights_sha256(),
            "imgsz": config.IMGSZ, "eval_conf": config.EVAL_CONF,
            "letterbox": config.ATTACK_LETTERBOX,
            "target": {k: target[k] for k in ("id", "file_name", "target_ann_id", "target_bbox")}}


def raw_vs_nms_iou(model, canvas_bgr, dets):
    """Check of the raw-output decoding: IoU of the top raw person box with the top NMS box."""
    if not dets:
        return float("nan")
    stride = attack.model_stride(model)
    with torch.no_grad():
        out = attack.raw_output(model, attack.to_tensor(canvas_bgr), stride)[0]
    i = int(out[4 + config.YOLO_PERSON_CLS].argmax())
    cx, cy, w, h = (float(out[k, i]) for k in range(4))
    top = max(dets, key=lambda d: d[4])
    return float(iou_xywh(np.array([cx - w / 2, cy - h / 2, w, h]), np.array([top[:4]]))[0])


def clean_conditions(predictor, src, bbox, f):
    """Render the three conditions and detect the target on each clean canvas."""
    conds, canvases = {}, {}
    for cond, (A, s) in placements(bbox, f).items():
        canvas = render(src, A, s, f)
        gt = canvas_box(bbox, A, s, f)
        dets = detect(predictor, canvas)
        conf = target_conf(dets, gt)
        canvases[cond] = canvas
        conds[cond] = {"A": list(A), "s": list(s), "gt": gt, "box_w": gt[2], "box_h": gt[3],
                       "box_area": gt[2] * gt[3], "touches_border": touches_border(gt),
                       "clean_conf": conf, "detected": conf is not None,
                       "dets_025": [d for d in dets if d[4] >= P["match_conf"]]}
    return conds, canvases


def run_target(target, f, get_models, force=False):
    """Render the three conditions, check eligibility, attack if eligible; cached per target."""
    cache_dir = config.PHASE1_DIR / "cache"
    path = cache_dir / f"{target['id']}.json"
    meta = cache_meta(target, f)
    if path.exists() and not force:
        with open(path) as fh:
            cached = json.load(fh)
        if cached.get("meta") == meta:
            return cached["result"]
        print(f"Phase 1 cache for target {target['id']} is stale; recomputing")
    model, predictor = get_models()
    src = cv2.imread(str(config.IMG_DIR / target["file_name"]))
    conds, canvases = clean_conditions(predictor, src, target["target_bbox"], f)
    eligible = all(c["detected"] for c in conds.values())
    result = {"id": target["id"], "eligible": eligible, "conditions": conds,
              "raw_vs_nms_iou_center": raw_vs_nms_iou(model, canvases["center"],
                                                      conds["center"]["dets_025"])}
    if eligible:
        for cond in config.PHASE1_CONDITIONS:
            conds[cond]["attack"] = min_eps(model, predictor, canvases[cond], conds[cond]["gt"],
                                            config.PHASE1_DIR / "adv" / cond, str(target["id"]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"meta": meta, "result": result}, fh)
    return result


def model_loader():
    """Lazily load the attack model and a separate predictor (predict() may fuse its model)."""
    models = {}

    def get():
        if not models:
            infer.set_seed()
            models["m"] = (attack.load_attack_model(), infer.load_model())
        return models["m"]

    return get
