"""Person-class mAP (pycocotools) and detection calibration error (D-ECE)."""
import contextlib
import io

import numpy as np
from pycocotools.cocoeval import COCOeval

import config


def coco_map(coco_gt, detections, img_ids):
    """Return (mAP50, mAP50_95) for the person class restricted to img_ids."""
    if not detections:
        return 0.0, 0.0
    coco_dt = coco_gt.loadRes(detections)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.imgIds = list(img_ids)
    ev.params.catIds = [config.PERSON_CAT_ID]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
    ev.summarize()
    return float(ev.stats[1]), float(ev.stats[0])


def iou_xywh(box, boxes):
    """IoU between one [x, y, w, h] box and an (N, 4) array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    y2 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def overlap_with_crowd(box, crowds):
    """Max over crowd boxes of intersection / detection area for one [x, y, w, h] box."""
    if not len(crowds) or box[2] * box[3] <= 0:
        return 0.0
    x1 = np.maximum(box[0], crowds[:, 0])
    y1 = np.maximum(box[1], crowds[:, 1])
    x2 = np.minimum(box[0] + box[2], crowds[:, 0] + crowds[:, 2])
    y2 = np.minimum(box[1] + box[3], crowds[:, 1] + crowds[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return float((inter / (box[2] * box[3])).max())


def _person_boxes(coco_gt, img_id, iscrowd):
    ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[config.PERSON_CAT_ID])
    anns = [a for a in coco_gt.loadAnns(ann_ids) if bool(a["iscrowd"]) == iscrowd]
    return np.array([a["bbox"] for a in anns], dtype=float).reshape(-1, 4)


def label_detections(coco_gt, detections, img_ids,
                     conf_thr=config.CALIB_CONF, iou_thr=config.MATCH_IOU):
    """Greedy TP/FP/ignored labelling of detections with score >= conf_thr (COCO crowd rule).

    Per image, detections are processed in descending confidence; each one is
    matched to the highest-IoU still-unmatched non-crowd person GT, and is a TP
    if that IoU >= iou_thr. An unmatched detection whose intersection / own area
    with any crowd person GT is >= iou_thr is ignored (neither TP nor FP).
    Returns a list of (detection, label) with label in {"TP", "FP", "ignored"}.
    """
    by_image = {}
    for d in detections:
        if d["score"] >= conf_thr:
            by_image.setdefault(d["image_id"], []).append(d)

    labeled = []
    for img_id in img_ids:
        dets = sorted(by_image.get(img_id, []), key=lambda d: -d["score"])
        if not dets:
            continue
        gts = _person_boxes(coco_gt, img_id, iscrowd=False)
        crowds = _person_boxes(coco_gt, img_id, iscrowd=True)
        matched = np.zeros(len(gts), dtype=bool)
        for d in dets:
            box = np.array(d["bbox"], dtype=float)
            tp = False
            if len(gts):
                ious = iou_xywh(box, gts)
                ious[matched] = -1.0
                best = int(np.argmax(ious))
                if ious[best] >= iou_thr:
                    matched[best] = True
                    tp = True
            if tp:
                labeled.append((d, "TP"))
            elif overlap_with_crowd(box, crowds) >= iou_thr:
                labeled.append((d, "ignored"))
            else:
                labeled.append((d, "FP"))
    return labeled


def match_detections(coco_gt, detections, img_ids,
                     conf_thr=config.CALIB_CONF, iou_thr=config.MATCH_IOU):
    """Returns (confidences, is_tp, n_ignored, labeled) for the non-ignored detections."""
    labeled = label_detections(coco_gt, detections, img_ids, conf_thr, iou_thr)
    kept = [(d, label) for d, label in labeled if label != "ignored"]
    confs = np.array([d["score"] for d, _ in kept], dtype=float)
    is_tp = np.array([label == "TP" for _, label in kept], dtype=bool)
    return confs, is_tp, len(labeled) - len(kept), labeled


def d_ece(confs, is_tp, n_bins=config.N_BINS):
    """D-ECE over equal-width bins on [0, 1]; returns (value, per-bin rows)."""
    n = len(confs)
    if n == 0:
        return float("nan"), []
    bin_idx = np.minimum((confs * n_bins).astype(int), n_bins - 1)
    total, rows = 0.0, []
    for b in range(n_bins):
        mask = bin_idx == b
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        mean_conf = float(confs[mask].mean())
        precision = float(is_tp[mask].mean())
        total += (n_bin / n) * abs(mean_conf - precision)
        rows.append({"bin": f"[{b / n_bins:.1f},{(b + 1) / n_bins:.1f})", "n": n_bin,
                     "mean_conf": mean_conf, "precision": precision})
    return float(total), rows


def signed_cal_gap(bins):
    """Sum over bins of (n_bin / N) * (precision - mean_conf); > 0 underconfident, < 0 overconfident."""
    n = sum(b["n"] for b in bins)
    if n == 0:
        return float("nan")
    return float(sum(b["n"] / n * (b["precision"] - b["mean_conf"]) for b in bins))


def count_gt(coco_gt, img_ids):
    """Number of non-crowd person GT boxes over img_ids."""
    return len(coco_gt.getAnnIds(imgIds=list(img_ids), catIds=[config.PERSON_CAT_ID], iscrowd=False))


def summarize(coco_gt, detections, img_ids):
    map50, map50_95 = coco_map(coco_gt, detections, img_ids)
    confs, is_tp, n_ignored, labeled = match_detections(coco_gt, detections, img_ids)
    ece, bins = d_ece(confs, is_tp)
    n_gt = count_gt(coco_gt, img_ids)
    row = {
        "n_images": len(img_ids),
        "mAP50": map50,
        "mAP50_95": map50_95,
        "d_ece": ece,
        "n_tp": int(is_tp.sum()),
        "n_fp": int((~is_tp).sum()),
        "n_ignored": n_ignored,
        "mean_conf_tp": float(confs[is_tp].mean()) if is_tp.any() else float("nan"),
        "mean_conf_fp": float(confs[~is_tp].mean()) if (~is_tp).any() else float("nan"),
        "n_gt": n_gt,
        "recall_025": int(is_tp.sum()) / n_gt if n_gt else float("nan"),
        "signed_cal_gap": signed_cal_gap(bins),
    }
    # recall with lower confidence thresholds: are missed people present at low confidence?
    precision = {}
    for col, thr in config.RECALL_CONFS.items():
        labels = [label for _, label in label_detections(coco_gt, detections, img_ids, conf_thr=thr)]
        n_tp_thr, n_fp_thr = labels.count("TP"), labels.count("FP")
        row[col] = n_tp_thr / n_gt if n_gt else float("nan")
        precision[col] = n_tp_thr / (n_tp_thr + n_fp_thr) if n_tp_thr + n_fp_thr else float("nan")
    n_kept = row["n_tp"] + row["n_fp"]
    row["precision_005"] = precision["recall_005"]
    row["precision_010"] = precision["recall_010"]
    row["precision_025"] = row["n_tp"] / n_kept if n_kept else float("nan")
    # share of people missed at 0.25 that are matched at 0.05
    row["recoverable_share"] = ((row["recall_005"] - row["recall_025"]) / (1 - row["recall_025"])
                                if row["recall_025"] < 1 else float("nan"))
    return row, bins, labeled
