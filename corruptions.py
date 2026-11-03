"""Deterministic image corruptions and fisheye ground-truth warping for stage "shift"."""
import contextlib
import io
import math

import cv2
import numpy as np
from pycocotools.coco import COCO

import config


def _to_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def gaussian_blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def gaussian_noise(img, std, image_id):
    """Additive N(0, std) noise on [0, 1]-scaled pixels, seeded by (SEED, image_id)."""
    rng = np.random.default_rng([config.SEED, image_id])
    x = img.astype(np.float64) / 255.0 + rng.normal(0.0, std, size=img.shape)
    return _to_uint8(np.clip(x, 0.0, 1.0) * 255.0)


def low_light(img, factor):
    return _to_uint8(img.astype(np.float64) * factor)


def jpeg(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"JPEG encoding failed at quality {quality}")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def fisheye_focal(width, height, theta_max_deg):
    """Focal length f such that the half-diagonal R maps to theta_max: f = R / theta_max."""
    return 0.5 * math.hypot(width, height) / math.radians(theta_max_deg)


_FISHEYE_MAPS = {}


def _fisheye_maps(width, height, theta_max_deg):
    """cv2.remap maps: output radius r_d -> theta = r_d / f -> source radius r_u = f * tan(theta)."""
    key = (width, height, theta_max_deg, config.FISHEYE_MAX_THETA_DEG)
    if key not in _FISHEYE_MAPS:
        f = fisheye_focal(width, height, theta_max_deg)
        cx, cy = width / 2.0, height / 2.0
        # continuous coordinates of output pixel centres, relative to the image centre
        dx, dy = np.meshgrid(np.arange(width) + 0.5 - cx, np.arange(height) + 0.5 - cy)
        r_d = np.hypot(dx, dy)
        theta = r_d / f
        valid = theta < math.radians(config.FISHEYE_MAX_THETA_DEG)
        r_u = f * np.tan(np.where(valid, theta, 0.0))
        scale = np.divide(r_u, r_d, out=np.ones_like(r_d), where=r_d > 0)
        # invalid pixels point far outside the source so BORDER_CONSTANT makes them black
        map_x = np.where(valid, cx + dx * scale - 0.5, -10.0).astype(np.float32)
        map_y = np.where(valid, cy + dy * scale - 0.5, -10.0).astype(np.float32)
        _FISHEYE_MAPS[key] = (map_x, map_y)
    return _FISHEYE_MAPS[key]


def fisheye(img, theta_max_deg):
    h, w = img.shape[:2]
    map_x, map_y = _fisheye_maps(w, h, theta_max_deg)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def apply(corruption, img, im):
    """Apply one SHIFT_CONDITIONS entry to a BGR uint8 image of subset entry `im`."""
    family = corruption["family"]
    if family == "gaussian_blur":
        return gaussian_blur(img, corruption["sigma"])
    if family == "gaussian_noise":
        return gaussian_noise(img, corruption["std"], im["id"])
    if family == "low_light":
        return low_light(img, corruption["factor"])
    if family == "jpeg":
        return jpeg(img, corruption["quality"])
    if family == "fisheye":
        return fisheye(img, corruption["theta_max_deg"])
    raise ValueError(f"unknown corruption family {family!r}")


def fisheye_forward(xs, ys, width, height, f):
    """Map source points to fisheye output: r_u -> theta = atan(r_u / f) -> r_d = f * theta."""
    cx, cy = width / 2.0, height / 2.0
    dx, dy = xs - cx, ys - cy
    r_u = np.hypot(dx, dy)
    r_d = f * np.arctan(r_u / f)
    scale = np.divide(r_d, r_u, out=np.ones_like(r_u), where=r_u > 0)
    return cx + dx * scale, cy + dy * scale


def warp_box(bbox, width, height, f):
    """Warp a COCO [x, y, w, h] box; returns the clipped bounding box or None if too small."""
    x, y, w, h = bbox
    t = np.linspace(0.0, 1.0, config.FISHEYE_EDGE_POINTS)
    xs = np.concatenate([x + t * w, x + t * w, np.full_like(t, x), np.full_like(t, x + w)])
    ys = np.concatenate([np.full_like(t, y), np.full_like(t, y + h), y + t * h, y + t * h])
    wx, wy = fisheye_forward(xs, ys, width, height, f)
    x1, x2 = max(0.0, wx.min()), min(float(width), wx.max())
    y1, y2 = max(0.0, wy.min()), min(float(height), wy.max())
    if x2 - x1 < config.FISHEYE_MIN_BOX or y2 - y1 < config.FISHEYE_MIN_BOX:
        return None
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def fisheye_coco(coco_gt, images, theta_max_deg):
    """Person-only COCO ground truth (crowd boxes included) warped into fisheye space.

    Returns (COCO object, per-image rows with f and box/drop counts).
    """
    anns, per_image = [], []
    for im in images:
        f = fisheye_focal(im["width"], im["height"], theta_max_deg)
        n_boxes = n_dropped = n_dropped_crowd = 0
        ann_ids = coco_gt.getAnnIds(imgIds=[im["id"]], catIds=[config.PERSON_CAT_ID])
        for a in coco_gt.loadAnns(ann_ids):
            n_boxes += 1
            box = warp_box(a["bbox"], im["width"], im["height"], f)
            if box is None:
                n_dropped += 1
                n_dropped_crowd += int(bool(a["iscrowd"]))
                continue
            old_area = a["bbox"][2] * a["bbox"][3]
            new_area = box[2] * box[3]
            # keep the original segment-to-box area ratio (only affects size-range breakdowns)
            area = a["area"] * new_area / old_area if old_area > 0 else new_area
            anns.append({"id": a["id"], "image_id": a["image_id"], "category_id": a["category_id"],
                         "iscrowd": a["iscrowd"], "bbox": box, "area": float(area)})
        per_image.append({"image_id": im["id"], "width": im["width"], "height": im["height"],
                          "f": f, "n_boxes": n_boxes, "n_dropped": n_dropped,
                          "n_dropped_crowd": n_dropped_crowd})
    coco = COCO()
    coco.dataset = {
        "images": coco_gt.loadImgs([im["id"] for im in images]),
        "categories": coco_gt.loadCats([config.PERSON_CAT_ID]),
        "annotations": anns,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco, per_image
