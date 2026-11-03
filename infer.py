"""YOLO inference with on-disk caching of person detections in COCO results format."""
import json
import os
import random

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

import config


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model():
    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    # ultralytics downloads missing weights into the current directory
    cwd = os.getcwd()
    os.chdir(config.WEIGHTS_DIR)
    try:
        model = YOLO(config.MODEL_WEIGHTS)
    finally:
        os.chdir(cwd)
    return model


def predict_persons(model, images, transform=None, view=None):
    """Run the model on each image and keep person boxes as COCO [x, y, w, h] dicts.

    `transform(img_bgr, im)`, if given, corrupts each image before inference.
    `view`, if given (a config.VIEWS entry), sets imgsz and optionally flips the
    (corrupted) image horizontally; flipped boxes are mapped back to the
    original orientation.
    """
    imgsz = config.IMGSZ if view is None else view["imgsz"]
    flip = view is not None and view["flip"]
    detections = []
    for im in tqdm(images, desc="Inference"):
        path = config.IMG_DIR / im["file_name"]
        if transform is None and not flip:
            source = str(path)
        else:
            source = cv2.imread(str(path))
            if transform is not None:
                source = transform(source, im)
            if flip:
                source = cv2.flip(source, 1)
        result = model.predict(
            source, imgsz=imgsz, conf=config.EVAL_CONF,
            device=config.DEVICE, verbose=False,
        )[0]
        h, w = result.orig_shape
        assert (w, h) == (im["width"], im["height"]), f"size mismatch for {im['file_name']}"
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()  # already in original image pixels
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            if k != config.YOLO_PERSON_CLS:
                continue
            if flip:
                x1, x2 = w - x2, w - x1
            detections.append({
                "image_id": im["id"],
                "category_id": config.PERSON_CAT_ID,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(c),
            })
    return detections


def cache_metadata(condition, images, corruption=None, view_name=None):
    """Settings that must match for a cached prediction file to be reused."""
    view = None if view_name is None else config.VIEWS[view_name]
    meta = {
        "model": config.MODEL_WEIGHTS,
        "imgsz": config.IMGSZ if view is None else view["imgsz"],
        "conf": config.EVAL_CONF,
        "device": config.DEVICE,
        "condition": condition,
        "image_ids": sorted(im["id"] for im in images),
    }
    if corruption is not None:  # clean caches carry no corruption field
        meta["corruption"] = corruption
        # settings outside the condition definition that still change the corrupted image
        if corruption["family"] == "fisheye":
            meta["fisheye_max_theta_deg"] = config.FISHEYE_MAX_THETA_DEG
        if corruption["family"] == "gaussian_noise":
            meta["seed"] = config.SEED
    if view is not None:  # original-view caches carry no view field
        meta["view"] = {"name": view_name, **view}
    return meta


def load_cache(cache_path, meta):
    """Return cached detections if the stored metadata matches `meta`, else None."""
    with open(cache_path) as f:
        cached = json.load(f)
    stored = cached.get("meta") if isinstance(cached, dict) else None
    if stored is None:
        print(f"Cache {cache_path} has no metadata; recomputing")
        return None
    # symmetric: a key present on only one side (e.g. "view" on an augmented-view
    # cache vs an original-view request) is a mismatch, absence meaning None
    mismatched = sorted(k for k in set(meta) | set(stored) if stored.get(k) != meta.get(k))
    if mismatched:
        for k in mismatched:
            if k == "image_ids":
                print(f"Cache mismatch in image_ids: cached {len(stored.get(k) or [])} ids, "
                      f"requested {len(meta.get(k) or [])} ids (or different ids)")
            else:
                print(f"Cache mismatch in {k}: cached {stored.get(k)!r}, requested {meta.get(k)!r}")
        print("Recomputing predictions")
        return None
    print(f"Loading cached predictions from {cache_path}")
    return cached["detections"]


def get_predictions(images, cache_path, condition, force=False, corruption=None, transform=None,
                    view_name=None, cache_only=False):
    """Return cached detections, computing (and caching) them if absent, stale or forced.

    With cache_only=True a missing or stale cache raises instead of running inference.
    """
    meta = cache_metadata(condition, images, corruption, view_name)
    if cache_path.exists() and not force:
        detections = load_cache(cache_path, meta)
        if detections is not None:
            return detections
    if cache_only:
        raise RuntimeError(f"No valid cached predictions at {cache_path}; run the stage that creates them")
    set_seed()
    model = load_model()
    view = None if view_name is None else config.VIEWS[view_name]
    detections = predict_persons(model, images, transform, view)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"meta": meta, "detections": detections}, f)
    print(f"Cached {len(detections)} person detections to {cache_path}")
    return detections
