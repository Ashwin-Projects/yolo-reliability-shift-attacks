"""Stage "attack": differentiable letterbox, PGD vanishing attack on persons, noise control.

Images are perturbed at their original resolution as RGB float tensors in [0, 1],
then rounded to uint8 and saved as PNG. Evaluation reads the saved PNGs through
the normal inference pipeline.
"""
import hashlib
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics import YOLO

import config


def load_attack_model():
    """Raw torch model (not fused, eval mode, float32); only the input needs gradients."""
    model = YOLO(str(config.WEIGHTS_DIR / config.MODEL_WEIGHTS)).model
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def model_stride(model):
    stride = int(model.stride.max())
    assert stride == config.ATTACK_LETTERBOX["stride"], f"model stride {stride} != configured letterbox stride"
    return stride


def weights_sha256():
    return hashlib.sha256((config.WEIGHTS_DIR / config.MODEL_WEIGHTS).read_bytes()).hexdigest()


def generation_metadata(spec, images):
    """Everything that determines the saved attacked images; any mismatch regenerates them."""
    return {
        "spec": spec,  # eps, alpha, steps, loss, top_k, best_iterate / seed for rand_eps8
        "model": config.MODEL_WEIGHTS,
        "weights_sha256": weights_sha256(),
        "imgsz": config.IMGSZ,
        "letterbox": config.ATTACK_LETTERBOX,
        "image_ids": [im["id"] for im in images],
    }


def letterbox_params(h, w, stride, imgsz=config.IMGSZ):
    """Same geometry as ultralytics LetterBox(auto=True) used by model.predict (rect=True):
    scale r = min(imgsz/h, imgsz/w), pad each side to the next multiple of `stride`."""
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = np.mod(imgsz - new_w, stride) / 2, np.mod(imgsz - new_h, stride) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return new_h, new_w, top, bottom, left, right


def letterbox_tensor(x, stride):
    """Differentiable letterbox of a (1, 3, H, W) RGB tensor in [0, 1]."""
    h, w = x.shape[-2:]
    new_h, new_w, top, bottom, left, right = letterbox_params(h, w, stride)
    if (new_h, new_w) != (h, w):
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return F.pad(x, (left, right, top, bottom), value=config.ATTACK_LETTERBOX["pad_value"] / 255)


def raw_output(model, x, stride):
    """(1, 4 + n_classes, N) prediction tensor; eval-mode Detect returns (y, features)."""
    out = model(letterbox_tensor(x, stride))
    return out[0] if isinstance(out, (tuple, list)) else out


def person_scores(model, x, stride):
    return raw_output(model, x, stride)[0, 4 + config.YOLO_PERSON_CLS, :]


def attack_loss(model, x, stride, top_k):
    """Sum of the top_k person scores over all predictions."""
    scores = person_scores(model, x, stride)
    return torch.topk(scores, k=min(top_k, scores.numel())).values.sum()


def to_tensor(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def to_uint8_bgr(x):
    rgb = np.rint(x[0].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def pgd_vanish(model, x0, eps, alpha, steps, stride, top_k):
    """L-inf PGD descent on the top-k person-score loss, keeping the best iterate.

    losses[i] is the loss of the iterate after i steps (losses[0] = starting image),
    each measured after projection and clipping. Returns (best_x, losses, best_iter).
    """
    x = x0.clone()
    losses, best_x, best_iter = [], x0, 0
    for i in range(steps + 1):
        last = i == steps
        x.requires_grad_(not last)
        with torch.set_grad_enabled(not last):
            loss = attack_loss(model, x, stride, top_k)
        losses.append(float(loss.detach()))
        if losses[-1] < losses[best_iter]:
            best_x, best_iter = x.detach().clone(), i
        if last:
            break
        loss.backward()
        if x.grad is None:
            raise RuntimeError("gradient w.r.t. the input image is None")
        x = x.detach() - alpha * x.grad.sign()
        x = torch.min(torch.max(x, x0 - eps), x0 + eps).clamp(0.0, 1.0)
    return best_x, losses, best_iter


def uniform_noise(x0, eps, image_id, seed):
    rng = np.random.default_rng([seed, image_id])
    noise = torch.from_numpy(rng.uniform(-eps, eps, size=tuple(x0.shape))).float()
    return (x0 + noise).clamp(0.0, 1.0)


def perturb(model, spec, img_bgr, image_id, stride):
    """Apply one ATTACK_CONDITIONS spec; returns (uint8 BGR image, info dict)."""
    x0 = to_tensor(img_bgr)
    eps = spec["eps_255"] / 255.0
    if spec["family"] == "uniform_noise":
        return to_uint8_bgr(uniform_noise(x0, eps, image_id, spec["seed"])), {}
    if spec["family"] == "pgd_vanish":
        x, losses, best_iter = pgd_vanish(model, x0, eps, spec["alpha_255"] / 255.0, spec["steps"],
                                          stride, spec["top_k"])
        return to_uint8_bgr(x), {"loss_start": losses[0], "loss_best": losses[best_iter],
                                 "loss_last": losses[-1], "best_iter": best_iter}
    raise ValueError(f"unknown attack family {spec['family']!r}")


def png_path(adv_dir, im):
    return adv_dir / (im["file_name"].rsplit(".", 1)[0] + ".png")


def generate(spec, images, adv_dir, force=False):
    """Create the saved PNGs (and attack_info.csv) for one condition unless an identical set exists.

    Returns (per-image info DataFrame, seconds spent generating; 0 when reused).
    """
    meta_path, info_path = adv_dir / "meta.json", adv_dir / "attack_info.csv"
    meta = generation_metadata(spec, images)
    if not force and meta_path.exists() and info_path.exists():
        with open(meta_path) as f:
            stored = json.load(f)
        mismatched = sorted(k for k in set(meta) | set(stored) if stored.get(k) != meta.get(k))
        if not mismatched and all(png_path(adv_dir, im).exists() for im in images):
            print(f"Using saved images in {adv_dir}")
            return pd.read_csv(info_path), 0.0
        print(f"Saved images in {adv_dir} are stale (mismatch in {', '.join(mismatched) or 'files'}); "
              "regenerating")
    adv_dir.mkdir(parents=True, exist_ok=True)
    model = load_attack_model()
    stride = model_stride(model)
    infos, t0 = [], time.perf_counter()
    for im in tqdm(images, desc=f"Generating {adv_dir.name}"):
        img = cv2.imread(str(config.IMG_DIR / im["file_name"]))
        adv, info = perturb(model, spec, img, im["id"], stride)
        assert np.abs(adv.astype(int) - img.astype(int)).max() <= spec["eps_255"], "L-inf bound violated"
        cv2.imwrite(str(png_path(adv_dir, im)), adv)
        infos.append({"image_id": im["id"], **info})
    info = pd.DataFrame(infos)
    info.to_csv(info_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    elapsed = time.perf_counter() - t0
    print(f"Generated {len(images)} images in {adv_dir} ({elapsed:.1f}s)")
    return info, elapsed


def digest(adv_dir, images):
    """sha256 over the saved PNG bytes, in image order (part of the prediction-cache metadata)."""
    h = hashlib.sha256()
    for im in images:
        h.update(png_path(adv_dir, im).read_bytes())
    return h.hexdigest()


def load_saved(adv_dir, img, im):
    """Inference transform: replace the clean image with the saved attacked PNG."""
    adv = cv2.imread(str(png_path(adv_dir, im)))
    assert adv is not None and adv.shape == img.shape, f"bad saved image for {im['file_name']}"
    return adv


def perturbation_panel(orig_bgr, adv_bgr, path, gain=10):
    """Original | attacked | perturbation x gain (mid-grey = no change)."""
    delta = adv_bgr.astype(np.float64) - orig_bgr.astype(np.float64)
    amp = np.clip(127.5 + gain * delta, 0, 255).astype(np.uint8)
    tiles = []
    font, scale, thick, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, 6
    for img, label in ((orig_bgr, "original"), (adv_bgr, "pgd_eps8"), (amp, f"perturbation x{gain}")):
        tile = img.copy()
        # solid label box sized from the text metrics, so the text cannot misalign
        (tw, th), base = cv2.getTextSize(label, font, scale, thick)
        cv2.rectangle(tile, (0, 0), (tw + 2 * pad, th + base + 2 * pad), (0, 0, 0), thickness=-1)
        cv2.putText(tile, label, (pad, pad + th), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        tiles.append(tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack(tiles))


def count_persons(predictor, img_bgr):
    r = predictor.predict(img_bgr, imgsz=config.IMGSZ, conf=config.CALIB_CONF,
                          device=config.DEVICE, verbose=False)[0]
    return int((r.boxes.cls.cpu().numpy().astype(int) == config.YOLO_PERSON_CLS).sum())


def gradcheck(images):
    """Print output format, letterbox agreement, per-iteration losses and detection counts."""
    from ultralytics.data.augment import LetterBox

    import infer

    model = load_attack_model()
    stride = model_stride(model)
    predictor = infer.load_model()  # separate instance: predict() may fuse its own model
    ok = True
    for im in images:
        img = cv2.imread(str(config.IMG_DIR / im["file_name"]))
        x0 = to_tensor(img)
        with torch.no_grad():
            out = model(letterbox_tensor(x0, stride))
        y = out[0] if isinstance(out, (tuple, list)) else out
        cls = y[0, 4:, :]
        print(f"\nimage {im['id']} ({im['width']}x{im['height']}): model output type {type(out).__name__}, "
              f"prediction tensor shape {tuple(y.shape)}; class rows min {float(cls.min()):.4g} "
              f"max {float(cls.max()):.4g}")
        ref = LetterBox(config.IMGSZ, auto=True, stride=stride)(image=img)[..., ::-1] / 255.0
        mine = letterbox_tensor(x0, stride)[0].permute(1, 2, 0).numpy()
        same = ref.shape == mine.shape
        diff = float(np.abs(ref - mine).max()) if same else float("nan")
        print(f"letterbox: ultralytics {ref.shape[:2]}, differentiable {mine.shape[:2]}, "
              f"max abs pixel difference {diff:.3g}")
        ok &= same
        n_before = count_persons(predictor, img)
        for name, spec in config.ATTACK_CONDITIONS.items():
            if spec["family"] != "pgd_vanish":
                continue
            eps = spec["eps_255"] / 255.0
            x, losses, best_iter = pgd_vanish(model, x0, eps, spec["alpha_255"] / 255.0, spec["steps"],
                                              stride, spec["top_k"])
            adv = to_uint8_bgr(x)  # best iterate, rounded as when saved
            n_after = count_persons(predictor, adv)
            ratio = losses[best_iter] / losses[0]
            passed = ratio <= config.ATTACK_GRADCHECK_PASS_RATIO and n_after <= n_before
            print(f"{name}: loss per iteration (0 = start) " + ", ".join(f"{v:.3f}" for v in losses))
            print(f"{name}: best loss {losses[best_iter]:.3f} at iteration {best_iter} "
                  f"({ratio:.1%} of start); persons >= {config.CALIB_CONF}: before {n_before}, "
                  f"after {n_after}; max |adv - orig| = "
                  f"{int(np.abs(adv.astype(int) - img.astype(int)).max())}/255 -> {'PASS' if passed else 'FAIL'}")
            ok &= passed
    print(f"\nGRADIENT CHECK {'PASSED' if ok else 'FAILED'} (best loss <= "
          f"{config.ATTACK_GRADCHECK_PASS_RATIO:.0%} of start and persons not increasing, every image)")
    return ok
