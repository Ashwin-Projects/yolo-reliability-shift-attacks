"""COCO val2017 download, person-subset selection and image fetching."""
import json
import time
import urllib.request
import zipfile

from pycocotools.coco import COCO
from tqdm import tqdm

import config


def _download(url, dest, retries=3):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            return
        except Exception as exc:  # network errors: retry, then re-raise
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            time.sleep(2 * attempt)


def ensure_annotations():
    """Download and extract instances_val2017.json if it is not present."""
    if config.ANN_FILE.exists():
        return config.ANN_FILE
    if not config.ANN_ZIP.exists():
        print(f"Downloading {config.ANN_ZIP_URL}")
        _download(config.ANN_ZIP_URL, config.ANN_ZIP)
    with zipfile.ZipFile(config.ANN_ZIP) as zf:
        zf.extract("annotations/instances_val2017.json", config.COCO_DIR)
    return config.ANN_FILE


def load_coco():
    return COCO(str(config.ANN_FILE))


def select_subset(coco):
    """First N_IMAGES val2017 images (by id) with >=1 non-crowd person annotation."""
    selected = []
    for img_id in sorted(coco.getImgIds()):
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[config.PERSON_CAT_ID], iscrowd=False)
        if ann_ids:
            selected.append(img_id)
        if len(selected) == config.N_IMAGES:
            break
    images = []
    for img in coco.loadImgs(selected):
        images.append({
            "id": img["id"],
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "coco_url": img["coco_url"],
        })
    return images


def validate_subset(saved, expected):
    """Raise if the saved subset differs from a fresh selection in count, order or ids."""
    saved_ids = [im["id"] for im in saved]
    expected_ids = [im["id"] for im in expected]
    if saved_ids == expected_ids:
        return
    problems = []
    if len(saved_ids) != len(expected_ids):
        problems.append(f"count {len(saved_ids)} saved vs {len(expected_ids)} expected")
    missing = sorted(set(expected_ids) - set(saved_ids))
    extra = sorted(set(saved_ids) - set(expected_ids))
    if missing:
        problems.append(f"{len(missing)} expected ids missing (first: {missing[:5]})")
    if extra:
        problems.append(f"{len(extra)} unexpected ids present (first: {extra[:5]})")
    if not missing and not extra:
        first = next(i for i, (a, b) in enumerate(zip(saved_ids, expected_ids)) if a != b)
        problems.append(f"same ids but different order; first difference at position {first}: "
                        f"saved {saved_ids[first]} vs expected {expected_ids[first]}")
    raise ValueError(f"{config.SUBSET_FILE} does not match select_subset(): " + "; ".join(problems)
                     + ". Delete the file to regenerate it.")


def load_or_create_subset(coco):
    images = select_subset(coco)
    if config.SUBSET_FILE.exists():
        with open(config.SUBSET_FILE) as f:
            saved = json.load(f)
        validate_subset(saved, images)
        return saved
    config.SUBSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SUBSET_FILE, "w") as f:
        json.dump(images, f, indent=1)
    print(f"Saved {len(images)} images to {config.SUBSET_FILE}")
    return images


def ensure_images(images):
    """Download any subset image missing from data/coco/val2017 (official COCO URLs)."""
    missing = [im for im in images if not (config.IMG_DIR / im["file_name"]).exists()]
    for im in tqdm(missing, desc="Downloading images", disable=not missing):
        _download(im["coco_url"], config.IMG_DIR / im["file_name"])
    return [config.IMG_DIR / im["file_name"] for im in images]
