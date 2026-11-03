"""Central configuration for detconf. All paths are relative to this file's folder."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Data
DATA_DIR = ROOT / "data"
COCO_DIR = DATA_DIR / "coco"
ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANN_ZIP = COCO_DIR / "annotations_trainval2017.zip"
ANN_FILE = COCO_DIR / "annotations" / "instances_val2017.json"
IMG_DIR = COCO_DIR / "val2017"
SUBSET_FILE = DATA_DIR / "subset.json"
N_IMAGES = 500
PERSON_CAT_ID = 1  # COCO category id for "person"

# Model / inference
SEED = 0
DEVICE = "cpu"
MODEL_WEIGHTS = "yolo11n.pt"
WEIGHTS_DIR = ROOT / "weights"
IMGSZ = 640
EVAL_CONF = 0.001
YOLO_PERSON_CLS = 0  # index of "person" in the ultralytics COCO class list

# Metrics
CALIB_CONF = 0.25  # detections used for D-ECE / TP / FP counts
MATCH_IOU = 0.5
N_BINS = 10

# Outputs
RESULTS_DIR = ROOT / "results"
PREDS_DIR = RESULTS_DIR / "preds"
VIS_DIR = RESULTS_DIR / "vis"
TABLE_FILE = RESULTS_DIR / "table.csv"
SMOKE_TABLE_FILE = RESULTS_DIR / "table_smoke.csv"  # --limit runs only
DETS_DIR = RESULTS_DIR / "dets"  # per-detection labels and radial distance
BINS_DIR = RESULTS_DIR / "bins"  # per-condition D-ECE bin breakdown
FISHEYE_DIR = RESULTS_DIR / "fisheye"  # per-image focal length and dropped GT counts
N_VIS = 5
N_VIS_SHIFT = 3

# Distribution shift conditions (stage "shift"); severities in increasing order per family
SHIFT_CONDITIONS = {
    "gaussian_blur_s1": {"family": "gaussian_blur", "sigma": 1.0},
    "gaussian_blur_s2": {"family": "gaussian_blur", "sigma": 2.0},
    "gaussian_blur_s3": {"family": "gaussian_blur", "sigma": 4.0},
    "gaussian_noise_s1": {"family": "gaussian_noise", "std": 0.04},
    "gaussian_noise_s2": {"family": "gaussian_noise", "std": 0.08},
    "gaussian_noise_s3": {"family": "gaussian_noise", "std": 0.12},
    "low_light_s1": {"family": "low_light", "factor": 0.5},
    "low_light_s2": {"family": "low_light", "factor": 0.3},
    "low_light_s3": {"family": "low_light", "factor": 0.15},
    "jpeg_s1": {"family": "jpeg", "quality": 30},
    "jpeg_s2": {"family": "jpeg", "quality": 15},
    "jpeg_s3": {"family": "jpeg", "quality": 5},
    "fisheye_s1": {"family": "fisheye", "theta_max_deg": 60.0},
    "fisheye_s2": {"family": "fisheye", "theta_max_deg": 80.0},
}
FISHEYE_MAX_THETA_DEG = 89.0  # output pixels at or beyond this angle are black
FISHEYE_EDGE_POINTS = 20  # samples per GT box edge for the forward warp
FISHEYE_MIN_BOX = 4  # warped GT boxes narrower or shorter than this (px) are dropped

TABLE_COLUMNS = [
    "condition", "n_images", "mAP50", "mAP50_95", "d_ece",
    "n_tp", "n_fp", "n_ignored", "mean_conf_tp", "mean_conf_fp",
    "n_gt", "recall_025", "signed_cal_gap", "recall_005", "recall_010",
    "precision_005", "precision_010", "precision_025", "recoverable_share",
]

# Stage "reliability"
RECALL_CONFS = {"recall_005": 0.05, "recall_010": 0.10}  # low-threshold recall columns
# Augmented views of each condition image (applied after the corruption)
VIEWS = {
    "hflip": {"flip": True, "imgsz": 640},
    "imgsz576": {"flip": False, "imgsz": 576},
}
MISS_HALF = 0.5  # image-level target: miss_rate >= MISS_HALF
RISK_COVERAGE_CONDITIONS = ["clean", "gaussian_blur_s3", "fisheye_s2"]

# Stage "attack": first ATTACK_N_IMAGES subset images; perturbations in 1/255 units
ATTACK_N_IMAGES = 100
ATTACK_CLEAN = "clean_100"
ATTACK_CONDITIONS = {
    "rand_eps8": {"family": "uniform_noise", "eps_255": 8, "seed": SEED},
    "pgd_eps4": {"family": "pgd_vanish", "eps_255": 4, "alpha_255": 1, "steps": 20,
                 "random_start": False, "loss": "topk_person_scores", "top_k": 100,
                 "best_iterate": True},
    "pgd_eps8": {"family": "pgd_vanish", "eps_255": 8, "alpha_255": 1, "steps": 20,
                 "random_start": False, "loss": "topk_person_scores", "top_k": 100,
                 "best_iterate": True},
}
# letterbox used inside the attack forward pass (matches model.predict, which sets rect=True)
ATTACK_LETTERBOX = {"auto": True, "stride": 32, "pad_value": 114, "resize": "bilinear"}
ATTACK_GRADCHECK_PASS_RATIO = 0.5  # best loss must be <= this share of the starting loss
ADV_DIR = RESULTS_DIR / "adv"  # saved (uint8 PNG) attacked images per condition
ATTACK_DETECT_POSITIVE = "pgd_eps8"
ATTACK_DETECT_NEGATIVES = ["clean_100", "rand_eps8"]
ATTACK_DETECT_SIGNALS = ["view_disagreement", "mean_conf"]
