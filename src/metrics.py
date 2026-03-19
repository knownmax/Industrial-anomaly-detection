"""
Evaluation metrics for PatchCore on MVTec AD.

1. Image-level AUROC  — standard ROC-AUC on image anomaly scores
2. Pixel-level AUROC  — ROC-AUC on all pixel-wise anomaly scores vs GT masks
3. PRO score          — Per-Region Overlap (MVTec official metric)
                        Averaged over connected components, integrated up to FPR=0.3
"""

from typing import List, Tuple, Dict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from skimage.measure import label as skimage_label


# ---------------------------------------------------------------------------
# 1. Image-level AUROC
# ---------------------------------------------------------------------------

def compute_image_auroc(
    labels: List[int],
    scores: List[float],
) -> Dict:
    """
    Compute image-level AUROC and optimal threshold (Youden's J).

    Returns dict with keys: auroc, threshold, fpr, tpr
    """
    labels = np.array(labels, dtype=int)
    scores = np.array(scores, dtype=float)

    auroc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Youden's J = TPR - FPR
    j_stat = tpr - fpr
    best_idx = int(np.argmax(j_stat))
    threshold = float(thresholds[best_idx])

    return {
        "auroc": float(auroc),
        "threshold": threshold,
        "fpr": fpr,
        "tpr": tpr,
    }


# ---------------------------------------------------------------------------
# 2. Pixel-level AUROC
# ---------------------------------------------------------------------------

def compute_pixel_auroc(
    gt_masks: List[np.ndarray],      # each [H, W], binary
    anomaly_maps: List[np.ndarray],  # each [H, W], float scores
) -> float:
    """
    Flatten all GT masks and anomaly maps, compute pixel-level AUROC.
    Good samples (zero GT mask) are included — their near-zero anomaly scores
    contribute true negatives correctly.
    """
    all_gt = np.concatenate([m.flatten() for m in gt_masks])
    all_sc = np.concatenate([m.flatten() for m in anomaly_maps])

    all_gt = (all_gt > 0.5).astype(int)

    if all_gt.sum() == 0:
        return 0.0

    return float(roc_auc_score(all_gt, all_sc))


# ---------------------------------------------------------------------------
# 3. PRO score (Per-Region Overlap)
# ---------------------------------------------------------------------------

def compute_pro_score(
    gt_masks: List[np.ndarray],       # each [H, W], binary (float 0/1)
    anomaly_maps: List[np.ndarray],   # each [H, W], float scores
    fpr_limit: float = 0.3,
    num_thresholds: int = 100,
) -> Dict:
    """
    Compute the PRO (Per-Region Overlap) curve and integrate up to fpr_limit.

    Steps:
      For each threshold t in [0, max_score]:
        - For each image with defects:
            - For each connected component in GT mask:
                overlap = (anomaly_map[t] & component).sum() / component.sum()
            - Average overlap over all components in this image
        - Average over all defective images → mean_pro at t
        - Compute mean_fpr at t: FP / total_normal_pixels across all normal pixels

    Returns dict with keys: pro_auc, fpr_array, pro_array
    """
    max_score = max(m.max() for m in anomaly_maps) + 1e-8
    thresholds = np.linspace(0, max_score, num_thresholds)

    # Pre-compute: which images have defects, their connected components
    defect_images = []   # list of (anomaly_map, list of component_masks)
    normal_pixels_per_image = []

    for gt, amap in zip(gt_masks, anomaly_maps):
        gt_bin = (gt > 0.5).astype(np.uint8)
        if gt_bin.sum() == 0:
            # Normal image — contributes to FPR denominator
            normal_pixels_per_image.append(amap)
            continue

        # Label connected components
        labeled = skimage_label(gt_bin, connectivity=2)
        n_components = labeled.max()
        if n_components == 0:
            normal_pixels_per_image.append(amap)
            continue

        components = []
        for comp_id in range(1, n_components + 1):
            comp_mask = (labeled == comp_id)
            components.append(comp_mask)

        defect_images.append((amap, components))

    if len(defect_images) == 0:
        return {"pro_auc": 0.0, "fpr_array": np.zeros(num_thresholds),
                "pro_array": np.zeros(num_thresholds)}

    # Total normal pixels (from images without any defect)
    total_normal_pixels = sum(m.size for m in normal_pixels_per_image)

    fpr_arr = np.zeros(num_thresholds)
    pro_arr = np.zeros(num_thresholds)

    for i, t in enumerate(thresholds):
        # --- PRO: per-region overlap averaged over components & images ---
        all_overlaps = []
        for amap, components in defect_images:
            pred_bin = (amap >= t)
            image_overlaps = []
            for comp in components:
                overlap = float(pred_bin[comp].sum()) / float(comp.sum())
                image_overlaps.append(overlap)
            all_overlaps.append(np.mean(image_overlaps))
        pro_arr[i] = np.mean(all_overlaps)

        # --- FPR: false positives on normal pixels ---
        if total_normal_pixels > 0:
            fp = sum((m >= t).sum() for m in normal_pixels_per_image)
            fpr_arr[i] = float(fp) / total_normal_pixels
        else:
            fpr_arr[i] = 0.0

    # Sort by FPR (ascending) for integration
    sort_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sort_idx]
    pro_sorted = pro_arr[sort_idx]

    # Integrate PRO curve up to fpr_limit, normalise by fpr_limit
    mask = fpr_sorted <= fpr_limit
    if mask.sum() < 2:
        pro_auc = 0.0
    else:
        pro_auc = float(np.trapz(pro_sorted[mask], fpr_sorted[mask])) / fpr_limit

    return {
        "pro_auc": pro_auc,
        "fpr_array": fpr_sorted,
        "pro_array": pro_sorted,
    }


# ---------------------------------------------------------------------------
# Convenience: run all metrics at once
# ---------------------------------------------------------------------------

def evaluate_all(
    labels: List[int],
    scores: List[float],
    gt_masks: List[np.ndarray],
    anomaly_maps: List[np.ndarray],
    pro_fpr_limit: float = 0.3,
    num_pro_thresholds: int = 100,
) -> Dict:
    img_metrics   = compute_image_auroc(labels, scores)
    px_auroc      = compute_pixel_auroc(gt_masks, anomaly_maps)
    pro_metrics   = compute_pro_score(
        gt_masks, anomaly_maps,
        fpr_limit=pro_fpr_limit,
        num_thresholds=num_pro_thresholds,
    )

    return {
        "img_auroc":   img_metrics["auroc"],
        "img_threshold": img_metrics["threshold"],
        "img_fpr":     img_metrics["fpr"],
        "img_tpr":     img_metrics["tpr"],
        "px_auroc":    px_auroc,
        "pro_auc":     pro_metrics["pro_auc"],
        "pro_fpr":     pro_metrics["fpr_array"],
        "pro_pro":     pro_metrics["pro_array"],
    }
