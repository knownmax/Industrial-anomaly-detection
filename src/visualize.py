"""
Visualization utilities for PatchCore anomaly detection results.

Produces:
  - anomaly_grid.png    : 8 anomalous + 8 normal samples with heatmap overlays
  - roc_curves.png      : Image AUROC | Pixel AUROC | PRO curve (3-subplot figure)
  - score_distribution.png : KDE of anomaly scores, normal vs anomalous
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import cv2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _denormalize(tensor_img) -> np.ndarray:
    """Convert [3, H, W] normalised tensor to [H, W, 3] uint8 numpy image."""
    img = tensor_img.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def _heatmap_overlay(img_np: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay jet colormap anomaly map onto RGB image."""
    score_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    heatmap = (cm.jet(score_norm)[:, :, :3] * 255).astype(np.uint8)
    overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    return overlay


def save_anomaly_grid(
    test_dataset,
    anomaly_maps: List[np.ndarray],
    scores: List[float],
    labels: List[int],
    save_path: str,
    n: int = 16,
) -> None:
    """
    Select 8 anomalous + 8 normal samples.
    For each: original image | GT mask | anomaly heatmap overlay.
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)

    anomaly_idx = [i for i, l in enumerate(labels) if l == 1]
    normal_idx  = [i for i, l in enumerate(labels) if l == 0]

    n_each = n // 2
    # Sample uniformly if more samples than needed
    if len(anomaly_idx) > n_each:
        step = len(anomaly_idx) // n_each
        anomaly_idx = anomaly_idx[::step][:n_each]
    if len(normal_idx) > n_each:
        step = len(normal_idx) // n_each
        normal_idx = normal_idx[::step][:n_each]

    selected = anomaly_idx[:n_each] + normal_idx[:n_each]
    actual_n = len(selected)

    fig, axes = plt.subplots(actual_n, 3, figsize=(9, actual_n * 3))
    if actual_n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(selected):
        img_tensor, label, gt_mask_tensor, defect_type = test_dataset[idx]

        img_np   = _denormalize(img_tensor)
        gt_np    = gt_mask_tensor.squeeze().cpu().numpy()
        amap     = anomaly_maps[idx]
        score    = scores[idx]
        tag      = "ANOMALY" if label == 1 else "NORMAL"
        color    = "red"    if label == 1 else "green"

        overlay  = _heatmap_overlay(img_np, amap)

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"[{tag}] score={score:.3f}", color=color, fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("GT Mask", fontsize=8)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Anomaly Heatmap", fontsize=8)
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved anomaly grid → {save_path}")


def save_roc_curves(
    image_roc: dict,
    pixel_roc_auroc: float,
    pro_curve: dict,
    save_path: str,
) -> None:
    """
    3-subplot figure: Image AUROC | Pixel AUROC | PRO curve.
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Image AUROC ---
    ax = axes[0]
    ax.plot(image_roc["fpr"], image_roc["tpr"], lw=2,
            label=f'AUROC={image_roc["auroc"]:.3f}')
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    # Mark operating point (Youden's J)
    thr = image_roc["threshold"]
    fpr_arr, tpr_arr = image_roc["fpr"], image_roc["tpr"]
    j = tpr_arr - fpr_arr
    best = int(np.argmax(j))
    ax.scatter(fpr_arr[best], tpr_arr[best], marker="o", color="red", zorder=5,
               label=f"Threshold={thr:.3f}")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Image-level ROC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Pixel AUROC (text only — full pixel ROC arrays not stored to save memory) ---
    ax = axes[1]
    ax.text(0.5, 0.5, f"Pixel AUROC\n{pixel_roc_auroc:.4f}",
            ha="center", va="center", fontsize=20, transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
    ax.set_title("Pixel-level AUROC")
    ax.axis("off")

    # --- PRO curve ---
    ax = axes[2]
    fpr_pro = pro_curve["fpr_array"]
    pro_pro = pro_curve["pro_array"]
    ax.plot(fpr_pro, pro_pro, lw=2, label=f'PRO-AUC={pro_curve["pro_auc"]:.3f}')
    ax.axvline(x=0.3, color="red", linestyle="--", lw=1, label="FPR=0.3 cutoff")
    ax.set_xlabel("Mean FPR")
    ax.set_ylabel("Mean Per-Region Overlap")
    ax.set_title("PRO Curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ROC curves    → {save_path}")


def save_score_distribution(
    scores: List[float],
    labels: List[int],
    save_path: str,
) -> None:
    """KDE plot of anomaly score distributions: normal vs anomalous with overlap shading."""
    os.makedirs(Path(save_path).parent, exist_ok=True)

    scores = np.array(scores)
    labels = np.array(labels)
    normal_scores   = scores[labels == 0]
    anomaly_scores  = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(normal_scores) > 1:
        sns.kdeplot(normal_scores, ax=ax, label="Normal", color="green", fill=True, alpha=0.4)
    if len(anomaly_scores) > 1:
        sns.kdeplot(anomaly_scores, ax=ax, label="Anomalous", color="red", fill=True, alpha=0.4)

    ax.set_xlabel("Anomaly Score (max patch distance)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: Normal vs Anomalous")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved score dist    → {save_path}")
