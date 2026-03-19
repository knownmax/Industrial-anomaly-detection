"""
Demo: train + evaluate PatchCore on a single MVTec AD category.

Usage:
    python src/run_single.py --category bottle \
        --data_root /data/max/kaggle/industrial-anomaly-detection/anomaly_ds \
        --output_dir results
"""

import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

# Allow imports from this directory
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders
from patchcore import PatchCore
from metrics import evaluate_all
from visualize import save_anomaly_grid, save_roc_curves, save_score_distribution


def resolve_device(device: str) -> str:
    """
    Resolve a user-provided device string against currently visible CUDA devices.

    Handles CUDA_VISIBLE_DEVICES remapping safely by falling back to cuda:0 when
    an invalid ordinal is requested.
    """
    if device == "cpu":
        return "cpu"

    if not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available; falling back to CPU.")
        return "cpu"

    if device == "cuda":
        return "cuda:0"

    if device.startswith("cuda:"):
        try:
            idx = int(device.split(":", 1)[1])
        except ValueError:
            warnings.warn(f"Unrecognized device '{device}'; using cuda:0.")
            return "cuda:0"

        visible_count = torch.cuda.device_count()
        if idx < 0 or idx >= visible_count:
            warnings.warn(
                f"Requested device '{device}' is invalid for current visibility "
                f"(available cuda indices: 0..{max(visible_count - 1, 0)}). Using cuda:0."
            )
            return "cuda:0"
        return f"cuda:{idx}"

    warnings.warn(f"Unknown device '{device}'; using cuda:0.")
    return "cuda:0"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="PatchCore single-category demo")
    p.add_argument("--category",   type=str, required=True,
                   help="MVTec AD category name (e.g. 'bottle')")
    p.add_argument("--data_root",  type=str,
                   default="/data/max/kaggle/industrial-anomaly-detection/anomaly_ds")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--backbone",   type=str, default="wide_resnet101_2")
    p.add_argument("--coreset_ratio", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--faiss_gpu",  action="store_true", default=True)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run_category(
    category:      str,
    data_root:     str,
    output_dir:    str,
    backbone:      str = "wide_resnet101_2",
    coreset_ratio: float = 0.01,
    batch_size:    int = 32,
    faiss_gpu:     bool = True,
    seed:          int = 42,
    device:        str = "cuda",
) -> dict:
    """Full fit + eval pipeline for one category. Returns metrics dict."""

    seed_everything(seed)
    device = resolve_device(device)
    print(f"  Using device   : {device}")

    out_dir = Path(output_dir) / category
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Data                                                              #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"  Category : {category}")
    print(f"{'='*60}")
    train_loader, test_loader = get_dataloaders(
        data_root, category, batch_size=batch_size
    )
    print(f"  Train samples : {len(train_loader.dataset)}")
    print(f"  Test  samples : {len(test_loader.dataset)}")

    # ------------------------------------------------------------------ #
    # 2. Fit                                                               #
    # ------------------------------------------------------------------ #
    t0 = time.time()
    model = PatchCore(
        backbone=backbone,
        coreset_ratio=coreset_ratio,
        device=device,
        faiss_gpu=faiss_gpu,
    )
    model.fit(train_loader)
    fit_time = time.time() - t0
    bank_mb = (model.memory_bank.element_size() *
               model.memory_bank.numel() / 1e6)
    print(f"  Fit time      : {fit_time:.1f}s")
    print(f"  Memory bank   : {len(model.memory_bank):,} vectors ({bank_mb:.1f} MB)")

    # Save model
    model.save(str(out_dir / "patchcore_model.pt"))

    # ------------------------------------------------------------------ #
    # 3. Predict on all test images                                        #
    # ------------------------------------------------------------------ #
    test_ds = test_loader.dataset
    scores, labels, gt_masks, anomaly_maps = [], [], [], []

    t1 = time.time()
    for img_tensor, label, gt_mask_tensor, _ in tqdm(
        test_ds, desc=f"  Predicting [{category}]", unit="img"
    ):
        score, amap = model.predict(img_tensor.unsqueeze(0))
        scores.append(score)
        labels.append(int(label))
        gt_masks.append(gt_mask_tensor.squeeze().numpy())
        anomaly_maps.append(amap)

    pred_time = time.time() - t1
    print(f"  Prediction    : {pred_time:.1f}s  ({pred_time/len(test_ds)*1000:.1f} ms/img)")

    # ------------------------------------------------------------------ #
    # 4. Metrics                                                           #
    # ------------------------------------------------------------------ #
    results = evaluate_all(labels, scores, gt_masks, anomaly_maps)

    table = [
        ["Image AUROC",  f"{results['img_auroc']*100:.2f}%"],
        ["Pixel AUROC",  f"{results['px_auroc']*100:.2f}%"],
        ["PRO Score",    f"{results['pro_auc']*100:.2f}%"],
        ["Fit time",     f"{fit_time:.1f}s"],
        ["Memory bank",  f"{len(model.memory_bank):,} ({bank_mb:.1f} MB)"],
    ]
    print("\n" + tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    # ------------------------------------------------------------------ #
    # 5. Save metrics JSON                                                 #
    # ------------------------------------------------------------------ #
    metrics_json = {
        "category":    category,
        "img_auroc":   round(results["img_auroc"], 4),
        "px_auroc":    round(results["px_auroc"], 4),
        "pro_auc":     round(results["pro_auc"], 4),
        "bank_size":   int(len(model.memory_bank)),
        "bank_mb":     round(bank_mb, 2),
        "fit_time_s":  round(fit_time, 2),
        "pred_time_s": round(pred_time, 2),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Metrics saved → {out_dir}/metrics.json")

    # ------------------------------------------------------------------ #
    # 6. Visualizations                                                    #
    # ------------------------------------------------------------------ #
    save_anomaly_grid(
        test_ds, anomaly_maps, scores, labels,
        save_path=str(out_dir / "anomaly_grid.png"),
    )
    save_roc_curves(
        image_roc={
            "auroc": results["img_auroc"],
            "threshold": results["img_threshold"],
            "fpr": results["img_fpr"],
            "tpr": results["img_tpr"],
        },
        pixel_roc_auroc=results["px_auroc"],
        pro_curve={
            "pro_auc": results["pro_auc"],
            "fpr_array": results["pro_fpr"],
            "pro_array": results["pro_pro"],
        },
        save_path=str(out_dir / "roc_curves.png"),
    )
    save_score_distribution(
        scores, labels,
        save_path=str(out_dir / "score_distribution.png"),
    )

    total_time = time.time() - t0
    print(f"\n  Total wall time: {total_time:.1f}s")
    return metrics_json


def main():
    args = parse_args()
    run_category(
        category=args.category,
        data_root=args.data_root,
        output_dir=args.output_dir,
        backbone=args.backbone,
        coreset_ratio=args.coreset_ratio,
        batch_size=args.batch_size,
        faiss_gpu=args.faiss_gpu,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
