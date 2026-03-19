"""
Benchmark PatchCore on all 15 MVTec AD categories sequentially.

Usage:
    python src/run_all.py \
        --data_root /data/max/kaggle/industrial-anomaly-detection/anomaly_ds \
        --output_dir results
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))
from run_single import run_category, seed_everything

CATEGORIES = [
    "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def parse_args():
    p = argparse.ArgumentParser(description="PatchCore all-category benchmark")
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
    p.add_argument("--categories", nargs="+", default=CATEGORIES,
                   help="Subset of categories to run (default: all 15)")
    return p.parse_args()


def save_summary_table_figure(df: pd.DataFrame, save_path: str) -> None:
    """Render a publication-ready summary table as a PNG figure."""
    fig, ax = plt.subplots(figsize=(10, 0.5 * (len(df) + 2)))
    ax.axis("off")

    # Format values as percentages
    display_df = df.copy()
    for col in ["img_auroc", "px_auroc", "pro_auc"]:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x*100:.1f}" if isinstance(x, float) else x
        )

    col_labels = ["Category", "Image AUROC (%)", "Pixel AUROC (%)", "PRO Score (%)"]
    table_data = display_df[["category", "img_auroc", "px_auroc", "pro_auc"]].values.tolist()

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Highlight header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight mean row
    mean_row = len(table_data)
    for j in range(len(col_labels)):
        table[mean_row, j].set_facecolor("#ecf0f1")
        table[mean_row, j].set_text_props(fontweight="bold")

    plt.title("PatchCore (WideResNet-101-2) on MVTec AD", fontsize=13,
              fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary table figure saved → {save_path}")


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    t_start = time.time()

    for cat in args.categories:
        try:
            metrics = run_category(
                category=cat,
                data_root=args.data_root,
                output_dir=args.output_dir,
                backbone=args.backbone,
                coreset_ratio=args.coreset_ratio,
                batch_size=args.batch_size,
                faiss_gpu=args.faiss_gpu,
                seed=args.seed,
                device=args.device,
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\n[ERROR] Category '{cat}' failed: {e}")
            all_metrics.append({
                "category": cat,
                "img_auroc": float("nan"),
                "px_auroc":  float("nan"),
                "pro_auc":   float("nan"),
            })
        finally:
            # Free GPU memory before next category
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time = time.time() - t_start
    print(f"\n\nTotal benchmark time: {total_time/60:.1f} min")

    # ------------------------------------------------------------------ #
    # Aggregate results                                                    #
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(all_metrics)

    # Compute mean ± std row
    numeric_cols = ["img_auroc", "px_auroc", "pro_auc"]
    means = df[numeric_cols].mean()
    stds  = df[numeric_cols].std()

    mean_row = {"category": "Mean ± Std"}
    for col in numeric_cols:
        mean_row[col] = f"{means[col]*100:.1f} ± {stds[col]*100:.1f}"

    # Print console summary
    print("\n" + "="*70)
    print("  BENCHMARK SUMMARY")
    print("="*70)
    table_rows = []
    for _, row in df.iterrows():
        table_rows.append([
            row["category"],
            f"{row['img_auroc']*100:.1f}" if not pd.isna(row.get("img_auroc", float("nan"))) else "ERR",
            f"{row['px_auroc']*100:.1f}"  if not pd.isna(row.get("px_auroc",  float("nan"))) else "ERR",
            f"{row['pro_auc']*100:.1f}"   if not pd.isna(row.get("pro_auc",   float("nan"))) else "ERR",
        ])
    table_rows.append([
        "Mean ± Std",
        f"{means['img_auroc']*100:.1f} ± {stds['img_auroc']*100:.1f}",
        f"{means['px_auroc']*100:.1f}  ± {stds['px_auroc']*100:.1f}",
        f"{means['pro_auc']*100:.1f}   ± {stds['pro_auc']*100:.1f}",
    ])
    print(tabulate(
        table_rows,
        headers=["Category", "Img AUROC (%)", "Px AUROC (%)", "PRO (%)"],
        tablefmt="fancy_grid",
    ))

    # Save CSV
    csv_path = output_dir / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved → {csv_path}")

    # Save figure
    png_path = output_dir / "summary_table.png"
    # Add mean row to df for figure (as string)
    display_df = df[["category", "img_auroc", "px_auroc", "pro_auc"]].copy()
    mean_display = pd.DataFrame([{
        "category": "Mean",
        "img_auroc": means["img_auroc"],
        "px_auroc":  means["px_auroc"],
        "pro_auc":   means["pro_auc"],
    }])
    display_df = pd.concat([display_df, mean_display], ignore_index=True)
    save_summary_table_figure(display_df, str(png_path))


if __name__ == "__main__":
    main()
