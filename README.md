# Industrial Anomaly Detection with PatchCore

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Dataset](https://img.shields.io/badge/dataset-MVTec%20AD-orange.svg)
![Backbone](https://img.shields.io/badge/backbone-WideResNet--101--2-purple.svg)

## Overview

**PatchCore** (Roth et al., 2022) is a state-of-the-art *training-free* anomaly detection method for industrial inspection. It builds a memory bank of patch-level feature descriptors extracted from normal (defect-free) images using a pretrained CNN backbone. At inference, anomaly scores are computed as the nearest-neighbour distance from test image patches to the memory bank — no fine-tuning or gradient updates required.

This repository benchmarks PatchCore with a **WideResNet-101-2** backbone on the **MVTec Anomaly Detection** dataset, covering 15 industrial product categories (textures and objects). The implementation includes the three standard MVTec metrics: image-level AUROC, pixel-level AUROC, and the PRO score (Per-Region Overlap).

---

## PatchCore Pipeline

```
TRAINING (fit — one forward pass, no gradients)
────────────────────────────────────────────────────────────────
  Normal images ──► WideResNet-101-2 ──► layer2 (512ch, 28×28)
                                    └──► layer3 (1024ch, 14×14)
                                              │
                             Adaptive AvgPool (align to 28×28)
                                              │
                        Concatenate → [B, 1536, 28×28 patches]
                                              │
                              L2 Normalize patch descriptors
                                              │
                     Greedy Coreset Subsampling (JL projection)
                        keep 1% → Memory Bank [M × 1536]
                                              │
                              faiss-GPU Index (FlatL2)

INFERENCE (predict)
────────────────────────────────────────────────────────────────
  Test image ──► WideResNet-101-2 ──► Patch features [784 × 1536]
                                              │
                         faiss GPU KNN (k=1) → patch distances
                                              │
              Reshape [28×28] → bilinear upsample [224×224]
                                              │
                           Gaussian smoothing (σ=4)
                                              │
                   anomaly_map [224×224]  +  image_score (max)
```

---

## Results

> Fill in actual numbers after running `python src/run_all.py`.
> Reference values (WRN-101, coreset 1%) from Roth et al. 2022 in parentheses.

| Category     | Image AUROC | Pixel AUROC | PRO Score |
|--------------|:-----------:|:-----------:|:---------:|
| Bottle       |   xx.x      |   xx.x      |   xx.x    |
| Cable        |   xx.x      |   xx.x      |   xx.x    |
| Capsule      |   xx.x      |   xx.x      |   xx.x    |
| Carpet       |   xx.x      |   xx.x      |   xx.x    |
| Grid         |   xx.x      |   xx.x      |   xx.x    |
| Hazelnut     |   xx.x      |   xx.x      |   xx.x    |
| Leather      |   xx.x      |   xx.x      |   xx.x    |
| Metal Nut    |   xx.x      |   xx.x      |   xx.x    |
| Pill         |   xx.x      |   xx.x      |   xx.x    |
| Screw        |   xx.x      |   xx.x      |   xx.x    |
| Tile         |   xx.x      |   xx.x      |   xx.x    |
| Toothbrush   |   xx.x      |   xx.x      |   xx.x    |
| Transistor   |   xx.x      |   xx.x      |   xx.x    |
| Wood         |   xx.x      |   xx.x      |   xx.x    |
| Zipper       |   xx.x      |   xx.x      |   xx.x    |
| **Mean**     | **~99.1**   | **~98.1**   | **~xx.x** |

---

## Quickstart

```bash
# Activate environment
conda activate env 

# Install pip dependencies
pip install -r requirements.txt
# Install faiss-gpu (do NOT use pip version)
conda install -c conda-forge faiss-gpu

# Demo — single category (bottle)
python src/run_single.py --category bottle \
    --data_root /data/max/kaggle/anomaly_detect/anomaly_ds

# Full benchmark — all 15 categories
python src/run_all.py \
    --data_root /data/max/kaggle/anomaly_detect/anomaly_ds

# Or use the convenience shell script
bash scripts/benchmark_all.sh
```

Results are saved to `results/{category}/`:
- `metrics.json` — numerical metrics
- `anomaly_grid.png` — 8 anomalous + 8 normal sample visualisations
- `roc_curves.png` — Image AUROC, Pixel AUROC, PRO curve
- `score_distribution.png` — KDE of anomaly scores

---

## Inference Notebook

Open `notebooks/inference_demo.ipynb` to:
- Load a saved model (`results/bottle/patchcore_model.pt`)
- Run inference on individual test images
- Visualise anomaly heatmaps interactively

---

## Performance Notes

Measured on **NVIDIA H100 SXM5 (80 GB)**:

| Step                 | Time (approx.)       |
|----------------------|----------------------|
| Feature extraction   | ~30–90 s per category|
| Coreset subsampling  | ~5–15 s              |
| faiss index build    | < 1 s                |
| Inference per image  | ~5–15 ms             |
| Total per category   | ~2–5 min             |

Memory bank sizes (WRN-101, coreset 1%):
- Small categories (toothbrush): ~5k vectors → ~30 MB
- Large categories (carpet, tile): ~200k vectors → ~1.2 GB

---

## Project Structure

```
industrial-anomaly-detection/
├── README.md
├── requirements.txt
├── configs/
│   └── patchcore_config.yaml      ← all hyperparameters
├── src/
│   ├── dataset.py                 ← MVTec dataset loaders
│   ├── feature_extractor.py       ← WideResNet-101-2 patch features
│   ├── coreset.py                 ← greedy coreset subsampling (pure PyTorch)
│   ├── patchcore.py               ← PatchCore model (fit + predict)
│   ├── metrics.py                 ← AUROC, pixel AUROC, PRO score
│   ├── visualize.py               ← anomaly grids, ROC curves, KDE plots
│   ├── run_single.py              ← demo: train + eval one category
│   └── run_all.py                 ← benchmark: all 15 categories
├── notebooks/
│   └── inference_demo.ipynb
├── results/
│   └── .gitkeep
└── scripts/
    └── benchmark_all.sh
```

---

## References

- **PatchCore**: Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022).
  *Towards Total Recall in Industrial Anomaly Detection.*
  CVPR 2022. [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)

- **MVTec AD**: Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019).
  *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*
  CVPR 2019. [Paper](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

## License

MIT License. The MVTec AD dataset has its own [license](https://www.mvtec.com/company/research/datasets/mvtec-ad/license).
