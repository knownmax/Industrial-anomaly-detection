"""
PatchCore model: fit (feature extraction + coreset indexing) and predict (KNN scoring).

No training loop — PatchCore is purely:
  fit  : one forward pass over normal images → coreset memory bank
  predict: KNN distance query for each test patch → anomaly map + image score
"""

import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from feature_extractor import PatchFeatureExtractor
from coreset import subsample_coreset


class PatchCore:
    """
    PatchCore anomaly detector.

    Args:
        backbone      : timm model name (default: 'wide_resnet101_2')
        coreset_ratio : fraction of patches to keep in memory bank (default: 0.01)
        device        : torch device string (default: 'cuda')
        faiss_gpu     : use faiss GPU index (default: True, fallback to CPU)
        gaussian_sigma: sigma for anomaly map smoothing (default: 4)
    """

    PATCH_GRID = 28    # spatial grid size for 224px input (both axes)
    FEAT_DIM   = 1536  # layer2 (512) + layer3 (1024)

    def __init__(
        self,
        backbone: str = "wide_resnet101_2",
        coreset_ratio: float = 0.01,
        device: str = "cuda",
        faiss_gpu: bool = True,
        gaussian_sigma: float = 4.0,
    ):
        self.backbone      = backbone
        self.coreset_ratio = coreset_ratio
        self.device        = torch.device(device if torch.cuda.is_available() else "cpu")
        self.faiss_gpu     = faiss_gpu
        self.gaussian_sigma = gaussian_sigma

        self.extractor = PatchFeatureExtractor(backbone=backbone, pretrained=True)
        self.extractor = self.extractor.to(self.device)

        self.memory_bank: torch.Tensor = None  # [M, 1536]
        self._faiss_index = None
        self._index_backend = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_loader) -> None:
        """
        Extract patch features from all normal training images,
        run coreset subsampling, build faiss index.
        """
        print("[PatchCore] Extracting features from training images …")
        all_features = []

        self.extractor.eval()
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="  Feature extraction", leave=False):
                if isinstance(batch, (list, tuple)):
                    imgs = batch[0]
                else:
                    imgs = batch
                imgs = imgs.to(self.device)
                feats = self.extractor.extract_patch_features(imgs)  # [B*784, 1536]
                all_features.append(feats.cpu())

        all_features = torch.cat(all_features, dim=0)  # [N_total, 1536]
        print(f"[PatchCore] Total patches before coreset: {len(all_features):,}")

        # Move to GPU for fast coreset computation
        all_features = all_features.to(self.device)

        print(f"[PatchCore] Running coreset subsampling (ratio={self.coreset_ratio}) …")
        self.memory_bank = subsample_coreset(all_features, self.coreset_ratio)
        print(f"[PatchCore] Memory bank size after coreset: {len(self.memory_bank):,}  "
              f"({self.memory_bank.element_size() * self.memory_bank.numel() / 1e6:.1f} MB)")

        self._build_faiss_index()

    def _build_faiss_index(self) -> None:
        """Build KNN backend from the memory bank (faiss preferred, torch fallback)."""
        bank_np = self.memory_bank.cpu().numpy().astype(np.float32)

        try:
            import faiss

            d = bank_np.shape[1]
            index_flat = faiss.IndexFlatL2(d)

            if self.faiss_gpu and torch.cuda.is_available():
                try:
                    res = faiss.StandardGpuResources()
                    self._faiss_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
                    self._index_backend = "faiss-gpu"
                    print("[PatchCore] Using faiss GPU index.")
                except Exception as e:
                    warnings.warn(f"[PatchCore] faiss GPU index failed ({e}); falling back to faiss CPU.")
                    self._faiss_index = index_flat
                    self._index_backend = "faiss-cpu"
            else:
                self._faiss_index = index_flat
                self._index_backend = "faiss-cpu"

            self._faiss_index.add(bank_np)

        except Exception as e:
            warnings.warn(
                "[PatchCore] faiss is unavailable/incompatible; "
                f"falling back to torch KNN search. Details: {e}"
            )
            # Keep a contiguous float bank on the configured device for fallback KNN.
            self.memory_bank = self.memory_bank.float().contiguous().to(self.device)
            self._faiss_index = None
            self._index_backend = "torch"

    def _search_knn(self, feats_np: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Return (squared_l2_distances, indices) for nearest neighbours."""
        if self._index_backend in ("faiss-gpu", "faiss-cpu"):
            return self._faiss_index.search(feats_np, k=k)

        # Torch fallback for environments where faiss import fails (e.g., NumPy ABI mismatch).
        query = torch.from_numpy(feats_np).to(self.device, dtype=torch.float32)
        bank = self.memory_bank

        # Chunking keeps peak memory usage bounded for larger memory banks.
        chunk_size = 1024
        dist_chunks = []
        idx_chunks = []

        with torch.no_grad():
            for start in range(0, query.shape[0], chunk_size):
                q = query[start:start + chunk_size]  # [Q, D]
                d2 = torch.sum((q[:, None, :] - bank[None, :, :]) ** 2, dim=-1)  # [Q, M]
                vals, idxs = torch.topk(d2, k=k, dim=1, largest=False)
                dist_chunks.append(vals)
                idx_chunks.append(idxs)

        distances = torch.cat(dist_chunks, dim=0).detach().cpu().numpy().astype(np.float32)
        indices = torch.cat(idx_chunks, dim=0).detach().cpu().numpy().astype(np.int64)
        return distances, indices

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Tuple[float, np.ndarray]:
        """
        Compute anomaly score and pixel-level anomaly map for a single image.

        Args:
            image_tensor: [1, 3, 224, 224] normalised image tensor

        Returns:
            image_score : float — max patch distance (image-level anomaly score)
            anomaly_map : np.ndarray [224, 224] — smoothed, upsampled patch distance map
        """
        image_tensor = image_tensor.to(self.device)

        # Extract patch features: [1*784, 1536]
        feats = self.extractor.extract_patch_features(image_tensor)
        feats_np = feats.cpu().numpy().astype(np.float32)

        # KNN distance query (k=1 nearest neighbour)
        distances, _ = self._search_knn(feats_np, k=1)  # [N_patches, 1]
        patch_scores = distances[:, 0]  # [N_patches]  squared L2 distances

        # Compute actual grid size from number of patches (robust to different feature extractors)
        num_patches = len(patch_scores)
        patch_grid = int(np.sqrt(num_patches))
        assert patch_grid * patch_grid == num_patches, \
            f"Non-square patch grid: {num_patches} patches (expected {patch_grid}²)"

        # Reshape to spatial grid
        score_map = patch_scores.reshape(patch_grid, patch_grid)

        # Upsample to 224×224 via bilinear interpolation
        score_tensor = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0)  # [1,1,G,G]
        score_upsampled = F.interpolate(
            score_tensor, size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze().numpy()  # [224, 224]

        # Gaussian smoothing
        anomaly_map = gaussian_filter(score_upsampled, sigma=self.gaussian_sigma)

        # Image-level score: max patch distance
        image_score = float(patch_scores.max())

        return image_score, anomaly_map

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize memory bank and config to a .pt file."""
        torch.save(
            {
                "memory_bank": self.memory_bank.cpu(),
                "backbone": self.backbone,
                "coreset_ratio": self.coreset_ratio,
                "gaussian_sigma": self.gaussian_sigma,
            },
            path,
        )
        print(f"[PatchCore] Model saved to {path}")

    def load(self, path: str) -> None:
        """Load memory bank and config from a .pt file, rebuild faiss index."""
        ckpt = torch.load(path, map_location="cpu")
        self.memory_bank    = ckpt["memory_bank"].to(self.device)
        self.backbone       = ckpt["backbone"]
        self.coreset_ratio  = ckpt["coreset_ratio"]
        self.gaussian_sigma = ckpt["gaussian_sigma"]
        self._build_faiss_index()
        print(f"[PatchCore] Model loaded from {path}  "
              f"(memory bank: {len(self.memory_bank):,} vectors)")
