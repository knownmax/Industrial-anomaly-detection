"""
Greedy coreset subsampling via Johnson-Lindenstrauss projection + furthest-point selection.
Pure PyTorch — no sklearn dependency at subsampling time.

Algorithm:
  1. Project features: F_proj = F @ R,  R ~ N(0, 1/sqrt(128)),  shape [N, 128]
  2. Pick a random starting point, seed the coreset C
  3. Maintain min-distance-to-coreset vector `min_dists` for all remaining points
  4. Greedily add the point with maximum min-distance
  5. Repeat until |C| = ceil(ratio * N)

"""

import math
import torch


PROJ_DIM   = 128
CHUNK_SIZE = 1000   # rows of the full set to process per distance batch


def _pairwise_l2sq_chunked(
    query: torch.Tensor,   # [M, D]
    bank:  torch.Tensor,   # [K, D]
) -> torch.Tensor:
    """
    Returns squared L2 distance matrix [M, K] without materializing a huge intermediate.
    Uses the (a-b)^2 = a^2 - 2ab + b^2 expansion.
    """
    # ||a||^2  [M, 1]
    a2 = (query ** 2).sum(dim=1, keepdim=True)
    # ||b||^2  [1, K]
    b2 = (bank ** 2).sum(dim=1, keepdim=True).t()
    # -2 <a, b>  [M, K]
    ab = torch.mm(query, bank.t())
    dist2 = a2 + b2 - 2 * ab
    dist2.clamp_(min=0.0)   # numerical safety
    return dist2


def subsample_coreset(features: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Greedy furthest-point coreset subsampling.

    Args:
        features : [N, D] float32 tensor on GPU (or CPU)
        ratio    : fraction of points to keep (e.g. 0.01 = 1%)

    Returns:
        [M, D] coreset tensor on the same device as `features`,
        where M = ceil(ratio * N).
    """
    N, D = features.shape
    M = max(1, math.ceil(ratio * N))
    device = features.device

    # --- Johnson-Lindenstrauss projection for distance computation only ---
    torch.manual_seed(42)
    R = torch.randn(D, PROJ_DIM, device=device) / math.sqrt(PROJ_DIM)
    F_proj = features @ R          # [N, PROJ_DIM]  (used only for distances)

    # --- Initialise with a random seed point ---
    start_idx = torch.randint(0, N, (1,)).item()
    coreset_indices = [start_idx]

    # min distance from each point to the current coreset (in projected space)
    # Initialise with distance to the first coreset point
    first = F_proj[start_idx].unsqueeze(0)             # [1, PROJ_DIM]
    min_dists = _pairwise_l2sq_chunked(F_proj, first).squeeze(1)  # [N]

    # --- Greedy furthest-point loop ---
    for _ in range(M - 1):
        new_idx = int(min_dists.argmax().item())
        coreset_indices.append(new_idx)

        # Update min_dists: distance from every point to the newly added coreset member
        new_pt = F_proj[new_idx].unsqueeze(0)  # [1, PROJ_DIM]

        # Chunked to avoid OOM
        for start in range(0, N, CHUNK_SIZE):
            end  = min(start + CHUNK_SIZE, N)
            chunk_dists = _pairwise_l2sq_chunked(
                F_proj[start:end], new_pt
            ).squeeze(1)                           # [chunk]
            min_dists[start:end] = torch.minimum(
                min_dists[start:end], chunk_dists
            )

    idx_tensor = torch.tensor(coreset_indices, dtype=torch.long, device=device)
    return features[idx_tensor]                    # [M, D] full-dim vectors
