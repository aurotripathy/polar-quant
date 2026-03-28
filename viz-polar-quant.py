import math
import numpy as np
import matplotlib.pyplot as plt


def visualize_level_quantization(
    psi_values: np.ndarray,
    boundaries: np.ndarray,
    centroids: np.ndarray,
    level: int,
    bins_hist: int = 80,
):
    """
    Visualize scalar quantization for one PolarQuant level.

    Args:
        psi_values: flattened angles from one level, shape (N,)
        boundaries: interval boundaries, shape (K+1,)
        centroids: quantization centroids, shape (K,)
        level: quantization level ell
    """
    psi_values = np.asarray(psi_values).ravel()
    boundaries = np.asarray(boundaries)
    centroids = np.asarray(centroids)

    # quantize
    q_idx = np.searchsorted(boundaries[1:-1], psi_values, side="right")
    q_vals = centroids[q_idx]

    # theoretical support
    if level == 1:
        lo, hi = 0.0, 2.0 * math.pi
    else:
        lo, hi = 0.0, math.pi / 2.0

    # ---- Plot 1: histogram + boundaries + centroids ----
    plt.figure(figsize=(10, 5))
    plt.hist(psi_values, bins=bins_hist, density=True, alpha=0.6)
    for x in boundaries:
        plt.axvline(x, linewidth=1)
    for c in centroids:
        plt.axvline(c, linestyle="--", linewidth=1.5)
    plt.xlim(lo, hi)
    plt.xlabel(f"Angle at level {level}")
    plt.ylabel("Density")
    plt.title(f"Level {level}: angle distribution with quantization boundaries and centroids")
    plt.show()

    # ---- Plot 2: original vs quantized ----
    plt.figure(figsize=(6, 6))
    plt.scatter(psi_values, q_vals, s=8, alpha=0.35)
    plt.xlabel("Original angle")
    plt.ylabel("Quantized angle")
    plt.title(f"Level {level}: original vs quantized angles")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.show()

    # ---- Plot 3: quantization error histogram ----
    plt.figure(figsize=(10, 4))
    plt.hist(psi_values - q_vals, bins=bins_hist, alpha=0.7)
    plt.xlabel("Quantization error (original - quantized)")
    plt.ylabel("Count")
    plt.title(f"Level {level}: quantization error")
    plt.show()
    plt.savefig(f"level_{level}_quantization_error.png")

from all_in_one import polarquant
from all_in_one import polar_transform_matrix

X = np.random.randn(100, 64)
S = np.eye(64)
b = 3

R, J_by_level, codebooks = polarquant(X, S, b)
_, Psi_by_level = polar_transform_matrix(X, S)

ell = 2
psi_vals = Psi_by_level[ell].ravel()
boundaries = codebooks[ell]["boundaries"]
centroids = codebooks[ell]["centroids"]

visualize_level_quantization(psi_vals, boundaries, centroids, ell)