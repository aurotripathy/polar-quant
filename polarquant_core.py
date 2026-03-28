""" PolarQuant core functions for polar quantization of data in polar coordinates.
https://arxiv.org/pdf/2502.02617
"""
import math
import numpy as np
from math import gamma


def random_rotation_matrix(rng: np.random.Generator, d: int) -> np.ndarray:
    """Random proper rotation in SO(d): orthogonal with det = +1."""
    M = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(M) # random rotation after starting from Gaussian matrix M
    if np.linalg.det(Q) < 0:
        Q = Q.copy()
        Q[:, -1] *= -1.0
    return Q


def init_polarquant_inputs(
    n: int,
    d: int,
    b: int,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    S = random_rotation_matrix(rng, d)
    return X, S, b


def polar_row(y: np.ndarray):
    y = np.asarray(y, dtype=float)

    if y.ndim != 1:
        raise ValueError("y must be 1D")
    d = y.shape[0]
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError("length of y must be a positive power of 2")

    r = y.copy()
    psi_levels = []
    rho_levels = []

    for _ in range(int(math.log2(d))):
        a = r[0::2]  # start at 0, step by 2
        b = r[1::2]  # start at 1, step by 2
        psi = np.arctan2(b, a) # the angle 
        r = np.hypot(a, b) # the magnitude 
        psi_levels.append(psi)
        rho_levels.append(r.copy())

    return float(r[0]), psi_levels, rho_levels


def polar_transform_matrix(X: np.ndarray, S: np.ndarray):
    X = np.asarray(X, dtype=float)
    S = np.asarray(S, dtype=float)

    n, d = X.shape
    if S.shape != (d, d):
        raise ValueError("S has wrong shape")

    X_preconditioned = X @ S
    L = int(math.log2(d))

    R = np.empty((n, 1), dtype=float)
    Psi_by_level = {ell: np.empty((n, d // (2**ell)), dtype=float) for ell in range(1, L + 1)}
    Rho_by_level = {ell: np.empty((n, d // (2**ell)), dtype=float) for ell in range(1, L + 1)}

    for i in range(n):
        r_i, psi_levels, rho_levels = polar_row(X_preconditioned[i])
        R[i, 0] = r_i
        for ell, psi in enumerate(psi_levels, start=1):
            Psi_by_level[ell][i, :] = psi
            Rho_by_level[ell][i, :] = rho_levels[ell - 1]

    return R, Psi_by_level, Rho_by_level


def angle_pdf(level: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if level == 1:
        pdf = np.zeros_like(x)
        mask = (x >= 0.0) & (x <= 2.0 * math.pi)
        pdf[mask] = 1.0 / (2.0 * math.pi)
        return pdf

    m = 2 ** (level - 1)
    c = gamma(m) / (2 ** (m - 1) * (gamma(m) ** 2))

    pdf = np.zeros_like(x)
    mask = (x >= 0.0) & (x <= math.pi / 2.0)
    pdf[mask] = c * (np.sin(2.0 * x[mask]) ** (m - 1))
    return pdf


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapz(y, x))


def _conditional_mean(grid: np.ndarray, pdf: np.ndarray, a: float, b: float) -> float:
    if b <= a:
        return 0.5 * (a + b)

    mask = (grid >= a) & (grid < b)
    if not np.any(mask):
        return 0.5 * (a + b)

    x_seg = grid[mask]
    p_seg = pdf[mask]

    mass = _trapz(p_seg, x_seg)
    if mass <= 1e-15:
        return 0.5 * (a + b)

    moment = _trapz(x_seg * p_seg, x_seg)
    return moment / mass


def lloyd_max_from_pdf(
    level: int,
    b: int,
    num_grid: int = 20000,
    max_iter: int = 200,
    tol: float = 1e-10,
):
    K = 2**b

    if level == 1:
        lo, hi = 0.0, 2.0 * math.pi
    else:
        lo, hi = 0.0, math.pi / 2.0

    grid = np.linspace(lo, hi, num_grid)
    pdf = angle_pdf(level, grid)

    boundaries = np.linspace(lo, hi, K + 1)
    centroids = 0.5 * (boundaries[:-1] + boundaries[1:])

    for _ in range(max_iter):
        old_boundaries = boundaries.copy()
        old_centroids = centroids.copy()

        for k in range(K):
            centroids[k] = _conditional_mean(grid, pdf, boundaries[k], boundaries[k + 1])

        boundaries[0] = lo
        boundaries[-1] = hi
        for k in range(1, K):
            boundaries[k] = 0.5 * (centroids[k - 1] + centroids[k])

        delta = max(
            np.max(np.abs(boundaries - old_boundaries)),
            np.max(np.abs(centroids - old_centroids)),
        )
        if delta < tol:
            break

    return boundaries, centroids


def build_polarquant_codebooks(d: int, b: int):
    L = int(math.log2(d))
    codebooks = {}

    for ell in range(1, L + 1):
        boundaries, centroids = lloyd_max_from_pdf(level=ell, b=b)
        codebooks[ell] = {
            "boundaries": boundaries,
            "centroids": centroids,
        }

    return codebooks


def quantize_angles(psi: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    K = len(boundaries) - 1
    J = np.searchsorted(boundaries[1:-1], psi, side="right")
    J = np.clip(J, 0, K - 1)
    return J.astype(np.int64)


def quantized_values_from_indices(J: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return centroids[J]


def polarquant(X: np.ndarray, S: np.ndarray, b: int):
    n, d = X.shape
    R, Psi_by_level, Rho_by_level = polar_transform_matrix(X, S)
    codebooks = build_polarquant_codebooks(d, b)

    J_by_level = {}
    Q_by_level = {}

    for ell, psi_mat in Psi_by_level.items():
        boundaries = codebooks[ell]["boundaries"]
        centroids = codebooks[ell]["centroids"]
        J = quantize_angles(psi_mat, boundaries)
        Q = quantized_values_from_indices(J, centroids)

        J_by_level[ell] = J
        Q_by_level[ell] = Q

    return R, Psi_by_level, Rho_by_level, J_by_level, Q_by_level, codebooks
