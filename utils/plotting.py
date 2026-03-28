import math

import matplotlib.pyplot as plt
import numpy as np

from polarquant_core import polar_transform_matrix, quantize_angles


def level_support(level: int):
    if level == 1:
        return 0.0, 2.0 * math.pi
    return 0.0, math.pi / 2.0


def plot_angles_all_levels_after_precondition(
    Psi_by_level: dict[int, np.ndarray] | None = None,
    *,
    X: np.ndarray | None = None,
    S: np.ndarray | None = None,
    bins_hist: int = 80,
    save_path: str | None = None,
) -> None:
    """Histogram of ψ at every level after preconditioning; x-axis ticks in units of π."""
    if Psi_by_level is None:
        if X is None or S is None:
            raise TypeError(
                "plot_angles_all_levels_after_precondition: "
                "pass Psi_by_level, or both X and S"
            )
        _, Psi_by_level, _ = polar_transform_matrix(X, S)

    levels = sorted(Psi_by_level.keys())
    nlev = len(levels)
    x_lo, x_hi = 0.0, 2.0 * math.pi
    bin_edges = np.linspace(x_lo, x_hi, bins_hist + 1)

    fig, axes = plt.subplots(
        nlev,
        1,
        figsize=(10, max(3.0, 2.8 * nlev)),
        sharex=True,
        squeeze=False,
    )
    ax_list = axes.ravel().tolist()

    pi = math.pi
    x_ticks = np.array([0.0, pi / 2, pi, 3 * pi / 2, 2 * pi])
    x_ticklabels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    x_label = r"$\psi_\ell$"

    for ax, ell in zip(ax_list, levels):
        psi = np.asarray(Psi_by_level[ell], dtype=float).ravel()
        ax.hist(psi, bins=bin_edges, density=True, alpha=0.75, color="C0")
        ax.set_xlim(x_lo, x_hi)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels)
        ax.set_ylabel("Density")
        ax.axvline(pi / 2.0, color="gray", linestyle=":", lw=0.9, alpha=0.7)
        ax.set_title(
            rf"Level $\ell={ell}$: $\psi$ ($X_{{\mathrm{{pre}}}} = X S$, polar)"
        )
        ax.set_xlabel(x_label)
        ax.tick_params(axis="x", labelbottom=True)
    fig.suptitle(
        "Polar angles at all levels after preconditioning (pre-quantization)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _normalize_radii_for_polar(
    rho: np.ndarray, r_inner: float = 0.12, r_outer: float = 1.0
) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    lo, hi = float(np.min(rho)), float(np.max(rho))
    if hi <= lo + 1e-15:
        return np.full(rho.shape, 0.5 * (r_inner + r_outer), dtype=float)
    t = (rho - lo) / (hi - lo)
    return r_inner + t * (r_outer - r_inner)


def plot_level_histogram_with_codebook(
    psi_values: np.ndarray,
    boundaries: np.ndarray,
    centroids: np.ndarray,
    level: int,
    bins_hist: int = 80,
):
    lo, hi = level_support(level)

    fig = plt.figure(figsize=(10, 5))
    plt.hist(psi_values, bins=bins_hist, density=True, alpha=0.6)
    for x in boundaries:
        plt.axvline(x, linewidth=1)
    for c in centroids:
        plt.axvline(c, linestyle="--", linewidth=1.5)
    plt.xlim(lo, hi)
    plt.xlabel(f"Angle at level {level}")
    plt.ylabel("Density")
    plt.title(f"Level {level}: histogram, boundaries, centroids")
    plt.close(fig)


def plot_quantizer_step(boundaries: np.ndarray, centroids: np.ndarray, level: int):
    lo, hi = level_support(level)
    x = np.linspace(lo, hi, 4000)
    idx = np.searchsorted(boundaries[1:-1], x, side="right")
    y = centroids[idx]

    fig = plt.figure(figsize=(9, 5))
    plt.plot(x, y)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Original angle")
    plt.ylabel("Quantized angle")
    plt.title(f"Level {level}: quantizer step function")
    plt.close(fig)


def plot_angle_scatter_minimized_quantized(
    psi_values: np.ndarray,
    q_values: np.ndarray,
    boundaries: np.ndarray,
    centroids: np.ndarray,
    level: int,
    max_points: int = 8000,
    seed: int = 0,
):
    lo, hi = level_support(level)
    psi = np.asarray(psi_values, dtype=float).ravel()
    q = np.asarray(q_values, dtype=float).ravel()
    if len(psi) != len(q):
        raise ValueError("psi_values and q_values must match in length")

    if len(psi) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(psi), size=max_points, replace=False)
        psi = psi[idx]
        q = q[idx]

    resid = psi - q
    rmse = float(np.sqrt(np.mean(resid**2)))
    J = quantize_angles(psi, boundaries)
    K = len(centroids)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.scatter(psi, q, s=5, alpha=0.22, c="C0", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.65, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\psi$ (continuous)")
    ax.set_ylabel(r"$\hat{\psi}$ (quantized)")
    ax.set_title(f"Level {level}: angles → discrete levels\nRMSE = {rmse:.4g}")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.scatter(psi, resid, s=5, alpha=0.22, c="C1", rasterized=True)
    ax.axhline(0.0, color="k", lw=0.85)
    for x in boundaries[1:-1]:
        ax.axvline(x, color="gray", lw=0.45, alpha=0.55)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"$\psi$")
    ax.set_ylabel(r"Residual $\psi - \hat{\psi}$")
    ax.set_title(
        "Residual scatter\n(Lloyd–Max minimizes " r"$\mathbb{E}[(\psi-\hat{\psi})^2]$" ")"
    )

    ax = axes[2]
    nonempty = [k for k in range(K) if np.any(J == k)]
    if nonempty:
        data = [psi[J == k] for k in nonempty]
        parts = ax.violinplot(
            data,
            positions=nonempty,
            widths=0.75,
            showmeans=True,
            showmedians=False,
            showextrema=False,
        )
        for b in parts["bodies"]:
            b.set_alpha(0.55)
        for k in nonempty:
            ax.plot(
                [k - 0.32, k + 0.32],
                [centroids[k], centroids[k]],
                color="crimson",
                lw=2.0,
                zorder=3,
            )
    ax.set_xticks(range(K))
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_xlabel("Quantization index")
    ax.set_ylabel(r"$\psi$ in bin")
    ax.set_title(r"Within-bin spread of $\psi$" "\n(red = centroid, MSE-minimal)")

    fig.suptitle(
        f"Level {level}: angle scatter under quantization",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out = f"level_{level}_angle_scatter_min_quantized.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_polar_original_vs_quantized(
    psi_values: np.ndarray,
    q_values: np.ndarray,
    boundaries: np.ndarray,
    level: int,
    rho_values: np.ndarray | None = None,
    max_points: int = 5000,
    seed: int = 0,
):
    lo, hi = level_support(level)
    psi = np.asarray(psi_values, dtype=float).ravel()
    q = np.asarray(q_values, dtype=float).ravel()
    if len(psi) != len(q):
        raise ValueError("psi_values and q_values must match in length")

    rho = None
    if rho_values is not None:
        rho = np.asarray(rho_values, dtype=float).ravel()
        if len(rho) != len(psi):
            raise ValueError("rho_values must match length of psi_values")

    if len(psi) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(psi), size=max_points, replace=False)
        psi = psi[idx]
        q = q[idx]
        if rho is not None:
            rho = rho[idx]

    if rho is not None:
        r_vis = _normalize_radii_for_polar(rho)
        r_ray = 1.06
        r_title_suffix = "r = ρ (min–max scaled); same ρ for quantized θ"
    else:
        r_vis = np.ones_like(psi)
        r_hat = np.full_like(q, 0.74)
        r_ray = 1.08
        r_title_suffix = "r = fixed rings (no ρ passed)"

    fig = plt.figure(figsize=(8.0, 7.5))
    ax = fig.add_subplot(111, projection="polar")

    if rho is not None:
        ax.scatter(
            psi,
            r_vis,
            s=12,
            alpha=0.3,
            c="C0",
            edgecolors="none",
            label=r"$(\psi,\rho)$ continuous",
            rasterized=True,
        )
        ax.scatter(
            q,
            r_vis,
            s=16,
            alpha=0.45,
            c="C3",
            marker="s",
            edgecolors="none",
            label=r"$(\hat{\psi},\rho)$ quantized θ only",
            rasterized=True,
        )
        ax.set_rticks(np.linspace(0.2, 1.0, 5))
    else:
        ax.scatter(
            psi,
            r_vis,
            s=10,
            alpha=0.28,
            c="C0",
            edgecolors="none",
            label=r"$\psi$ (fixed $r$)",
            rasterized=True,
        )
        ax.scatter(
            q,
            r_hat,
            s=14,
            alpha=0.4,
            c="C3",
            marker="s",
            edgecolors="none",
            label=r"$\hat{\psi}$ (fixed $r$)",
            rasterized=True,
        )
        ax.set_rticks([0.74, 1.0])

    for x in boundaries:
        ax.plot([x, x], [0.0, r_ray], color="gray", lw=0.7, alpha=0.55, zorder=1)

    ax.set_rlim(0.0, r_ray + 0.02)
    ax.set_rlabel_position(22.5)

    if level == 1:
        ax.set_thetamin(0.0)
        ax.set_thetamax(360.0)
    else:
        ax.set_thetamin(0.0)
        ax.set_thetamax(math.degrees(hi))

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(
        f"Level {level}: polar — θ angle, {r_title_suffix}",
        y=1.08,
        fontsize=10,
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=8)

    fig.tight_layout()
    out = f"level_{level}_polar_original_vs_quantized.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_original_vs_quantized(
    psi_values: np.ndarray,
    q_values: np.ndarray,
    level: int,
    max_points: int = 3000,
    seed: int = 0,
):
    lo, hi = level_support(level)

    psi_values = np.asarray(psi_values).ravel()
    q_values = np.asarray(q_values).ravel()

    if len(psi_values) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(psi_values), size=max_points, replace=False)
        psi_values = psi_values[idx]
        q_values = q_values[idx]

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(psi_values, q_values, s=8, alpha=0.35)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Original angle")
    plt.ylabel("Quantized angle")
    plt.title(f"Level {level}: original vs quantized")
    plt.close(fig)


def plot_error_histogram(
    psi_values: np.ndarray,
    q_values: np.ndarray,
    level: int,
    bins_hist: int = 80,
):
    err = np.asarray(psi_values).ravel() - np.asarray(q_values).ravel()

    fig = plt.figure(figsize=(10, 4))
    plt.hist(err, bins=bins_hist, alpha=0.7)
    plt.xlabel("Quantization error (original - quantized)")
    plt.ylabel("Count")
    plt.title(f"Level {level}: quantization error")
    plt.close(fig)


def plot_mse_by_level(Psi_by_level, Q_by_level):
    levels = sorted(Psi_by_level.keys())
    mse = []

    for ell in levels:
        psi = Psi_by_level[ell]
        q = Q_by_level[ell]
        mse.append(np.mean((psi - q) ** 2))

    fig = plt.figure(figsize=(8, 4))
    plt.plot(levels, mse, marker="o")
    plt.xlabel("Level")
    plt.ylabel("Mean squared quantization error")
    plt.title("Quantization MSE by level")
    plt.close(fig)


def visualize_all_levels(
    Psi_by_level,
    Q_by_level,
    codebooks,
    Rho_by_level: dict | None = None,
):
    for ell in sorted(Psi_by_level.keys()):
        psi_values = Psi_by_level[ell].ravel()
        q_values = Q_by_level[ell].ravel()
        boundaries = codebooks[ell]["boundaries"]
        centroids = codebooks[ell]["centroids"]
        rho_flat = None
        if Rho_by_level is not None:
            rho_flat = Rho_by_level[ell].ravel()

        print(f"Visualizing level {ell}...")
        plot_level_histogram_with_codebook(psi_values, boundaries, centroids, ell)
        plot_quantizer_step(boundaries, centroids, ell)
        plot_original_vs_quantized(psi_values, q_values, ell)
        plot_angle_scatter_minimized_quantized(
            psi_values, q_values, boundaries, centroids, ell
        )
        plot_polar_original_vs_quantized(
            psi_values, q_values, boundaries, ell, rho_values=rho_flat
        )
        plot_error_histogram(psi_values, q_values, ell)

    plot_mse_by_level(Psi_by_level, Q_by_level)
