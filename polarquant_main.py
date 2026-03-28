import matplotlib.pyplot as plt

from polarquant_core import *
from utils.plotting import (
    level_support,
    plot_angle_scatter_minimized_quantized,
    plot_angles_all_levels_after_precondition,
    plot_error_histogram,
    plot_level_histogram_with_codebook,
    plot_mse_by_level,
    plot_original_vs_quantized,
    plot_polar_original_vs_quantized,
    plot_quantizer_step,
    visualize_all_levels,
)

if __name__ == "__main__":
    nb_samples = 100 # number of samples
    dim = 32 # dimension of the data
    nb_bits = 4 # number of bits per sample

    X, S, b = init_polarquant_inputs(n=nb_samples, d=dim, b=nb_bits, seed=42)

    R, Psi_by_level, Rho_by_level, J_by_level, Q_by_level, codebooks = polarquant(
        X, S, b
    )

    plot_angles_all_levels_after_precondition(
        Psi_by_level, save_path="angles_all_levels_after_precondition.png"
    )

    print("R shape:", R.shape)
    for ell in sorted(J_by_level.keys()):
        print(f"Level {ell}:")
        print("  Psi shape:", Psi_by_level[ell].shape)
        print("  Rho shape:", Rho_by_level[ell].shape)
        print("  J shape  :", J_by_level[ell].shape)
        print("  Q shape  :", Q_by_level[ell].shape)

    visualize_all_levels(Psi_by_level, Q_by_level, codebooks, Rho_by_level)
    plt.close("all")
