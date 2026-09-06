"""
Two-phase spectral density for a binary sphere packing.

Outputs:
1. Spectral-density CSV.
2. Fit-results CSV.
3. Full and zoomed linear plots.
4. Log-log plot showing the low-q power-law fit.

Fit model:
    chi(q) = A * q**alpha
    
"""

import os
import numpy as np
import pandas as pd
import gsd.hoomd
import matplotlib.pyplot as plt


base_dir = "/home/akash/binary_colloids"

particle_chunk_size = 25000
mode_chunk_size = 20

# Stronger directional averaging at low q
max_modes_low_q = 200
low_q_sampling_cutoff = 1.0

# Fewer modes at higher q
max_modes_high_q = 40

random_seed = 42


# Power-law fitting intervals

low_fit_q_min = 0.0
low_fit_q_max = 1.0


# ============================================================
# SPHERE FORM FACTOR
# ============================================================

def sphere_form_factor(q, radius):

    q = np.asarray(
        q,
        dtype=np.float64
    )

    x = q * radius

    volume = (
        4.0
        * np.pi
        * radius**3
        / 3.0
    )

    result = np.empty_like(q)

    near_zero = (
        np.abs(x) < 1e-10
    )

    result[near_zero] = volume

    xx = x[~near_zero]

    result[~near_zero] = (
        volume
        * 3.0
        * (
            np.sin(xx)
            - xx * np.cos(xx)
        )
        / xx**3
    )

    return result


# ============================================================
# MEMORY-SAFE FOURIER SUM
# ============================================================

def calculate_fourier_sum(
    positions,
    q_vectors,
    particle_chunk_size,
    mode_chunk_size
):

    number_of_particles = len(
        positions
    )

    number_of_modes = len(
        q_vectors
    )

    rho = np.zeros(
        number_of_modes,
        dtype=np.complex128
    )


    if number_of_particles == 0:

        return rho


    for mode_start in range(
        0,
        number_of_modes,
        mode_chunk_size
    ):

        mode_stop = min(
            mode_start + mode_chunk_size,
            number_of_modes
        )

        q_part = q_vectors[
            mode_start:mode_stop
        ]

        rho_part = np.zeros(
            mode_stop - mode_start,
            dtype=np.complex128
        )


        for particle_start in range(
            0,
            number_of_particles,
            particle_chunk_size
        ):

            particle_stop = min(
                particle_start + particle_chunk_size,
                number_of_particles
            )

            position_part = positions[
                particle_start:particle_stop
            ]

            phase = (
                position_part
                @ q_part.T
            )

            rho_part += np.exp(
                1j * phase
            ).sum(axis=0)


        rho[
            mode_start:mode_stop
        ] = rho_part


    return rho


# ============================================================
# POWER-LAW FIT
# ============================================================

def fit_power_law(
    df,
    fit_name,
    q_lower,
    q_upper
):

    fit_data = df[
        (df["q"] >= q_lower)
        &
        (df["q"] <= q_upper)
        &
        np.isfinite(df["q"])
        &
        np.isfinite(df["chi"])
        &
        (df["q"] > 0)
        &
        (df["chi"] > 0)
    ].copy()


    if len(fit_data) < 3:

        raise RuntimeError(
            f"Not enough data points for {fit_name}."
        )


    log_q = np.log(
        fit_data["q"].to_numpy()
    )

    log_chi = np.log(
        fit_data["chi"].to_numpy()
    )


    alpha, log_A = np.polyfit(
        log_q,
        log_chi,
        1
    )

    A = np.exp(
        log_A
    )


    predicted_log_chi = (
        log_A
        + alpha * log_q
    )


    residual_sum = np.sum(
        (
            log_chi
            - predicted_log_chi
        )**2
    )

    total_sum = np.sum(
        (
            log_chi
            - np.mean(log_chi)
        )**2
    )


    if total_sum > 0:

        r_squared = (
            1.0
            - residual_sum / total_sum
        )

    else:

        r_squared = np.nan


    q_fit = np.linspace(
        fit_data["q"].min(),
        fit_data["q"].max(),
        300
    )

    chi_fit = (
        A
        * q_fit**alpha
    )


    result = {
        "fit_name": fit_name,
        "q_min": fit_data["q"].min(),
        "q_max": fit_data["q"].max(),
        "n_points": len(fit_data),
        "alpha": alpha,
        "A": A,
        "r_squared": r_squared
    }


    return result, q_fit, chi_fit


# ============================================================
# MAIN FUNCTION
# ============================================================

def spectral_density(
    L,
    xs,
    d1,
    d2,
    q_min,
    q_max,
    number_of_qs
):

    folder_name = (
        f"final_rcp_xs_{xs:.2f}_L_{L:g}"
    )

    folder_path = os.path.join(
        base_dir,
        folder_name
    )

    input_file = os.path.join(
        folder_path,
        "seed_00.gsd"
    )


    if not os.path.isfile(input_file):

        raise FileNotFoundError(
            f"GSD file not found:\n{input_file}"
        )


    print("=" * 70)
    print("Reading:")
    print(input_file)
    print("=" * 70)


    # ========================================================
    # READ CONFIGURATION
    # ========================================================

    with gsd.hoomd.open(
        input_file,
        mode="r"
    ) as trajectory:

        frame = trajectory[-1]


    positions = np.asarray(
        frame.particles.position,
        dtype=np.float64
    )

    typeid = np.asarray(
        frame.particles.typeid,
        dtype=int
    )

    particle_types = list(
        frame.particles.types
    )


    Lx, Ly, Lz, xy, xz, yz = (
        frame.configuration.box
    )


    if (
        abs(xy) > 1e-10
        or abs(xz) > 1e-10
        or abs(yz) > 1e-10
    ):

        raise RuntimeError(
            "The box must be orthorhombic."
        )


    if not (
        np.isclose(Lx, Ly)
        and np.isclose(Lx, Lz)
    ):

        raise RuntimeError(
            "The box must be cubic."
        )


    if "small" not in particle_types:

        raise RuntimeError(
            f"'small' type not found: {particle_types}"
        )


    if "large" not in particle_types:

        raise RuntimeError(
            f"'large' type not found: {particle_types}"
        )


    V = Lx * Ly * Lz


    # ========================================================
    # SEPARATE PARTICLES
    # ========================================================

    small_id = particle_types.index(
        "small"
    )

    large_id = particle_types.index(
        "large"
    )


    r_small = positions[
        typeid == small_id
    ]

    r_large = positions[
        typeid == large_id
    ]


    N_small = len(
        r_small
    )

    N_large = len(
        r_large
    )

    N_total = (
        N_small
        + N_large
    )


    R_small = d1 / 2.0
    R_large = d2 / 2.0


    v_small = (
        4.0
        * np.pi
        * R_small**3
        / 3.0
    )

    v_large = (
        4.0
        * np.pi
        * R_large**3
        / 3.0
    )


    total_small_volume = (
        N_small * v_small
    )

    total_large_volume = (
        N_large * v_large
    )

    total_particle_volume = (
        total_small_volume
        + total_large_volume
    )


    measured_xs = (
        total_small_volume
        / total_particle_volume
    )

    packing_fraction = (
        total_particle_volume
        / V
    )


    print("N_small          =", N_small)
    print("N_large          =", N_large)
    print("N_total          =", N_total)
    print("Box              =", Lx, Ly, Lz)
    print("Measured xs      =", measured_xs)
    print("Packing fraction =", packing_fraction)


    # ========================================================
    # q BINS
    # ========================================================

    q_edges = np.linspace(
        q_min,
        q_max,
        number_of_qs + 1
    )

    dq = (
        q_edges[1]
        - q_edges[0]
    )

    reciprocal_spacing = (
        2.0
        * np.pi
        / Lx
    )


    print("Smallest accessible q =", reciprocal_spacing)
    print("q-bin width           =", dq)
    print("Requested q range     =", q_min, q_max)


    # ========================================================
    # RECIPROCAL-LATTICE VECTORS
    # ========================================================

    n_max = int(
        np.ceil(
            q_max
            / reciprocal_spacing
        )
    )


    n_values = np.arange(
        -n_max,
        n_max + 1,
        dtype=np.int16
    )


    nx, ny, nz = np.meshgrid(
        n_values,
        n_values,
        n_values,
        indexing="ij"
    )


    nx = nx.ravel()
    ny = ny.ravel()
    nz = nz.ravel()


    h2 = (
        nx.astype(np.int32)**2
        + ny.astype(np.int32)**2
        + nz.astype(np.int32)**2
    )


    q_magnitude = (
        reciprocal_spacing
        * np.sqrt(h2)
    )


    keep = (
        (h2 > 0)
        &
        (q_magnitude >= q_min)
        &
        (q_magnitude <= q_max)
    )


    nx = nx[keep]
    ny = ny[keep]
    nz = nz[keep]
    q_magnitude = q_magnitude[keep]


    # Keep only one member of each q and -q pair.

    independent = (
        (nx > 0)
        |
        (
            (nx == 0)
            &
            (ny > 0)
        )
        |
        (
            (nx == 0)
            &
            (ny == 0)
            &
            (nz > 0)
        )
    )


    nx = nx[independent]
    ny = ny[independent]
    nz = nz[independent]
    q_magnitude = q_magnitude[independent]


    # ========================================================
    # BIN ASSIGNMENT
    # ========================================================

    bin_id = np.searchsorted(
        q_edges,
        q_magnitude,
        side="right"
    ) - 1


    bin_id[
        np.isclose(
            q_magnitude,
            q_max
        )
    ] = number_of_qs - 1


    valid = (
        (bin_id >= 0)
        &
        (bin_id < number_of_qs)
    )


    nx = nx[valid]
    ny = ny[valid]
    nz = nz[valid]

    q_magnitude = q_magnitude[valid]
    bin_id = bin_id[valid]


    # ========================================================
    # SELECT DIRECTIONS FROM EACH BIN
    # ========================================================

    rng = np.random.default_rng(
        random_seed
    )


    selected_indices = []
    selected_bins = []
    available_modes = {}


    for current_bin in range(
        number_of_qs
    ):

        indices = np.where(
            bin_id == current_bin
        )[0]


        available_modes[
            current_bin
        ] = len(indices)


        if len(indices) == 0:
            continue


        q_bin_center = 0.5 * (
            q_edges[current_bin]
            + q_edges[current_bin + 1]
        )


        if (
            q_bin_center
            <= low_q_sampling_cutoff
        ):

            modes_to_use = min(
                len(indices),
                max_modes_low_q
            )

        else:

            modes_to_use = min(
                len(indices),
                max_modes_high_q
            )


        if len(indices) > modes_to_use:

            indices = rng.choice(
                indices,
                size=modes_to_use,
                replace=False
            )


        selected_indices.append(
            indices
        )

        selected_bins.append(
            np.full(
                len(indices),
                current_bin,
                dtype=int
            )
        )


    if len(selected_indices) == 0:

        raise RuntimeError(
            "No reciprocal vectors were found."
        )


    selected_indices = np.concatenate(
        selected_indices
    )

    selected_bins = np.concatenate(
        selected_bins
    )


    selected_q = q_magnitude[
        selected_indices
    ]


    q_vectors = reciprocal_spacing * np.column_stack(
        (
            nx[selected_indices],
            ny[selected_indices],
            nz[selected_indices]
        )
    )


    print(
        "Selected independent q modes =",
        len(q_vectors)
    )


    del nx
    del ny
    del nz
    del h2
    del q_magnitude
    del bin_id


    # ========================================================
    # FOURIER AMPLITUDES
    # ========================================================

    print()
    print("Calculating small-particle amplitudes...")


    rho_small = calculate_fourier_sum(
        r_small,
        q_vectors,
        particle_chunk_size,
        mode_chunk_size
    )


    print("Calculating large-particle amplitudes...")


    rho_large = calculate_fourier_sum(
        r_large,
        q_vectors,
        particle_chunk_size,
        mode_chunk_size
    )


    # ========================================================
    # SPECTRAL DENSITY
    # ========================================================

    m_small = sphere_form_factor(
        selected_q,
        R_small
    )

    m_large = sphere_form_factor(
        selected_q,
        R_large
    )


    amplitude = (
        m_small * rho_small
        +
        m_large * rho_large
    )


    chi_modes = (
        np.abs(amplitude)**2
        / V
    )


    # ========================================================
    # RADIAL AVERAGING
    # ========================================================

    results = []


    for current_bin in range(
        number_of_qs
    ):

        current_mask = (
            selected_bins == current_bin
        )


        if not np.any(current_mask):
            continue


        q_values = selected_q[
            current_mask
        ]

        chi_values = chi_modes[
            current_mask
        ]


        number_used = len(
            chi_values
        )


        chi_mean = np.mean(
            chi_values
        )


        if number_used > 1:

            chi_std = np.std(
                chi_values,
                ddof=1
            )

        else:

            chi_std = 0.0


        chi_sem = (
            chi_std
            / np.sqrt(number_used)
        )


        results.append(
            {
                "xs": xs,
                "q": np.mean(q_values),
                "q_bin_left": q_edges[current_bin],
                "q_bin_right": q_edges[current_bin + 1],
                "chi": chi_mean,
                "chi_std": chi_std,
                "chi_sem": chi_sem,
                "n_modes_used": number_used,
                "n_modes_available": available_modes[current_bin],
                "L": Lx,
                "d_small": d1,
                "d_large": d2,
                "N_small": N_small,
                "N_large": N_large,
                "packing_fraction": packing_fraction
            }
        )


    df = pd.DataFrame(
        results
    ).sort_values(
        "q"
    ).reset_index(
        drop=True
    )


    # Moving average is only a visual guide.

    df["chi_rolling"] = (
        df["chi"]
        .rolling(
            window=5,
            center=True,
            min_periods=1
        )
        .mean()
    )


    # ========================================================
    # POWER-LAW FITS
    # ========================================================

    low_result, low_q_fit, low_chi_fit = fit_power_law(
        df=df,
        fit_name="low_q",
        q_lower=low_fit_q_min,
        q_upper=low_fit_q_max
    )


    fit_df = pd.DataFrame(
        [
            low_result
        ]
    )


    print()
    print("Power-law fits:")
    print(fit_df.to_string(index=False))


    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    output_csv = os.path.join(
        folder_path,
        (
            f"spectral_density_xs_{xs:.2f}"
            f"_q_{q_min:.2f}_to_{q_max:.2f}.csv"
        )
    )

    fit_output_csv = os.path.join(
        folder_path,
        (
            f"spectral_density_fits_xs_{xs:.2f}.csv"
        )
    )


    df.to_csv(
        output_csv,
        index=False
    )

    fit_df.to_csv(
        fit_output_csv,
        index=False
    )


    print()
    print("Saved spectral density:")
    print(output_csv)

    print("Saved fits:")
    print(fit_output_csv)



    # ========================================================
    # LINEAR FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 10)
    )


    # --------------------------------------------------------
    # Panel 1: complete q range
    # --------------------------------------------------------

    axes[0].errorbar(
        df["q"],
        df["chi"],
        yerr=df["chi_sem"],
        marker="o",
        markersize=3,
        linewidth=0.8,
        capsize=1.5,
        color="tab:blue",
        alpha=0.65,
        label="Spectral density"
    )


    axes[0].plot(
        df["q"],
        df["chi_rolling"],
        color="darkred",
        linewidth=2.0,
        label="Five-bin moving average"
    )


    axes[0].plot(
        low_q_fit,
        low_chi_fit,
        color="green",
        linewidth=2.2,
        linestyle="--",
        label=(
            fr"$0<q\leq1$: "
            fr"$\alpha={low_result['alpha']:.2f}$, "
            fr"$R^2={low_result['r_squared']:.3f}$"
        )
    )


    axes[0].set_xlim(
        q_min,
        q_max
    )


    axes[0].set_xlabel(
        r"$q$ ($\mu\mathrm{m}^{-1}$)"
    )


    axes[0].set_ylabel(
        r"$\tilde{\chi}_V(q)$ ($\mu\mathrm{m}^3$)"
    )


    axes[0].set_title(
        fr"Full spectral density, $x_s={xs:.2f}$"
    )


    axes[0].legend(
        frameon=False,
        fontsize=9
    )


    # --------------------------------------------------------
    # Panel 2: detailed low-q fitting region
    # --------------------------------------------------------

    zoom_2 = df[
        df["q"] <= 2.0
    ]


    axes[1].errorbar(
        zoom_2["q"],
        zoom_2["chi"],
        yerr=zoom_2["chi_sem"],
        marker="o",
        markersize=4,
        linewidth=0.9,
        capsize=2,
        color="tab:blue",
        alpha=0.70,
        label="Spectral density"
    )


    axes[1].plot(
        zoom_2["q"],
        zoom_2["chi_rolling"],
        color="darkred",
        linewidth=2.0,
        label="Five-bin moving average"
    )


    axes[1].plot(
        low_q_fit,
        low_chi_fit,
        color="green",
        linewidth=2.2,
        linestyle="--",
        label=(
            fr"$0<q\leq1$: "
            fr"$\alpha={low_result['alpha']:.2f}$, "
            fr"$R^2={low_result['r_squared']:.3f}$"
        )
    )


    axes[1].set_xlim(
        0.0,
        2.0
    )


    axes[1].set_xlabel(
        r"$q$ ($\mu\mathrm{m}^{-1}$)"
    )


    axes[1].set_ylabel(
        r"$\tilde{\chi}_V(q)$ ($\mu\mathrm{m}^3$)"
    )


    axes[1].set_title(
        r"Low-$q$ power-law fit"
    )


    axes[1].legend(
        frameon=False,
        fontsize=9
    )


    plt.tight_layout()


    linear_figure = os.path.join(
        folder_path,
        f"spectral_density_xs_{xs:.2f}_fits_linear.png"
    )


    plt.savefig(
        linear_figure,
        dpi=300,
        bbox_inches="tight"
    )


    print("Saved linear figure:")
    print(linear_figure)


    plt.show()


    # ========================================================
    # LOG-LOG FIGURE
    # ========================================================

    positive_df = df[
        (df["q"] > 0)
        &
        (df["chi"] > 0)
    ]


    plt.figure(
        figsize=(8, 6)
    )


    plt.errorbar(
        positive_df["q"],
        positive_df["chi"],
        yerr=positive_df["chi_sem"],
        marker="o",
        markersize=4,
        linewidth=0.8,
        capsize=2,
        color="tab:blue",
        alpha=0.70,
        label="Spectral density"
    )


    plt.plot(
        low_q_fit,
        low_chi_fit,
        color="green",
        linewidth=2.2,
        linestyle="--",
        label=(
            fr"$0<q\leq1$: "
            fr"$\alpha={low_result['alpha']:.2f}$, "
            fr"$R^2={low_result['r_squared']:.3f}$"
        )
    )


    plt.xscale(
        "log"
    )


    plt.yscale(
        "log"
    )


    plt.xlim(
        positive_df["q"].min(),
        q_max
    )


    plt.xlabel(
        r"$q$ ($\mu\mathrm{m}^{-1}$)"
    )


    plt.ylabel(
        r"$\tilde{\chi}_V(q)$ ($\mu\mathrm{m}^3$)"
    )


    plt.title(
        fr"Power-law behavior, $x_s={xs:.2f}$"
    )


    plt.legend(
        frameon=False,
        fontsize=9
    )


    plt.tight_layout()


    log_figure = os.path.join(
        folder_path,
        f"spectral_density_xs_{xs:.2f}_fits_loglog.png"
    )


    plt.savefig(
        log_figure,
        dpi=300,
        bbox_inches="tight"
    )


    print("Saved log-log figure:")
    print(log_figure)


    plt.show()


    return df, fit_df


# ============================================================
# RUN CALCULATION
# ============================================================

df, fit_df = spectral_density(
    L=50.8,
    xs=0.50,
    d1=0.5,
    d2=0.7,
    q_min=0.0,
    q_max=6.0,
    number_of_qs=100
)