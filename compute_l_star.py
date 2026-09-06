import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import miepython as mie


# ============================================================
# INPUT AND OUTPUT FILES
# ============================================================

filename = (
    "/home/akash/binary_colloids/"
    "partial_structure_factors_all_xs.csv"
)

output_file = (
    "/home/akash/binary_colloids/"
    "lstar_klstar_all_xs.csv"
)

lstar_plot_file = (
    "/home/akash/binary_colloids/"
    "lstar_vs_wavelength_all_xs.png"
)

klstar_plot_file = (
    "/home/akash/binary_colloids/"
    "klstar_vs_wavelength_all_xs.png"
)


# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

d_small = 0.5       # micron
d_large = 0.7       # micron

r_small = d_small / 2.0
r_large = d_large / 2.0

# Air holes embedded in silicon
n_hole = 1.0
n_si = 3.48


# ============================================================
# WAVELENGTH RANGE
# ============================================================

wavelengths = np.linspace(
    2.0,
    3.0,
    200
)


# ============================================================
# SCATTERING ANGLES
# ============================================================

theta = np.linspace(
    0.0,
    np.pi,
    2001
)

mu = np.cos(theta)


# ============================================================
# READ PARTIAL STRUCTURE-FACTOR CSV
# ============================================================

df = pd.read_csv(filename)


# ============================================================
# GET q VALUES FROM COLUMN NAMES
# ============================================================

S11_columns = [
    col for col in df.columns
    if col.startswith("S11_q_")
]

S22_columns = [
    col for col in df.columns
    if col.startswith("S22_q_")
]

S12_columns = [
    col for col in df.columns
    if col.startswith("S12_q_")
]


q11 = np.array([
    float(col.replace("S11_q_", ""))
    for col in S11_columns
])

q22 = np.array([
    float(col.replace("S22_q_", ""))
    for col in S22_columns
])

q12 = np.array([
    float(col.replace("S12_q_", ""))
    for col in S12_columns
])


# Sort q values and corresponding column names
order11 = np.argsort(q11)
order22 = np.argsort(q22)
order12 = np.argsort(q12)

q11 = q11[order11]
q22 = q22[order22]
q12 = q12[order12]

S11_columns = [
    S11_columns[i]
    for i in order11
]

S22_columns = [
    S22_columns[i]
    for i in order22
]

S12_columns = [
    S12_columns[i]
    for i in order12
]


# ============================================================
# INTERPOLATION FUNCTION
#
# Removes NaN bins before interpolation.
# ============================================================

def interpolate_S(q_target, q_data, S_data):

    good = (
        np.isfinite(q_data)
        & np.isfinite(S_data)
    )

    q_good = q_data[good]
    S_good = S_data[good]

    if len(q_good) == 0:
        return np.zeros_like(q_target)

    return np.interp(
        q_target,
        q_good,
        S_good
    )


# ============================================================
# MIE AMPLITUDES
# ============================================================

def mie_amplitudes(radius, wavelength):

    # Wavevector inside silicon
    k = (
        2.0
        * np.pi
        * n_si
        / wavelength
    )

    # Mie size parameter
    x = k * radius

    # Relative refractive index:
    # air hole divided by silicon host
    m = n_hole / n_si

    S1, S2 = mie.S1_S2(
        m,
        x,
        mu,
        norm="wiscombe"
    )

    # Convert dimensionless Mie amplitudes
    # into physical scattering amplitudes
    f1 = S1 / k
    f2 = S2 / k

    return f1, f2


# ============================================================
# STORAGE
# ============================================================

all_results = []


# ============================================================
# LOOP OVER COMPOSITIONS
# ============================================================

for row_index, row in df.iterrows():

    xs = float(row["xs"])

    N1 = int(row["N_small"])
    N2 = int(row["N_large"])

    L = float(row["Lx"])

    volume = L**3

    rho1 = N1 / volume
    rho2 = N2 / volume

    print("\n============================================")
    print(f"x_s = {xs:.2f}")
    print("N_small =", N1)
    print("N_large =", N2)
    print("rho_small =", rho1)
    print("rho_large =", rho2)
    print("============================================")


    # ========================================================
    # STRUCTURE FACTORS FOR THIS COMPOSITION
    # ========================================================

    S11_data = row[S11_columns].to_numpy(
        dtype=float
    )

    S22_data = row[S22_columns].to_numpy(
        dtype=float
    )

    S12_data = row[S12_columns].to_numpy(
        dtype=float
    )


    # ========================================================
    # LOOP OVER WAVELENGTH
    # ========================================================

    for wavelength in wavelengths:

        # Wavevector in silicon
        k = (
            2.0
            * np.pi
            * n_si
            / wavelength
        )


        # ----------------------------------------------------
        # Momentum-transfer wavevector q(theta)
        # ----------------------------------------------------

        q = (
            2.0
            * k
            * np.sin(theta / 2.0)
        )


        # ----------------------------------------------------
        # Interpolate partial structure factors
        # ----------------------------------------------------

        if N1 > 0:

            S11 = interpolate_S(
                q,
                q11,
                S11_data
            )

        else:

            S11 = np.zeros_like(q)


        if N2 > 0:

            S22 = interpolate_S(
                q,
                q22,
                S22_data
            )

        else:

            S22 = np.zeros_like(q)


        if N1 > 0 and N2 > 0:

            S12 = interpolate_S(
                q,
                q12,
                S12_data
            )

        else:

            S12 = np.zeros_like(q)


        # ----------------------------------------------------
        # SMALL-HOLE MIE SCATTERING
        # ----------------------------------------------------

        if N1 > 0:

            f1_perp, f1_para = mie_amplitudes(
                r_small,
                wavelength
            )

            dsigma1 = 0.5 * (
                np.abs(f1_perp)**2
                +
                np.abs(f1_para)**2
            )

        else:

            f1_perp = None
            f1_para = None

            dsigma1 = np.zeros_like(theta)


        # ----------------------------------------------------
        # LARGE-HOLE MIE SCATTERING
        # ----------------------------------------------------

        if N2 > 0:

            f2_perp, f2_para = mie_amplitudes(
                r_large,
                wavelength
            )

            dsigma2 = 0.5 * (
                np.abs(f2_perp)**2
                +
                np.abs(f2_para)**2
            )

        else:

            f2_perp = None
            f2_para = None

            dsigma2 = np.zeros_like(theta)


        # ----------------------------------------------------
        # CROSS MIE TERM
        # ----------------------------------------------------

        if N1 > 0 and N2 > 0:

            cross = 0.5 * np.real(

                f1_perp
                * np.conjugate(f2_perp)

                +

                f1_para
                * np.conjugate(f2_para)
            )

        else:

            cross = np.zeros_like(theta)


        # ----------------------------------------------------
        # CORRELATED DIFFERENTIAL SCATTERING
        # ----------------------------------------------------

        K = (
            rho1
            * dsigma1
            * S11

            +

            rho2
            * dsigma2
            * S22

            +

            2.0
            * np.sqrt(rho1 * rho2)
            * cross
            * S12
        )


        # ----------------------------------------------------
        # TRANSPORT INTEGRAL
        # ----------------------------------------------------

        integrand = (
            K
            * (1.0 - np.cos(theta))
            * np.sin(theta)
        )


        inverse_lstar = (
            2.0
            * np.pi
            * np.trapezoid(
                integrand,
                theta
            )
        )


        if inverse_lstar > 0.0:

            lstar = 1.0 / inverse_lstar

            klstar = k * lstar

        else:

            lstar = np.inf
            klstar = np.inf


        all_results.append([
            xs,
            wavelength,
            inverse_lstar,
            lstar,
            klstar
        ])


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

results = pd.DataFrame(
    all_results,
    columns=[
        "xs",
        "wavelength_um",
        "inverse_lstar_um^-1",
        "lstar_um",
        "klstar"
    ]
)

results.to_csv(
    output_file,
    index=False
)

print("\nSaved numerical results:")
print(output_file)


# ============================================================
# PLOT AND SAVE l* vs WAVELENGTH
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

for xs in sorted(results["xs"].unique()):

    sub = results[
        np.isclose(
            results["xs"],
            xs
        )
    ].sort_values(
        "wavelength_um"
    )

    ax.plot(
        sub["wavelength_um"],
        sub["lstar_um"],
        linewidth=2,
        label=fr"$x_s={xs:.2f}$"
    )


ax.set_xlabel(
    r"Wavelength $\lambda$ ($\mu$m)",
    fontsize=14
)

ax.set_ylabel(
    r"Transport mean free path $l^*$ ($\mu$m)",
    fontsize=14
)

ax.tick_params(
    axis="both",
    labelsize=12
)

ax.legend(
    fontsize=9,
    ncol=2,
    frameon=False
)

ax.grid(
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    lstar_plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Saved l* plot:")
print(lstar_plot_file)


# ============================================================
# PLOT AND SAVE k l* vs WAVELENGTH
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

for xs in sorted(results["xs"].unique()):

    sub = results[
        np.isclose(
            results["xs"],
            xs
        )
    ].sort_values(
        "wavelength_um"
    )

    ax.plot(
        sub["wavelength_um"],
        sub["klstar"],
        linewidth=2,
        label=fr"$x_s={xs:.2f}$"
    )


# Ioffe–Regel reference
ax.axhline(
    y=1.0,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=r"Ioffe–Regel: $k l^*=1$"
)

ax.set_xlabel(
    r"Wavelength $\lambda$ ($\mu$m)",
    fontsize=14
)

ax.set_ylabel(
    r"$k l^*$",
    fontsize=14
)

ax.tick_params(
    axis="both",
    labelsize=12
)

ax.legend(
    fontsize=9,
    ncol=2,
    frameon=False
)

ax.grid(
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    klstar_plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Saved k l* plot:")
print(klstar_plot_file)