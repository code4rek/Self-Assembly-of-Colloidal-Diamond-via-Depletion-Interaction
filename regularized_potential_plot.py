# -*- coding: utf-8 -*-

"""
Plot the effective one-dimensional hybrid Hamiltonian

    U_eff(x; s)
        = s U_real(x)
        + 0.5 (1-s) lambda_max (r-r_reference)^2

for several s values.

Assumption:
One particle is fixed and the other moves directly toward it.
Therefore, this is a diagnostic slice through the full many-particle
Hamiltonian, not the complete crystal energy.
"""

import math

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Parameters
# =============================================================================

# Select interaction pair:
PAIR = "core-core"
# PAIR = "core-lobe"
# PAIR = "lobe-lobe"

RL = 0.5
SZ = 1.15

CORE_RADIUS = SZ * RL
LOBE_RADIUS = RL

BRUSH_LENGTH = 0.01
SIGMA_G = 0.09 * 1000.0**2

DEPLETANT_VOLUME_FRACTION = 0.05
DEPLETANT_RADIUS = 0.02

LAMBDA_MAX = 15000.0
# Reference separation for the selected INTER-TLPP pair.
#
# core-core:
#   First core RDF peak measured for this structure.
#
# core-lobe and lobe-lobe:
#   Brush-contact separation = r1 + r2 + 2L.

if PAIR == "core-core":
    R_REFERENCE = 1.24

elif PAIR == "core-lobe":
    R_REFERENCE = (
        CORE_RADIUS
        + LOBE_RADIUS
        + 2.0 * BRUSH_LENGTH
    )

elif PAIR == "lobe-lobe":
    R_REFERENCE = (
        2.0 * LOBE_RADIUS
        + 2.0 * BRUSH_LENGTH
    )

else:
    raise ValueError(
        "PAIR must be 'core-core', "
        "'core-lobe', or 'lobe-lobe'."
    )

print(
    f"{PAIR} reference separation: "
    f"{R_REFERENCE:.6f}"
)

S_VALUES = [
    0.0,
    1.0e-6,
    3.0e-6,
    1.0e-5,
    3.0e-5,
    7.0e-5,
    1.0e-4,
    1.5e-4,
    3.0e-4,
    1.0e-3,
    1.0e-2
]

# Surface-gap range.
X_MIN = -0.10
X_MAX = 0.2
NUMBER_OF_POINTS = 5000

# Plot the original or contact-regularized physical potential.
USE_CONTACT_REGULARIZATION = False

CONTACT_SWITCH_GAP = 4.0e-5
CONTACT_DERIVATIVE_STEP = 4.0e-5


# =============================================================================
# Choose radii
# =============================================================================

if PAIR == "core-core":
    R1 = CORE_RADIUS
    R2 = CORE_RADIUS

elif PAIR == "core-lobe":
    R1 = CORE_RADIUS
    R2 = LOBE_RADIUS

elif PAIR == "lobe-lobe":
    R1 = LOBE_RADIUS
    R2 = LOBE_RADIUS

else:
    raise ValueError(
        "PAIR must be 'core-core', 'core-lobe', or 'lobe-lobe'."
    )


CONTACT_DISTANCE = R1 + R2
X_REFERENCE = R_REFERENCE - CONTACT_DISTANCE


# =============================================================================
# Original brush + depletion potential
# =============================================================================

def physical_potential_original(rcc):
    r1_eff = BRUSH_LENGTH + R1
    r2_eff = BRUSH_LENGTH + R2

    x = rcc - R1 - R2

    q1 = r1_eff / DEPLETANT_RADIUS
    q2 = r2_eff / DEPLETANT_RADIUS
    n = rcc / DEPLETANT_RADIUS

    r_average = 2.0 / (
        1.0 / R1 + 1.0 / R2
    )

    if (
        x <= 2.0 * BRUSH_LENGTH
        and x > 0.0
        and rcc
        <= r1_eff + r2_eff
        + 2.0 * DEPLETANT_RADIUS
    ):
        brush_energy = (
            16.0
            * math.pi
            * BRUSH_LENGTH**2
            * r_average
            * SIGMA_G**1.5
            / 35.0
            * (
                28.0
                * (
                    (
                        2.0
                        * BRUSH_LENGTH
                        / x
                    )**0.25
                    - 1.0
                )
                + 20.0
                / 11.0
                * (
                    1.0
                    - (
                        x
                        / (
                            2.0
                            * BRUSH_LENGTH
                        )
                    )**2.75
                )
                + 12.0
                * (
                    x
                    / (
                        2.0
                        * BRUSH_LENGTH
                    )
                    - 1.0
                )
            )
        )

        depletion_energy = (
            -DEPLETANT_VOLUME_FRACTION
            / 16.0
            * (
                q1 + q2 + 2.0 - n
            )**2
            * (
                n
                + 2.0
                * (
                    q1 + q2 + 2.0
                )
                - 3.0
                / n
                * (
                    q1**2
                    + q2**2
                    - 2.0 * q1 * q2
                )
            )
        )

    elif (
        x > 2.0 * BRUSH_LENGTH
        and rcc
        <= r1_eff + r2_eff
        + 2.0 * DEPLETANT_RADIUS
    ):
        brush_energy = 0.0

        depletion_energy = (
            -DEPLETANT_VOLUME_FRACTION
            / 16.0
            * (
                q1 + q2 + 2.0 - n
            )**2
            * (
                n
                + 2.0
                * (
                    q1 + q2 + 2.0
                )
                - 3.0
                / n
                * (
                    q1**2
                    + q2**2
                    - 2.0 * q1 * q2
                )
            )
        )

    else:
        brush_energy = 0.0
        depletion_energy = 0.0

    return brush_energy + depletion_energy


# =============================================================================
# Contact-regularized potential
# =============================================================================

def physical_potential_regularized(rcc):
    x = rcc - CONTACT_DISTANCE

    if x >= CONTACT_SWITCH_GAP:
        return physical_potential_original(rcc)

    r_switch = (
        CONTACT_DISTANCE
        + CONTACT_SWITCH_GAP
    )

    r_next = (
        r_switch
        + CONTACT_DERIVATIVE_STEP
    )

    u_switch = physical_potential_original(
        r_switch
    )

    u_next = physical_potential_original(
        r_next
    )

    force_switch = -(
        u_next - u_switch
    ) / CONTACT_DERIVATIVE_STEP

    return (
        u_switch
        + force_switch
        * (
            CONTACT_SWITCH_GAP - x
        )
    )


def physical_potential(rcc):
    if USE_CONTACT_REGULARIZATION:
        return physical_potential_regularized(
            rcc
        )

    return physical_potential_original(rcc)


# =============================================================================
# Calculate energy curves
# =============================================================================

x_values = np.linspace(
    X_MIN,
    X_MAX,
    NUMBER_OF_POINTS,
)

r_values = CONTACT_DISTANCE + x_values

u_real = np.asarray(
    [
        physical_potential(r)
        for r in r_values
    ],
    dtype=float,
)

# One particle moves from its reference position.
displacement = r_values - R_REFERENCE

u_einstein_unscaled = (
    0.5 * displacement**2
)

u_einstein_full = (
    LAMBDA_MAX
    * u_einstein_unscaled
)


# =============================================================================
# Plot
# =============================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(17, 5.2),
)

colors = plt.cm.viridis(
    np.linspace(
        0.0,
        1.0,
        len(S_VALUES),
    )
)


# Panel 1: scaled physical potential
for s, color in zip(S_VALUES, colors):
    axes[0].plot(
        x_values,
        s * u_real,
        color=color,
        linewidth=1.8,
        label=f"s={s:g}",
    )

axes[0].axvline(
    0.0,
    color="black",
    linestyle="--",
    linewidth=1.0,
)

axes[0].set_title(
    r"Scaled physical energy: $sU_{\rm real}$"
)

axes[0].set_xlabel(
    r"Surface gap $x=r-r_1-r_2$"
)

axes[0].set_ylabel(
    r"Energy [$k_BT$]"
)

axes[0].set_yscale("symlog", linthresh=1.0)
axes[0].grid(alpha=0.25)


# Panel 2: Einstein tether energy
for s, color in zip(S_VALUES, colors):
    axes[1].plot(
        x_values,
        (1.0 - s) * u_einstein_full,
        color=color,
        linewidth=1.8,
    )

axes[1].axvline(
    0.0,
    color="black",
    linestyle="--",
    linewidth=1.0,
)

axes[1].axvline(
    X_REFERENCE,
    color="red",
    linestyle=":",
    linewidth=1.3,
    label="Reference gap",
)

axes[1].set_title(
    r"Tether energy: "
    r"$(1-s)\lambda_{\max}U_E$"
)

axes[1].set_xlabel(
    r"Surface gap $x=r-r_1-r_2$"
)

axes[1].set_ylabel(
    r"Energy [$k_BT$]"
)

axes[1].set_yscale("symlog", linthresh=1.0)
axes[1].grid(alpha=0.25)
axes[1].legend(frameon=False)


# Panel 3: total effective Hamiltonian
for s, color in zip(S_VALUES, colors):
    u_total = (
        s * u_real
        + (1.0 - s) * u_einstein_full
    )

    axes[2].plot(
        x_values,
        u_total,
        color=color,
        linewidth=1.8,
        label=f"s={s:g}",
    )

axes[2].axvline(
    0.0,
    color="black",
    linestyle="--",
    linewidth=1.0,
)

axes[2].axvline(
    X_REFERENCE,
    color="red",
    linestyle=":",
    linewidth=1.3,
)

axes[2].set_title(
    r"$U_{\rm eff}=sU_{\rm real}"
    r"+(1-s)\lambda_{\max}U_E$"
)

axes[2].set_xlabel(
    r"Surface gap $x=r-r_1-r_2$"
)

axes[2].set_ylabel(
    r"Energy [$k_BT$]"
)

axes[2].set_yscale("symlog", linthresh=1.0)
axes[2].grid(alpha=0.25)


fig.suptitle(
    (
        f"{PAIR}, "
        f"reference separation={R_REFERENCE:.5f}, "
        f"reference gap={X_REFERENCE:.5f}"
    ),
    fontsize=14,
    fontweight="bold",
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="center right",
    bbox_to_anchor=(1.015, 0.5),
    frameon=False,
)

fig.tight_layout(
    rect=[0.0, 0.0, 0.90, 0.95]
)

output_name = (
    f"effective_hamiltonian_{PAIR}.png"
)

fig.savefig(
    output_name,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"Saved: {output_name}")
print(
    f"Contact distance = {CONTACT_DISTANCE:.8f}"
)
print(
    f"Reference separation = {R_REFERENCE:.8f}"
)
print(
    f"Reference surface gap = {X_REFERENCE:.8f}"
)