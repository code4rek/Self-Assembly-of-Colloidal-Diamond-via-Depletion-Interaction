"""Measure intermolecular sphere overlaps around a trajectory collapse.

Edit GSD_PATH and the frame interval below, then run this file in Spyder or:

    python check_overlap_around_collapse.py

Pairs belonging to the same rigid TLPP are excluded. Surface gap is

    x = center_distance - radius_i - radius_j

The contact-regularized pair table covers x >= -0.05. Therefore, any pair with
x < -0.05 has crossed below the table minimum.
"""

import csv
from pathlib import Path

import freud
import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Edit these settings
# ---------------------------------------------------------------------------
GSD_PATH = Path(
    "/home/akash/hpc_hoomd-files/free_energy_contact_regularized/"
    "diamond_sz_1.15_cr_0.9_rl_0.5_phi_0.050_sigmap_0.020_lambda_5000.0_N_144_contact_regularized_v1/s_0.0001000000000000/seed_1002/attempt_20260830_220041_765900_pid312502/production.gsd"
)

# The collapse was observed near frame 3300.
FIRST_FRAME = 0
LAST_FRAME = 3400       # inclusive

SZ = 1.15
RL = 0.5
CONTACT_TABLE_MINIMUM_GAP = -0.05
REGULARIZATION_SWITCH_GAP = 4.0e-5

OUTPUT_CSV = GSD_PATH.parent / "overlap_around_collapse.csv"
OUTPUT_PLOT = GSD_PATH.parent / "overlap_around_collapse.png"


def minimum_image_displacements(box, positions_a, positions_b):
    """Return all A-to-B displacement vectors with periodic wrapping."""
    displacement = (
        positions_b[np.newaxis, :, :]
        - positions_a[:, np.newaxis, :]
    )
    original_shape = displacement.shape
    displacement = box.wrap(displacement.reshape(-1, 3))
    return np.asarray(displacement).reshape(original_shape)


def pair_gap_statistics(
    box,
    positions_a,
    positions_b,
    body_a,
    body_b,
    indices_a,
    indices_b,
    contact_distance,
    same_particle_type,
):
    displacement = minimum_image_displacements(
        box,
        positions_a,
        positions_b,
    )
    distance = np.linalg.norm(displacement, axis=2)

    # Intermolecular pairs only: exclude core/lobe or lobe/lobe pairs belonging
    # to the same rigid TLPP.
    valid = body_a[:, np.newaxis] != body_b[np.newaxis, :]

    # For core-core and lobe-lobe, retain each unordered pair only once.
    if same_particle_type:
        valid &= np.triu(
            np.ones(distance.shape, dtype=bool),
            k=1,
        )

    row_index, column_index = np.nonzero(valid)
    if len(row_index) == 0:
        raise RuntimeError("No valid intermolecular pairs were found.")

    gaps = (
        distance[row_index, column_index]
        - contact_distance
    )
    closest = int(np.argmin(gaps))

    return {
        "minimum_gap": float(gaps[closest]),
        "number_below_contact": int(np.count_nonzero(gaps < 0.0)),
        "number_below_switch": int(
            np.count_nonzero(gaps < REGULARIZATION_SWITCH_GAP)
        ),
        "number_below_table_minimum": int(
            np.count_nonzero(
                gaps < CONTACT_TABLE_MINIMUM_GAP
            )
        ),
        "closest_particle_a": int(indices_a[row_index[closest]]),
        "closest_particle_b": int(indices_b[column_index[closest]]),
        "closest_body_a": int(body_a[row_index[closest]]),
        "closest_body_b": int(body_b[column_index[closest]]),
    }


def prefixed(row, prefix, values):
    for key, value in values.items():
        row[f"{prefix}_{key}"] = value


def analyze_frame(frame_number, frame):
    type_names = list(frame.particles.types)
    for required in ("core", "tethered_lobe"):
        if required not in type_names:
            raise RuntimeError(
                f"Required particle type {required!r} is absent. "
                f"Available types: {type_names}"
            )

    core_typeid = type_names.index("core")
    lobe_typeid = type_names.index("tethered_lobe")

    positions = np.asarray(frame.particles.position, dtype=float)
    typeid = np.asarray(frame.particles.typeid)
    body = np.asarray(frame.particles.body)
    particle_indices = np.arange(len(positions), dtype=int)

    core_mask = typeid == core_typeid
    lobe_mask = typeid == lobe_typeid

    core_positions = positions[core_mask]
    lobe_positions = positions[lobe_mask]
    core_body = body[core_mask]
    lobe_body = body[lobe_mask]
    core_indices = particle_indices[core_mask]
    lobe_indices = particle_indices[lobe_mask]

    if len(core_positions) != 144 or len(lobe_positions) != 576:
        print(
            f"Warning at frame {frame_number}: found "
            f"{len(core_positions)} cores and {len(lobe_positions)} lobes."
        )

    box = freud.box.Box.from_box(frame.configuration.box)

    core_radius = SZ * RL
    lobe_radius = RL

    cc = pair_gap_statistics(
        box,
        core_positions,
        core_positions,
        core_body,
        core_body,
        core_indices,
        core_indices,
        contact_distance=2.0 * core_radius,
        same_particle_type=True,
    )
    cl = pair_gap_statistics(
        box,
        core_positions,
        lobe_positions,
        core_body,
        lobe_body,
        core_indices,
        lobe_indices,
        contact_distance=core_radius + lobe_radius,
        same_particle_type=False,
    )
    ll = pair_gap_statistics(
        box,
        lobe_positions,
        lobe_positions,
        lobe_body,
        lobe_body,
        lobe_indices,
        lobe_indices,
        contact_distance=2.0 * lobe_radius,
        same_particle_type=True,
    )

    row = {
        "frame": frame_number,
        "timestep": int(frame.configuration.step),
    }
    prefixed(row, "core_core", cc)
    prefixed(row, "core_lobe", cl)
    prefixed(row, "lobe_lobe", ll)
    return row


def write_csv(rows):
    with OUTPUT_CSV.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows):
    frames = np.asarray([row["frame"] for row in rows])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
    )

    pair_styles = [
        ("core_core", "core-core", "#3264A8"),
        ("core_lobe", "core-lobe", "#D26A3A"),
        ("lobe_lobe", "lobe-lobe", "#148F77"),
    ]

    for prefix, label, color in pair_styles:
        minimum_gap = np.asarray(
            [row[f"{prefix}_minimum_gap"] for row in rows]
        )
        below_table = np.asarray(
            [
                row[f"{prefix}_number_below_table_minimum"]
                for row in rows
            ]
        )

        axes[0].plot(
            frames,
            minimum_gap,
            label=label,
            color=color,
            linewidth=1.5,
        )
        axes[1].plot(
            frames,
            below_table,
            label=label,
            color=color,
            linewidth=1.5,
        )

    axes[0].axhline(
        0.0,
        color="black",
        linestyle=":",
        label="contact",
    )
    axes[0].axhline(
        CONTACT_TABLE_MINIMUM_GAP,
        color="red",
        linestyle="--",
        label=r"table minimum $x=-0.05$",
    )
    axes[0].set_ylabel(r"Minimum surface gap $x$")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("GSD frame")
    axes[1].set_ylabel(r"Number of pairs with $x<-0.05$")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    fig.suptitle("Intermolecular overlaps around structural collapse")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    if not GSD_PATH.exists():
        raise FileNotFoundError(
            "Edit GSD_PATH at the top of the script. File not found: "
            f"{GSD_PATH}"
        )

    with gsd.hoomd.open(name=str(GSD_PATH), mode="r") as trajectory:
        number_of_frames = len(trajectory)
        first = max(0, FIRST_FRAME)
        last = min(LAST_FRAME, number_of_frames - 1)

        if first > last:
            raise ValueError(
                f"Requested frames {FIRST_FRAME} to {LAST_FRAME}, but the "
                f"trajectory contains {number_of_frames} frames."
            )

        print(
            f"Reading frames {first} through {last} "
            f"from {number_of_frames} total frames..."
        )

        rows = []
        for frame_number in range(first, last + 1):
            rows.append(
                analyze_frame(
                    frame_number,
                    trajectory[frame_number],
                )
            )
            if (frame_number - first + 1) % 25 == 0:
                print(f"Processed through frame {frame_number}")

    write_csv(rows)
    make_plot(rows)

    print("\nOVERLAP CHECK")
    print("=" * 88)
    for prefix, label in (
        ("core_core", "core-core"),
        ("core_lobe", "core-lobe"),
        ("lobe_lobe", "lobe-lobe"),
    ):
        minimum_row = min(
            rows,
            key=lambda row: row[f"{prefix}_minimum_gap"],
        )
        first_below = next(
            (
                row for row in rows
                if row[f"{prefix}_number_below_table_minimum"] > 0
            ),
            None,
        )

        print(
            f"{label:10s}: minimum x="
            f"{minimum_row[f'{prefix}_minimum_gap']:+.8f} "
            f"at frame={minimum_row['frame']}, "
            f"timestep={minimum_row['timestep']}"
        )
        if first_below is None:
            print(" " * 12 + "No pair crossed below x=-0.05.")
        else:
            print(
                " " * 12
                + f"First x<-0.05 at frame={first_below['frame']}, "
                f"timestep={first_below['timestep']}, "
                f"count={first_below[f'{prefix}_number_below_table_minimum']}"
            )

    print("\nSaved:")
    print(OUTPUT_CSV)
    print(OUTPUT_PLOT)


if __name__ == "__main__":
    main()