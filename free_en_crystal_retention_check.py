# -*- coding: utf-8 -*-

"""RDF-retention audit for diamond/MTAB at cr=0.5 and 0.9."""

import csv
import re
from pathlib import Path

import freud
import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

PARENT_FOLDER = Path(
    "/home/akash/hpc_hoomd-files/free_energy_contact_regularized"
)

CRYSTALS = ["diamond","mtab"]
CR_VALUES = [0.5,0.9]
SEEDS = [1001, 1002, 1003, 1004, 1005]

SZ = 1.15
RL = 0.5
PHI = 0.050
SIGMAP = 0.020
LAMBDA_MAX = 5000.0
N_TLPPS = 144

# Use the exact production folders. The attempt_* directory names inside the
# seed folders are intentionally not specified because each run has a unique
# timestamp/job/PID suffix.
STATE_FOLDERS = {
    ("diamond", 0.5): PARENT_FOLDER / (
        "diamond_sz_1.15_cr_0.5_rl_0.5_phi_0.050_sigmap_0.020_"
        "lambda_5000.0_N_144_contact_regularized_extended_table_v2"
    ),
    ("diamond", 0.9): PARENT_FOLDER / (
        "diamond_sz_1.15_cr_0.9_rl_0.5_phi_0.050_sigmap_0.020_"
        "lambda_5000.0_N_144_contact_regularized_extended_table_v2"
    ),
    ("mtab", 0.5): PARENT_FOLDER / (
        "mtab_sz_1.15_cr_0.5_rl_0.5_phi_0.050_sigmap_0.020_"
        "lambda_5000.0_N_144_contact_regularized_extended_table_v2"
    ),
    ("mtab", 0.9): PARENT_FOLDER / (
        "mtab_sz_1.15_cr_0.9_rl_0.5_phi_0.050_sigmap_0.020_"
        "lambda_5000.0_N_144_contact_regularized_extended_table_v2"
    ),
}

LAST_N_FRAMES = 1000
NUMBER_OF_BLOCKS = 5
REQUIRED_RETAINED_BLOCKS = 4

R_MAX = 2.105
BIN_WIDTH = 0.005
SMOOTH_SIGMA_BINS = 2.5

# A detected local maximum is counted only if its smoothed RDF height is at
# least this fraction of the first detected peak's smoothed height.
MIN_RELATIVE_PEAK_HEIGHT = 0.15

MIN_MTAB_PEAKS = 4
MAX_MTAB_PEAKS = 6

MIN_DIAMOND_PEAKS = 2
MAX_DIAMOND_PEAKS = 3
MAX_DIAMOND_FIRST_PEAK = 1.30

OUTPUT_FOLDER = PARENT_FOLDER / "all_states_rdf_retention_last_1000"
OUTPUT_RUN_CSV = OUTPUT_FOLDER / "all_states_rdf_retention_by_run.csv"
OUTPUT_S_CSV = OUTPUT_FOLDER / "all_states_rdf_retention_by_s.csv"
OUTPUT_HEATMAP = OUTPUT_FOLDER / "all_states_rdf_retention_heatmap.png"
OUTPUT_COUNT_PLOT = OUTPUT_FOLDER / "all_states_retained_seeds_by_s.png"


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------

def extract_s_value(path):
    match = re.fullmatch(r"s_(.+)", path.name)
    if match is None:
        raise ValueError(f"Invalid s-folder name: {path.name}")
    return float(match.group(1))


def discover_s_folders(state_folder):
    rows = []
    for path in state_folder.glob("s_*"):
        if not path.is_dir():
            continue
        try:
            rows.append((extract_s_value(path), path))
        except ValueError:
            continue
    return sorted(rows, key=lambda item: item[0])


def trajectory_score(path):
    try:
        with gsd.hoomd.open(name=str(path), mode="r") as trajectory:
            if len(trajectory) == 0:
                return None
            final_step = int(trajectory[-1].configuration.step)
        return final_step, path.stat().st_mtime
    except (OSError, RuntimeError, EOFError):
        return None


def find_trajectory(seed_folder):
    candidates = []

    for name in ("production.gsd", "restart.gsd"):
        direct = seed_folder / name
        if direct.is_file():
            candidates.append(direct)

    for pattern in (
        "attempt_*/production.gsd",
        "attempt_*/restart.gsd",
    ):
        candidates.extend(seed_folder.glob(pattern))

    ranked = []
    for path in candidates:
        score = trajectory_score(path)
        if score is not None:
            ranked.append((*score, path))

    return max(ranked)[2] if ranked else None


# -----------------------------------------------------------------------------
# RDF analysis
# -----------------------------------------------------------------------------

def get_core_system(frame):
    types = list(frame.particles.types)
    if "core" not in types:
        raise RuntimeError("Particle type 'core' was not found")

    core_typeid = types.index("core")
    typeids = np.asarray(frame.particles.typeid)
    positions = np.asarray(frame.particles.position, dtype=float)
    core_positions = positions[typeids == core_typeid]

    if len(core_positions) < 2:
        raise RuntimeError(f"Only {len(core_positions)} cores were found")

    box = freud.box.Box.from_box(frame.configuration.box)
    return box, core_positions


def retained_by_rule(crystal, number_of_peaks, peak_positions):
    if crystal == "mtab":
        return MIN_MTAB_PEAKS <= number_of_peaks <= MAX_MTAB_PEAKS

    if crystal == "diamond":
        if not (MIN_DIAMOND_PEAKS <= number_of_peaks <= MAX_DIAMOND_PEAKS):
            return False
        return len(peak_positions) > 0 and peak_positions[0] < MAX_DIAMOND_FIRST_PEAK

    raise ValueError(f"Unknown crystal: {crystal}")


def calculate_rdf(trajectory, frame_indices, crystal):
    rdf = freud.density.RDF(
        bins=int(round(R_MAX / BIN_WIDTH)),
        r_max=R_MAX,
    )
    timesteps = []

    for count, frame_index in enumerate(frame_indices):
        frame = trajectory[int(frame_index)]
        box, core_positions = get_core_system(frame)
        rdf.compute(
            system=(box, core_positions),
            reset=(count == 0),
        )
        timesteps.append(int(frame.configuration.step))

    r = np.asarray(rdf.bin_centers, dtype=float)
    g = np.asarray(rdf.rdf, dtype=float)
    g_smooth = gaussian_filter1d(
        g,
        sigma=SMOOTH_SIGMA_BINS,
        mode="nearest",
    )

    candidate_indices, _ = find_peaks(g_smooth)
    candidate_indices = candidate_indices[r[candidate_indices] < R_MAX]

    # Remove weak ripples relative to the first RDF peak. The first candidate
    # means the lowest-r local maximum returned by find_peaks.
    if len(candidate_indices) > 0:
        first_peak_height = float(g_smooth[candidate_indices[0]])
        minimum_peak_height = MIN_RELATIVE_PEAK_HEIGHT * first_peak_height

        peak_indices = candidate_indices[
            g_smooth[candidate_indices] >= minimum_peak_height
        ]
    else:
        first_peak_height = np.nan
        minimum_peak_height = np.nan
        peak_indices = candidate_indices

    peak_positions = r[peak_indices]
    peak_heights = g_smooth[peak_indices]
    number_of_peaks = len(peak_positions)

    return {
        "r": r,
        "g_smooth": g_smooth,
        "peak_indices": peak_indices,
        "peak_positions": peak_positions,
        "peak_heights": peak_heights,
        "first_peak_height": first_peak_height,
        "minimum_peak_height": minimum_peak_height,
        "number_of_peaks": number_of_peaks,
        "retained": retained_by_rule(
            crystal,
            number_of_peaks,
            peak_positions,
        ),
        "first_timestep": min(timesteps),
        "last_timestep": max(timesteps),
    }


def analyze_trajectory(path, crystal, cr, s_value, seed):
    with gsd.hoomd.open(name=str(path), mode="r") as trajectory:
        total_frames = len(trajectory)
        if total_frames == 0:
            raise RuntimeError("Trajectory has zero frames")

        start = max(0, total_frames - LAST_N_FRAMES)
        indices = np.arange(start, total_frames, dtype=int)
        full = calculate_rdf(trajectory, indices, crystal)
        blocks = []

        for block_indices in np.array_split(
            indices,
            min(NUMBER_OF_BLOCKS, len(indices)),
        ):
            if len(block_indices):
                blocks.append(calculate_rdf(trajectory, block_indices, crystal))

    retained_blocks = sum(item["retained"] for item in blocks)
    retained_run = full["retained"] and retained_blocks >= min(
        REQUIRED_RETAINED_BLOCKS,
        len(blocks),
    )

    if retained_run:
        status = "retained"
    elif full["retained"] or retained_blocks > 0:
        status = "intermittent"
    else:
        status = "failed"

    row = {
        "crystal": crystal,
        "cr": cr,
        "s": s_value,
        "seed": seed,
        "state_folder": str(path.parents[1]),
        "s_folder": str(path.parents[0]),
        "trajectory": str(path),
        "total_gsd_frames": total_frames,
        "tail_frames_analyzed": len(indices),
        "first_tail_timestep": full["first_timestep"],
        "last_tail_timestep": full["last_timestep"],
        "full_tail_number_of_peaks": full["number_of_peaks"],
        "full_tail_peak_positions": ";".join(
            f"{value:.6f}" for value in full["peak_positions"]
        ),
        "full_tail_peak_heights": ";".join(
            f"{value:.6f}" for value in full["peak_heights"]
        ),
        "full_tail_first_peak_height": full["first_peak_height"],
        "minimum_counted_peak_height": full["minimum_peak_height"],
        "full_tail_retained": full["retained"],
        "block_peak_counts": ";".join(
            str(item["number_of_peaks"]) for item in blocks
        ),
        "retained_blocks": retained_blocks,
        "number_of_blocks": len(blocks),
        "retained_block_fraction": retained_blocks / len(blocks),
        "retained_run": retained_run,
        "retention_status": status,
        "error_type": "",
        "error": "",
    }
    return row


def error_row(crystal, cr, s_value, seed, state_folder, s_folder,
              trajectory, error_type, error):
    return {
        "crystal": crystal,
        "cr": cr,
        "s": s_value,
        "seed": seed,
        "state_folder": str(state_folder) if state_folder else "",
        "s_folder": str(s_folder) if s_folder else "",
        "trajectory": str(trajectory) if trajectory else "",
        "total_gsd_frames": "",
        "tail_frames_analyzed": "",
        "first_tail_timestep": "",
        "last_tail_timestep": "",
        "full_tail_number_of_peaks": "",
        "full_tail_peak_positions": "",
        "full_tail_peak_heights": "",
        "full_tail_first_peak_height": "",
        "minimum_counted_peak_height": "",
        "full_tail_retained": False,
        "block_peak_counts": "",
        "retained_blocks": "",
        "number_of_blocks": "",
        "retained_block_fraction": "",
        "retained_run": False,
        "retention_status": "ERROR",
        "error_type": error_type,
        "error": str(error),
    }


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_s(run_rows):
    groups = {}
    for row in run_rows:
        key = (row["crystal"], float(row["cr"]), float(row["s"]))
        groups.setdefault(key, []).append(row)

    rows = []
    for (crystal, cr, s_value), group in sorted(groups.items()):
        valid = [row for row in group if row["retention_status"] != "ERROR"]
        retained = [row for row in valid if row["retained_run"]]
        rows.append({
            "crystal": crystal,
            "cr": cr,
            "s": s_value,
            "expected_seeds": len(SEEDS),
            "analyzed_seeds": len(valid),
            "retained_seeds": len(retained),
            "failed_or_intermittent_seeds": len(valid) - len(retained),
            "error_or_missing_seeds": len(group) - len(valid),
            "retained_fraction_of_analyzed": (
                len(retained) / len(valid) if valid else ""
            ),
            "retained_seed_ids": ";".join(
                str(row["seed"]) for row in retained
            ),
        })
    return rows


def plot_outputs(summary_rows, run_rows):
    states = [(crystal, cr) for crystal in CRYSTALS for cr in CR_VALUES]
    all_s = sorted({float(row["s"]) for row in summary_rows})
    x = np.arange(len(all_s))
    s_index = {round(value, 14): index for index, value in enumerate(all_s)}

    # Create exactly one subplot per requested state. np.atleast_1d keeps this
    # working when only one state is selected, for example diamond at cr=0.9.
    number_of_states = len(states)
    fig, axes = plt.subplots(
        number_of_states,
        1,
        figsize=(16, max(3.5, 2.0 * number_of_states)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    status_value = {"failed": 0.0, "intermittent": 0.5, "retained": 1.0}

    for axis, (crystal, cr) in zip(axes, states):
        matrix = np.full((len(SEEDS), len(all_s)), np.nan)
        for row in run_rows:
            if row["crystal"] != crystal or float(row["cr"]) != cr:
                continue
            if row["retention_status"] == "ERROR":
                continue
            i = SEEDS.index(int(row["seed"]))
            j = s_index[round(float(row["s"]), 14)]
            matrix[i, j] = status_value[row["retention_status"]]

        image = axis.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            vmin=0,
            vmax=1,
            cmap="RdYlGn",
            interpolation="nearest",
        )
        axis.set_yticks(range(len(SEEDS)))
        axis.set_yticklabels([str(seed) for seed in SEEDS])
        axis.set_ylabel(f"{crystal}\ncr={cr:.1f}\nseed")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [f"{value:.3g}" for value in all_s],
        rotation=90,
        fontsize=7,
    )
    axes[-1].set_xlabel("s")
    colorbar = fig.colorbar(
        image,
        ax=list(axes),
        fraction=0.015,
        pad=0.01,
    )

    # Use separate calls for tick locations and labels. This is compatible
    # with older Matplotlib versions where Colorbar.set_ticks() accepts only
    # the tick-location argument.
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels([
        "failed",
        "intermittent",
        "retained",
    ])
    fig.suptitle("Crystal retention from last-1000-frame core-core RDF")
    fig.subplots_adjust(left=0.10, right=0.92, bottom=0.20, top=0.94, hspace=0.18)
    fig.savefig(OUTPUT_HEATMAP, dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 5.5))
    styles = {
        ("diamond", 0.5): ("#1f77b4", "o"),
        ("diamond", 0.9): ("#17becf", "s"),
        ("mtab", 0.5): ("#d62728", "o"),
        ("mtab", 0.9): ("#ff7f0e", "s"),
    }
    for crystal, cr in states:
        selected = {
            round(float(row["s"]), 14): row
            for row in summary_rows
            if row["crystal"] == crystal and float(row["cr"]) == cr
        }
        y = [
            selected.get(round(value, 14), {}).get("retained_seeds", np.nan)
            for value in all_s
        ]
        color, marker = styles[(crystal, cr)]
        axis.plot(x, y, color=color, marker=marker, markersize=4,
                  linewidth=1.5, label=f"{crystal}, cr={cr:.1f}")

    axis.set_ylim(-0.2, len(SEEDS) + 0.2)
    axis.set_yticks(range(len(SEEDS) + 1))
    axis.set_xticks(x)
    axis.set_xticklabels([f"{value:.3g}" for value in all_s], rotation=90,
                         fontsize=7)
    axis.set_xlabel("s")
    axis.set_ylabel("retained seeds out of 5")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_COUNT_PLOT, dpi=250, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    run_rows = []

    for crystal in CRYSTALS:
        for cr in CR_VALUES:
            state_folder = STATE_FOLDERS[(crystal, cr)]
            print("\n" + "=" * 96)
            print(f"{crystal}, cr={cr:.1f}")

            if not state_folder.is_dir():
                print(f"STATE FOLDER MISSING: {state_folder}")
                continue

            print(f"Using: {state_folder}")
            s_folders = discover_s_folders(state_folder)
            print(f"Discovered {len(s_folders)} numerical s folders")

            if len(s_folders) != 61:
                print(f"WARNING: expected 61 s folders, found {len(s_folders)}")

            for s_value, s_folder in s_folders:
                for seed in SEEDS:
                    seed_folder = s_folder / f"seed_{seed}"

                    if not seed_folder.is_dir():
                        print(f"s={s_value:.8g}, seed={seed}: SEED FOLDER MISSING")
                        run_rows.append(error_row(
                            crystal, cr, s_value, seed, state_folder, s_folder,
                            None, "seed_folder_missing", seed_folder,
                        ))
                        continue

                    trajectory = find_trajectory(seed_folder)
                    if trajectory is None:
                        print(f"s={s_value:.8g}, seed={seed}: GSD MISSING/UNREADABLE")
                        run_rows.append(error_row(
                            crystal, cr, s_value, seed, state_folder, s_folder,
                            None, "trajectory_missing", seed_folder,
                        ))
                        continue

                    try:
                        row = analyze_trajectory(
                            trajectory, crystal, cr, s_value, seed,
                        )
                    except Exception as error:
                        print(f"s={s_value:.8g}, seed={seed}: ANALYSIS ERROR: {error}")
                        row = error_row(
                            crystal, cr, s_value, seed, state_folder, s_folder,
                            trajectory, "analysis_error", error,
                        )
                    run_rows.append(row)

    if not run_rows:
        raise RuntimeError("No s folders were found in any requested state")

    run_fields = list(run_rows[0].keys())
    write_csv(OUTPUT_RUN_CSV, run_rows, run_fields)

    summary_rows = summarize_by_s(run_rows)
    summary_fields = list(summary_rows[0].keys())
    write_csv(OUTPUT_S_CSV, summary_rows, summary_fields)
    plot_outputs(summary_rows, run_rows)

    print("\nBY STATE")
    print("-" * 96)
    for crystal in CRYSTALS:
        for cr in CR_VALUES:
            selected = [
                row for row in run_rows
                if row["crystal"] == crystal and float(row["cr"]) == cr
            ]
            analyzed = [row for row in selected if row["retention_status"] != "ERROR"]
            retained = sum(bool(row["retained_run"]) for row in analyzed)
            errors = len(selected) - len(analyzed)
            s_count = len({float(row["s"]) for row in selected})
            print(
                f"{crystal:8s} cr={cr:.1f}: s folders={s_count:2d}, "
                f"analyzed={len(analyzed):3d}, retained={retained:3d}, errors={errors:3d}"
            )

    print("\nSaved:")
    print(OUTPUT_RUN_CSV)
    print(OUTPUT_S_CSV)
    print(OUTPUT_HEATMAP)
    print(OUTPUT_COUNT_PLOT)


if __name__ == "__main__":
    main()
