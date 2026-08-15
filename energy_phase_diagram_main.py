#!/usr/bin/env python3
"""
Energy phase diagram gated by smoothed-RDF crystal classification.

Protocol per (sz, cr):

  1. For each candidate crystal file:
        stabSPRING_diamond_...
        stabSPRING_mtab_...
        stabSPRING_rhombo_...

     compute the core-core RDF of the last frame, smooth g(r), detect peaks,
     and classify the structure by RDF peak count:

        0 peaks -> amorphous
        1 peak  -> amorphous
        4 peaks -> mtab
        2 or 3 peaks and first peak < 1.3  -> diamond
        2 or 3 peaks and first peak >= 1.3 -> rhombo
        >4 peaks -> amorphous

  2. A candidate crystal is included in the energy comparison only if
     the RDF classification matches the candidate file type.

     Example:
        diamond file classified as diamond -> include diamond energy
        diamond file classified as rhombo  -> exclude diamond energy
        mtab file classified as mtab       -> include mtab energy
        rhombo file classified as rhombo   -> include rhombo energy

  3. Among the included crystals, compute mean energy per TLPP over the
     last n_last_energy frames.

  4. Lowest mean energy wins.

Markers:
    diamond   = D
    mtab      = s
    rhombo    = h
    amorphous = gray triangle
"""

import os
import csv
import numpy as np
import gsd.hoomd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import freud
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# ============================================================
# Energy log settings
# ============================================================

KE_KEY = "md/compute/ThermodynamicQuantities/kinetic_energy"
PE_KEY = "md/compute/ThermodynamicQuantities/potential_energy"

CORE_TYPE_NAMES = ("core", "frozen_core")


# ============================================================
# Smoothed-RDF classification settings
# ============================================================

RDF_R_MAX = 2.105
RDF_BIN_WIDTH = 0.005
RDF_SEARCH_PAST_R_MAX = 0.10

RDF_SMOOTH = True
RDF_SMOOTH_SIGMA_BINS = 2.5

# Baseline value
RDF_MIN_PEAK_G = 2.0

# Actual counted-peak cutoff
# A peak is counted only if its smoothed height is above this value.
RDF_PEAK_COUNT_MIN_G = 26.0

# Keep 0.0 if you want pure height > 20 rule.
RDF_MIN_PEAK_HEIGHT_FRACTION = 0.0

# Prominence cutoff to suppress small shoulders
RDF_MIN_PEAK_PROMINENCE = 2.0

# Diamond vs rhombo first-peak cutoff
RDF_FIRST_PEAK_R_CUTOFF = 1.3

# RDF particle selection
RDF_CORE_TYPE_NAMES = ("core", "frozen_core")

# Set to None for no z-cut.
# For stability/bulk files, None is usually better.
RDF_Z_CUT = None


# ============================================================
# File naming
# ============================================================

def default_fname(folder, crystal, sz, cr):
    return (
        f"{folder}/stabSPRING_{crystal}_"
        f"sz_{sz:.2f}_cr_{cr:.1f}_"
        f"dpl_fr_0.050_dpl_sz_0.02_kT_1.0_"
        f"spring_k100_hold10000000_pad8.0.gsd"
    )


# ============================================================
# RDF helper functions
# ============================================================

def get_core_positions_for_rdf(frame, core_type_names=RDF_CORE_TYPE_NAMES, z_cut=RDF_Z_CUT):
    """
    Select core positions for RDF.

    Includes:
        core
        frozen_core

    Excludes:
        lobe
        frozen_lobe
    """
    types = frame.particles.types
    typeid = frame.particles.typeid

    core_tids = [
        i for i, t in enumerate(types)
        if t in core_type_names
    ]

    is_core = np.isin(typeid, core_tids)
    pos = frame.particles.position[is_core].astype(np.float32)

    if z_cut is not None:
        pos = pos[pos[:, 2] < z_cut]

    return pos


def rdf_from_positions(pos, box, r_max=RDF_R_MAX):
    """
    Compute RDF from selected positions.

    Periodic boundary conditions are included through the freud/HOOMD box.
    """
    r_requested_max = r_max + RDF_SEARCH_PAST_R_MAX

    n_bins = int(np.ceil(r_requested_max / RDF_BIN_WIDTH))
    r_compute_max = n_bins * RDF_BIN_WIDTH

    rdf = freud.density.RDF(bins=n_bins, r_max=r_compute_max)
    rdf.compute(system=(freud.box.Box.from_box(box), pos))

    return rdf.bin_centers, rdf.rdf


def smooth_rdf(g, sigma_bins=RDF_SMOOTH_SIGMA_BINS):
    """
    Smooth RDF using a Gaussian filter.

    sigma_bins is in units of RDF bins.
    With RDF_BIN_WIDTH = 0.005 and sigma_bins = 2,
    smoothing length is about 0.01 in r.
    """
    if not RDF_SMOOTH:
        return g

    return gaussian_filter1d(g, sigma=sigma_bins)


def find_counted_peaks(r, g, r_max=RDF_R_MAX):
    """
    Smooth g(r), then detect and count RDF peaks.

    A counted peak must have smoothed height above RDF_PEAK_COUNT_MIN_G.
    Peaks are returned sorted by increasing r.
    """
    if len(r) < 2 or np.max(g) <= 0:
        return np.array([]), np.array([]), g

    g_smooth = smooth_rdf(g)

    height_cutoff = max(
        RDF_MIN_PEAK_G,
        RDF_PEAK_COUNT_MIN_G,
        RDF_MIN_PEAK_HEIGHT_FRACTION * np.max(g_smooth)
    )

    peak_idx, props = find_peaks(
        g_smooth,
        height=height_cutoff,
        prominence=RDF_MIN_PEAK_PROMINENCE
    )

    peak_r = r[peak_idx]
    peak_g = g_smooth[peak_idx]

    keep = peak_r < r_max
    peak_r = peak_r[keep]
    peak_g = peak_g[keep]

    order = np.argsort(peak_r)

    return peak_r[order], peak_g[order], g_smooth


def classify_from_rdf_peaks(peak_r, peak_g):
    """
    Smoothed-RDF classification rule:

        0 peaks -> amorphous
        1 peak  -> amorphous
        4 peaks -> mtab
        2 or 3 peaks and first peak < 1.3  -> diamond
        2 or 3 peaks and first peak >= 1.3 -> rhombo
        >4 peaks -> amorphous
    """
    n_peaks = len(peak_r)

    if n_peaks == 0:
        return "amorphous", "no_peaks"

    first_peak_r = peak_r[0]

    if n_peaks == 1:
        return "amorphous", f"one_peak_first_peak_{first_peak_r:.4f}"

    if n_peaks == 4:
        return "mtab", f"four_peaks_first_peak_{first_peak_r:.4f}"

    if n_peaks in (2, 3):
        if first_peak_r < RDF_FIRST_PEAK_R_CUTOFF:
            return "diamond", f"n_peaks_{n_peaks}_first_peak_{first_peak_r:.4f}_below_1.3"
        else:
            return "rhombo", f"n_peaks_{n_peaks}_first_peak_{first_peak_r:.4f}_above_1.3"

    return "amorphous", f"n_peaks_{n_peaks}_first_peak_{first_peak_r:.4f}"


def choose_verdict(labels):
    """
    Choose final verdict from frame labels.

    If there is a tie, classify conservatively as amorphous.
    """
    if len(labels) == 0:
        return "amorphous", 0.0

    unique, counts = np.unique(labels, return_counts=True)

    max_count = np.max(counts)
    tied = unique[counts == max_count]

    if len(tied) > 1:
        return "amorphous", max_count / len(labels)

    verdict = tied[0]
    frac = max_count / len(labels)

    return verdict, frac


def check_persistence(path, crystal=None, n_last=1, verbose=False):
    """
    Local replacement for mtab_rdf_peaks_sweep.check_persistence().

    Computes smoothed RDF for the last n_last frames and classifies each frame.
    The verdict is the most frequent classification among those frames.

    If n_last=1, this uses only the final frame.
    """
    labels = []
    reasons = []
    peak_counts = []
    first_peak_rs = []
    peak_positions_all = []
    peak_heights_all = []

    with gsd.hoomd.open(path, "r") as traj:
        n_frames = len(traj)

        if n_frames == 0:
            return {
                "verdict": "amorphous",
                "frac_persist": 0.0,
                "labels": [],
                "reasons": ["empty_trajectory"],
                "peak_counts": [],
                "first_peak_rs": [],
                "peak_positions": [],
                "peak_heights": [],
            }

        start = max(0, n_frames - n_last)

        for i in range(start, n_frames):
            frame = traj[i]

            pos = get_core_positions_for_rdf(frame)

            if len(pos) < 5:
                labels.append("amorphous")
                reasons.append("too_few_core_particles")
                peak_counts.append(0)
                first_peak_rs.append(np.nan)
                peak_positions_all.append([])
                peak_heights_all.append([])
                continue

            r, g = rdf_from_positions(pos, frame.configuration.box)
            peak_r, peak_g, g_smooth = find_counted_peaks(r, g)

            label, reason = classify_from_rdf_peaks(peak_r, peak_g)

            labels.append(label)
            reasons.append(reason)
            peak_counts.append(len(peak_r))

            if len(peak_r) > 0:
                first_peak_rs.append(float(peak_r[0]))
            else:
                first_peak_rs.append(np.nan)

            peak_positions_all.append([float(x) for x in peak_r])
            peak_heights_all.append([float(x) for x in peak_g])

    verdict, frac_persist = choose_verdict(labels)

    median_peak_count = float(np.median(peak_counts)) if peak_counts else 0.0

    if np.any(~np.isnan(first_peak_rs)):
        median_first_peak_r = float(np.nanmedian(first_peak_rs))
    else:
        median_first_peak_r = np.nan

    if verbose:
        print(
            f"{os.path.basename(path)}: verdict={verdict}, "
            f"frac={frac_persist:.2f}, labels={labels}, "
            f"peak_counts={peak_counts}, first_peak_rs={first_peak_rs}"
        )

    return {
        "verdict": verdict,
        "frac_persist": frac_persist,
        "labels": labels,
        "reasons": reasons,
        "peak_counts": peak_counts,
        "first_peak_rs": first_peak_rs,
        "median_peak_count": median_peak_count,
        "median_first_peak_r": median_first_peak_r,
        "peak_positions": peak_positions_all,
        "peak_heights": peak_heights_all,
    }


# ============================================================
# Energy helper functions
# ============================================================

def read_energy_per_frame(path, core_type_names=CORE_TYPE_NAMES, use_pe_only=False):
    """
    Returns:
        frames, E_per_N, N_total

    E_per_N = (KE + PE) / N_total_cores by default,
    or PE / N_total_cores if use_pe_only=True.
    """
    with gsd.hoomd.open(path, "r") as f:
        frame0 = f[0]

        types = frame0.particles.types
        typeid = frame0.particles.typeid

        core_tids = [
            k for k, t in enumerate(types)
            if t in core_type_names
        ]

        is_core = np.isin(typeid, core_tids)
        n_total = int(is_core.sum())

        n_frames = len(f)

        KE = np.empty(n_frames)
        PE = np.empty(n_frames)

        for i, frame in enumerate(f):
            KE[i] = frame.log[KE_KEY][0]
            PE[i] = frame.log[PE_KEY][0]

    E_per_N = (PE / n_total) if use_pe_only else ((KE + PE) / n_total)

    return np.arange(n_frames), E_per_N, n_total


def mean_energy_last_n(path, n_last=50, use_pe_only=False):
    """
    Mean and std of logged energy per TLPP over the last n_last frames.
    """
    frames, E_per_N, n_total = read_energy_per_frame(
        path,
        use_pe_only=use_pe_only
    )

    n_use = min(n_last, len(E_per_N))
    window = E_per_N[-n_use:]

    return float(window.mean()), float(window.std()), n_total, n_use


# ============================================================
# Energy phase diagram
# ============================================================

def energy_phase_diagram(
        sz_values,
        cr_values,
        folder,
        crystals=("diamond", "mtab", "rhombo"),
        n_last_energy=50,
        n_last_persistence=1,
        use_pe_only=False,
        save_png="energy_phase_diagram.png",
        save_csv="energy_phase_diagram.csv",
        fname_fn=None,
        verbose=True):
    """
    For each (sz, cr):

      1. GATE:
         classify each crystal's GSD file by smoothed RDF.

      2. Include candidate crystal in energy comparison only if:
             RDF verdict == candidate crystal name

      3. Among included crystals:
             compute mean energy over the last n_last_energy frames.

      4. Lowest energy wins.

    If no candidate passes the RDF gate, point is amorphous.
    """
    if fname_fn is None:
        fname_fn = lambda crystal, sz, cr: default_fname(folder, crystal, sz, cr)

    ENERGY_TIE_THRESHOLD = 0.005

    MARKER_SIZE = 150
    MARKER_SIZE_CORRECTION = {
        "D": 1.0,
        "s": 1.0,
        "h": 1.0 / 0.8,
        "^": 1.0,
    }

    marker_map = {
        "diamond": "D",
        "mtab": "s",
        "rhombo": "h",
    }

    color_map = {
        "diamond": "#1f77b4",
        "mtab": "#d62728",
        "rhombo": "#2ca02c",
    }

    label_map = {
        "diamond": "diamond",
        "mtab": "mtab",
        "rhombo": "rhombohedral",
    }

    results = []

    fig, ax = plt.subplots(figsize=(8, 6))

    legend_used = {c: False for c in crystals}
    legend_used["amorphous"] = False

    for sz in sz_values:
        for cr in cr_values:

            # ------------------------------------------------
            # STEP 1: smoothed-RDF classification gate
            # ------------------------------------------------
            persisting = []
            gate_results = {}

            for crystal in crystals:
                path = fname_fn(crystal, sz, cr)

                if not os.path.exists(path):
                    gate_results[crystal] = {
                        "verdict": "missing",
                        "frac_persist": 0.0,
                        "median_peak_count": np.nan,
                        "median_first_peak_r": np.nan,
                    }

                    if verbose:
                        print(f"missing: {crystal:8s} sz={sz:.2f} cr={cr:.1f}")

                    continue

                res = check_persistence(
                    path,
                    crystal=crystal,
                    n_last=n_last_persistence,
                    verbose=False
                )

                gate_results[crystal] = res

                if res["verdict"] == crystal:
                    persisting.append(crystal)
                elif verbose:
                    print(
                        f"  [{crystal}] sz={sz:.2f} cr={cr:.1f}: "
                        f"classified as {res['verdict']} "
                        f"(frac {res['frac_persist']*100:.0f}%, "
                        f"median peaks {res.get('median_peak_count', np.nan)}, "
                        f"median first peak {res.get('median_first_peak_r', np.nan)}) "
                        f"-- excluded from energy comparison"
                    )

            # ------------------------------------------------
            # If no crystal survived gate
            # ------------------------------------------------
            if not persisting:
                results.append({
                    "sz": sz,
                    "cr": cr,
                    "winner": "amorphous",
                    "n_compared": 0,
                    "n_tied": 0,
                    "tied_with": "",
                    "E_diamond": np.nan,
                    "E_mtab": np.nan,
                    "E_rhombo": np.nan,
                    "std_diamond": np.nan,
                    "std_mtab": np.nan,
                    "std_rhombo": np.nan,
                    "gate_diamond": gate_results.get("diamond", {}).get("verdict", "missing"),
                    "gate_mtab": gate_results.get("mtab", {}).get("verdict", "missing"),
                    "gate_rhombo": gate_results.get("rhombo", {}).get("verdict", "missing"),
                    "peaks_diamond": gate_results.get("diamond", {}).get("median_peak_count", np.nan),
                    "peaks_mtab": gate_results.get("mtab", {}).get("median_peak_count", np.nan),
                    "peaks_rhombo": gate_results.get("rhombo", {}).get("median_peak_count", np.nan),
                    "first_peak_diamond": gate_results.get("diamond", {}).get("median_first_peak_r", np.nan),
                    "first_peak_mtab": gate_results.get("mtab", {}).get("median_first_peak_r", np.nan),
                    "first_peak_rhombo": gate_results.get("rhombo", {}).get("median_first_peak_r", np.nan),
                })

                if verbose:
                    print(
                        f"sz={sz:.2f} cr={cr:.1f}: "
                        f"winner=amorphous; no candidate matched its RDF classification"
                    )

                ax.scatter(
                    sz,
                    cr,
                    marker="^",
                    s=MARKER_SIZE * MARKER_SIZE_CORRECTION["^"],
                    color="gray",
                    edgecolor="black",
                    linewidth=1.0,
                    label="amorphous" if not legend_used["amorphous"] else None,
                    zorder=3
                )

                legend_used["amorphous"] = True

                continue

            # ------------------------------------------------
            # STEP 2: energy comparison among persisting crystals
            # ------------------------------------------------
            energies = {}
            stds = {}

            for crystal in persisting:
                path = fname_fn(crystal, sz, cr)

                try:
                    mean_E, std_E, n_total, n_use = mean_energy_last_n(
                        path,
                        n_last=n_last_energy,
                        use_pe_only=use_pe_only
                    )

                except KeyError as exc:
                    if verbose:
                        print(f"missing energy log key in {path}: {exc}")
                    continue

                energies[crystal] = mean_E
                stds[crystal] = std_E

            if not energies:
                if verbose:
                    print(
                        f"skip sz={sz:.2f} cr={cr:.1f}: "
                        f"RDF gate passed but no usable energy files"
                    )
                continue

            winner = min(energies, key=energies.get)
            min_E = energies[winner]

            near_tied = sorted(
                [
                    c for c, E in energies.items()
                    if E - min_E < ENERGY_TIE_THRESHOLD
                ],
                key=lambda c: energies[c]
            )

            results.append({
                "sz": sz,
                "cr": cr,
                "winner": winner,
                "n_compared": len(energies),
                "n_tied": len(near_tied),
                "tied_with": ",".join(near_tied),
                "E_diamond": energies.get("diamond", np.nan),
                "E_mtab": energies.get("mtab", np.nan),
                "E_rhombo": energies.get("rhombo", np.nan),
                "std_diamond": stds.get("diamond", np.nan),
                "std_mtab": stds.get("mtab", np.nan),
                "std_rhombo": stds.get("rhombo", np.nan),
                "gate_diamond": gate_results.get("diamond", {}).get("verdict", "missing"),
                "gate_mtab": gate_results.get("mtab", {}).get("verdict", "missing"),
                "gate_rhombo": gate_results.get("rhombo", {}).get("verdict", "missing"),
                "peaks_diamond": gate_results.get("diamond", {}).get("median_peak_count", np.nan),
                "peaks_mtab": gate_results.get("mtab", {}).get("median_peak_count", np.nan),
                "peaks_rhombo": gate_results.get("rhombo", {}).get("median_peak_count", np.nan),
                "first_peak_diamond": gate_results.get("diamond", {}).get("median_first_peak_r", np.nan),
                "first_peak_mtab": gate_results.get("mtab", {}).get("median_first_peak_r", np.nan),
                "first_peak_rhombo": gate_results.get("rhombo", {}).get("median_first_peak_r", np.nan),
            })

            if verbose:
                tie_str = (
                    f" [within {ENERGY_TIE_THRESHOLD} kT: {near_tied}]"
                    if len(near_tied) > 1 else ""
                )

                print(
                    f"sz={sz:.2f} cr={cr:.1f}: "
                    f"RDF-persisting={persisting}, "
                    f"winner={winner:8s}, "
                    f"energies={ {k: round(v, 4) for k, v in energies.items()} }"
                    f"{tie_str}"
                )

            # ------------------------------------------------
            # Plot tied/winning crystals
            # ------------------------------------------------
            alpha = 1.0 / len(near_tied)

            for crystal in near_tied:
                marker = marker_map[crystal]

                label = label_map[crystal] if not legend_used[crystal] else None

                ax.scatter(
                    sz,
                    cr,
                    marker=marker,
                    s=MARKER_SIZE * MARKER_SIZE_CORRECTION[marker],
                    color=color_map[crystal],
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=alpha,
                    label=label,
                    zorder=3,
                )

                legend_used[crystal] = True

    # ========================================================
    # Kissing curve
    # ========================================================

    cr_curve = np.linspace(min(cr_values), max(cr_values), 400)
    sz_curve = (cr_curve / np.sqrt(6.0)) + np.sqrt(1.0 - (cr_curve**2 / 3.0))

    ax.plot(
        sz_curve,
        cr_curve,
        "k--",
        linewidth=2.5,
        label="kissing curve",
        zorder=2
    )

    energy_label = "PE/N" if use_pe_only else "(KE+PE)/N"

    ax.set_xlabel("Size ratio, sz", fontsize=13)
    ax.set_ylabel("Compression ratio, cr", fontsize=13)

    ax.set_title(
        f"Energy phase diagram gated by smoothed-RDF classification\n"
        f"RDF gate: final {n_last_persistence} frame(s); "
        f"lowest mean {energy_label} over last {n_last_energy} frames wins",
        fontsize=12,
        fontweight="bold"
    )

    ax.grid(alpha=0.3)

    ax.legend(
        fontsize=10,
        title="Result",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0)
    )

    fig.tight_layout()
    fig.savefig(save_png, dpi=250, bbox_inches="tight")

    print(f"saved figure -> {save_png}")

    # ========================================================
    # Save CSV
    # ========================================================

    if save_csv is not None:
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sz",
                    "cr",
                    "winner",
                    "n_compared",
                    "n_tied",
                    "tied_with",
                    "E_diamond",
                    "E_mtab",
                    "E_rhombo",
                    "std_diamond",
                    "std_mtab",
                    "std_rhombo",
                    "gate_diamond",
                    "gate_mtab",
                    "gate_rhombo",
                    "peaks_diamond",
                    "peaks_mtab",
                    "peaks_rhombo",
                    "first_peak_diamond",
                    "first_peak_mtab",
                    "first_peak_rhombo",
                ],
            )

            writer.writeheader()
            writer.writerows(results)

        print(f"saved csv -> {save_csv}")

    return results


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    FOLDER = "/home/akash/hpc_hoomd-files/stability"

    SZ_VALUES = [1.15,1.16,1.17,1.18,1.19,1.20,1.21,1.22,1.23]
    CR_VALUES = [0.5,0.6,0.7,0.8,0.9,1.0]

    energy_phase_diagram(
        sz_values=SZ_VALUES,
        cr_values=CR_VALUES,
        folder=FOLDER,
        crystals=("diamond", "mtab", "rhombo"),
        n_last_energy=50,

        # Use final frame only for RDF classification.
        # Change to 10 if you want persistence over last 10 frames.
        n_last_persistence=1,

        # Set True if you want PE/N instead of (KE+PE)/N.
        use_pe_only=False,

        save_png=f"{FOLDER}/energy_phase_diagram_smoothed_rdf_gate_last50.png",
        save_csv=f"{FOLDER}/energy_phase_diagram_smoothed_rdf_gate_last50.csv",
    )