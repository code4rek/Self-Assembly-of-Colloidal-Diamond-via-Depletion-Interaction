#!/usr/bin/env python3
"""
E_diamond - E_rhombo (simple logged (KE+PE)/N_total, per TLPP) vs cr, one
line per sz.

Gated by the RDF persistence check first: at a given (sz, cr), if EITHER
diamond or rhombo comes back amorphous, that point is skipped entirely for
this line -- a diff is only meaningful between two crystals that actually
persisted as themselves, not between a real crystal and whatever an
amorphous blob's energy happens to be.

Sign convention: diff < 0 -> diamond more favorable (lower/more negative
energy); diff > 0 -> rhombo more favorable.
"""
import os
import numpy as np
import gsd.hoomd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mtab_rdf_peaks_sweep import check_persistence

# ---- Self-contained energy-reading (no dependency on energy_phase_diagram.py
# being present as a separate file -- copied in directly to avoid import
# fragility across files/notebook sessions) ----
KE_KEY = "md/compute/ThermodynamicQuantities/kinetic_energy"
PE_KEY = "md/compute/ThermodynamicQuantities/potential_energy"
CORE_TYPE_NAMES = ("core", "frozen_core")


def default_fname(folder, crystal, sz, cr):
    return (
        f"{folder}/stabSPRING_{crystal}_"
        f"sz_{sz:.2f}_cr_{cr:.1f}_"
        f"dpl_fr_0.050_dpl_sz_0.02_kT_1.0_"
        f"spring_k100_hold10000000_pad8.0.gsd"
    )


def read_energy_per_frame(path, core_type_names=CORE_TYPE_NAMES):
    """Returns frames, E_per_N, N_total. E_per_N = (KE + PE) / N_total_cores."""
    with gsd.hoomd.open(path, "r") as f:
        frame0 = f[0]
        types = frame0.particles.types
        typeid = frame0.particles.typeid
        core_tids = [k for k, t in enumerate(types) if t in core_type_names]
        is_core = np.isin(typeid, core_tids)
        n_total = int(is_core.sum())
        n_frames = len(f)
        KE = np.empty(n_frames)
        PE = np.empty(n_frames)
        for i, frame in enumerate(f):
            KE[i] = frame.log[KE_KEY][0]
            PE[i] = frame.log[PE_KEY][0]
    E_per_N = (KE + PE) / n_total
    return np.arange(n_frames), E_per_N, n_total


def mean_energy_last_n(path, n_last=50):
    """Mean and std of logged total energy per TLPP over the last n_last frames."""
    frames, E_per_N, n_total = read_energy_per_frame(path)
    n_use = min(n_last, len(E_per_N))
    window = E_per_N[-n_use:]
    return float(window.mean()), float(window.std()), n_total, n_use


def energy_diff_vs_cr(crystal_a, crystal_b, sz_values, cr_values, folder, n_last_energy=50,
                       n_last_persistence=10, require_persistence=True, save=None,
                       fname_fn=None, verbose=True):
    """E_{crystal_a} - E_{crystal_b} (simple logged (KE+PE)/N_total, per TLPP)
    vs cr, one line per sz.

    require_persistence=True (default): gated by the RDF persistence check
    first -- a point is skipped entirely for this line unless BOTH crystals
    persisted as themselves at that (sz, cr). require_persistence=False:
    skip the gate entirely and compare energy for every point where both
    files exist, regardless of whether either structure is actually still
    crystalline -- the energy of an amorphous blob is still a real logged
    number, just not necessarily a meaningful "which crystal is more
    favorable" comparison the way it is for two genuinely persisting
    crystals. Use this if you specifically want the raw comparison anyway.

    Sign convention: diff < 0 -> crystal_a more favorable; diff > 0 ->
    crystal_b more favorable.
    """
    if fname_fn is None:
        fname_fn = lambda crystal, sz, cr: default_fname(folder, crystal, sz, cr)
    if save is None:
        save = f"{crystal_a}_minus_{crystal_b}_vs_cr.png"

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#e377c2", "#7f7f7f"]
    sz_colors = {sz: palette[i % len(palette)] for i, sz in enumerate(sz_values)}

    fig, ax = plt.subplots(figsize=(8, 6))
    any_plotted = False

    for sz in sz_values:
        cr_ok, diffs = [], []
        for cr in cr_values:
            path_a = fname_fn(crystal_a, sz, cr)
            path_b = fname_fn(crystal_b, sz, cr)

            if not (os.path.exists(path_a) and os.path.exists(path_b)):
                if verbose:
                    print(f"  sz={sz:.2f} cr={cr:.1f}: skip, missing file(s)")
                continue

            if require_persistence:
                res_a = check_persistence(path_a, crystal=crystal_a, n_last=n_last_persistence, verbose=False)
                res_b = check_persistence(path_b, crystal=crystal_b, n_last=n_last_persistence, verbose=False)

                if res_a["verdict"] != crystal_a or res_b["verdict"] != crystal_b:
                    if verbose:
                        print(f"  sz={sz:.2f} cr={cr:.1f}: skip, "
                              f"{crystal_a}={res_a['verdict']}  {crystal_b}={res_b['verdict']}")
                    continue

            E_a, _, _, _ = mean_energy_last_n(path_a, n_last=n_last_energy)
            E_b, _, _, _ = mean_energy_last_n(path_b, n_last=n_last_energy)
            diff = E_a - E_b

            cr_ok.append(cr)
            diffs.append(diff)
            if verbose:
                print(f"  sz={sz:.2f} cr={cr:.1f}: E_{crystal_a}={E_a:.4f}  E_{crystal_b}={E_b:.4f}  "
                      f"diff={diff:+.4f}")

        if cr_ok:
            ax.plot(cr_ok, diffs, "o-", color=sz_colors[sz], lw=2.2, ms=7, label=f"sz={sz:.2f}")
            any_plotted = True

    if not any_plotted:
        print("no comparison points at all -- not saving")
        plt.close(fig)
        return None

    gate_str = "gated by persistence check" if require_persistence else "UNGATED -- amorphous structures included"
    ax.axhline(0, color="k", lw=1, ls=":")
    ax.set_xlabel("Compression ratio (cr)", fontsize=13)
    ax.set_ylabel(rf"$E_{{{crystal_a}}} - E_{{{crystal_b}}}$  (kT per TLPP)", fontsize=12)
    ax.set_title(f"{crystal_a.capitalize()} $-$ {crystal_b}: logged (KE+PE)/N vs cr\n"
                 f"(negative = {crystal_a} favored; {gate_str})",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=10, title="sz", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(save, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {save}")


def diamond_minus_rhombo_vs_cr(sz_values, cr_values, folder, **kwargs):
    return energy_diff_vs_cr("diamond", "rhombo", sz_values, cr_values, folder, **kwargs)


def diamond_minus_mtab_vs_cr(sz_values, cr_values, folder, **kwargs):
    return energy_diff_vs_cr("diamond", "mtab", sz_values, cr_values, folder, **kwargs)


if __name__ == "__main__":
    FOLDER = "/home/akash/hpc_hoomd-files/stability"
    SZ_VALUES = [1.15, 1.16, 1.17, 1.18, 1.19, 1.21, 1.22, 1.23]
    CR_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    diamond_minus_rhombo_vs_cr(
        sz_values=SZ_VALUES, cr_values=CR_VALUES, folder=FOLDER,
        save=f"{FOLDER}/diamond_minus_rhombo_vs_cr.png",
    )
    diamond_minus_mtab_vs_cr(
        sz_values=SZ_VALUES, cr_values=CR_VALUES, folder=FOLDER,
        save=f"{FOLDER}/diamond_minus_mtab_vs_cr.png",
    )