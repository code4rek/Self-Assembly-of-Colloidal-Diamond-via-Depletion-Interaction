#!/usr/bin/env python3
"""
Plot  E/N(diamond) - E/N(mtab)  -- the per-particle lobe-lobe binding-energy
difference between the two polymorphs -- vs compression ratio cr, one line per
sz. Each E/N is the mean over the LAST N_LAST frames of that stability run.

Sign convention (both E/N are negative binding energies):
    diff = E_diam/N - E_mtab/N
    diff < 0  -> diamond is more strongly bound (lower energy) -> diamond favored
    diff > 0  -> mtab is more strongly bound                   -> mtab favored

The two analysis modules define the SAME function names, so they are loaded as
separate module objects to keep their bands / constants distinct.
"""
import os
import importlib.util
import numpy as np
import gsd.hoomd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- paths to the two analysis modules ----
MTAB_MODULE = "/home/akash/hoomd-files/mtab_lobe_energy.py"
DIAM_MODULE = "/home/akash/hoomd-files/diam_lobe_energy.py"

# ---- trajectory locations & naming (EDIT to match your files) ----
STAB_DIR = "/home/akash/hpc_hoomd-files/stability"
def mtab_file(sz, cr):
    return (f"{STAB_DIR}/stabSPRING_mtab_sz_{sz:1.2f}_cr_{cr:1.1f}_"
        f"dpl_fr_0.050_dpl_sz_0.02_kT_1.0_spring_k100_hold10000000_pad8.0.gsd")
def diam_file(sz, cr):
    return (f"{STAB_DIR}/stabSPRING_diamond_sz_{sz:1.2f}_cr_{cr:1.1f}_"
        f"dpl_fr_0.050_dpl_sz_0.02_kT_1.0_spring_k100_hold10000000_pad8.0.gsd")

N_LAST    = 50
SZ_VALUES = [1.15,1.16,1.17,1.18,1.19, 1.20,1.22]
CR_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SAVE      = f"{STAB_DIR}/diam_minus_mtab_vs_cr.png"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def last_n_EpN(module, path, n_last=N_LAST):
    """(<E/N>, std) over the last n_last frames; (nan, nan) if the file is missing."""
    if not os.path.exists(path):
        return np.nan, np.nan
    with gsd.hoomd.open(path, 'r') as t:
        nfr = len(t)
    start = max(0, nfr - n_last)
    # only common kwargs -- the modules differ in require_both_mtab vs _diam
    _, s = module.analyze(path, start_fr=start, end_fr=None, z_cut=None, verbose=False)
    return s["E_per_particle_mean"], s["E_per_particle_std"]


def main(mtab_module=MTAB_MODULE, diam_module=DIAM_MODULE,
         mtab_fname=mtab_file, diam_fname=diam_file, save=SAVE):
    M = load_module(mtab_module, "mtab_mod")
    D = load_module(diam_module, "diam_mod")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {1.15: "#1f77b4", 1.21: "#d62728"}

    for sz in SZ_VALUES:
        cr_ok, diff, err = [], [], []
        for cr in CR_VALUES:
            e_m, s_m = last_n_EpN(M, mtab_fname(sz, cr))
            e_d, s_d = last_n_EpN(D, diam_fname(sz, cr))
            if np.isnan(e_m) or np.isnan(e_d):
                print(f"  skip sz={sz:.2f} cr={cr:.1f}: E_mtab={e_m}  E_diam={e_d}")
                continue
            cr_ok.append(cr)
            diff.append(e_d - e_m)                 # diamond - mtab
            err.append(float(np.hypot(s_d, s_m)))  # independent runs -> quadrature
            print(f"  sz={sz:.2f} cr={cr:.1f}: E_diam={e_d:8.3f}  E_mtab={e_m:8.3f}  "
                  f"diff={e_d - e_m:+8.3f}")
        if cr_ok:
            ax.errorbar(cr_ok, diff, yerr=err, marker="o", lw=2.2, ms=7, capsize=4,
                        color=colors.get(sz), label=f"sz={sz:.2f}")

    ax.axhline(0.0, color="k", lw=1.0, ls=":")
    ax.set_xlabel("Compression ratio (cr)", fontsize=13)
    ax.set_ylabel(r"$E_{\mathrm{diam}}/N - E_{\mathrm{mtab}}/N$   (kT)", fontsize=13)
    ax.set_title("Per-particle lobe-lobe binding energy:  diamond $-$ mtab\n"
                 "(negative = diamond favored)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {save}")


if __name__ == "__main__":
    main()