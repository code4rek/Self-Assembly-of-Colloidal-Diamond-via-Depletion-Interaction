import numpy as np
import gsd.hoomd

# ── Parameters ────────────────────────────────────────────────────────────────
sz_values  = [round(1.15 + i*0.01, 2) for i in range(8)]   # 1.15 to 1.22
cr_values  = [round(0.5  + i*0.1,  1) for i in range(6)]   # 0.5  to 1.0
lobe_radius = 0.5
L           = 0.01
n_cols      = 10
n_rows      = 10
n_repeats   = 10
output_dir  = "/home/akash/hoomd-results/diamond_vs_mtab/"

# ── Nearest-neighbour distance ─────────────────────────────────────────────────
def find_nndistance(sz, cr, L, rl=0.5):
    a = (4/np.sqrt(6)) * (2*rl*sz + 2*L)
    if sz < cr/np.sqrt(6) + np.sqrt(1 - cr**2/3):
        a = (4/np.sqrt(6)) * (2*cr*rl/np.sqrt(6) + 2*rl*np.sqrt(1 - cr**2/3) + 2*L)
    return a

def generate_template(size_ratio, compression_ratio):
    a            = find_nndistance(size_ratio, compression_ratio, L)
    core_radius  = size_ratio * lobe_radius
    lobe_scaling = compression_ratio * 2 * lobe_radius

    # ── Lobe offsets ──────────────────────────────────────────────────────────
    lobe_offsets = np.array([
        [-1/2,  -1/(2*np.sqrt(3)), -1/(2*np.sqrt(6))],
        [ 1/2,  -1/(2*np.sqrt(3)), -1/(2*np.sqrt(6))],
        [ 0,     1/np.sqrt(3),     -1/(2*np.sqrt(6))],
        [ 0,     0,                 np.sqrt(3/8)     ]
    ]) * lobe_scaling

    # ── dz from lobe contact constraint ───────────────────────────────────────
    xy_shift  = np.array([a/2, a/(2*np.sqrt(3))])
    min_xy_sq = min(
        np.dot(xy_shift + o2[:2] - o1[:2], xy_shift + o2[:2] - o1[:2])
        for o1 in lobe_offsets for o2 in lobe_offsets
    )
    dz = np.sqrt(4*(lobe_radius + L)**2 - min_xy_sq)

    # ── Box and base grid ──────────────────────────────────────────────────────
    Lx = n_cols * a              + a/2
    Ly = n_rows * a*np.sqrt(3)/2 + a/np.sqrt(3)
    Lz = 2 * max(Lx, Ly)

    core_positions = np.array([
        [-Lx/2 + (j + 0.5)*a + (a/2 if i%2 else 0),
         -Ly/2 + (i + 0.5)*a*np.sqrt(3)/2,
         -Lz/2 + lobe_radius]
        for i in range(n_rows) for j in range(n_cols)
    ])

    # ── Lattice vectors and parameters ───────────────────────────────────────
    a1 = np.array([a,    0.0,             0.0])
    a2 = np.array([a/2,  a*np.sqrt(3)/2,  0.0])
    a3 = np.array([a/2,  a/(2*np.sqrt(3)), dz ])

    a_len = np.linalg.norm(a1)
    b_len = np.linalg.norm(a2)
    c_len = np.linalg.norm(a3)
    alpha = np.degrees(np.arccos(np.dot(a2, a3) / (b_len * c_len)))
    beta  = np.degrees(np.arccos(np.dot(a1, a3) / (a_len * c_len)))
    gamma = np.degrees(np.arccos(np.dot(a1, a2) / (a_len * b_len)))

    print(f"  Lattice parameters:")
    print(f"    |a1|={a_len:.4f}  |a2|={b_len:.4f}  |a3|={c_len:.4f}")
    print(f"    alpha={alpha:.2f}°  beta={beta:.2f}°  gamma={gamma:.2f}°")
    if np.isclose(a_len, b_len) and np.isclose(a_len, c_len) and np.isclose(alpha, beta) and np.isclose(beta, gamma):
        print(f"    → Rhombohedral ✓")
    else:
        print(f"    → Triclinic (not rhombohedral)")

    # ── 3-layer basis ──────────────────────────────────────────────────────────
    basis_xy = [
        np.array([0.0,  0.0             ]),
        np.array([0.0,  a/np.sqrt(3)    ]),
        np.array([a/2,  a/(2*np.sqrt(3))]),
    ]

    # ── Build all layers ───────────────────────────────────────────────────────
    # all_bodies: each TLPP (1 core + 4 lobes) gets a unique body ID equal to the
    # core's own particle index, matching HOOMD's rigid-body convention
    # (central particle body == its tag; constituents share it). Analysis code
    # groups lobes by this value, so it must be shared core<->lobes and unique
    # per TLPP.
    all_positions, all_typeids, all_diameters, all_bodies = [], [], [], []

    for r in range(n_repeats):
        for k, xy in enumerate(basis_xy):
            n     = r * 3 + k
            z_off = n * dz

            cores       = core_positions.copy()
            cores[:, 0] = ((cores[:, 0] + xy[0] + Lx/2) % Lx) - Lx/2
            cores[:, 1] = ((cores[:, 1] + xy[1] + Ly/2) % Ly) - Ly/2
            cores[:, 2] += z_off

            c_tid, l_tid = (2, 3) if n == 0 else (0, 1)

            for cp in cores:
                core_idx = len(all_positions)          # this core's particle tag
                all_positions.append(cp);         all_typeids.append(c_tid); all_diameters.append(2*core_radius); all_bodies.append(core_idx)
                for o in lobe_offsets:
                    all_positions.append(cp + o); all_typeids.append(l_tid); all_diameters.append(2*lobe_radius); all_bodies.append(core_idx)

    all_positions = np.array(all_positions)
    all_typeids   = np.array(all_typeids,   dtype=int)
    all_diameters = np.array(all_diameters, dtype=float)
    all_bodies    = np.array(all_bodies,    dtype=int)

    # ── Write GSD ─────────────────────────────────────────────────────────────
    frame = gsd.hoomd.Frame()
    frame.configuration.box        = [Lx, Ly, Lz, 0, 0, 0]
    frame.particles.N              = len(all_positions)
    frame.particles.types          = ["bottom_core", "bottom_lobe", "frozen_core", "frozen_lobe"]
    frame.particles.typeid         = all_typeids
    frame.particles.position       = all_positions
    frame.particles.diameter       = all_diameters
    frame.particles.body           = all_bodies      # <-- the missing field
    frame.particles.mass           = [0] * len(all_positions)
    frame.particles.moment_inertia = [[0, 0, 0]] * len(all_positions)

    filename = f"{output_dir}/rhombo_sz_{size_ratio:.2f}_cr_{compression_ratio:.1f}.gsd"
    with gsd.hoomd.open(name=filename, mode="w") as f:
        f.append(frame)

    print(f"  Wrote: {filename}")

# ── Loop over all sz and cr ────────────────────────────────────────────────────
total = len(sz_values) * len(cr_values)
count = 0
for sz in sz_values:
    for cr in cr_values:
        print(f"[{count+1}/{total}] sz={sz:.2f}  cr={cr:.1f}")
        generate_template(sz, cr)
        count += 1

print(f"\nDone. {total} templates written to {output_dir}")