"""
Single-s visualization run for diamond, MTAB, and rhombohedral TLPP crystals.

Hamiltonian:

    U_s = s U_real + (1-s) lambda_max U_E

At every TLPP, U_E tethers the core center and all four lobes to their own
reference positions. The four equivalent lobe tethers give tetrahedrally
symmetric small-angle rotational stiffness.

The restrained lobes use the particle type ``tethered_lobe`` so they can be
colored separately from the cores in OVITO.

One execution writes one trajectory for each crystal at the same state point,
so their reference geometries and tethered fluctuations can be compared in
OVITO. This does not perform the final free-energy integration.
"""

import math
import os
import time

import gsd.hoomd
import hoomd
import numpy as np
from scipy.spatial.transform import Rotation


# ===========================================================================
# Thermodynamic state and one coupling window
# ===========================================================================
KT = 1.0
PHI = 0.05
SIGMAP = 0.02
RDEP = 0.5 * SIGMAP
L_BRUSH = 0.01
SIGMAG = 0.09 * (1000) ** 2
LOBE_R = 0.5

SZ = 1.21
CR = 0.60

S = 0.0
LAMBDA_MAX = 5000.0
K_ACTIVE = (1.0 - S) * LAMBDA_MAX

OUT_DIR = "/home/akash/hpc_hoomd-files/free_energy/visual_check"

# All three periodic cells contain exactly 144 TLPPs.
CRYSTALS = ( "mtab", "diamond", "rhombo")
DIAMOND_DIMS = (3, 4, 3)  # 3* 4* 3 * 4 = 144
MTAB_DIMS = (3, 4, 3)     # 3* 4* 3 * 4 = 144
RHOMBO_DIMS = (4, 4, 9)   # 4 * 4 * 9 = 144

DT = 1e-4
EQUIL_STEPS = 100_000
PRODUCTION_STEPS = 1_000_000
WRITE_EVERY = 10_000
N_TABLE = 1000
SEED = 35
USE_GPU = False

# Translate the complete periodic crystal before rigid bodies and reference
# sites are created. This only changes where the periodic box cuts the crystal.
INITIAL_SHIFT = np.array([0.5, 0.5, 0.25], dtype=np.float64)


# ===========================================================================
# Brush + depletion interaction
# ===========================================================================
def P_Implicit_Dep(r1, r2, sigma_g, L, phi, rdep, rcc):
    r1_eff = L + r1
    r2_eff = L + r2
    x = rcc - r1 - r2
    q1 = r1_eff / rdep
    q2 = r2_eff / rdep
    n = rcc / rdep
    r_avg = 2.0 / (1.0 / r1 + 1.0 / r2)

    if (x <= 2.0 * L and x > 0.0) and (
        rcc <= r1_eff + r2_eff + 2.0 * rdep
    ):
        U_B = (
            16.0
            * math.pi
            * L**2
            * r_avg
            * sigma_g ** (3.0 / 2.0)
            / 35.0
            * (
                28.0 * ((2.0 * L / x) ** (1.0 / 4.0) - 1.0)
                + (20.0 / 11.0) * (1.0 - (x / (2.0 * L)) ** (11.0 / 4.0))
                + 12.0 * (x / (2.0 * L) - 1.0)
            )
        )
        U_AO = (
            -phi
            / 16.0
            * (q1 + q2 + 2.0 - n) ** 2
            * (
                n
                + 2.0 * (q1 + q2 + 2.0)
                - 3.0 / n * (q1**2 + q2**2 - 2.0 * q1 * q2)
            )
        )
    elif (x > 2.0 * L) and (rcc <= r1_eff + r2_eff + 2.0 * rdep):
        U_B = 0.0
        U_AO = (
            -phi
            / 16.0
            * (q1 + q2 + 2.0 - n) ** 2
            * (
                n
                + 2.0 * (q1 + q2 + 2.0)
                - 3.0 / n * (q1**2 + q2**2 - 2.0 * q1 * q2)
            )
        )
    else:
        U_AO = 0.0
        U_B = 0.0

    return U_B + U_AO


def tabulate_P_implicit(
    r1, r2, sigma_g, L, phi, rdep, r_min, r_max, n_points=1000
):
    r = np.linspace(r_min, r_max, n_points)
    U = np.array(
        [P_Implicit_Dep(r1, r2, sigma_g, L, phi, rdep, ri) for ri in r]
    )
    F = -np.gradient(U, r)
    return U, F


# ===========================================================================
# Diamond, MTAB, and rhombohedral crystal geometry
# ===========================================================================
THETAX = np.arccos(1.0 / np.sqrt(3.0))
RX = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, np.cos(THETAX), -np.sin(THETAX)],
     [0.0, np.sin(THETAX), np.cos(THETAX)]],
    dtype=np.float64,
)
RZ90 = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
RX180 = np.diag([1.0, -1.0, -1.0])
TETRA = np.array(
    [[-1 / 2, -1 / (2 * np.sqrt(3)), -1 / (2 * np.sqrt(6))],
     [1 / 2, -1 / (2 * np.sqrt(3)), -1 / (2 * np.sqrt(6))],
     [0.0, 1 / np.sqrt(3), -1 / (2 * np.sqrt(6))],
     [0.0, 0.0, np.sqrt(3 / 8)]],
    dtype=np.float64,
)


def mat_to_quat_hoomd(matrix):
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def wrap_to_box(x, box_length):
    return (x + box_length / 2.0) % box_length - box_length / 2.0


def find_diamond_mtab_spacing(sz, cr, L=L_BRUSH, rl=LOBE_R):
    a = (4.0 / np.sqrt(6.0)) * (2.0 * rl * sz + 2.0 * L)
    if sz < cr / np.sqrt(6.0) + np.sqrt(1.0 - cr**2 / 3.0):
        a = (4.0 / np.sqrt(6.0)) * (
            2.0 * cr * rl / np.sqrt(6.0)
            + 2.0 * rl * np.sqrt((1.0 + L / rl) ** 2 - cr**2 / 3.0)
        )
    return a


def find_rhombo_spacing(sz, cr, L=L_BRUSH, rl=LOBE_R):
    a = (4.0 / np.sqrt(6.0)) * (2.0 * rl * sz + 2.0 * L)
    if sz < cr / np.sqrt(6.0) + np.sqrt(1.0 - cr**2 / 3.0):
        a = (4.0 / np.sqrt(6.0)) * (
            2.0 * cr * rl / np.sqrt(6.0)
            + 2.0 * rl * np.sqrt(1.0 - cr**2 / 3.0)
            + 2.0 * L
        )
    return a


def diamond_centers(nx, ny, nz, a):
    ax = ay = a
    az = a * np.sqrt(2.0)
    Lx, Ly, Lz = nx * ax, ny * ay, nz * az
    basis = a * np.array(
        [[0.0, 0.0, 0.0],
         [0.5, 0.0, 1.0 / (2.0 * np.sqrt(2.0))],
         [0.5, 0.5, 1.0 / np.sqrt(2.0)],
         [0.0, 0.5, 3.0 / (2.0 * np.sqrt(2.0))]]
    )
    dz = a / (2.0 * np.sqrt(2.0))
    centers, layers = [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                translation = np.array([i * ax, j * ay, k * az])
                for offset in basis:
                    p = translation + offset
                    p[0] = wrap_to_box(p[0] - Lx / 2.0, Lx)
                    p[1] = wrap_to_box(p[1] - Ly / 2.0, Ly)
                    p[2] = wrap_to_box(p[2] - Lz / 2.0, Lz)
                    centers.append(p)
                    layers.append(int(np.floor((p[2] + Lz / 2.0) / dz + 1e-9)))
    return np.asarray(centers), np.asarray(layers, dtype=int), (Lx, Ly, Lz)


def compute_mtab_h(cr, L=L_BRUSH, rl=LOBE_R):
    return rl * (cr / np.sqrt(2.0) + np.sqrt(1.0 - cr**2 / 2.0)) + L


def compute_mtab_staggered_h(sz, cr, L=L_BRUSH, rl=LOBE_R):
    a = find_diamond_mtab_spacing(sz, cr, L, rl)
    return np.sqrt((np.sqrt(6.0) * a / 4.0) ** 2 - (a / 2.0) ** 2)


def mtab_centers(nx, ny, nz, a, h, hs):
    if nz % nx != 0:
        raise ValueError("For this orthorhombic MTAB cell, nz must be a multiple of nx.")
    Lx, Ly, Lz = nx * a, ny * a, nz * 2.0 * (h + hs)
    centers = np.zeros((nx * ny * nz * 4, 3), dtype=np.float64)
    basis_index = np.zeros(len(centers), dtype=np.int32)
    basis = np.array([[0.0, 0.0, 0.0], [0.0, a / 2.0, hs]])
    index = 0
    for k_cell in range(nz):
        for z_step in range(2):
            k_total = 2 * k_cell + z_step
            z_lattice = k_total * (h + hs)
            x_shift = k_total * a / 2.0
            for i in range(nx):
                for j in range(ny):
                    for b, offset in enumerate(basis):
                        x = i * a + x_shift + offset[0]
                        y = j * a + offset[1]
                        z = z_lattice + offset[2]
                        centers[index] = [
                            wrap_to_box(x - Lx / 2.0 + 0.25 * a, Lx),
                            wrap_to_box(y - Ly / 2.0 + 0.25 * a, Ly),
                            wrap_to_box(z - Lz / 2.0 + 0.20 * a, Lz),
                        ]
                        basis_index[index] = b
                        index += 1
    return centers, basis_index, (Lx, Ly, Lz)


def find_rhombo_dz(a, lobe_offsets, L=L_BRUSH, rl=LOBE_R):
    xy_shift = np.array([a / 2.0, a / (2.0 * np.sqrt(3.0))])
    min_xy_sq = min(
        np.dot(xy_shift + o2[:2] - o1[:2], xy_shift + o2[:2] - o1[:2])
        for o1 in lobe_offsets for o2 in lobe_offsets
    )
    return np.sqrt(4.0 * (rl + L) ** 2 - min_xy_sq)


def rhombohedral_centers(ncols, nrows, nlayers, a, dz):
    if nrows % 2 or nlayers % 3:
        raise ValueError("Rhombo requires even nrows and nlayers divisible by 3.")
    Lx = ncols * a
    Ly = nrows * a * np.sqrt(3.0) / 2.0
    Lz = nlayers * dz
    basis_xy = [
        np.array([0.0, 0.0]),
        np.array([0.0, a / np.sqrt(3.0)]),
        np.array([a / 2.0, a / (2.0 * np.sqrt(3.0))]),
    ]
    centers = []
    for layer in range(nlayers):
        sx, sy = basis_xy[layer % 3]
        z = -Lz / 2.0 + layer * dz
        for row in range(nrows):
            row_offset = a / 2.0 if row % 2 else 0.0
            y = -Ly / 2.0 + (row + 0.5) * a * np.sqrt(3.0) / 2.0 + sy
            for col in range(ncols):
                x = -Lx / 2.0 + (col + 0.5) * a + row_offset + sx
                centers.append([wrap_to_box(x, Lx), wrap_to_box(y, Ly), wrap_to_box(z, Lz)])
    return np.asarray(centers), (Lx, Ly, Lz)


def build_cores_snapshot(crystal, sz, cr):
    body_positions = TETRA * (cr * 2.0 * LOBE_R)

    if crystal == "diamond":
        a = find_diamond_mtab_spacing(sz, cr)
        centers, layers, box_lengths = diamond_centers(*DIAMOND_DIMS, a)
        q_even = mat_to_quat_hoomd(RX)
        q_odd = mat_to_quat_hoomd(RZ90 @ RX)
        orientations = np.asarray([q_even if k % 2 == 0 else q_odd for k in layers])
    elif crystal == "mtab":
        a = find_diamond_mtab_spacing(sz, cr)
        h = compute_mtab_h(cr)
        hs = compute_mtab_staggered_h(sz, cr)
        centers, basis_index, box_lengths = mtab_centers(*MTAB_DIMS, a, h, hs)
        rotation_0 = RZ90 @ RX
        q_0 = mat_to_quat_hoomd(rotation_0)
        q_1 = mat_to_quat_hoomd(rotation_0 @ RX180)
        orientations = np.asarray([q_1 if b else q_0 for b in basis_index])
    elif crystal == "rhombo":
        a = find_rhombo_spacing(sz, cr)
        dz = find_rhombo_dz(a, body_positions)
        centers, box_lengths = rhombohedral_centers(*RHOMBO_DIMS, a, dz)
        orientations = np.tile([1.0, 0.0, 0.0, 0.0], (len(centers), 1))
    else:
        raise ValueError("crystal must be diamond, mtab, or rhombo")

    Lx, Ly, Lz = box_lengths
    centers = centers + INITIAL_SHIFT
    centers[:, 0] = wrap_to_box(centers[:, 0], Lx)
    centers[:, 1] = wrap_to_box(centers[:, 1], Ly)
    centers[:, 2] = wrap_to_box(centers[:, 2], Lz)

    inertia_tensor = np.zeros((3, 3))
    for p in body_positions:
        inertia_tensor += (p @ p) * np.eye(3) - np.outer(p, p)
    moment_inertia = np.diag(inertia_tensor).copy()

    snapshot = gsd.hoomd.Frame()
    snapshot.configuration.box = [Lx, Ly, Lz, 0.0, 0.0, 0.0]
    snapshot.particles.N = len(centers)
    snapshot.particles.types = ["core", "tethered_lobe"]
    snapshot.particles.typeid = np.zeros(len(centers), dtype=np.uint32)
    snapshot.particles.position = centers.astype(np.float32)
    snapshot.particles.orientation = orientations.astype(np.float32)
    snapshot.particles.body = np.arange(len(centers), dtype=np.int32)
    snapshot.particles.mass = np.full(len(centers), 5.0, dtype=np.float32)
    snapshot.particles.moment_inertia = np.tile(moment_inertia, (len(centers), 1)).astype(np.float32)
    snapshot.particles.diameter = np.full(len(centers), 2.0 * sz * LOBE_R, dtype=np.float32)
    return snapshot, body_positions, centers, orientations, box_lengths


def build_reference_positions(centers, orientations, body_positions, box_lengths):
    """Reference coordinates indexed by the global particle tag.

    After create_bodies, HOOMD assigns core tags first, followed by four
    constituent tags per body in body-major order.
    """
    n_cores = len(centers)
    n_total = 5 * n_cores
    reference = np.zeros((n_total, 3), dtype=np.float64)
    reference[:n_cores] = centers

    Lx, Ly, Lz = box_lengths
    rotations = Rotation.from_quat(orientations[:, [1, 2, 3, 0]])
    for body_index, center in enumerate(centers):
        rotated_offsets = rotations[body_index].apply(body_positions)
        for lobe_index, offset in enumerate(rotated_offsets):
            tag = n_cores + 4 * body_index + lobe_index
            position = center + offset
            reference[tag] = [
                wrap_to_box(position[0], Lx),
                wrap_to_box(position[1], Ly),
                wrap_to_box(position[2], Lz),
            ]

    return reference


# ===========================================================================
# Einstein tether: every core plus all four tethered_lobe constituents
# ===========================================================================
class CoreFourLobeTether(hoomd.md.force.Custom):
    """Apply harmonic forces and report their harmonic potential energy."""

    def __init__(self, k_active, reference, tethered_lobe_typeid):
        super().__init__(aniso=False)
        self.k_active = float(k_active)
        self.reference = np.asarray(reference, dtype=np.float64)
        self.tethered_lobe_typeid = int(tethered_lobe_typeid)

    def set_forces(self, timestep):
        with self._state.cpu_local_snapshot as snapshot, self.cpu_local_force_arrays as arrays:
            tag = np.asarray(snapshot.particles.tag)
            body = np.asarray(snapshot.particles.body)
            typeid = np.asarray(snapshot.particles.typeid)
            position = np.asarray(snapshot.particles.position, dtype=np.float64)

            displacement = position - self.reference[tag]
            box = self._state.box
            lengths = np.array([box.Lx, box.Ly, box.Lz], dtype=np.float64)
            displacement -= lengths * np.rint(displacement / lengths)

            is_core = body == tag
            is_selected_lobe = typeid == self.tethered_lobe_typeid
            selected = is_core | is_selected_lobe

            force = np.zeros_like(position)
            force[selected] = -self.k_active * displacement[selected]

            energy = np.zeros(len(tag), dtype=np.float64)
            energy[selected] = 0.5 * self.k_active * np.sum(
                displacement[selected] ** 2, axis=1
            )

            arrays.force[:] = force
            arrays.potential_energy[:] = energy


# ===========================================================================
# HOOMD run
# ===========================================================================
def set_table_pair(table, pair, r_min, r_cut, U, F):
    table.params[pair] = dict(r_min=r_min, U=U, F=F)
    table.r_cut[pair] = r_cut


def run_single_s_window(crystal):
    os.makedirs(OUT_DIR, exist_ok=True)
    output = os.path.join(
        OUT_DIR,
        f"ein_visual_{crystal}_sz_{SZ:.2f}_cr_{CR:.1f}_"
        f"s_{S:.3f}_all_4lobes.gsd",
    )

    snapshot, body_positions, centers, orientations, box_lengths = (
        build_cores_snapshot(crystal, SZ, CR)
    )

    device = hoomd.device.GPU() if USE_GPU else hoomd.device.CPU()
    simulation = hoomd.Simulation(device=device, seed=SEED)
    simulation.create_state_from_snapshot(snapshot)

    rigid = hoomd.md.constrain.Rigid()
    rigid.body["core"] = {
        "constituent_types": [
            "tethered_lobe",
            "tethered_lobe",
            "tethered_lobe",
            "tethered_lobe",
        ],
        "positions": body_positions.tolist(),
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * 4,
    }
    rigid.create_bodies(simulation.state)

    n_cores = len(centers)
    expected_total = 5 * n_cores
    if simulation.state.N_particles != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} total particles, got "
            f"{simulation.state.N_particles}."
        )

    reference = build_reference_positions(
        centers, orientations, body_positions, box_lengths
    )

    core_radius = SZ * LOBE_R
    lobe_radius = LOBE_R

    rcc_min = 2.0 * core_radius
    rcc_cut = rcc_min + 2.0 * L_BRUSH + SIGMAP
    rcl_min = core_radius + lobe_radius
    rcl_cut = rcl_min + 2.0 * L_BRUSH + SIGMAP
    rll_min = 2.0 * lobe_radius
    rll_cut = rll_min + 2.0 * L_BRUSH + SIGMAP

    U_cc_0, F_cc_0 = tabulate_P_implicit(
        core_radius,
        core_radius,
        SIGMAG,
        L_BRUSH,
        PHI,
        RDEP,
        rcc_min,
        rcc_cut,
        N_TABLE,
    )
    U_cl_0, F_cl_0 = tabulate_P_implicit(
        core_radius,
        lobe_radius,
        SIGMAG,
        L_BRUSH,
        PHI,
        RDEP,
        rcl_min,
        rcl_cut,
        N_TABLE,
    )
    U_ll_0, F_ll_0 = tabulate_P_implicit(
        lobe_radius,
        lobe_radius,
        SIGMAG,
        L_BRUSH,
        PHI,
        RDEP,
        rll_min,
        rll_cut,
        N_TABLE,
    )

    # The physical interaction contribution in U_s is s U_real.
    U_cc, F_cc = S * U_cc_0, S * F_cc_0
    U_cl, F_cl = S * U_cl_0, S * F_cl_0
    U_ll, F_ll = S * U_ll_0, S * F_ll_0

    neighbor_list = hoomd.md.nlist.Cell(buffer=0.0, exclusions=["body"])
    table = hoomd.md.pair.Table(nlist=neighbor_list, default_r_cut=0.0)

    set_table_pair(table, ("core", "core"), rcc_min, rcc_cut, U_cc, F_cc)
    set_table_pair(
        table,
        ("core", "tethered_lobe"),
        rcl_min,
        rcl_cut,
        U_cl,
        F_cl,
    )
    set_table_pair(
        table,
        ("tethered_lobe", "tethered_lobe"),
        rll_min,
        rll_cut,
        U_ll,
        F_ll,
    )

    tether = CoreFourLobeTether(
        k_active=K_ACTIVE,
        reference=reference,
        tethered_lobe_typeid=1,
    )

    centers_free = hoomd.filter.Rigid(("center", "free"))
    integrator = hoomd.md.Integrator(dt=DT, integrate_rotational_dof=True)
    integrator.rigid = rigid
    integrator.forces.extend([table, tether])
    integrator.methods.append(
        hoomd.md.methods.Langevin(filter=centers_free, kT=KT)
    )
    simulation.operations.integrator = integrator
    simulation.state.thermalize_particle_momenta(filter=centers_free, kT=KT)

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=centers_free)
    simulation.operations.computes.append(thermo)

    logger = hoomd.logging.Logger()
    logger.add(simulation, quantities=["timestep", "tps"])
    logger.add(
        thermo,
        quantities=["kinetic_temperature", "potential_energy"],
    )
    logger.add(table, quantities=["energy"])
    logger.add(tether, quantities=["energy"])

    writer = hoomd.write.GSD(
        filename=output,
        trigger=hoomd.trigger.Periodic(WRITE_EVERY),
        mode="wb",
        logger=logger,
        dynamic=["property"],
    )
    writer.write_diameter = True
    simulation.operations.writers.append(writer)

    simulation.run(0)
    print(
        f"crystal={crystal}, cores={n_cores}, "
        f"total particles={simulation.state.N_particles}, "
        f"s={S}, active spring k={K_ACTIVE}, shift={INITIAL_SHIFT.tolist()}"
    )

    start = time.perf_counter()
    simulation.run(EQUIL_STEPS)
    simulation.run(PRODUCTION_STEPS)

    # These identities are valid because 0 < S < 1.
    U_real = table.energy / S
    lambda_U_E = tether.energy / (1.0 - S)
    g_s = U_real - lambda_U_E

    print(
        f"done in {time.perf_counter() - start:.1f} s, "
        f"T={thermo.kinetic_temperature:.3f}"
    )
    print(f"scaled pair energy       = {table.energy:.6f}")
    print(f"scaled tether energy     = {tether.energy:.6f}")
    print(f"instantaneous g(s)/TLPP  = {g_s / n_cores:.6f}")
    print(f"trajectory               = {output}")


if __name__ == "__main__":
    for crystal_name in CRYSTALS:
        run_single_s_window(crystal_name)