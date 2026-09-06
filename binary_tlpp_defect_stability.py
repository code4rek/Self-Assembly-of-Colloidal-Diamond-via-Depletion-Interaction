"""
Bidisperse substitutional-defect stability test for a bulk diamond TLPP crystal.

System
------
* 5 x 5 x 5 orthorhombic diamond cells (4 TLPPs/cell = 500 TLPPs).
* 90% host TLPPs:   lobe radius = 0.50.
* 10% defect TLPPs: lobe radius = 0.45, chosen at random without replacement.
* Both species: size ratio sz = 1.15 and compression ratio cr = 0.60.
* Full periodic boundary conditions in x, y, and z. No walls and no gravity.
* Effective AdG brush + Asakura-Osawa depletion interaction on every physical
  core/lobe type pair, using the actual radii of the two species. Each table
  extends to r = 0, with a finite repulsive continuation in the x < 0 overlap
  region instead of HOOMD's zero-below-r_min behavior.

Spring protocol
---------------
* k = 100 for 10,000,000 steps.
* k =  75 for  1,000,000 steps.
* k =  50 for  1,000,000 steps.
* k =  25 for  1,000,000 steps.
* k =   0 for 10,000,000 steps (fully free stability test).

The translational spring is implemented efficiently as a harmonic bond of
rest length zero between every rigid TLPP center and a fixed, non-integrated
anchor at its original lattice site. The anchor type has no pair interactions.
The spring does not restrain TLPP orientation.

The trajectory contains cores, lobes, and zero-diameter anchor particles.
For structural analysis, ignore type "anchor" and select core_050/core_045 as
the TLPP centers.
"""

import csv
import itertools
import math
import os
import time

import gsd.hoomd
import hoomd
import numpy as np
from scipy.spatial.transform import Rotation


# =============================================================================
# USER INPUTS
# =============================================================================

SZ = 1.15
CR = 0.60

HOST_LOBE_R = 0.50
DEFECT_LOBE_R = 0.45
DEFECT_FRACTION = 0.10

# These are numbers of the orthorhombic four-site diamond cells used by the
# builder below. Therefore, 5 x 5 x 5 gives 4*5*5*5 = 500 TLPPs.
NX = 5
NY = 5
NZ = 5

KT = 1.0
PHI = 0.05
SIGMAP = 0.02                 # depletant diameter
RDEP = 0.5 * SIGMAP           # depletant radius
L_BRUSH = 0.01
SIGMAG = 0.09 * (1000.0**2)

DT = 1.0e-4
# r_min=0 covers the complete overlap interval. This width keeps the spacing
# near contact comparable to the narrower 1000-point table in the reference.
N_TABLE = 20_000
NLIST_BUFFER = 0.0

# Short-range regularization of the divergent AdG brush potential.
# The original AO+AdG expression is used unchanged for x >= X_REG. Below
# X_REG, a tangent-line continuation keeps both U and F continuous and gives
# a strictly repulsive force throughout the overlap region x < 0.
X_REG = L_BRUSH

INITIAL_K = 100.0
INITIAL_HOLD_STEPS = 10_000_000
RAMP_HOLD_STEPS = 1_000_000
FREE_STEPS = 10_000_000

WRITE_EVERY = 100_000
LOG_EVERY = 100_000

DEFECT_SEED = 24680
DYNAMICS_SEED = 13579
USE_GPU = False

OUT_DIR = "/home/akash/hpc_hoomd-files/bidisperse_tlpp_defect_stability"


# Particle type names.
CORE_HOST = "core_050"
CORE_DEFECT = "core_045"
ANCHOR = "anchor"
LOBE_HOST = "lobe_050"
LOBE_DEFECT = "lobe_045"

PHYSICAL_TYPES = [CORE_HOST, CORE_DEFECT, LOBE_HOST, LOBE_DEFECT]
ALL_TYPES = [CORE_HOST, CORE_DEFECT, ANCHOR, LOBE_HOST, LOBE_DEFECT]

RADIUS_BY_TYPE = {
    CORE_HOST: SZ * HOST_LOBE_R,
    CORE_DEFECT: SZ * DEFECT_LOBE_R,
    LOBE_HOST: HOST_LOBE_R,
    LOBE_DEFECT: DEFECT_LOBE_R,
}


# =============================================================================
# EFFECTIVE AdG + AO POTENTIAL
# =============================================================================

def P_Implicit_Dep_raw(r1, r2, sigma_g, L, phi, rdep, rcc):
    """Unregularized effective AdG brush + AO depletion potential for x>0."""
    r1_eff = L + r1
    r2_eff = L + r2
    x = rcc - r1 - r2
    q1 = r1_eff / rdep
    q2 = r2_eff / rdep
    n = rcc / rdep
    r_avg = 2.0 / (1.0 / r1 + 1.0 / r2)

    if (
        x <= 2.0 * L
        and x > 0.0
        and rcc <= r1_eff + r2_eff + 2.0 * rdep
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
    elif x > 2.0 * L and rcc <= r1_eff + r2_eff + 2.0 * rdep:
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
        U_B = 0.0
        U_AO = 0.0

    return U_B + U_AO


def P_Implicit_Dep(r1, r2, sigma_g, L, phi, rdep, rcc):
    """Regularized AdG+AO potential with a repulsive continuation for x<0.

    The analytic brush expression diverges as x -> 0 from above, so no finite
    continuation can be joined exactly at x=0. Instead, match the original
    potential and its force at the small positive gap X_REG. For all smaller
    gaps, use the tangent line

        U(x) = U(x_reg) + F(x_reg) * (x_reg - x),

    which has the constant outward force F(x_reg) > 0. The original potential
    is completely unchanged for x >= X_REG.
    """
    contact = r1 + r2
    x = rcc - contact

    if x >= X_REG:
        return P_Implicit_Dep_raw(r1, r2, sigma_g, L, phi, rdep, rcc)

    r_match = contact + X_REG
    U_match = P_Implicit_Dep_raw(
        r1, r2, sigma_g, L, phi, rdep, r_match
    )

    # Centered numerical derivative of the unchanged analytic potential at
    # the matching point. X_REG is positive, so both samples remain physical.
    derivative_step = min(1.0e-6, 0.01 * X_REG)
    U_plus = P_Implicit_Dep_raw(
        r1,
        r2,
        sigma_g,
        L,
        phi,
        rdep,
        r_match + derivative_step,
    )
    U_minus = P_Implicit_Dep_raw(
        r1,
        r2,
        sigma_g,
        L,
        phi,
        rdep,
        r_match - derivative_step,
    )
    F_match = -(U_plus - U_minus) / (2.0 * derivative_step)

    if not np.isfinite(F_match) or F_match <= 0.0:
        raise RuntimeError(
            "The short-range continuation did not produce a positive "
            f"repulsive force: r1={r1}, r2={r2}, F_match={F_match}."
        )

    return U_match + F_match * (X_REG - x)


def tabulate_P_implicit(
    r1,
    r2,
    sigma_g,
    L,
    phi,
    rdep,
    r_min,
    r_max,
    n_points,
):
    # HOOMD interprets table values on linspace(..., endpoint=False).
    r = np.linspace(r_min, r_max, n_points, endpoint=False)
    U = np.array(
        [P_Implicit_Dep(r1, r2, sigma_g, L, phi, rdep, ri) for ri in r]
    )
    F = -np.gradient(U, r)

    overlap = r < (r1 + r2)
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(F)):
        raise RuntimeError(f"Non-finite AO+AdG table for radii {r1}, {r2}.")
    if np.any(U[overlap] <= 0.0) or np.any(F[overlap] <= 0.0):
        raise RuntimeError(
            "The regularized overlap branch must have U>0 and F>0 for "
            f"radii {r1}, {r2}."
        )

    return U, F


def make_pair_table():
    """Build all ten physical pair tables plus zero anchor interactions."""
    nlist = hoomd.md.nlist.Cell(
        buffer=NLIST_BUFFER,
        exclusions=["body", "bond"],
    )
    table = hoomd.md.pair.Table(nlist=nlist, default_r_cut=0.0)

    for type_i, type_j in itertools.combinations_with_replacement(
        PHYSICAL_TYPES, 2
    ):
        r_i = RADIUS_BY_TYPE[type_i]
        r_j = RADIUS_BY_TYPE[type_j]
        contact = r_i + r_j
        # Table returns U=F=0 below r_min. Using r_min=0 therefore prevents a
        # zero-potential hole anywhere in the physically possible x<0 range.
        r_min = 0.0
        r_cut = r_i + r_j + 2.0 * L_BRUSH + SIGMAP

        U, F = tabulate_P_implicit(
            r_i,
            r_j,
            SIGMAG,
            L_BRUSH,
            PHI,
            RDEP,
            r_min,
            r_cut,
            N_TABLE,
        )
        table.params[(type_i, type_j)] = {
            "r_min": r_min,
            "U": U,
            "F": F,
        }
        table.r_cut[(type_i, type_j)] = r_cut

    # Anchor particles only define fixed reference positions. They must not
    # interact through the physical pair potential. Setting r_cut=0 excludes
    # every pair involving an anchor from the neighbor list.
    zero_U = np.zeros(N_TABLE, dtype=float)
    zero_F = np.zeros(N_TABLE, dtype=float)
    for type_i in ALL_TYPES:
        pair = (type_i, ANCHOR)
        table.params[pair] = {
            "r_min": 0.0,
            "U": zero_U,
            "F": zero_F,
        }
        table.r_cut[pair] = 0.0

    return table


# =============================================================================
# DIAMOND GEOMETRY
# =============================================================================

THETAX = np.arccos(1.0 / np.sqrt(3.0))
RX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, np.cos(THETAX), -np.sin(THETAX)],
        [0.0, np.sin(THETAX), np.cos(THETAX)],
    ]
)
RZ90 = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

TETRA = np.array(
    [
        [-1.0 / 2.0, -1.0 / (2.0 * np.sqrt(3.0)), -1.0 / (2.0 * np.sqrt(6.0))],
        [1.0 / 2.0, -1.0 / (2.0 * np.sqrt(3.0)), -1.0 / (2.0 * np.sqrt(6.0))],
        [0.0, 1.0 / np.sqrt(3.0), -1.0 / (2.0 * np.sqrt(6.0))],
        [0.0, 0.0, np.sqrt(3.0 / 8.0)],
    ]
)


def find_host_lattice_spacing():
    """Return the orthorhombic builder's in-plane diamond lattice spacing."""
    rl = HOST_LOBE_R
    a = (4.0 / np.sqrt(6.0)) * (2.0 * rl * SZ + 2.0 * L_BRUSH)

    if SZ < CR / np.sqrt(6.0) + np.sqrt(1.0 - CR**2 / 3.0):
        a = (4.0 / np.sqrt(6.0)) * (
            2.0 * CR * rl / np.sqrt(6.0)
            + 2.0
            * rl
            * np.sqrt((1.0 + L_BRUSH / rl) ** 2 - CR**2 / 3.0)
        )
    return a


def wrap_to_box(x, box_length):
    return (x + box_length / 2.0) % box_length - box_length / 2.0


def diamond_centers(nx, ny, nz, a):
    """Generate the four-site orthorhombic diamond cell used in the reference."""
    ax = a
    ay = a
    az = a * np.sqrt(2.0)
    Lx = nx * ax
    Ly = ny * ay
    Lz = nz * az

    basis = a * np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 1.0 / (2.0 * np.sqrt(2.0))],
            [0.5, 0.5, 1.0 / np.sqrt(2.0)],
            [0.0, 0.5, 3.0 / (2.0 * np.sqrt(2.0))],
        ]
    )

    dz = a / (2.0 * np.sqrt(2.0))
    centers = []
    layer_indices = []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                translation = np.array([i * ax, j * ay, k * az])
                for basis_position in basis:
                    position = translation + basis_position
                    position[0] = wrap_to_box(position[0] - Lx / 2.0, Lx)
                    position[1] = wrap_to_box(position[1] - Ly / 2.0, Ly)
                    position[2] = wrap_to_box(position[2] - Lz / 2.0, Lz)
                    centers.append(position)
                    layer_indices.append(
                        int(np.floor((position[2] + Lz / 2.0) / dz + 1.0e-9))
                    )

    return (
        np.asarray(centers, dtype=float),
        np.asarray(layer_indices, dtype=int),
        (Lx, Ly, Lz),
    )


def matrix_to_quaternion(rotation_matrix):
    x, y, z, w = Rotation.from_matrix(rotation_matrix).as_quat()
    return [w, x, y, z]


def rigid_body_properties(lobe_radius):
    """Return body-space lobe positions and the reference-code moment tensor."""
    body_positions = TETRA * (CR * 2.0 * lobe_radius)

    inertia = np.zeros((3, 3), dtype=float)
    for position in body_positions:
        inertia += (
            np.dot(position, position) * np.eye(3)
            - np.outer(position, position)
        )

    return body_positions, np.diag(inertia).copy()


def build_initial_snapshot():
    """Create rigid centers, fixed anchors, and one tether bond per center."""
    a = find_host_lattice_spacing()
    centers, layer_indices, (Lx, Ly, Lz) = diamond_centers(NX, NY, NZ, a)
    n_centers = len(centers)

    n_defects = int(round(DEFECT_FRACTION * n_centers))
    rng = np.random.default_rng(DEFECT_SEED)
    defect_tags = np.sort(
        rng.choice(n_centers, size=n_defects, replace=False)
    ).astype(np.int64)
    is_defect = np.zeros(n_centers, dtype=bool)
    is_defect[defect_tags] = True

    q_even = matrix_to_quaternion(RX)
    q_odd = matrix_to_quaternion(RZ90 @ RX)
    center_orientations = np.asarray(
        [q_even if layer % 2 == 0 else q_odd for layer in layer_indices],
        dtype=np.float32,
    )

    _, moi_host = rigid_body_properties(HOST_LOBE_R)
    _, moi_defect = rigid_body_properties(DEFECT_LOBE_R)

    # Initial particles are the N rigid centers followed by N fixed anchors.
    # Rigid constituent lobes are added later by rigid.create_bodies().
    n_initial = 2 * n_centers
    snap = gsd.hoomd.Frame()
    snap.configuration.box = [Lx, Ly, Lz, 0.0, 0.0, 0.0]
    snap.configuration.step = 0

    snap.particles.N = n_initial
    snap.particles.types = ALL_TYPES

    typeid = np.full(n_initial, ALL_TYPES.index(ANCHOR), dtype=np.uint32)
    typeid[:n_centers] = np.where(
        is_defect,
        ALL_TYPES.index(CORE_DEFECT),
        ALL_TYPES.index(CORE_HOST),
    ).astype(np.uint32)
    snap.particles.typeid = typeid

    positions = np.vstack([centers, centers]).astype(np.float32)
    snap.particles.position = positions

    orientations = np.zeros((n_initial, 4), dtype=np.float32)
    orientations[:, 0] = 1.0
    orientations[:n_centers] = center_orientations
    snap.particles.orientation = orientations

    body = np.full(n_initial, -1, dtype=np.int32)
    body[:n_centers] = np.arange(n_centers, dtype=np.int32)
    snap.particles.body = body

    # Keep the same total rigid-body mass convention as the reference script.
    mass = np.ones(n_initial, dtype=np.float32)
    mass[:n_centers] = 5.0
    snap.particles.mass = mass

    moment_inertia = np.zeros((n_initial, 3), dtype=np.float32)
    host_indices = np.flatnonzero(~is_defect)
    defect_indices = np.flatnonzero(is_defect)
    moment_inertia[host_indices] = moi_host
    moment_inertia[defect_indices] = moi_defect
    snap.particles.moment_inertia = moment_inertia

    diameter = np.zeros(n_initial, dtype=np.float32)
    diameter[:n_centers] = np.where(
        is_defect,
        2.0 * SZ * DEFECT_LOBE_R,
        2.0 * SZ * HOST_LOBE_R,
    )
    snap.particles.diameter = diameter

    # Each center tag i is tethered to its coincident anchor tag N+i.
    snap.bonds.N = n_centers
    snap.bonds.types = ["tether"]
    snap.bonds.typeid = np.zeros(n_centers, dtype=np.uint32)
    snap.bonds.group = np.column_stack(
        [
            np.arange(n_centers, dtype=np.int32),
            np.arange(n_centers, 2 * n_centers, dtype=np.int32),
        ]
    )

    metadata = {
        "a_builder": a,
        "a_diamond_cubic": np.sqrt(2.0) * a,
        "box": (Lx, Ly, Lz),
        "n_centers": n_centers,
        "n_defects": n_defects,
        "defect_tags": defect_tags,
        "centers": centers,
        "is_defect": is_defect,
    }
    return snap, metadata


def set_created_particle_diameters(sim):
    """Set constituent lobe diameters after rigid.create_bodies()."""
    snapshot = sim.state.get_snapshot()

    if sim.device.communicator.rank == 0:
        type_names = list(snapshot.particles.types)
        typeids = np.asarray(snapshot.particles.typeid)
        diameters = np.asarray(snapshot.particles.diameter).copy()

        target_diameters = {
            CORE_HOST: 2.0 * SZ * HOST_LOBE_R,
            CORE_DEFECT: 2.0 * SZ * DEFECT_LOBE_R,
            ANCHOR: 0.0,
            LOBE_HOST: 2.0 * HOST_LOBE_R,
            LOBE_DEFECT: 2.0 * DEFECT_LOBE_R,
        }

        for type_name, value in target_diameters.items():
            type_index = type_names.index(type_name)
            diameters[typeids == type_index] = value

        # HOOMD Snapshot array properties cannot be replaced, but their
        # contents are writable. Assign in place instead of using the
        # property setter.
        snapshot.particles.diameter[:] = diameters.astype(np.float32)

    sim.state.set_snapshot(snapshot)


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def save_defect_table(metadata, filename):
    if not metadata["defect_tags"].size:
        return

    with open(filename, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["center_tag", "lobe_radius", "x", "y", "z"])
        for tag in metadata["defect_tags"]:
            x, y, z = metadata["centers"][tag]
            writer.writerow([int(tag), DEFECT_LOBE_R, x, y, z])


def save_schedule(filename):
    stages = [
        ("restrained", INITIAL_K, INITIAL_HOLD_STEPS),
        ("ramp_k75", 75.0, RAMP_HOLD_STEPS),
        ("ramp_k50", 50.0, RAMP_HOLD_STEPS),
        ("ramp_k25", 25.0, RAMP_HOLD_STEPS),
        ("free", 0.0, FREE_STEPS),
    ]

    start = 0
    with open(filename, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "spring_k", "start_timestep", "end_timestep"])
        for label, spring_k, steps in stages:
            end = start + steps
            writer.writerow([label, spring_k, start, end])
            start = end


# =============================================================================
# SIMULATION
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tag = (
        f"diamond_{NX}x{NY}x{NZ}_sz_{SZ:.2f}_cr_{CR:.2f}"
        f"_rhost_{HOST_LOBE_R:.2f}_rdef_{DEFECT_LOBE_R:.2f}"
        f"_xdef_{DEFECT_FRACTION:.2f}_phi_{PHI:.3f}"
    )
    trajectory_file = os.path.join(OUT_DIR, f"{tag}.gsd")
    thermo_file = os.path.join(OUT_DIR, f"{tag}_thermo.txt")
    defect_file = os.path.join(OUT_DIR, f"{tag}_defect_sites.csv")
    schedule_file = os.path.join(OUT_DIR, f"{tag}_spring_schedule.csv")

    initial_snapshot, metadata = build_initial_snapshot()
    save_defect_table(metadata, defect_file)
    save_schedule(schedule_file)

    device = hoomd.device.GPU() if USE_GPU else hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=DYNAMICS_SEED)
    sim.create_state_from_snapshot(initial_snapshot)

    host_body_positions, _ = rigid_body_properties(HOST_LOBE_R)
    defect_body_positions, _ = rigid_body_properties(DEFECT_LOBE_R)

    rigid = hoomd.md.constrain.Rigid()
    rigid.body[CORE_HOST] = {
        "constituent_types": [LOBE_HOST] * 4,
        "positions": host_body_positions.tolist(),
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * 4,
    }
    rigid.body[CORE_DEFECT] = {
        "constituent_types": [LOBE_DEFECT] * 4,
        "positions": defect_body_positions.tolist(),
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * 4,
    }
    rigid.create_bodies(sim.state)
    set_created_particle_diameters(sim)

    pair_table = make_pair_table()

    spring = hoomd.md.bond.Harmonic()
    spring.params["tether"] = {"k": INITIAL_K, "r0": 0.0}

    # Only the two rigid-center types are integrated. Anchors remain fixed,
    # and the rigid constraint updates the lobe constituents automatically.
    centers_filter = hoomd.filter.Type([CORE_HOST, CORE_DEFECT])
    langevin = hoomd.md.methods.Langevin(filter=centers_filter, kT=KT)

    integrator = hoomd.md.Integrator(
        dt=DT,
        methods=[langevin],
        forces=[pair_table, spring],
        integrate_rotational_dof=True,
    )
    integrator.rigid = rigid
    sim.operations.integrator = integrator

    sim.state.thermalize_particle_momenta(filter=centers_filter, kT=KT)

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=centers_filter)
    sim.operations.computes.append(thermo)

    # hoomd.write.Table accepts only scalar/string logger categories.
    # Every quantity below is scalar, so restrict the Logger accordingly.
    logger = hoomd.logging.Logger(categories=["scalar"])
    logger.add(sim, quantities=["timestep", "tps"])
    logger.add(
        thermo,
        quantities=[
            "kinetic_temperature",
            "potential_energy",
            "kinetic_energy",
            "translational_kinetic_energy",
            "rotational_kinetic_energy",
        ],
    )
    logger.add(pair_table, quantities=["energy"])
    logger.add(spring, quantities=["energy"])

    gsd_writer = hoomd.write.GSD(
        filename=trajectory_file,
        trigger=hoomd.trigger.Periodic(WRITE_EVERY),
        mode="wb",
        dynamic=["property", "momentum", "attribute", "topology"],
        logger=logger,
    )
    gsd_writer.write_diameter = True
    sim.operations.writers.append(gsd_writer)

    thermo_handle = open(thermo_file, "w")
    table_writer = hoomd.write.Table(
        trigger=hoomd.trigger.Periodic(LOG_EVERY),
        logger=logger,
        output=thermo_handle,
    )
    sim.operations.writers.append(table_writer)

    sim.run(0)

    print("\nBidisperse TLPP defect-stability run")
    print(f"  HOOMD version               : {hoomd.version.version}")
    print(f"  cell dimensions             : {NX} x {NY} x {NZ}")
    print(f"  TLPP centers                : {metadata['n_centers']}")
    print(f"  r_l=0.45 defect TLPPs       : {metadata['n_defects']}")
    print(f"  r_l=0.50 host TLPPs         : {metadata['n_centers'] - metadata['n_defects']}")
    print(f"  builder lattice spacing a   : {metadata['a_builder']:.8f}")
    print(f"  cubic diamond constant      : {metadata['a_diamond_cubic']:.8f}")
    print(f"  box                          : {metadata['box']}")
    print(f"  overlap continuation        : r_min=0, match x={X_REG:g}")
    print(f"  pair-table points           : {N_TABLE:,}")
    print(f"  total particles incl. anchors/lobes: {sim.state.N_particles}")
    print(f"  trajectory                  : {trajectory_file}")
    print(f"  defect list                 : {defect_file}")
    print(f"  schedule                    : {schedule_file}\n")

    stages = [
        ("restrained", INITIAL_K, INITIAL_HOLD_STEPS),
        ("ramp_k75", 75.0, RAMP_HOLD_STEPS),
        ("ramp_k50", 50.0, RAMP_HOLD_STEPS),
        ("ramp_k25", 25.0, RAMP_HOLD_STEPS),
        ("free", 0.0, FREE_STEPS),
    ]

    total_start = time.perf_counter()
    for stage_index, (label, spring_k, n_steps) in enumerate(stages):
        spring.params["tether"] = {"k": spring_k, "r0": 0.0}
        start_step = sim.timestep
        stage_start = time.perf_counter()

        print(
            f"[{label}] k={spring_k:.1f}, steps={n_steps:,}, "
            f"timestep {start_step:,} -> {start_step + n_steps:,}"
        )

        sim.run(n_steps, write_at_start=(stage_index == 0))
        gsd_writer.flush()

        checkpoint = os.path.join(
            OUT_DIR,
            f"{tag}_after_{label}_step_{sim.timestep}.gsd",
        )
        hoomd.write.GSD.write(
            state=sim.state,
            filename=checkpoint,
            mode="wb",
            logger=logger,
        )

        elapsed = time.perf_counter() - stage_start
        pe_per_tlpp = thermo.potential_energy / metadata["n_centers"]
        print(
            f"[{label}] completed in {elapsed:.1f} s; "
            f"T={thermo.kinetic_temperature:.5f}; "
            f"PE/TLPP={pe_per_tlpp:.8f}; checkpoint={checkpoint}"
        )

    gsd_writer.flush()
    thermo_handle.flush()
    thermo_handle.close()

    total_elapsed = time.perf_counter() - total_start
    print(f"\nAll stages completed in {total_elapsed:.1f} s")
    print(f"Final timestep: {sim.timestep:,}")
    print(f"Final trajectory: {trajectory_file}")


if __name__ == "__main__":
    main()
