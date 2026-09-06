"""
Bidisperse substitutional-defect stability test for a bulk diamond TLPP crystal.

Batch system
------------
* One CSV row defines one independent HPC job.
* The supplied scan crosses defect fractions 0.10, 0.15, 0.20, and 0.25
  with defect-lobe radii 0.30, 0.35, and 0.40 (12 jobs total).
* 5 x 5 x 5 conventional cubic-diamond cells (8 TLPPs/cell = 1000 TLPPs).
* Host lobe radius = 0.50; defect sites are random without replacement.
* Both species: size ratio sz = 1.15 and compression ratio cr = 0.60.
* Full periodic boundary conditions in x, y, and z. No walls and no gravity.
* Single-rank CPU execution is enabled by default.
* Effective AdG brush + Asakura-Osawa depletion interaction on every physical
  core/lobe type pair, using the actual radii of the two species. Each table
  extends to r = 0, with a finite repulsive continuation in the x < 0 overlap
  region instead of HOOMD's zero-below-r_min behavior.

Spring protocol
---------------
* k = 100 for  1,000,000 steps.
* k =  75 for  1,000,000 steps.
* k =  50 for  1,000,000 steps.
* k =  25 for  1,000,000 steps.
* k =   0 for  1,000,000 steps (free equilibration).
* k =   0 for  1,000,000 steps (stationarity measurement).

Pressure and structure protocol
-------------------------------
Pressure evaluation is deliberately enabled only when k reaches zero. During
each 1,000,000-step free window, Pxx, Pyy, and Pzz are sampled every 10,000
steps and averaged into ten 100,000-step blocks. The uncertainty reported for
the pressure is the standard deviation of the ten block means (with the SEM
also written for deciding whether the mean pressure is statistically zero).

At the end of each block, freud computes four-neighbor Steinhardt q4 and q6
from the rigid-center positions only. A center is diamond-like when it has
four neighbors in the first shell and both order parameters are close to the
ideal tetrahedral values. The diamond-like centers must also form a connected
cluster. Box adjustment is allowed only if every diagonal pressure component,
q4, q6, the local diamond-like count, and the largest-cluster count are
stationary when the first five and last five block means are compared. At
least the configured fraction of all centers must belong to the largest
diamond-like cluster.

If the stationary isotropic pressure is positive, the box is expanded
isotropically; if it is negative, the box is contracted. Each small box change
is followed by 1,000,000 equilibration steps and a fresh 1,000,000-step
measurement window. A pressure sign change halves the fractional box step.

The translational spring is implemented efficiently as a harmonic bond of
rest length zero between every rigid TLPP center and a fixed, non-integrated
anchor at its original lattice site. The anchor type has no pair interactions.
The spring does not restrain TLPP orientation.

The trajectory contains cores, lobes, and zero-diameter anchor particles.
For structural analysis, ignore type "anchor" and select core_host/core_defect
as the TLPP centers.
"""

import argparse
import csv
import itertools
import math
import os
import time

import freud
import gsd.hoomd
import hoomd
import numpy as np
from scipy.spatial.transform import Rotation


# =============================================================================
# USER INPUTS
# =============================================================================

# Defaults are overwritten by the selected CSV row before the system is built.
RUN_ID = "000"
SZ = 1.15
CR = 0.60

HOST_LOBE_R = 0.50
DEFECT_LOBE_R = 0.30
DEFECT_FRACTION = 0.10

# These are numbers of conventional eight-site cubic-diamond cells.
# Therefore, 5 x 5 x 5 gives 8*5*5*5 = 1000 TLPPs in a cubic box.
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
INITIAL_HOLD_STEPS = 1_000_000
RAMP_HOLD_STEPS = 1_000_000
FREE_EQUIL_STEPS = 1_000_000
FREE_MEASURE_STEPS = 1_000_000

# Pressure/block analysis. These choices give 10 instantaneous samples per
# block and 10 statistically coarser block means per measurement window.
PRESSURE_SAMPLE_EVERY = 10_000
PRESSURE_BLOCK_STEPS = 100_000
PRESSURE_Z = 2.0
PRESSURE_ABS_TOL = 1.0e-4

# Structure and stationarity guards for automatic box adjustment. The ideal
# q_l values below are for a center with four perfect tetrahedral neighbors.
# First-shell q4/q6 detect diamond-like tetrahedral order but do not distinguish
# cubic diamond from hexagonal diamond; the output is named accordingly.
IDEAL_DIAMOND_Q4 = 0.5091750772173156
IDEAL_DIAMOND_Q6 = 0.6285393610547089
Q4_TOLERANCE = 0.10
Q6_TOLERANCE = 0.10
Q_DRIFT_ABS_TOLERANCE = 0.02
FIRST_SHELL_CUTOFF_FACTOR = 1.35
MIN_DIAMOND_LIKE_FRACTION = 0.90
DIAMOND_DRIFT_TOLERANCE_FRACTION = 0.01
MAX_STATIONARITY_EXTENSIONS = 5

# Isotropic pressure relaxation after the initial 6,000,000-step protocol.
BOX_SCALE_STEP = 0.002          # 0.2% in each box length per adjustment
MIN_BOX_SCALE_STEP = 0.00025    # 0.025% after repeated sign changes
BOX_EQUIL_STEPS = 1_000_000
MAX_BOX_ADJUSTMENTS = 12

WRITE_EVERY = 100_000
LOG_EVERY = 100_000

DEFECT_SEED = 24680
DYNAMICS_SEED = 13579
USE_GPU = False

DEFAULT_OUTPUT_ROOT = "/scratch/agg7742/defect_interplay"
DEFAULT_INPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "defect_scan_inputs.csv",
)
OUT_DIR = DEFAULT_OUTPUT_ROOT


# Particle type names.
CORE_HOST = "core_host"
CORE_DEFECT = "core_defect"
ANCHOR = "anchor"
LOBE_HOST = "lobe_host"
LOBE_DEFECT = "lobe_defect"

PHYSICAL_TYPES = [CORE_HOST, CORE_DEFECT, LOBE_HOST, LOBE_DEFECT]
ALL_TYPES = [CORE_HOST, CORE_DEFECT, ANCHOR, LOBE_HOST, LOBE_DEFECT]

RADIUS_BY_TYPE = {
    CORE_HOST: SZ * HOST_LOBE_R,
    CORE_DEFECT: SZ * DEFECT_LOBE_R,
    LOBE_HOST: HOST_LOBE_R,
    LOBE_DEFECT: DEFECT_LOBE_R,
}


# =============================================================================
# CSV-DRIVEN HPC INPUT
# =============================================================================

def parse_command_line():
    parser = argparse.ArgumentParser(
        description=(
            "Run one row of the bidisperse TLPP defect scan. When "
            "--row-index is omitted, SLURM_ARRAY_TASK_ID is used; outside "
            "Slurm the default row is 0."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help="CSV containing one simulation parameter set per row.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Zero-based data-row index. Defaults to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory under which a separate run directory is made.",
    )
    return parser.parse_args()


def _csv_value(row, key, default, converter):
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return default
    try:
        return converter(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid CSV value for '{key}': {value!r}."
        ) from error


def select_csv_row(csv_filename, requested_row_index):
    """Load one zero-based row, using the Slurm array index when available."""
    if requested_row_index is None:
        slurm_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        row_index = int(slurm_index) if slurm_index is not None else 0
    else:
        row_index = requested_row_index

    with open(csv_filename, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows found in input CSV: {csv_filename}")
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(
            f"CSV row index {row_index} is outside 0..{len(rows) - 1}."
        )
    return row_index, rows[row_index], len(rows)


def apply_csv_configuration(csv_filename, requested_row_index, output_root):
    """Apply the selected CSV row to the module-level simulation inputs."""
    global RUN_ID
    global SZ, CR, HOST_LOBE_R, DEFECT_LOBE_R, DEFECT_FRACTION
    global NX, NY, NZ, KT, PHI, SIGMAP, RDEP
    global DEFECT_SEED, DYNAMICS_SEED, USE_GPU, OUT_DIR

    row_index, row, n_rows = select_csv_row(
        csv_filename, requested_row_index
    )

    RUN_ID = str(row.get("run_id", f"{row_index:03d}")).strip()
    if not RUN_ID:
        RUN_ID = f"{row_index:03d}"

    DEFECT_FRACTION = _csv_value(
        row, "defect_fraction", DEFECT_FRACTION, float
    )
    DEFECT_LOBE_R = _csv_value(
        row, "defect_lobe_radius", DEFECT_LOBE_R, float
    )
    HOST_LOBE_R = _csv_value(
        row, "host_lobe_radius", HOST_LOBE_R, float
    )
    SZ = _csv_value(row, "sz", SZ, float)
    CR = _csv_value(row, "cr", CR, float)
    NX = _csv_value(row, "nx", NX, int)
    NY = _csv_value(row, "ny", NY, int)
    NZ = _csv_value(row, "nz", NZ, int)
    KT = _csv_value(row, "kT", KT, float)
    PHI = _csv_value(row, "phi", PHI, float)
    SIGMAP = _csv_value(row, "sigmap", SIGMAP, float)
    RDEP = 0.5 * SIGMAP
    DEFECT_SEED = _csv_value(row, "defect_seed", DEFECT_SEED, int)
    DYNAMICS_SEED = _csv_value(
        row, "dynamics_seed", DYNAMICS_SEED, int
    )
    USE_GPU = bool(_csv_value(row, "use_gpu", 0, int))

    if not (0.0 < DEFECT_FRACTION < 1.0):
        raise ValueError("defect_fraction must be strictly between 0 and 1.")
    if DEFECT_LOBE_R <= 0.0 or HOST_LOBE_R <= 0.0:
        raise ValueError("Both lobe radii must be positive.")
    if DEFECT_LOBE_R >= HOST_LOBE_R:
        raise ValueError(
            "This scan expects defect_lobe_radius < host_lobe_radius."
        )
    if min(NX, NY, NZ) <= 0:
        raise ValueError("nx, ny, and nz must be positive integers.")
    n_tlpps = 8 * NX * NY * NZ
    if n_tlpps != 1000:
        raise ValueError(
            f"The selected row builds {n_tlpps} TLPPs; this batch requires "
            "exactly 1000 (8*nx*ny*nz)."
        )
    RADIUS_BY_TYPE.update(
        {
            CORE_HOST: SZ * HOST_LOBE_R,
            CORE_DEFECT: SZ * DEFECT_LOBE_R,
            LOBE_HOST: HOST_LOBE_R,
            LOBE_DEFECT: DEFECT_LOBE_R,
        }
    )

    run_directory_name = (
        f"run_{RUN_ID}_fdef_{DEFECT_FRACTION:.2f}"
        f"_rdef_{DEFECT_LOBE_R:.2f}"
    )
    OUT_DIR = os.path.join(output_root, run_directory_name)

    resolved = {
        "row_index": row_index,
        "n_csv_rows": n_rows,
        "input_csv": os.path.abspath(csv_filename),
        "run_id": RUN_ID,
        "defect_fraction": DEFECT_FRACTION,
        "defect_lobe_radius": DEFECT_LOBE_R,
        "host_lobe_radius": HOST_LOBE_R,
        "sz": SZ,
        "cr": CR,
        "nx": NX,
        "ny": NY,
        "nz": NZ,
        "n_tlpps": n_tlpps,
        "kT": KT,
        "phi": PHI,
        "sigmap": SIGMAP,
        "defect_seed": DEFECT_SEED,
        "dynamics_seed": DYNAMICS_SEED,
        "use_gpu": int(USE_GPU),
        "output_directory": OUT_DIR,
    }
    return resolved


def save_resolved_configuration(configuration, filename):
    with open(filename, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(configuration))
        writer.writeheader()
        writer.writerow(configuration)


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

TETRA = np.array(
    [
        [-1.0 / 2.0, -1.0 / (2.0 * np.sqrt(3.0)), -1.0 / (2.0 * np.sqrt(6.0))],
        [1.0 / 2.0, -1.0 / (2.0 * np.sqrt(3.0)), -1.0 / (2.0 * np.sqrt(6.0))],
        [0.0, 1.0 / np.sqrt(3.0), -1.0 / (2.0 * np.sqrt(6.0))],
        [0.0, 0.0, np.sqrt(3.0 / 8.0)],
    ]
)

# Nearest-neighbor directions from the A sublattice of a conventional cubic
# diamond cell. The B-sublattice directions are their negatives.
DIAMOND_A_DIRECTIONS = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]
) / np.sqrt(3.0)


def find_host_lattice_spacing():
    """Return a where the conventional cubic diamond constant is sqrt(2)*a."""
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
    """Generate an eight-site conventional cubic-diamond supercell."""
    a_cubic = np.sqrt(2.0) * a
    Lx = nx * a_cubic
    Ly = ny * a_cubic
    Lz = nz * a_cubic

    fcc_basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )
    basis = np.vstack((fcc_basis, fcc_basis + 0.25)) * a_cubic
    basis_sublattice = np.array([0] * 4 + [1] * 4, dtype=np.int8)

    centers = []
    sublattices = []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                translation = a_cubic * np.array([i, j, k], dtype=float)
                for basis_position, sublattice in zip(
                    basis, basis_sublattice
                ):
                    position = translation + basis_position
                    position[0] = wrap_to_box(position[0] - Lx / 2.0, Lx)
                    position[1] = wrap_to_box(position[1] - Ly / 2.0, Ly)
                    position[2] = wrap_to_box(position[2] - Lz / 2.0, Lz)
                    centers.append(position)
                    sublattices.append(sublattice)

    return (
        np.asarray(centers, dtype=float),
        np.asarray(sublattices, dtype=np.int8),
        (Lx, Ly, Lz),
    )


def matrix_to_quaternion(rotation_matrix):
    x, y, z, w = Rotation.from_matrix(rotation_matrix).as_quat()
    return [w, x, y, z]


def cubic_sublattice_quaternions():
    """Orient the four lobes along the diamond bonds on each sublattice."""
    body_directions = TETRA / np.linalg.norm(TETRA, axis=1)[:, None]

    # These fixed vertex correspondences give proper rotations (det=+1) and
    # map all four body directions exactly onto the A/B diamond bonds.
    target_A = DIAMOND_A_DIRECTIONS[[0, 1, 3, 2]]
    target_B = -DIAMOND_A_DIRECTIONS
    rotation_A, error_A = Rotation.align_vectors(target_A, body_directions)
    rotation_B, error_B = Rotation.align_vectors(target_B, body_directions)
    if max(error_A, error_B) > 1.0e-10:
        raise RuntimeError("Could not align TLPP lobes with cubic diamond bonds.")

    return (
        matrix_to_quaternion(rotation_A.as_matrix()),
        matrix_to_quaternion(rotation_B.as_matrix()),
    )


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
    centers, sublattices, (Lx, Ly, Lz) = diamond_centers(NX, NY, NZ, a)
    n_centers = len(centers)

    n_defects = int(round(DEFECT_FRACTION * n_centers))
    rng = np.random.default_rng(DEFECT_SEED)
    # Taking prefixes of one seeded random permutation makes the scan
    # controlled and nested: the 10% defect sites are contained in the 15%
    # set, which is contained in the 20% and 25% sets. Rows with the same
    # fraction but different radii therefore use exactly the same sites.
    defect_tags = np.sort(rng.permutation(n_centers)[:n_defects]).astype(
        np.int64
    )
    is_defect = np.zeros(n_centers, dtype=bool)
    is_defect[defect_tags] = True

    q_A, q_B = cubic_sublattice_quaternions()
    center_orientations = np.asarray(
        [q_A if sublattice == 0 else q_B for sublattice in sublattices],
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
        ("free_equilibration", 0.0, FREE_EQUIL_STEPS),
        ("free_stationarity", 0.0, FREE_MEASURE_STEPS),
    ]

    start = 0
    with open(filename, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "spring_k", "start_timestep", "end_timestep"])
        for label, spring_k, steps in stages:
            end = start + steps
            writer.writerow([label, spring_k, start, end])
            start = end


def validate_protocol():
    """Fail before a long run if the block protocol is inconsistent."""
    if FREE_EQUIL_STEPS % PRESSURE_BLOCK_STEPS != 0:
        raise ValueError("FREE_EQUIL_STEPS must contain whole pressure blocks.")
    if FREE_MEASURE_STEPS % PRESSURE_BLOCK_STEPS != 0:
        raise ValueError("FREE_MEASURE_STEPS must contain whole pressure blocks.")
    if PRESSURE_BLOCK_STEPS % PRESSURE_SAMPLE_EVERY != 0:
        raise ValueError(
            "PRESSURE_BLOCK_STEPS must contain whole pressure-sampling intervals."
        )

    n_blocks = FREE_MEASURE_STEPS // PRESSURE_BLOCK_STEPS
    if n_blocks < 10 or n_blocks % 2 != 0:
        raise ValueError(
            "FREE_MEASURE_STEPS must provide an even number of at least 10 blocks."
        )
    if not (0.0 < MIN_DIAMOND_LIKE_FRACTION <= 1.0):
        raise ValueError("MIN_DIAMOND_LIKE_FRACTION must be in (0, 1].")
    if Q4_TOLERANCE <= 0.0 or Q6_TOLERANCE <= 0.0:
        raise ValueError("Q4_TOLERANCE and Q6_TOLERANCE must be positive.")
    if FIRST_SHELL_CUTOFF_FACTOR <= 1.0:
        raise ValueError("FIRST_SHELL_CUTOFF_FACTOR must be greater than 1.")
    if not (0.0 < MIN_BOX_SCALE_STEP <= BOX_SCALE_STEP < 1.0):
        raise ValueError(
            "Require 0 < MIN_BOX_SCALE_STEP <= BOX_SCALE_STEP < 1."
        )


def checkpoint_state(sim, logger, filename):
    hoomd.write.GSD.write(
        state=sim.state,
        filename=filename,
        mode="wb",
        logger=logger,
    )


def current_box_values(sim):
    box = sim.state.box
    return float(box.Lx), float(box.Ly), float(box.Lz)


def analyze_diamond_order(sim):
    """Return freud q4/q6 and connectivity metrics for rigid TLPP centers.

    Only core_host and core_defect are analyzed. Lobes and anchors are
    intentionally excluded. First-shell q4/q6 identify tetrahedral
    diamond-like order, not the cubic-versus-hexagonal stacking sequence.
    Connectivity prevents disconnected tetrahedral fragments from being
    counted as a surviving crystal.
    """
    if sim.device.communicator.num_ranks != 1:
        raise RuntimeError(
            "The in-run freud structure check requires one MPI rank so every "
            "pressure/box decision uses the complete set of TLPP centers."
        )

    snapshot = sim.state.get_snapshot()
    type_names = list(snapshot.particles.types)
    typeids = np.asarray(snapshot.particles.typeid)
    center_typeids = [
        type_names.index(CORE_HOST),
        type_names.index(CORE_DEFECT),
    ]
    center_mask = np.isin(typeids, center_typeids)
    positions = np.asarray(
        snapshot.particles.position, dtype=np.float32
    )[center_mask].copy()
    n_centers = len(positions)
    if n_centers < 5:
        raise RuntimeError("At least five TLPP centers are required for q4/q6.")

    Lx, Ly, Lz, xy, xz, yz = map(float, snapshot.configuration.box)
    box = freud.box.Box(
        Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz
    )
    query = freud.locality.AABBQuery(box, positions)

    # q4/q6 use exactly the four nearest center neighbors.
    nearest_four = query.query(
        positions,
        {"num_neighbors": 4, "exclude_ii": True},
    ).toNeighborList()
    neighbor_counts = np.bincount(
        np.asarray(nearest_four.query_point_indices, dtype=np.int64),
        minlength=n_centers,
    )
    if not np.all(neighbor_counts == 4):
        raise RuntimeError("freud did not find four neighbors for every center.")

    q4_compute = freud.order.Steinhardt(l=4, average=False)
    q6_compute = freud.order.Steinhardt(l=6, average=False)
    q4_compute.compute((box, positions), neighbors=nearest_four)
    q6_compute.compute((box, positions), neighbors=nearest_four)
    q4 = np.asarray(q4_compute.particle_order, dtype=float)
    q6 = np.asarray(q6_compute.particle_order, dtype=float)

    # The cubic-diamond first-neighbor distance is sqrt(3)*a_cubic/4.
    # Scale the cutoff with the current box, so an isotropic box move does not
    # by itself change the coordination classification.
    current_cubic_a = np.mean([Lx / NX, Ly / NY, Lz / NZ])
    ideal_first_neighbor = np.sqrt(3.0) * current_cubic_a / 4.0
    first_shell_cutoff = FIRST_SHELL_CUTOFF_FACTOR * ideal_first_neighbor
    first_shell = query.query(
        positions,
        {"r_max": first_shell_cutoff, "exclude_ii": True},
    ).toNeighborList()
    shell_counts = np.bincount(
        np.asarray(first_shell.query_point_indices, dtype=np.int64),
        minlength=n_centers,
    )

    finite = np.isfinite(q4) & np.isfinite(q6)
    diamond_like = (
        finite
        & (shell_counts == 4)
        & (np.abs(q4 - IDEAL_DIAMOND_Q4) <= Q4_TOLERANCE)
        & (np.abs(q6 - IDEAL_DIAMOND_Q6) <= Q6_TOLERANCE)
    )

    # Union connected diamond-like centers using first-shell bonds. The
    # neighbor list is directed, which is harmless for union operations.
    parent = np.arange(n_centers, dtype=np.int64)
    component_size = np.ones(n_centers, dtype=np.int64)

    def find_root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        root_first = find_root(first)
        root_second = find_root(second)
        if root_first == root_second:
            return
        if component_size[root_first] < component_size[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        component_size[root_first] += component_size[root_second]

    query_indices = np.asarray(
        first_shell.query_point_indices, dtype=np.int64
    )
    point_indices = np.asarray(first_shell.point_indices, dtype=np.int64)
    for first, second in zip(query_indices, point_indices):
        if diamond_like[first] and diamond_like[second]:
            union(int(first), int(second))

    local_indices = np.flatnonzero(diamond_like)
    if local_indices.size:
        roots = np.fromiter(
            (find_root(int(index)) for index in local_indices),
            dtype=np.int64,
            count=local_indices.size,
        )
        largest_cluster = int(np.max(np.unique(roots, return_counts=True)[1]))
    else:
        largest_cluster = 0

    return {
        "total": n_centers,
        "n_four_coordinated": int(np.count_nonzero(shell_counts == 4)),
        "n_diamond_local": int(local_indices.size),
        "largest_diamond_cluster": largest_cluster,
        "q4_mean": float(np.nanmean(q4)),
        "q4_std": float(np.nanstd(q4, ddof=1)),
        "q6_mean": float(np.nanmean(q6)),
        "q6_std": float(np.nanstd(q6, ddof=1)),
        "first_shell_cutoff": float(first_shell_cutoff),
    }


def read_diagonal_pressure(thermo):
    """Return Pxx, Pyy, Pzz, and their isotropic mean."""
    pressure_tensor = np.asarray(thermo.pressure_tensor, dtype=float)
    if pressure_tensor.size != 6:
        raise RuntimeError(
            "Expected HOOMD pressure_tensor order xx,xy,xz,yy,yz,zz."
        )
    Pxx = float(pressure_tensor[0])
    Pyy = float(pressure_tensor[3])
    Pzz = float(pressure_tensor[5])
    Piso = (Pxx + Pyy + Pzz) / 3.0
    values = np.array([Pxx, Pyy, Pzz, Piso])
    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            "Pressure is non-finite. Confirm sim.always_compute_pressure=True "
            "and inspect the trajectory for particle overlaps."
        )
    return Pxx, Pyy, Pzz, Piso


def run_pressure_window(
    sim,
    thermo,
    metadata,
    label,
    iteration,
    n_steps,
    pressure_writer,
    pressure_handle,
    block_writer,
    block_handle,
):
    """Run one sampled free window and return instantaneous/block data."""
    if n_steps % PRESSURE_BLOCK_STEPS != 0:
        raise ValueError("A pressure window must contain complete blocks.")

    samples_per_block = PRESSURE_BLOCK_STEPS // PRESSURE_SAMPLE_EVERY
    n_blocks = n_steps // PRESSURE_BLOCK_STEPS
    window_start = int(sim.timestep)
    all_samples = []
    blocks = []

    for block_index in range(n_blocks):
        block_samples = []
        pending_rows = []

        for sample_index in range(samples_per_block):
            sim.run(PRESSURE_SAMPLE_EVERY)
            Pxx, Pyy, Pzz, Piso = read_diagonal_pressure(thermo)
            Lx, Ly, Lz = current_box_values(sim)
            values = {
                "Pxx": Pxx,
                "Pyy": Pyy,
                "Pzz": Pzz,
                "Piso": Piso,
            }
            block_samples.append(values)
            all_samples.append(values)
            pending_rows.append(
                [
                    int(sim.timestep),
                    label,
                    iteration,
                    block_index,
                    sample_index,
                    Pxx,
                    Pyy,
                    Pzz,
                    Piso,
                    Lx,
                    Ly,
                    Lz,
                ]
            )

        order = analyze_diamond_order(sim)
        for row in pending_rows:
            pressure_writer.writerow(
                row
                + [
                    order["n_four_coordinated"],
                    order["n_diamond_local"],
                    order["largest_diamond_cluster"],
                    order["q4_mean"],
                    order["q4_std"],
                    order["q6_mean"],
                    order["q6_std"],
                    order["first_shell_cutoff"],
                ]
            )

        block = {
            field: float(np.mean([sample[field] for sample in block_samples]))
            for field in ("Pxx", "Pyy", "Pzz", "Piso")
        }
        block.update(
            {
                "start_timestep": int(
                    sim.timestep - PRESSURE_BLOCK_STEPS
                ),
                "end_timestep": int(sim.timestep),
                "n_four_coordinated": order["n_four_coordinated"],
                "n_diamond_local": order["n_diamond_local"],
                "largest_diamond_cluster": order[
                    "largest_diamond_cluster"
                ],
                "q4_mean": order["q4_mean"],
                "q4_std": order["q4_std"],
                "q6_mean": order["q6_mean"],
                "q6_std": order["q6_std"],
                "first_shell_cutoff": order["first_shell_cutoff"],
            }
        )
        blocks.append(block)

        Lx, Ly, Lz = current_box_values(sim)
        current_cubic_a = metadata["a_diamond_cubic"] * (
            Lx / metadata["box"][0]
        )
        block_writer.writerow(
            [
                label,
                iteration,
                block_index,
                block["start_timestep"],
                block["end_timestep"],
                block["Pxx"],
                block["Pyy"],
                block["Pzz"],
                block["Piso"],
                block["n_four_coordinated"],
                block["n_diamond_local"],
                block["largest_diamond_cluster"],
                block["q4_mean"],
                block["q4_std"],
                block["q6_mean"],
                block["q6_std"],
                block["first_shell_cutoff"],
                Lx,
                Ly,
                Lz,
                current_cubic_a,
            ]
        )
        pressure_handle.flush()
        block_handle.flush()
        print(
            f"[{label}] block {block_index + 1:02d}/{n_blocks:02d}: "
            f"Piso={block['Piso']:+.6e}, "
            f"diamond-like={block['n_diamond_local']}/"
            f"{metadata['n_centers']}, "
            f"largest cluster={block['largest_diamond_cluster']}/"
            f"{metadata['n_centers']}, "
            f"q4={block['q4_mean']:.4f}, q6={block['q6_mean']:.4f}"
        )

    return {
        "label": label,
        "iteration": iteration,
        "start_timestep": window_start,
        "end_timestep": int(sim.timestep),
        "samples": all_samples,
        "blocks": blocks,
    }


def _sample_std(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def analyze_pressure_window(window, n_centers):
    """Compute block uncertainty and early/late stationarity tests."""
    blocks = window["blocks"]
    midpoint = len(blocks) // 2
    if midpoint < 2:
        raise ValueError("Need at least four blocks for a stationarity test.")

    stats = {
        "label": window["label"],
        "iteration": window["iteration"],
        "start_timestep": window["start_timestep"],
        "end_timestep": window["end_timestep"],
    }
    component_stationarity = []

    for field in ("Pxx", "Pyy", "Pzz", "Piso"):
        values = np.asarray([block[field] for block in blocks], dtype=float)
        early = values[:midpoint]
        late = values[-midpoint:]
        std = _sample_std(values)
        sem = std / math.sqrt(len(values))
        early_sem = _sample_std(early) / math.sqrt(len(early))
        late_sem = _sample_std(late) / math.sqrt(len(late))
        drift = float(np.mean(late) - np.mean(early))
        drift_limit = max(
            PRESSURE_ABS_TOL,
            PRESSURE_Z * math.sqrt(early_sem**2 + late_sem**2),
        )

        stats[f"{field}_mean"] = float(np.mean(values))
        stats[f"{field}_block_std"] = std
        stats[f"{field}_sem"] = sem
        stats[f"{field}_early_mean"] = float(np.mean(early))
        stats[f"{field}_late_mean"] = float(np.mean(late))
        stats[f"{field}_drift"] = drift
        stats[f"{field}_drift_limit"] = drift_limit
        component_stationarity.append(abs(drift) <= drift_limit)

    raw_piso = [sample["Piso"] for sample in window["samples"]]
    stats["Piso_raw_std"] = _sample_std(raw_piso)
    stats["pressure_stationary"] = bool(all(component_stationarity))

    def structure_drift(field, absolute_tolerance):
        values = np.asarray([block[field] for block in blocks], dtype=float)
        early = values[:midpoint]
        late = values[-midpoint:]
        early_sem = _sample_std(early) / math.sqrt(midpoint)
        late_sem = _sample_std(late) / math.sqrt(midpoint)
        drift = float(np.mean(late) - np.mean(early))
        drift_limit = max(
            absolute_tolerance,
            PRESSURE_Z * math.sqrt(early_sem**2 + late_sem**2),
        )
        return {
            "mean": float(np.mean(values)),
            "std": _sample_std(values),
            "early_mean": float(np.mean(early)),
            "late_mean": float(np.mean(late)),
            "drift": drift,
            "drift_limit": drift_limit,
            "stationary": abs(drift) <= drift_limit,
        }

    count_tolerance = max(
        1.0, DIAMOND_DRIFT_TOLERANCE_FRACTION * n_centers
    )
    local_stats = structure_drift("n_diamond_local", count_tolerance)
    cluster_stats = structure_drift(
        "largest_diamond_cluster", count_tolerance
    )
    q4_stats = structure_drift("q4_mean", Q_DRIFT_ABS_TOLERANCE)
    q6_stats = structure_drift("q6_mean", Q_DRIFT_ABS_TOLERANCE)
    four_coord = np.asarray(
        [block["n_four_coordinated"] for block in blocks], dtype=float
    )
    stats.update(
        {
            "n_four_coordinated_mean": float(np.mean(four_coord)),
            "n_diamond_local_mean": local_stats["mean"],
            "n_diamond_local_std": local_stats["std"],
            "diamond_local_fraction": local_stats["mean"] / n_centers,
            "n_diamond_local_early_mean": local_stats["early_mean"],
            "n_diamond_local_late_mean": local_stats["late_mean"],
            "n_diamond_local_drift": local_stats["drift"],
            "n_diamond_local_drift_limit": local_stats["drift_limit"],
            "diamond_local_stationary": local_stats["stationary"],
            "largest_diamond_cluster_mean": cluster_stats["mean"],
            "largest_diamond_cluster_std": cluster_stats["std"],
            "largest_diamond_cluster_fraction": (
                cluster_stats["mean"] / n_centers
            ),
            "largest_diamond_cluster_early_mean": cluster_stats[
                "early_mean"
            ],
            "largest_diamond_cluster_late_mean": cluster_stats["late_mean"],
            "largest_diamond_cluster_drift": cluster_stats["drift"],
            "largest_diamond_cluster_drift_limit": cluster_stats[
                "drift_limit"
            ],
            "diamond_cluster_stationary": cluster_stats["stationary"],
            "q4_mean": q4_stats["mean"],
            "q4_block_std": q4_stats["std"],
            "q4_early_mean": q4_stats["early_mean"],
            "q4_late_mean": q4_stats["late_mean"],
            "q4_drift": q4_stats["drift"],
            "q4_drift_limit": q4_stats["drift_limit"],
            "q4_stationary": q4_stats["stationary"],
            "q6_mean": q6_stats["mean"],
            "q6_block_std": q6_stats["std"],
            "q6_early_mean": q6_stats["early_mean"],
            "q6_late_mean": q6_stats["late_mean"],
            "q6_drift": q6_stats["drift"],
            "q6_drift_limit": q6_stats["drift_limit"],
            "q6_stationary": q6_stats["stationary"],
        }
    )
    stats["structure_ok"] = (
        stats["largest_diamond_cluster_fraction"]
        >= MIN_DIAMOND_LIKE_FRACTION
    )
    stats["diamond_stationary"] = bool(
        stats["diamond_local_stationary"]
        and stats["diamond_cluster_stationary"]
        and stats["q4_stationary"]
        and stats["q6_stationary"]
    )
    stats["pressure_zero_limit"] = max(
        PRESSURE_ABS_TOL,
        PRESSURE_Z * stats["Piso_sem"],
    )
    stats["pressure_zero"] = (
        abs(stats["Piso_mean"]) <= stats["pressure_zero_limit"]
    )
    stats["stationary"] = (
        stats["pressure_stationary"] and stats["diamond_stationary"]
    )
    return stats


SUMMARY_FIELDS = [
    "label",
    "iteration",
    "start_timestep",
    "end_timestep",
    "Lx",
    "Ly",
    "Lz",
    "cubic_lattice_constant",
    "box_fractional_step",
    "Pxx_mean",
    "Pxx_block_std",
    "Pxx_sem",
    "Pxx_early_mean",
    "Pxx_late_mean",
    "Pxx_drift",
    "Pxx_drift_limit",
    "Pyy_mean",
    "Pyy_block_std",
    "Pyy_sem",
    "Pyy_early_mean",
    "Pyy_late_mean",
    "Pyy_drift",
    "Pyy_drift_limit",
    "Pzz_mean",
    "Pzz_block_std",
    "Pzz_sem",
    "Pzz_early_mean",
    "Pzz_late_mean",
    "Pzz_drift",
    "Pzz_drift_limit",
    "Piso_mean",
    "Piso_raw_std",
    "Piso_block_std",
    "Piso_sem",
    "Piso_early_mean",
    "Piso_late_mean",
    "Piso_drift",
    "Piso_drift_limit",
    "pressure_zero_limit",
    "pressure_stationary",
    "pressure_zero",
    "n_four_coordinated_mean",
    "n_diamond_local_mean",
    "n_diamond_local_std",
    "diamond_local_fraction",
    "n_diamond_local_early_mean",
    "n_diamond_local_late_mean",
    "n_diamond_local_drift",
    "n_diamond_local_drift_limit",
    "diamond_local_stationary",
    "largest_diamond_cluster_mean",
    "largest_diamond_cluster_std",
    "largest_diamond_cluster_fraction",
    "largest_diamond_cluster_early_mean",
    "largest_diamond_cluster_late_mean",
    "largest_diamond_cluster_drift",
    "largest_diamond_cluster_drift_limit",
    "diamond_cluster_stationary",
    "q4_mean",
    "q4_block_std",
    "q4_early_mean",
    "q4_late_mean",
    "q4_drift",
    "q4_drift_limit",
    "q4_stationary",
    "q6_mean",
    "q6_block_std",
    "q6_early_mean",
    "q6_late_mean",
    "q6_drift",
    "q6_drift_limit",
    "q6_stationary",
    "diamond_stationary",
    "structure_ok",
    "stationary",
    "decision",
]


def write_window_summary(
    summary_writer,
    summary_handle,
    stats,
    sim,
    metadata,
    decision,
    box_fractional_step=0.0,
):
    Lx, Ly, Lz = current_box_values(sim)
    row = {field: stats.get(field, "") for field in SUMMARY_FIELDS}
    row.update(
        {
            "Lx": Lx,
            "Ly": Ly,
            "Lz": Lz,
            "cubic_lattice_constant": metadata["a_diamond_cubic"]
            * (Lx / metadata["box"][0]),
            "box_fractional_step": box_fractional_step,
            "decision": decision,
        }
    )
    summary_writer.writerow(row)
    summary_handle.flush()


def print_window_summary(stats):
    print(
        f"[{stats['label']}] Pxx={stats['Pxx_mean']:+.6e}, "
        f"Pyy={stats['Pyy_mean']:+.6e}, "
        f"Pzz={stats['Pzz_mean']:+.6e}"
    )
    print(
        f"[{stats['label']}] Piso={stats['Piso_mean']:+.6e}; "
        f"raw std={stats['Piso_raw_std']:.3e}; "
        f"block std={stats['Piso_block_std']:.3e}; "
        f"SEM={stats['Piso_sem']:.3e}; "
        f"early-late drift={stats['Piso_drift']:+.3e}"
    )
    print(
        f"[{stats['label']}] diamond-like local="
        f"{stats['n_diamond_local_mean']:.1f}; largest cluster="
        f"{stats['largest_diamond_cluster_mean']:.1f} "
        f"({stats['largest_diamond_cluster_fraction']:.4f}); "
        f"q4={stats['q4_mean']:.4f}; q6={stats['q6_mean']:.4f}; "
        f"pressure stationary={stats['pressure_stationary']}; "
        f"diamond stationary={stats['diamond_stationary']}"
    )


def resize_box_isotropically(sim, scale_factor):
    """Instantaneously scale the box and all particle reference positions."""
    old_box = sim.state.box
    new_box = hoomd.Box(
        Lx=old_box.Lx * scale_factor,
        Ly=old_box.Ly * scale_factor,
        Lz=old_box.Lz * scale_factor,
        xy=old_box.xy,
        xz=old_box.xz,
        yz=old_box.yz,
    )
    # Scaling every particle also moves the k=0 anchors consistently. On the
    # following integration step, the rigid constraint restores each TLPP's
    # undeformed internal core-lobe geometry.
    hoomd.update.BoxResize.update(state=sim.state, box=new_box)


# =============================================================================
# SIMULATION
# =============================================================================

def main():
    args = parse_command_line()
    configuration = apply_csv_configuration(
        csv_filename=args.input_csv,
        requested_row_index=args.row_index,
        output_root=args.output_root,
    )
    validate_protocol()
    os.makedirs(OUT_DIR, exist_ok=True)

    tag = (
        f"run_{RUN_ID}_diamond_{NX}x{NY}x{NZ}_sz_{SZ:.2f}_cr_{CR:.2f}"
        f"_rhost_{HOST_LOBE_R:.2f}_rdef_{DEFECT_LOBE_R:.2f}"
        f"_xdef_{DEFECT_FRACTION:.2f}_phi_{PHI:.3f}"
    )
    trajectory_file = os.path.join(OUT_DIR, f"{tag}.gsd")
    thermo_file = os.path.join(OUT_DIR, f"{tag}_thermo.txt")
    defect_file = os.path.join(OUT_DIR, f"{tag}_defect_sites.csv")
    schedule_file = os.path.join(OUT_DIR, f"{tag}_spring_schedule.csv")
    pressure_file = os.path.join(OUT_DIR, f"{tag}_pressure_samples.csv")
    block_file = os.path.join(OUT_DIR, f"{tag}_pressure_blocks.csv")
    summary_file = os.path.join(OUT_DIR, f"{tag}_pressure_summary.csv")
    resolved_config_file = os.path.join(
        OUT_DIR, f"{tag}_resolved_input.csv"
    )
    save_resolved_configuration(configuration, resolved_config_file)

    initial_snapshot, metadata = build_initial_snapshot()
    save_defect_table(metadata, defect_file)
    save_schedule(schedule_file)

    device = hoomd.device.GPU() if USE_GPU else hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=DYNAMICS_SEED)
    sim.create_state_from_snapshot(initial_snapshot)
    sim.always_compute_pressure = False

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
    pressure_handle = open(pressure_file, "w", newline="")
    block_handle = open(block_file, "w", newline="")
    summary_handle = open(summary_file, "w", newline="")

    pressure_writer = csv.writer(pressure_handle)
    pressure_writer.writerow(
        [
            "timestep",
            "label",
            "iteration",
            "block_index",
            "sample_index",
            "Pxx",
            "Pyy",
            "Pzz",
            "Piso",
            "Lx",
            "Ly",
            "Lz",
            "n_four_coordinated_at_block_end",
            "n_diamond_local_at_block_end",
            "largest_diamond_cluster_at_block_end",
            "q4_mean_at_block_end",
            "q4_std_at_block_end",
            "q6_mean_at_block_end",
            "q6_std_at_block_end",
            "first_shell_cutoff_at_block_end",
        ]
    )
    block_writer = csv.writer(block_handle)
    block_writer.writerow(
        [
            "label",
            "iteration",
            "block_index",
            "start_timestep",
            "end_timestep",
            "Pxx_mean",
            "Pyy_mean",
            "Pzz_mean",
            "Piso_mean",
            "n_four_coordinated",
            "n_diamond_local",
            "largest_diamond_cluster",
            "q4_mean",
            "q4_std",
            "q6_mean",
            "q6_std",
            "first_shell_cutoff",
            "Lx",
            "Ly",
            "Lz",
            "cubic_lattice_constant",
        ]
    )
    summary_writer = csv.DictWriter(summary_handle, fieldnames=SUMMARY_FIELDS)
    summary_writer.writeheader()

    table_writer = hoomd.write.Table(
        trigger=hoomd.trigger.Periodic(LOG_EVERY),
        logger=logger,
        output=thermo_handle,
    )
    sim.operations.writers.append(table_writer)

    sim.run(0)
    initial_order = analyze_diamond_order(sim)

    print("\nBidisperse TLPP defect-stability run")
    print(
        f"  CSV row                     : {configuration['row_index']} of "
        f"{configuration['n_csv_rows'] - 1}"
    )
    print(f"  run ID                      : {RUN_ID}")
    print(f"  HOOMD version               : {hoomd.version.version}")
    print(f"  device                      : {sim.device}")
    print(f"  cell dimensions             : {NX} x {NY} x {NZ}")
    print(f"  TLPP centers                : {metadata['n_centers']}")
    print(
        f"  r_l={DEFECT_LOBE_R:.2f} defect TLPPs       : "
        f"{metadata['n_defects']}"
    )
    print(
        f"  r_l={HOST_LOBE_R:.2f} host TLPPs         : "
        f"{metadata['n_centers'] - metadata['n_defects']}"
    )
    print(f"  defect fraction             : {DEFECT_FRACTION:.4f}")
    print(f"  builder lattice spacing a   : {metadata['a_builder']:.8f}")
    print(f"  cubic diamond constant      : {metadata['a_diamond_cubic']:.8f}")
    print(f"  box                          : {metadata['box']}")
    print(f"  overlap continuation        : r_min=0, match x={X_REG:g}")
    print(f"  pair-table points           : {N_TABLE:,}")
    print(f"  total particles incl. anchors/lobes: {sim.state.N_particles}")
    print(
        "  initial freud diamond-like  : "
        f"{initial_order['n_diamond_local']}/{metadata['n_centers']} local, "
        f"largest cluster {initial_order['largest_diamond_cluster']}/"
        f"{metadata['n_centers']}"
    )
    print(
        "  initial q4, q6              : "
        f"{initial_order['q4_mean']:.6f}, {initial_order['q6_mean']:.6f}"
    )
    print("  pressure calculation        : OFF until k=0")
    print(f"  trajectory                  : {trajectory_file}")
    print(f"  defect list                 : {defect_file}")
    print(f"  schedule                    : {schedule_file}")
    print(f"  resolved input              : {resolved_config_file}")
    print(f"  pressure samples            : {pressure_file}")
    print(f"  pressure blocks             : {block_file}")
    print(f"  pressure summaries          : {summary_file}\n")

    restrained_stages = [
        ("restrained", INITIAL_K, INITIAL_HOLD_STEPS),
        ("ramp_k75", 75.0, RAMP_HOLD_STEPS),
        ("ramp_k50", 50.0, RAMP_HOLD_STEPS),
        ("ramp_k25", 25.0, RAMP_HOLD_STEPS),
    ]

    total_start = time.perf_counter()
    for stage_index, (label, spring_k, n_steps) in enumerate(
        restrained_stages
    ):
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
        checkpoint_state(sim, logger, checkpoint)

        elapsed = time.perf_counter() - stage_start
        pe_per_tlpp = thermo.potential_energy / metadata["n_centers"]
        print(
            f"[{label}] completed in {elapsed:.1f} s; "
            f"T={thermo.kinetic_temperature:.5f}; "
            f"PE/TLPP={pe_per_tlpp:.8f}; checkpoint={checkpoint}"
        )

    # k=0 is the exact point at which pressure evaluation and recording start.
    # Keeping the bond object with k=0 is dynamically equivalent to removing
    # the tethers, while preserving a simple restartable topology.
    spring.params["tether"] = {"k": 0.0, "r0": 0.0}
    sim.always_compute_pressure = True
    print(
        f"[free_equilibration] k=0.0; pressure calculation ON; "
        f"timestep {sim.timestep:,} -> "
        f"{sim.timestep + FREE_EQUIL_STEPS:,}"
    )

    free_equil_window = run_pressure_window(
        sim=sim,
        thermo=thermo,
        metadata=metadata,
        label="free_equilibration",
        iteration=0,
        n_steps=FREE_EQUIL_STEPS,
        pressure_writer=pressure_writer,
        pressure_handle=pressure_handle,
        block_writer=block_writer,
        block_handle=block_handle,
    )
    free_equil_stats = analyze_pressure_window(
        free_equil_window, metadata["n_centers"]
    )
    print_window_summary(free_equil_stats)
    write_window_summary(
        summary_writer,
        summary_handle,
        free_equil_stats,
        sim,
        metadata,
        decision="equilibration_only_no_box_decision",
    )
    gsd_writer.flush()
    checkpoint_state(
        sim,
        logger,
        os.path.join(
            OUT_DIR,
            f"{tag}_after_free_equilibration_step_{sim.timestep}.gsd",
        ),
    )

    # The first decision is made after the second k=0 window, at timestep 6e6
    # unless an earlier user setting changed one of the fixed durations.
    current_stats = None
    stop_reason = None
    for extension in range(MAX_STATIONARITY_EXTENSIONS + 1):
        label = (
            "free_stationarity"
            if extension == 0
            else f"free_stationarity_extension_{extension}"
        )
        print(
            f"[{label}] k=0.0; timestep {sim.timestep:,} -> "
            f"{sim.timestep + FREE_MEASURE_STEPS:,}"
        )
        window = run_pressure_window(
            sim=sim,
            thermo=thermo,
            metadata=metadata,
            label=label,
            iteration=extension,
            n_steps=FREE_MEASURE_STEPS,
            pressure_writer=pressure_writer,
            pressure_handle=pressure_handle,
            block_writer=block_writer,
            block_handle=block_handle,
        )
        current_stats = analyze_pressure_window(
            window, metadata["n_centers"]
        )
        print_window_summary(current_stats)

        if not current_stats["structure_ok"]:
            decision = "stop_diamond_order_below_threshold"
            stop_reason = decision
        elif not current_stats["stationary"]:
            if extension < MAX_STATIONARITY_EXTENSIONS:
                decision = "extend_k0_stationarity_window"
            else:
                decision = "stop_not_stationary_after_extensions"
                stop_reason = decision
        elif current_stats["pressure_zero"]:
            decision = "stop_pressure_consistent_with_zero"
            stop_reason = decision
        elif current_stats["Piso_mean"] > 0.0:
            decision = "stationary_expand_box_next"
        else:
            decision = "stationary_contract_box_next"

        write_window_summary(
            summary_writer,
            summary_handle,
            current_stats,
            sim,
            metadata,
            decision=decision,
        )
        gsd_writer.flush()
        checkpoint_state(
            sim,
            logger,
            os.path.join(
                OUT_DIR,
                f"{tag}_after_{label}_step_{sim.timestep}.gsd",
            ),
        )

        if stop_reason is not None or current_stats["stationary"]:
            break

    # Pressure-directed isotropic box relaxation. No box move is made unless
    # both observables are stationary and the structure passes the survival
    # threshold. The sign convention is positive -> expand, negative -> shrink.
    fractional_step = BOX_SCALE_STEP
    previous_pressure_sign = None
    adjustments_completed = 0

    if stop_reason is None:
        for adjustment in range(1, MAX_BOX_ADJUSTMENTS + 1):
            current_sign = 1 if current_stats["Piso_mean"] > 0.0 else -1
            if (
                previous_pressure_sign is not None
                and current_sign != previous_pressure_sign
            ):
                fractional_step = max(
                    MIN_BOX_SCALE_STEP, 0.5 * fractional_step
                )
                print(
                    "[box relaxation] pressure sign changed; reducing "
                    f"box-length step to {100.0 * fractional_step:.4f}%"
                )

            signed_step = current_sign * fractional_step
            scale_factor = 1.0 + signed_step
            action = "expand" if current_sign > 0 else "contract"
            old_box = current_box_values(sim)
            resize_box_isotropically(sim, scale_factor)
            new_box = current_box_values(sim)
            previous_pressure_sign = current_sign

            print(
                f"[box_adjustment_{adjustment:02d}] {action} each length by "
                f"{100.0 * fractional_step:.4f}%: "
                f"L=({old_box[0]:.8f}, {old_box[1]:.8f}, {old_box[2]:.8f}) "
                f"-> ({new_box[0]:.8f}, {new_box[1]:.8f}, "
                f"{new_box[2]:.8f})"
            )

            equil_label = f"box_adjustment_{adjustment:02d}_equilibration"
            equil_start = int(sim.timestep)
            print(
                f"[{equil_label}] timestep {equil_start:,} -> "
                f"{equil_start + BOX_EQUIL_STEPS:,}"
            )
            sim.run(BOX_EQUIL_STEPS)
            gsd_writer.flush()
            checkpoint_state(
                sim,
                logger,
                os.path.join(
                    OUT_DIR,
                    f"{tag}_after_{equil_label}_step_{sim.timestep}.gsd",
                ),
            )

            measurement_passed = False
            for extension in range(MAX_STATIONARITY_EXTENSIONS + 1):
                label = f"box_adjustment_{adjustment:02d}_measurement"
                if extension:
                    label += f"_extension_{extension}"

                window = run_pressure_window(
                    sim=sim,
                    thermo=thermo,
                    metadata=metadata,
                    label=label,
                    iteration=adjustment,
                    n_steps=FREE_MEASURE_STEPS,
                    pressure_writer=pressure_writer,
                    pressure_handle=pressure_handle,
                    block_writer=block_writer,
                    block_handle=block_handle,
                )
                current_stats = analyze_pressure_window(
                    window, metadata["n_centers"]
                )
                print_window_summary(current_stats)

                if not current_stats["structure_ok"]:
                    decision = "stop_diamond_order_below_threshold"
                    stop_reason = decision
                elif not current_stats["stationary"]:
                    if extension < MAX_STATIONARITY_EXTENSIONS:
                        decision = "extend_post_resize_stationarity_window"
                    else:
                        decision = "stop_not_stationary_after_extensions"
                        stop_reason = decision
                elif current_stats["pressure_zero"]:
                    decision = "stop_pressure_consistent_with_zero"
                    stop_reason = decision
                    measurement_passed = True
                elif current_stats["Piso_mean"] > 0.0:
                    decision = "stationary_expand_box_next"
                    measurement_passed = True
                else:
                    decision = "stationary_contract_box_next"
                    measurement_passed = True

                write_window_summary(
                    summary_writer,
                    summary_handle,
                    current_stats,
                    sim,
                    metadata,
                    decision=decision,
                    box_fractional_step=signed_step,
                )
                gsd_writer.flush()
                checkpoint_state(
                    sim,
                    logger,
                    os.path.join(
                        OUT_DIR,
                        f"{tag}_after_{label}_step_{sim.timestep}.gsd",
                    ),
                )

                if stop_reason is not None or measurement_passed:
                    break

            adjustments_completed = adjustment
            if stop_reason is not None:
                break

        if stop_reason is None:
            stop_reason = "stop_maximum_box_adjustments_reached"

    gsd_writer.flush()
    thermo_handle.flush()
    pressure_handle.flush()
    block_handle.flush()
    summary_handle.flush()
    thermo_handle.close()
    pressure_handle.close()
    block_handle.close()
    summary_handle.close()

    total_elapsed = time.perf_counter() - total_start
    print(f"\nAll stages completed in {total_elapsed:.1f} s")
    print(f"Final timestep: {sim.timestep:,}")
    print(f"Box adjustments completed: {adjustments_completed}")
    print(f"Stop reason: {stop_reason}")
    print(f"Final box: {current_box_values(sim)}")
    print(f"Final trajectory: {trajectory_file}")
    print(f"Pressure summary: {summary_file}")


if __name__ == "__main__":
    main()
