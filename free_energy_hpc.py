"""
Incremental Frenkel-Ladd sampling for diamond, MTAB, and rhombohedral TLPP
crystals.

Hamiltonian:

    U_s = s U_real + (1-s) lambda_max U_E

The physical brush + depletion potential is unchanged outside an extremely
small contact regularization region. At and below contact, the tabulated
potential is continued with a linear repulsive branch. This prevents
hoomd.md.pair.Table from assigning zero energy and zero force to overlapping
pairs below r_min.

At every TLPP, U_E tethers the core center and all four lobes to their own
reference positions. The four equivalent lobe tethers give tetrahedrally
symmetric small-angle rotational stiffness.

Each CSV row defines one independent (crystal, sz, cr, rl, s, seed,
lambda_max) window. Completed windows are never overwritten, incomplete
attempts remain preserved, and adding rows later does not waste finished work.
"""

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
import gsd.hoomd
import hoomd
import numpy as np
from scipy.spatial.transform import Rotation


# ===========================================================================
# Fixed thermodynamic state
# ===========================================================================
KT = 1.0
PHI = 0.05
SIGMAP = 0.02
RDEP = 0.5*SIGMAP
L_BRUSH = 0.01
SIGMAG = 0.09*(1000)**2

# The Python file owns the output location and creates it when needed.
OUT_DIR = Path("/scratch/agg7742/free_energy")

# All three periodic cells contain exactly 144 TLPPs.
# diamond: 4 basis sites * 3 * 4 * 3 = 144
# mtab:    4 sites per repeated cell * 3 * 4 * 3 = 144
# rhombo:  4 columns * 4 rows * 9 ABC layers = 144
DIAMOND_DIMS = (3, 4, 3)
MTAB_DIMS = (3, 4, 3)
RHOMBO_DIMS = (4, 4, 9)

def sampling_settings(s_value):
    """Select timestep and existing sampling schedule for this s window."""
    if 0.00001 <= s_value <= 0.008:
        return 5e-5, 200_000, 100_000_000, 20_000, 20_000
    return 1e-4, 100_000, 50_000_000, 10_000, 10_000


USE_GPU = False
FIX_REFERENCE_TLPP = True

# Contact-table regularization. The original table began exactly at contact,
# where P_Implicit_Dep returned zero, and HOOMD also sets U=F=0 below r_min.
# The new table extends almost to zero center-to-center separation and joins
# the original potential at a small positive surface gap with a continuous,
# constant-force repulsive branch. The original potential is unchanged for
# x >= CONTACT_SWITCH_GAP.
CONTACT_SWITCH_GAP = 4.0e-5
CONTACT_DERIVATIVE_STEP = 4.0e-5
TABLE_R_MIN = 0.0001
TABLE_TARGET_BIN_WIDTH = 4.0e-5
POTENTIAL_VERSION = "contact_regularized_extended_table_v3"

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


def P_Implicit_Dep_contact_regularized(
    r1,
    r2,
    sigma_g,
    L,
    phi,
    rdep,
    rcc,
):
    """Return the physical potential with a repulsive contact continuation.

    For surface gaps x >= CONTACT_SWITCH_GAP, this returns the original
    P_Implicit_Dep potential exactly.

    For x < CONTACT_SWITCH_GAP, it uses the tangent estimated immediately to
    the right of CONTACT_SWITCH_GAP and continues that repulsive branch
    linearly. Thus, U and F remain finite, continuous, and repulsive through
    contact and into the overlap region sampled by the Einstein endpoint.
    """
    contact_distance = r1 + r2
    x = rcc - contact_distance

    if x >= CONTACT_SWITCH_GAP:
        return P_Implicit_Dep(
            r1,
            r2,
            sigma_g,
            L,
            phi,
            rdep,
            rcc,
        )

    r_switch = contact_distance + CONTACT_SWITCH_GAP
    r_next = r_switch + CONTACT_DERIVATIVE_STEP

    U_switch = P_Implicit_Dep(
        r1,
        r2,
        sigma_g,
        L,
        phi,
        rdep,
        r_switch,
    )
    U_next = P_Implicit_Dep(
        r1,
        r2,
        sigma_g,
        L,
        phi,
        rdep,
        r_next,
    )

    F_switch = -(
        U_next - U_switch
    ) / CONTACT_DERIVATIVE_STEP

    if not np.isfinite(F_switch) or F_switch <= 0.0:
        raise RuntimeError(
            "Contact continuation did not produce a finite repulsive force: "
            f"F_switch={F_switch}."
        )

    return U_switch + F_switch * (
        CONTACT_SWITCH_GAP - x
    )


def tabulate_P_implicit(
    r1, r2, sigma_g, L, phi, rdep, r_min, r_max, n_points=1000
):
    # HOOMD's implicit table grid is equivalent to endpoint=False.
    r = np.linspace(r_min, r_max, n_points, endpoint=False)
    U = np.array(
        [
            P_Implicit_Dep_contact_regularized(
                r1,
                r2,
                sigma_g,
                L,
                phi,
                rdep,
                ri,
            )
            for ri in r
        ]
    )
    F = -np.gradient(U, r)

    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(F)):
        raise RuntimeError("Non-finite value in the regularized pair table.")

    return U, F


# ===========================================================================
# Diamond, MTAB, and rhombohedral crystal geometry
# ===========================================================================
THETAX = np.arccos(1.0 / np.sqrt(3.0))
RX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, np.cos(THETAX), -np.sin(THETAX)],
        [0.0, np.sin(THETAX), np.cos(THETAX)],
    ],
    dtype=np.float64,
)
RZ90 = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
RX180 = np.diag([1.0, -1.0, -1.0])

TETRA = np.array(
    [
        [-1 / 2, -1 / (2 * np.sqrt(3)), -1 / (2 * np.sqrt(6))],
        [1 / 2, -1 / (2 * np.sqrt(3)), -1 / (2 * np.sqrt(6))],
        [0.0, 1 / np.sqrt(3), -1 / (2 * np.sqrt(6))],
        [0.0, 0.0, np.sqrt(3 / 8)],
    ],
    dtype=np.float64,
)


def normalize_crystal_name(crystal):
    name = str(crystal).strip().lower()
    aliases = {
        "diamond": "diamond",
        "cubic_diamond": "diamond",
        "mtab": "mtab",
        "monoclinic": "mtab",
        "rhombo": "rhombo",
        "rhombohedral": "rhombo",
    }
    if name not in aliases:
        raise ValueError(
            f"Unknown crystal {crystal!r}; use diamond, mtab, or rhombo."
        )
    return aliases[name]


def mat_to_quat_hoomd(matrix):
    """Convert a rotation matrix to HOOMD quaternion order [w, x, y, z]."""
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def wrap_to_box(x, box_length):
    return (x + box_length / 2.0) % box_length - box_length / 2.0


def find_diamond_mtab_spacing(sz, cr, rl, L=L_BRUSH):
    a = (4.0 / np.sqrt(6.0)) * (2.0 * rl * sz + 2.0 * L)
    if sz < cr / np.sqrt(6.0) + np.sqrt(1.0 - cr**2 / 3.0):
        a = (4.0 / np.sqrt(6.0)) * (
            2.0 * cr * rl / np.sqrt(6.0)
            + 2.0 * rl * np.sqrt((1.0 + L / rl) ** 2 - cr**2 / 3.0)
        )
    return a


def find_rhombo_spacing(sz, cr, rl, L=L_BRUSH):
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
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 1.0 / (2.0 * np.sqrt(2.0))],
            [0.5, 0.5, 1.0 / np.sqrt(2.0)],
            [0.0, 0.5, 3.0 / (2.0 * np.sqrt(2.0))],
        ]
    )
    dz = a / (2.0 * np.sqrt(2.0))
    centers, layer_index = [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                translation = np.array([i * ax, j * ay, k * az])
                for offset in basis:
                    position = translation + offset
                    position[0] = wrap_to_box(position[0] - Lx / 2.0, Lx)
                    position[1] = wrap_to_box(position[1] - Ly / 2.0, Ly)
                    position[2] = wrap_to_box(position[2] - Lz / 2.0, Lz)
                    centers.append(position)
                    layer_index.append(
                        int(np.floor((position[2] + Lz / 2.0) / dz + 1e-9))
                    )
    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(layer_index, dtype=np.int32),
        (Lx, Ly, Lz),
    )


def compute_mtab_h(cr, rl, L=L_BRUSH):
    return rl * (cr / np.sqrt(2.0) + np.sqrt(1.0 - cr**2 / 2.0)) + L


def compute_mtab_staggered_h(sz, cr, rl, L=L_BRUSH):
    a = find_diamond_mtab_spacing(sz, cr, rl, L)
    a_bcd = a / 2.0
    dnn = np.sqrt(6.0) * a / 4.0
    return np.sqrt(dnn**2 - a_bcd**2)


def mtab_centers(nx, ny, nz, a, h, hs):
    if nz % nx != 0:
        raise ValueError("For this orthorhombic MTAB cell, nz must be a multiple of nx.")

    Lx = nx * a
    Ly = ny * a
    Lz = nz * 2.0 * (h + hs)
    n_cores = nx * ny * nz * 4
    centers = np.zeros((n_cores, 3), dtype=np.float64)
    basis_index = np.zeros(n_cores, dtype=np.int32)
    basis = np.array([[0.0, 0.0, 0.0], [0.0, a / 2.0, hs]])

    index = 0
    for k_cell in range(nz):
        for z_step in range(2):
            k_total = 2 * k_cell + z_step
            z_lattice = k_total * (h + hs)
            x_shift = k_total * a / 2.0
            for i in range(nx):
                x_base = i * a + x_shift
                for j in range(ny):
                    y_base = j * a
                    for b, offset in enumerate(basis):
                        x = x_base + offset[0]
                        y = y_base + offset[1]
                        z = z_lattice + offset[2]
                        centers[index] = [
                            wrap_to_box(x - Lx / 2.0 + 0.25 * a, Lx),
                            wrap_to_box(y - Ly / 2.0 + 0.25 * a, Ly),
                            wrap_to_box(z - Lz / 2.0 + 0.20 * a, Lz),
                        ]
                        basis_index[index] = b
                        index += 1
    return centers, basis_index, (Lx, Ly, Lz)


def find_rhombo_dz(a, lobe_offsets, rl, L=L_BRUSH):
    xy_shift = np.array([a / 2.0, a / (2.0 * np.sqrt(3.0))])
    min_xy_sq = min(
        np.dot(
            xy_shift + o2[:2] - o1[:2],
            xy_shift + o2[:2] - o1[:2],
        )
        for o1 in lobe_offsets
        for o2 in lobe_offsets
    )
    return np.sqrt(4.0 * (rl + L) ** 2 - min_xy_sq)


def rhombohedral_centers(ncols, nrows, nlayers, a, dz):
    if nrows % 2 != 0 or nlayers % 3 != 0:
        raise ValueError("Rhombo requires an even nrows and nlayers divisible by 3.")

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
                centers.append(
                    [wrap_to_box(x, Lx), wrap_to_box(y, Ly), wrap_to_box(z, Lz)]
                )
    return np.asarray(centers, dtype=np.float64), (Lx, Ly, Lz)


def crystal_geometry(crystal, sz, cr, rl):
    """Return centers, HOOMD core orientations, body offsets, and box lengths."""
    crystal = normalize_crystal_name(crystal)
    body_positions = TETRA * (cr * 2.0 * rl)

    if crystal == "diamond":
        a = find_diamond_mtab_spacing(sz, cr, rl)
        centers, layers, box_lengths = diamond_centers(*DIAMOND_DIMS, a)
        q_even = mat_to_quat_hoomd(RX)
        q_odd = mat_to_quat_hoomd(RZ90 @ RX)
        orientations = np.asarray(
            [q_even if layer % 2 == 0 else q_odd for layer in layers]
        )
    elif crystal == "mtab":
        a = find_diamond_mtab_spacing(sz, cr, rl)
        h = compute_mtab_h(cr, rl)
        hs = compute_mtab_staggered_h(sz, cr, rl)
        centers, basis_indices, box_lengths = mtab_centers(
            *MTAB_DIMS, a, h, hs
        )
        rotation_0 = RZ90 @ RX
        q_0 = mat_to_quat_hoomd(rotation_0)
        q_1 = mat_to_quat_hoomd(rotation_0 @ RX180)
        orientations = np.asarray(
            [q_1 if basis_index else q_0 for basis_index in basis_indices]
        )
    else:
        a = find_rhombo_spacing(sz, cr, rl)
        dz = find_rhombo_dz(a, body_positions, rl)
        centers, box_lengths = rhombohedral_centers(*RHOMBO_DIMS, a, dz)
        orientations = np.tile([1.0, 0.0, 0.0, 0.0], (len(centers), 1))

    Lx, Ly, Lz = box_lengths
    centers = centers + INITIAL_SHIFT
    centers[:, 0] = wrap_to_box(centers[:, 0], Lx)
    centers[:, 1] = wrap_to_box(centers[:, 1], Ly)
    centers[:, 2] = wrap_to_box(centers[:, 2], Lz)
    return centers, orientations, body_positions, box_lengths


def build_cores_snapshot(crystal, sz, cr, rl):
    centers, orientations, body_positions, box_lengths = crystal_geometry(
        crystal, sz, cr, rl
    )
    n_cores = len(centers)

    inertia_tensor = np.zeros((3, 3))
    for position in body_positions:
        inertia_tensor += (
            (position @ position) * np.eye(3) - np.outer(position, position)
        )
    moment_inertia = np.diag(inertia_tensor).copy()

    snapshot = gsd.hoomd.Frame()
    snapshot.configuration.box = [*box_lengths, 0.0, 0.0, 0.0]
    snapshot.particles.N = n_cores
    snapshot.particles.types = ["core", "tethered_lobe"]
    snapshot.particles.typeid = np.zeros(n_cores, dtype=np.uint32)
    snapshot.particles.position = centers.astype(np.float32)
    snapshot.particles.orientation = orientations.astype(np.float32)
    snapshot.particles.body = np.arange(n_cores, dtype=np.int32)
    snapshot.particles.mass = np.full(n_cores, 5.0, dtype=np.float32)
    snapshot.particles.moment_inertia = np.tile(
        moment_inertia, (n_cores, 1)
    ).astype(np.float32)
    snapshot.particles.diameter = np.full(
        n_cores, 2.0 * sz * rl, dtype=np.float32
    )
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
    scipy_quaternions = orientations[:, [1, 2, 3, 0]]
    rotations = Rotation.from_quat(scipy_quaternions)
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


class EnergyRecorder(hoomd.custom.Action):
    """Record both scaled force energies and the reconstructed g(s)."""

    FIELDNAMES = [
        "timestep",
        "scaled_pair_energy",
        "scaled_tether_energy",
        "U_real",
        "lambda_max_U_E",
        "g_total",
        "g_per_tlpp",
    ]

    def __init__(
        self,
        table,
        tether,
        s_value,
        n_cores,
        csv_path,
        append=False,
        checkpoint_timestep=None,
    ):
        self.table = table
        self.tether = tether
        self.s_value = float(s_value)
        self.n_cores = int(n_cores)
        self.csv_path = Path(csv_path)
        self.g_per_tlpp = []
        self.last_recorded_timestep = -1

        if append:
            if checkpoint_timestep is None:
                raise ValueError(
                    "checkpoint_timestep is required when appending energies."
                )
            self._prepare_existing_csv(int(checkpoint_timestep))
        else:
            with self.csv_path.open("w", newline="") as stream:
                csv.DictWriter(
                    stream,
                    fieldnames=self.FIELDNAMES,
                ).writeheader()

    def _prepare_existing_csv(self, checkpoint_timestep):
        """Keep valid rows through the GSD checkpoint and load their g values."""
        if not self.csv_path.is_file() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="") as stream:
                csv.DictWriter(
                    stream,
                    fieldnames=self.FIELDNAMES,
                ).writeheader()
            return

        with self.csv_path.open("r", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != self.FIELDNAMES:
                raise RuntimeError(
                    f"Unexpected energy CSV header in {self.csv_path}: "
                    f"{reader.fieldnames}"
                )

            kept_rows = []
            for row in reader:
                try:
                    timestep = int(float(row["timestep"]))
                    g_value = float(row["g_per_tlpp"])
                except (TypeError, ValueError, KeyError):
                    # Ignore a partially written final CSV row.
                    continue

                if timestep <= checkpoint_timestep:
                    kept_rows.append(row)
                    self.g_per_tlpp.append(g_value)
                    self.last_recorded_timestep = max(
                        self.last_recorded_timestep,
                        timestep,
                    )

        # Rewrite atomically so energy rows later than the usable GSD
        # checkpoint cannot be duplicated after continuation.
        temporary_path = self.csv_path.with_suffix(".csv.tmp")
        with temporary_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(kept_rows)
        os.replace(temporary_path, self.csv_path)

        print(
            f"ENERGY RESUME: kept {len(kept_rows)} rows through timestep "
            f"{checkpoint_timestep} in {self.csv_path}",
            flush=True,
        )

    def act(self, timestep):
        timestep = int(timestep)
        if timestep <= self.last_recorded_timestep:
            return

        scaled_pair = float(self.table.energy)
        scaled_tether = float(self.tether.energy)
        U_real = scaled_pair / self.s_value
        lambda_max_U_E = scaled_tether / (1.0 - self.s_value)
        g_total = U_real - lambda_max_U_E
        g_per_tlpp = g_total / self.n_cores
        self.g_per_tlpp.append(g_per_tlpp)
        self.last_recorded_timestep = timestep

        with self.csv_path.open("a", newline="") as stream:
            csv.DictWriter(
                stream,
                fieldnames=self.FIELDNAMES,
            ).writerow(
                {
                    "timestep": timestep,
                    "scaled_pair_energy": scaled_pair,
                    "scaled_tether_energy": scaled_tether,
                    "U_real": U_real,
                    "lambda_max_U_E": lambda_max_U_E,
                    "g_total": g_total,
                    "g_per_tlpp": g_per_tlpp,
                }
            )


# ===========================================================================
# HOOMD run
# ===========================================================================
def set_table_pair(table, pair, r_min, r_cut, U, F):
    table.params[pair] = dict(r_min=r_min, U=U, F=F)
    table.r_cut[pair] = r_cut


def s_label(s_value):
    return f"{float(s_value):.16f}"


def read_parameter_row(csv_path, row_index):
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        raise ValueError(f"No parameter rows found in {csv_path}.")
    if not 0 <= row_index < len(rows):
        raise IndexError(
            f"Requested row {row_index}, but {csv_path} has {len(rows)} data rows."
        )

    required = {"crystal", "sz", "cr", "rl", "s", "seed", "lambda_max"}
    missing = required.difference(rows[row_index])
    if missing:
        raise ValueError(
            f"Missing CSV columns: {', '.join(sorted(missing))}. "
            "Required header: crystal,sz,cr,rl,s,seed,lambda_max"
        )

    raw = rows[row_index]
    parameters = {
        "crystal": normalize_crystal_name(raw["crystal"]),
        "sz": float(raw["sz"]),
        "cr": float(raw["cr"]),
        "rl": float(raw["rl"]),
        "s": float(raw["s"]),
        "seed": int(float(raw["seed"])),
        "lambda_max": float(raw["lambda_max"]),
        "csv_row": int(row_index),
        "csv_path": str(csv_path.resolve()),
    }

    if not 0.0 < parameters["s"] < 1.0:
        raise ValueError("Every production s value must satisfy 0 < s < 1.")
    if parameters["sz"] <= 0.0 or parameters["cr"] <= 0.0:
        raise ValueError("sz and cr must be positive.")
    if parameters["rl"] <= 0.0:
        raise ValueError("rl must be positive.")
    if parameters["lambda_max"] <= 0.0:
        raise ValueError("lambda_max must be positive.")
    return parameters


def state_output_dir(parameters, n_cores):
    crystal = parameters["crystal"]
    sz = parameters["sz"]
    cr = parameters["cr"]
    rl = parameters["rl"]
    lambda_max = parameters["lambda_max"]
    return OUT_DIR / (
        f"{crystal}_sz_{sz:.2f}_cr_{cr:.1f}_rl_{rl:.1f}_"
        f"phi_{PHI:.3f}_sigmap_{SIGMAP:.3f}_lambda_{lambda_max}_N_{n_cores}_"
        f"{POTENTIAL_VERSION}"
    )


def find_earliest_attempt_trajectory(run_dir):
    """Read the checkpoint in the first-created attempt for one seed.

    Attempt directory names begin with an ISO-like timestamp, so sorting their
    names selects the attempt created first.  We deliberately resume only that
    attempt, keeping one production.gsd and one energies.csv for the run.

    Progress is read from ``last_frame.configuration.step``.  It must not be
    reconstructed from the frame count because equilibration and resumed jobs
    can change the relationship between frame index and simulation timestep.
    """
    attempt_dirs = sorted(
        path
        for path in Path(run_dir).glob("attempt_*")
        if path.is_dir()
    )

    if not attempt_dirs:
        return None

    attempt_dir = attempt_dirs[0]
    trajectory_path = attempt_dir / "production.gsd"
    summary_path = attempt_dir / "summary.json"

    if not trajectory_path.is_file():
        raise RuntimeError(
            f"The earliest attempt has no production.gsd: {attempt_dir}"
        )

    try:
        with gsd.hoomd.open(name=str(trajectory_path), mode="r") as trajectory:
            number_of_frames = len(trajectory)
            if number_of_frames == 0:
                raise RuntimeError(
                    f"The earliest attempt contains an empty production.gsd: "
                    f"{trajectory_path}"
                )

            last_frame_index = number_of_frames - 1
            last_timestep = int(
                trajectory[last_frame_index].configuration.step
            )
    except (OSError, EOFError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"Cannot read the earliest attempt checkpoint {trajectory_path}: "
            f"{type(error).__name__}: {error}"
        ) from error

    summary = None
    if summary_path.is_file():
        try:
            with summary_path.open("r") as stream:
                summary = json.load(stream)
        except (OSError, json.JSONDecodeError):
            summary = None

    summary_status = (
        summary.get("status", "unknown") if summary is not None else "missing"
    )

    return {
        "attempt_dir": attempt_dir,
        "summary_path": summary_path if summary_path.is_file() else None,
        "summary": summary,
        "summary_status": summary_status,
        "trajectory_path": trajectory_path,
        "number_of_frames": number_of_frames,
        "last_frame_index": last_frame_index,
        "last_timestep": last_timestep,
    }


def atomic_write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def block_statistics(values, maximum_blocks=10):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise RuntimeError("No production energy samples were recorded.")

    n_blocks = min(maximum_blocks, len(values))
    blocks = np.array_split(values, n_blocks)
    block_means = np.asarray([np.mean(block) for block in blocks])
    mean = float(np.mean(values))
    if len(block_means) > 1:
        sem = float(np.std(block_means, ddof=1) / np.sqrt(len(block_means)))
    else:
        sem = float("nan")
    return mean, sem, block_means.tolist()


def expected_n_cores(crystal):
    crystal = normalize_crystal_name(crystal)
    if crystal == "diamond":
        return 4 * int(np.prod(DIAMOND_DIMS))
    if crystal == "mtab":
        return 4 * int(np.prod(MTAB_DIMS))
    return int(np.prod(RHOMBO_DIMS))


def run_window(parameters):
    crystal = parameters["crystal"]
    sz = parameters["sz"]
    cr = parameters["cr"]
    rl = parameters["rl"]
    s_value = parameters["s"]
    DT, EQUIL_STEPS, TARGET_FINAL_TIMESTEP, WRITE_EVERY, SAMPLE_EVERY = (
        sampling_settings(s_value)
    )
    print(
        f"s={s_value:g}, DT={DT:g}, equil={EQUIL_STEPS}, "
        f"target={TARGET_FINAL_TIMESTEP}, "
        f"write every={WRITE_EVERY}, sample every={SAMPLE_EVERY}",
        flush=True,
    )
    seed = parameters["seed"]
    lambda_max = parameters["lambda_max"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state_dir = state_output_dir(parameters, expected_n_cores(crystal))
    run_dir = state_dir / f"s_{s_label(s_value)}" / f"seed_{int(seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    previous = find_earliest_attempt_trajectory(run_dir)
    if previous is not None:
        previous_timestep = previous["last_timestep"]
        previous_frame_count = previous["number_of_frames"]
        source_trajectory = previous["trajectory_path"]
        source_summary_status = previous["summary_status"]

        print(
            f"FOUND CHECKPOINT crystal={crystal} "
            f"s={s_label(s_value)} seed={seed}: "
            f"{source_trajectory} has {previous_frame_count} frames; "
            f"last frame index={previous['last_frame_index']}; "
            f"last timestep={previous_timestep}; "
            f"summary status={source_summary_status}",
            flush=True,
        )

        if previous_timestep >= TARGET_FINAL_TIMESTEP:
            print(
                f"SKIP: target timestep {TARGET_FINAL_TIMESTEP} has already "
                f"been reached.",
                flush=True,
            )
            if previous["summary"] is not None:
                return previous["summary"]
            return {
                "status": "checkpoint_at_target_without_summary",
                "trajectory": str(source_trajectory),
                "final_timestep": int(previous_timestep),
            }

        is_continuation = True
    else:
        previous_timestep = 0
        previous_frame_count = 0
        source_trajectory = None
        source_summary_status = None
        is_continuation = False

    if is_continuation:
        # Continue inside the same attempt directory. This keeps exactly one
        # production.gsd and one energies.csv for this run.
        attempt_dir = previous["attempt_dir"]
        trajectory_path = previous["trajectory_path"]
        energy_csv_path = attempt_dir / "energies.csv"
        summary_path = attempt_dir / "summary.json"

        print(
            f"APPENDING TO EXISTING ATTEMPT: {attempt_dir}",
            flush=True,
        )
    else:
        attempt_name = datetime.now().strftime("attempt_%Y%m%d_%H%M%S_%f")
        attempt_dir = run_dir / f"{attempt_name}_pid{os.getpid()}"
        attempt_dir.mkdir(parents=False, exist_ok=False)

        trajectory_path = attempt_dir / "production.gsd"
        energy_csv_path = attempt_dir / "energies.csv"
        summary_path = attempt_dir / "summary.json"

    snapshot, body_positions, centers, orientations, box_lengths = (
        build_cores_snapshot(crystal, sz, cr, rl)
    )

    device = hoomd.device.GPU() if USE_GPU else hoomd.device.CPU()
    simulation = hoomd.Simulation(device=device, seed=int(seed))
    if is_continuation:
        # create_state_from_snapshot(gsd_frame) resets the HOOMD timestep to
        # zero. Loading the file directly preserves the timestep stored in the
        # selected GSD frame.
        simulation.create_state_from_gsd(
            filename=str(source_trajectory),
            frame=-1,
        )
        restored_timestep = int(simulation.timestep)
        if restored_timestep != previous_timestep:
            raise RuntimeError(
                "Checkpoint timestep was not restored exactly: "
                f"GSD last timestep={previous_timestep}, "
                f"simulation timestep={restored_timestep}."
            )
    else:
        simulation.create_state_from_snapshot(snapshot)

    rigid = hoomd.md.constrain.Rigid()
    rigid.body["core"] = {
        "constituent_types": ["tethered_lobe"] * 4,
        "positions": body_positions.tolist(),
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * 4,
    }
    # A saved production frame already contains the rigid constituents. Calling
    # create_bodies() again would create four duplicate lobes per TLPP.
    if not is_continuation:
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

    core_radius = sz * rl
    lobe_radius = rl
    rcc_contact = 2.0 * core_radius
    rcl_contact = core_radius + lobe_radius
    rll_contact = 2.0 * lobe_radius

    # Use the same small absolute center-to-center lower limit for all pair
    # types. Avoid r=0, where the direction of the central force is undefined.
    rcc_min = TABLE_R_MIN
    rcl_min = TABLE_R_MIN
    rll_min = TABLE_R_MIN

    rcc_cut = rcc_contact + 2.0 * L_BRUSH + SIGMAP
    rcl_cut = rcl_contact + 2.0 * L_BRUSH + SIGMAP
    rll_cut = rll_contact + 2.0 * L_BRUSH + SIGMAP

    n_table_cc = max(
        2,
        int(round((rcc_cut - rcc_min) / TABLE_TARGET_BIN_WIDTH)),
    )
    n_table_cl = max(
        2,
        int(round((rcl_cut - rcl_min) / TABLE_TARGET_BIN_WIDTH)),
    )
    n_table_ll = max(
        2,
        int(round((rll_cut - rll_min) / TABLE_TARGET_BIN_WIDTH)),
    )

    U_cc_0, F_cc_0 = tabulate_P_implicit(
        core_radius,
        core_radius,
        SIGMAG,
        L_BRUSH,
        PHI,
        RDEP,
        rcc_min,
        rcc_cut,
        n_table_cc,
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
        n_table_cl,
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
        n_table_ll,
    )

    # A concise startup diagnostic. For every pair type, the first table force
    # and the force at contact must be finite and repulsive.
    for pair_name, U_0, F_0, r_min, r_contact, r_cut, n_table in (
        (
            "core-core", U_cc_0, F_cc_0,
            rcc_min, rcc_contact, rcc_cut, n_table_cc,
        ),
        (
            "core-lobe", U_cl_0, F_cl_0,
            rcl_min, rcl_contact, rcl_cut, n_table_cl,
        ),
        (
            "lobe-lobe", U_ll_0, F_ll_0,
            rll_min, rll_contact, rll_cut, n_table_ll,
        ),
    ):
        table_spacing = (r_cut - r_min) / n_table
        contact_index = int(round((r_contact - r_min) / table_spacing))
        contact_index = min(max(contact_index, 0), len(U_0) - 1)

        if F_0[0] <= 0.0 or F_0[contact_index] <= 0.0:
            raise RuntimeError(
                f"{pair_name} contact table is not repulsive: "
                f"F(first)={F_0[0]}, F(contact)={F_0[contact_index]}."
            )

        print(
            f"TABLE {pair_name}: n={n_table} "
            f"dr={table_spacing:.8g} "
            f"U(contact)={U_0[contact_index]:.8g} "
            f"F(contact)={F_0[contact_index]:.8g}"
        )

    # The pair force contributes s U_real to U_s.
    U_cc, F_cc = s_value * U_cc_0, s_value * F_cc_0
    U_cl, F_cl = s_value * U_cl_0, s_value * F_cl_0
    U_ll, F_ll = s_value * U_ll_0, s_value * F_ll_0

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

    k_active = (1.0 - s_value) * lambda_max
    tether = CoreFourLobeTether(
        k_active=k_active,
        reference=reference,
        tethered_lobe_typeid=1,
    )

    all_centers = hoomd.filter.Rigid(("center", "free"))
    if FIX_REFERENCE_TLPP:
        mobile_centers = hoomd.filter.SetDifference(
            all_centers, hoomd.filter.Tags([0])
        )
    else:
        mobile_centers = all_centers

    integrator = hoomd.md.Integrator(dt=DT, integrate_rotational_dof=True)
    integrator.rigid = rigid
    integrator.forces.extend([table, tether])
    integrator.methods.append(
        hoomd.md.methods.Langevin(filter=mobile_centers, kT=KT)
    )
    simulation.operations.integrator = integrator
    # Preserve the continued positions and orientations, but draw valid thermal
    # momenta because this GSD is a trajectory rather than an exact RNG checkpoint.
    simulation.state.thermalize_particle_momenta(filter=mobile_centers, kT=KT)

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=mobile_centers)
    simulation.operations.computes.append(thermo)

    simulation.run(0)
    print(
        f"RUN crystal={crystal} row={parameters['csv_row']} "
        f"s={s_label(s_value)} seed={seed} "
        f"k_active={k_active:.6f} attempt={attempt_dir.name}"
    )

    start = time.perf_counter()
    if not is_continuation:
        simulation.run(EQUIL_STEPS)

    current_timestep = int(simulation.timestep)
    remaining_steps = TARGET_FINAL_TIMESTEP - current_timestep
    if remaining_steps <= 0:
        raise RuntimeError(
            f"Current timestep {current_timestep} is already at or beyond "
            f"target {TARGET_FINAL_TIMESTEP}."
        )

    print(
        f"CONTINUATION={is_continuation} current timestep={current_timestep} "
        f"target timestep={TARGET_FINAL_TIMESTEP} "
        f"steps to run={remaining_steps}"
    )

    logger = hoomd.logging.Logger()
    logger.add(simulation, quantities=["timestep", "tps"])
    logger.add(thermo, quantities=["kinetic_temperature", "potential_energy"])
    logger.add(table, quantities=["energy"])
    logger.add(tether, quantities=["energy"])

    gsd_writer = hoomd.write.GSD(
        filename=str(trajectory_path),
        trigger=hoomd.trigger.Periodic(WRITE_EVERY),
        mode=("ab" if is_continuation else "wb"),
        logger=logger,
        dynamic=["property"],
    )
    gsd_writer.write_diameter = True
    simulation.operations.writers.append(gsd_writer)

    recorder = EnergyRecorder(
        table=table,
        tether=tether,
        s_value=s_value,
        n_cores=n_cores,
        csv_path=energy_csv_path,
        append=is_continuation,
        checkpoint_timestep=(current_timestep if is_continuation else None),
    )
    recorder_writer = hoomd.write.CustomWriter(
        trigger=hoomd.trigger.Periodic(SAMPLE_EVERY),
        action=recorder,
    )
    simulation.operations.writers.append(recorder_writer)

    simulation.run(remaining_steps)
    elapsed = time.perf_counter() - start

    g_mean, g_block_sem, block_means = block_statistics(recorder.g_per_tlpp)
    summary = {
        "status": "complete",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "crystal": crystal,
        "s": s_value,
        "seed": int(seed),
        "lambda_max": float(lambda_max),
        "k_active": float(k_active),
        "g_mean_per_tlpp": g_mean,
        "g_block_sem_per_tlpp": g_block_sem,
        "g_block_means_per_tlpp": block_means,
        "n_energy_samples": len(recorder.g_per_tlpp),
        "n_tlpps": n_cores,
        "box": [float(value) for value in box_lengths],
        "temperature": float(KT),
        "sz": float(sz),
        "cr": float(cr),
        "rl": float(rl),
        "phi": float(PHI),
        "sigmap": float(SIGMAP),
        "potential_version": POTENTIAL_VERSION,
        "contact_switch_gap": float(CONTACT_SWITCH_GAP),
        "contact_derivative_step": float(CONTACT_DERIVATIVE_STEP),
        "table_r_min": float(TABLE_R_MIN),
        "table_target_bin_width": float(TABLE_TARGET_BIN_WIDTH),
        "number_of_table_points_core_core": int(n_table_cc),
        "number_of_table_points_core_lobe": int(n_table_cl),
        "number_of_table_points_lobe_lobe": int(n_table_ll),
        "dt": float(DT),
        "equil_steps": int(EQUIL_STEPS),
        "production_steps": int(remaining_steps),
        "target_final_timestep": int(TARGET_FINAL_TIMESTEP),
        "starting_timestep": int(current_timestep),
        "final_timestep": int(simulation.timestep),
        "is_continuation": bool(is_continuation),
        "source_trajectory": (
            str(source_trajectory) if source_trajectory is not None else None
        ),
        "source_number_of_frames": int(previous_frame_count),
        "source_summary_status": source_summary_status,
        "sample_every": int(SAMPLE_EVERY),
        "write_every": int(WRITE_EVERY),
        "fixed_reference_tlpp": bool(FIX_REFERENCE_TLPP),
        "initial_shift": INITIAL_SHIFT.tolist(),
        "csv_row": int(parameters["csv_row"]),
        "parameter_csv": parameters["csv_path"],
        "elapsed_seconds": float(elapsed),
        "trajectory": trajectory_path.name,
        "energy_series": energy_csv_path.name,
    }
    atomic_write_json(summary_path, summary)

    print(
        f"DONE crystal={crystal} s={s_label(s_value)} seed={seed}: "
        f"g/N={g_mean:.8f} block_SEM={g_block_sem:.3g} -> {summary_path}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run one Frenkel-Ladd window selected from a CSV file."
    )
    parser.add_argument(
        "--csv",
        default="free_energy_parameters.csv",
        help="CSV with crystal,sz,cr,rl,s,seed,lambda_max columns.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Zero-based CSV data-row index. Defaults to SLURM_ARRAY_TASK_ID.",
    )
    args = parser.parse_args()

    row_index = args.row
    if row_index is None:
        slurm_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_index is None:
            parser.error("provide --row or run inside a Slurm job array")
        row_index = int(slurm_index)

    parameters = read_parameter_row(args.csv, row_index)
    run_window(parameters)


if __name__ == "__main__":
    main()
