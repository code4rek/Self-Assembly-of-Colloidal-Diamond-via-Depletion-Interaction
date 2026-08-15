"""
stability ("does it melt?") test for the bulk RHOMBOHEDRAL TLPP crystal, using a
HARMONIC SPRING TETHER as an equilibration scaffold (Frenkel-Ladd style):

  stage 1 (0 .. SPRING_STEPS): a spring  F_i = -k (r_i - r0_i)  pins each rigid
          CORE to its ideal rhombohedral site r0_i. The lattice skeleton can't
          drift or disperse, so the lobes are free to librate (gamma_r > 0) and
          settle into the depletion well while the cores stay put.
  stage 2 (SPRING_STEPS .. +FREE_STEPS): springs are REMOVED (the box is left
          UNCHANGED the whole run). If the now-bonded crystal stays crystalline
          it is intrinsically stable; if it only held because of the springs,
          it melts here.

The spring tethers only TRANSLATION of the cores -- it does nothing to the
orientations, so the lobes rotate freely into the well. That rotation still
needs gamma_r > 0 to damp/seat (a positional spring alone won't seat bonds).

Geometry: triangular layers in ABC stacking, in-plane spacing a=find_nndistance,
inter-layer spacing dz from the lobe-contact constraint. Every TLPP carries the
SAME (identity) orientation. NROWS even, NLAYERS a multiple of 3.

Interaction : P_Implicit_Dep (brush U_B + depletion U_AO), all sphere pairs.
Ensemble    : Langevin on the rigid centers (translational + rotational).
"""
import hoomd
import gsd.hoomd
import numpy as np
import os, math, time

KT       = 1.0
PHI      = 0.05
SIGMAP   = 0.02
RDEP     = 0.5 * SIGMAP
L_BRUSH  = 0.01
SIGMAG   = 0.09 * (1000)**2
LOBE_R   = 0.5
OUT_DIR  = "/home/akash/hoomd-results/stability"

NCOLS, NROWS, NLAYERS = 6, 6, 24   # NROWS even, NLAYERS multiple of 3 -> N = 324
DT           = 1e-4
WRITE_EVERY  = 1000
N_TABLE      = 1000
SEED         = 35
USE_GPU      = False
SZ0, CR0     = 1.21, 0.60

# ---- spring / two-stage controls ----
K_SPRING      = 100.0        # spring constant (kT / length^2) on the core centers.
                             # stiff enough to pin the lattice, soft enough to let it
                             # relax; thermal wobble ~ sqrt(kT/k) ~ 0.1 here. Tune.
SPRING_STEPS  = 100_000      # tethered thermalization (bump toward 10^7 for a real hold)
FREE_STEPS    = 1_000_000    # free run after the springs are removed
BOX_PAD       = 8.0          # vacuum on EACH side. Box BUILT at this size, kept CONSTANT.
                             # Set 0.0 for a tight bulk-PBC box (strict bulk-lattice test).
GAMMA_R       = 1.0          # rotational Langevin drag (per axis). REQUIRED so the lobes
                             # rotate into the well and STAY (spring does not touch rotation).


# ===========================================================================
# Depletion + brush potential
# ===========================================================================
def P_Implicit_Dep(r1, r2, sigma_g, L, phi, rdep, rcc):
    r1_eff = L + r1; r2_eff = L + r2; x = rcc - r1 - r2
    q1 = r1_eff / rdep; q2 = r2_eff / rdep; n = rcc / rdep
    r_Avg = 2 / (1 / r1 + 1 / r2)
    if ((x <= 2 * L and x > 0) and (rcc <= (r1_eff + r2_eff + 2 * rdep))):
        U_B = 16 * math.pi * L**2 * r_Avg * sigma_g**(3/2) / 35 * (
            28 * ((2 * L / x)**(1/4) - 1) + (20/11) * (1 - (x / (2 * L))**(11/4))
            + 12 * (x / (2 * L) - 1))
        U_AO = -phi/16 * (q1+q2+2-n)**2 * (n+2*(q1+q2+2) - 3/n*(q1**2+q2**2-2*q1*q2))
    elif ((x > 2 * L) and (rcc <= r1_eff + r2_eff + 2 * rdep)):
        U_B = 0
        U_AO = -phi/16 * (q1+q2+2-n)**2 * (n+2*(q1+q2+2) - 3/n*(q1**2+q2**2-2*q1*q2))
    else:
        U_AO = 0; U_B = 0
    return U_B + U_AO


def tabulate_P_implicit(r1, r2, sigma_g, L, phi, rdep, r_min, r_max, n_points=1000):
    r = np.linspace(r_min, r_max, n_points)
    U = np.array([P_Implicit_Dep(r1, r2, sigma_g, L, phi, rdep, ri) for ri in r])
    F = -np.gradient(U, r)
    return U, F


# ===========================================================================
# Geometry  (rhombohedral builder)
# ===========================================================================
TETRA = np.array([[-1/2, -1/(2*np.sqrt(3)), -1/(2*np.sqrt(6))],
                  [ 1/2, -1/(2*np.sqrt(3)), -1/(2*np.sqrt(6))],
                  [ 0.0,  1/np.sqrt(3),     -1/(2*np.sqrt(6))],
                  [ 0.0,  0.0,               np.sqrt(3/8)]])


def find_nndistance(sz, cr, L=L_BRUSH, rl=LOBE_R):
    """In-plane nearest-neighbour spacing `a` (rhombohedral / template form)."""
    a = (4.0/np.sqrt(6.0)) * (2.0*rl*sz + 2.0*L)
    if sz < cr/np.sqrt(6.0) + np.sqrt(1.0 - cr**2/3.0):
        a = (4.0/np.sqrt(6.0)) * (2.0*cr*rl/np.sqrt(6.0)
                                  + 2.0*rl*np.sqrt(1.0 - cr**2/3.0) + 2.0*L)
    return a


def find_dz(a, lobe_offsets, rl=LOBE_R, L=L_BRUSH):
    """Inter-layer spacing from the lobe-contact constraint (template form).
    The closest inter-layer lobe pair sits exactly at 3D distance 2*(rl+L)."""
    xy_shift = np.array([a/2.0, a/(2.0*np.sqrt(3.0))])
    min_xy_sq = min(
        np.dot(xy_shift + o2[:2] - o1[:2], xy_shift + o2[:2] - o1[:2])
        for o1 in lobe_offsets for o2 in lobe_offsets
    )
    return np.sqrt(4.0*(rl + L)**2 - min_xy_sq)


def wrap_to_box(x, Lb):
    return (x + Lb/2.0) % Lb - Lb/2.0


def rhombohedral_centers(ncols, nrows, nlayers, a, dz):
    """Periodic bulk rhombohedral core centres: triangular layers, ABC stacking."""
    assert nrows % 2 == 0,   "NROWS must be even for in-plane periodicity"
    assert nlayers % 3 == 0, "NLAYERS must be a multiple of 3 (ABC stacking)"
    Lx = ncols * a
    Ly = nrows * a * np.sqrt(3.0) / 2.0
    Lz = nlayers * dz
    basis_xy = [np.array([0.0,      0.0]),
                np.array([0.0,      a/np.sqrt(3.0)]),
                np.array([a/2.0,    a/(2.0*np.sqrt(3.0))])]
    C = []
    for nl in range(nlayers):
        sx, sy = basis_xy[nl % 3]
        z = -Lz/2.0 + nl*dz
        for i in range(nrows):
            row_off = a/2.0 if (i % 2) else 0.0
            y = -Ly/2.0 + (i + 0.5)*a*np.sqrt(3.0)/2.0 + sy
            for j in range(ncols):
                x = -Lx/2.0 + (j + 0.5)*a + row_off + sx
                C.append([wrap_to_box(x, Lx), wrap_to_box(y, Ly), wrap_to_box(z, Lz)])
    return np.asarray(C, dtype=float), (Lx, Ly, Lz)


def build_cores_snapshot(sz, cr):
    lobe_scaling = cr * 2.0 * LOBE_R
    body_pos = TETRA * lobe_scaling

    a  = find_nndistance(sz, cr)
    dz = find_dz(a, body_pos)
    centers, (Lx, Ly, Lz) = rhombohedral_centers(NCOLS, NROWS, NLAYERS, a, dz)
    Nc = len(centers)

    # isotropic MoI of the 4 lobes (unit mass each); regular tetrahedron
    I = np.zeros((3, 3))
    for p in body_pos:
        I += (p @ p)*np.eye(3) - np.outer(p, p)
    moi = np.diag(I).copy()

    orient = np.tile([1.0, 0.0, 0.0, 0.0], (Nc, 1))   # identity for all cores

    box = [Lx + 2*BOX_PAD, Ly + 2*BOX_PAD, Lz + 2*BOX_PAD, 0.0, 0.0, 0.0]  # padded, CONSTANT

    snap = gsd.hoomd.Frame()
    snap.configuration.box = box
    snap.particles.N = Nc
    snap.particles.types = ["core", "lobe"]
    snap.particles.typeid = np.zeros(Nc, dtype=np.uint32)
    snap.particles.position = centers.astype(np.float32)
    snap.particles.orientation = orient.astype(np.float32)
    snap.particles.body = np.arange(Nc, dtype=np.int32)
    snap.particles.mass = np.full(Nc, 5.0, dtype=np.float32)
    snap.particles.moment_inertia = np.tile(moi, (Nc, 1)).astype(np.float32)
    snap.particles.diameter = np.full(Nc, 2.0*sz*LOBE_R, dtype=np.float32)
    return snap, body_pos.tolist(), centers.astype(np.float64), (Lx, Ly, Lz)


# ===========================================================================
# Harmonic spring tether on the rigid CENTERS  (custom force)
# ===========================================================================
class HarmonicTether(hoomd.md.force.Custom):
    """F_i = -k (r_i - r0_i) on central particles only (body == tag).
    `reference` is an (N_total, 3) array indexed by particle TAG."""
    def __init__(self, k, reference):
        super().__init__(aniso=False)
        self.k = float(k)
        self.reference = np.asarray(reference, dtype=np.float64)

    def set_forces(self, timestep):
        with self._state.cpu_local_snapshot as snap, \
             self.cpu_local_force_arrays as arr:
            tag  = snap.particles.tag
            pos  = np.array(snap.particles.position, dtype=np.float64)
            body = snap.particles.body
            d = pos - self.reference[tag]
            box = self._state.box
            L = np.array([box.Lx, box.Ly, box.Lz])
            d -= L * np.rint(d / L)                 # orthorhombic min image (guard)
            f = -self.k * d
            f[body != tag] = 0.0                    # tether only the rigid centers
            arr.force[:] = f


# ===========================================================================
# One run
# ===========================================================================
def run_stability(sz, cr):
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = (f"sz_{sz:.2f}_cr_{cr:.1f}_dpl_fr_{PHI:.3f}_dpl_sz_{SIGMAP:.2f}_kT_{KT:.1f}"
           f"_spring_k{K_SPRING:.0f}_hold{SPRING_STEPS}_pad{BOX_PAD:.1f}")
    traj = os.path.join(OUT_DIR, f"stabSPRING_rhombo_{tag}.gsd")

    snap, body_pos, centers, (Lx, Ly, Lz) = build_cores_snapshot(sz, cr)

    dev = hoomd.device.GPU() if USE_GPU else hoomd.device.CPU()
    sim = hoomd.Simulation(device=dev, seed=SEED)
    sim.create_state_from_snapshot(snap)

    rigid = hoomd.md.constrain.Rigid()
    rigid.body["core"] = {
        "constituent_types": ["lobe", "lobe", "lobe", "lobe"],
        "positions": body_pos,
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * 4,
    }
    rigid.create_bodies(sim.state)

    R_c = sz * LOBE_R
    rl  = LOBE_R
    rcc_min, rcc = 2*R_c,        2*R_c + 2*L_BRUSH + SIGMAP
    rcl_min, rcl = R_c + rl,     R_c + rl + 2*L_BRUSH + SIGMAP
    rll_min, rll = 2*rl,         2*rl + 2*L_BRUSH + SIGMAP
    U_cc, F_cc = tabulate_P_implicit(R_c, R_c, SIGMAG, L_BRUSH, PHI, RDEP, rcc_min, rcc, N_TABLE)
    U_cl, F_cl = tabulate_P_implicit(R_c, rl,  SIGMAG, L_BRUSH, PHI, RDEP, rcl_min, rcl, N_TABLE)
    U_ll, F_ll = tabulate_P_implicit(rl,  rl,  SIGMAG, L_BRUSH, PHI, RDEP, rll_min, rll, N_TABLE)

    cell = hoomd.md.nlist.Cell(buffer=0.0, exclusions=["body"])
    table = hoomd.md.pair.Table(nlist=cell, default_r_cut=0.0)
    table.params[('core', 'core')] = dict(r_min=rcc_min, U=U_cc, F=F_cc); table.r_cut[('core', 'core')] = rcc
    table.params[('core', 'lobe')] = dict(r_min=rcl_min, U=U_cl, F=F_cl); table.r_cut[('core', 'lobe')] = rcl
    table.params[('lobe', 'lobe')] = dict(r_min=rll_min, U=U_ll, F=F_ll); table.r_cut[('lobe', 'lobe')] = rll

    centers_free = hoomd.filter.Rigid(("center", "free"))

    integrator = hoomd.md.Integrator(dt=DT, integrate_rotational_dof=True)
    integrator.rigid = rigid
    integrator.forces.append(table)

    # ---- spring reference: ideal core sites, indexed by tag (cores are tags 0..Nc-1) ----
    Ntot = sim.state.N_particles
    Nc = len(centers)
    ref = np.zeros((Ntot, 3), dtype=np.float64)
    ref[:Nc] = centers                       # lobes (tags >= Nc) are masked out in the force
    tether = HarmonicTether(k=K_SPRING, reference=ref)
    integrator.forces.append(tether)

    sim.operations.integrator = integrator

    langevin = hoomd.md.methods.Langevin(filter=centers_free, kT=KT)
    langevin.gamma_r['core'] = (GAMMA_R, GAMMA_R, GAMMA_R)
    integrator.methods.append(langevin)
    sim.state.thermalize_particle_momenta(filter=centers_free, kT=KT)

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=centers_free)
    sim.operations.computes.append(thermo)

    logger = hoomd.logging.Logger()
    logger.add(sim, quantities=["timestep", "tps"])
    logger.add(thermo, quantities=["kinetic_temperature", "potential_energy",
                                   "kinetic_energy", "translational_kinetic_energy",
                                   "rotational_kinetic_energy"])
    logger.add(table, quantities=["energy"])   # watch this drop + plateau as lobes seat

    # ---- STAGE-1 writer: dense, captures the tethered thermalization ----
    hold_traj  = os.path.join(OUT_DIR, f"stabSPRING_rhombo_{tag}_HOLD.gsd")
    hold_every = max(1, SPRING_STEPS // 50)
    hold_writer = hoomd.write.GSD(filename=hold_traj,
                                  trigger=hoomd.trigger.Periodic(hold_every, phase=0),
                                  mode="wb", logger=logger, dynamic=["property"])
    hold_writer.write_diameter = True
    sim.operations.writers.append(hold_writer)

    # ---- STAGE-2 writer: coarse, the free run ----
    gsd_writer = hoomd.write.GSD(filename=traj, trigger=hoomd.trigger.Periodic(WRITE_EVERY),
                                 mode="wb", logger=logger, dynamic=["property"])
    gsd_writer.write_diameter = True

    sim.run(0)
    print(f"[{tag}] cores={Nc}  N_total={Ntot}  k_spring={K_SPRING}  "
          f"box=({Lx+2*BOX_PAD:.2f},{Ly+2*BOX_PAD:.2f},{Lz+2*BOX_PAD:.2f}) CONSTANT")
    t0 = time.perf_counter()

    # ================= STAGE 1: tethered (densely written) =================
    sim.run(SPRING_STEPS)
    print(f"[{tag}] stage-1 done at step {sim.timestep}  "
          f"T={thermo.kinetic_temperature:.3f}  PE/N={thermo.potential_energy/Nc:.3f}  "
          f"table.E={table.energy:.1f}")

    # hand off the writer, remove springs (box stays the same)
    sim.operations.writers.remove(hold_writer)
    sim.operations.writers.append(gsd_writer)

    integrator.forces.remove(tether)          # RELEASE the springs -- box is left UNCHANGED
    print(f"[{tag}] >>> released springs at step {sim.timestep} (box unchanged)")

    # ================= STAGE 2: free =================
    sim.run(FREE_STEPS)
    print(f"[{tag}] done in {time.perf_counter()-t0:.1f}s  T={thermo.kinetic_temperature:.3f}  "
          f"PE/N={thermo.potential_energy/Nc:.3f}  table.E={table.energy:.1f}  -> {traj}")


if __name__ == "__main__":
    run_stability(SZ0, CR0)