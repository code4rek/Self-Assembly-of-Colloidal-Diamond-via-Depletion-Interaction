"""
stability ("does it melt?") test for the bulk MONOCLINIC TWO-ATOM-BASIS (mtab)
TLPP crystal, using a HARMONIC SPRING TETHER as an equilibration scaffold:

  stage 1 (0 .. SPRING_STEPS): a spring  F_i = -k (r_i - r0_i)  pins each rigid
          CORE to its ideal mtab site. The lattice skeleton can't drift, so the
          lobes librate (gamma_r > 0) and settle into the depletion well.
  stage 2 (SPRING_STEPS .. +FREE_STEPS): springs are REMOVED. The box is left
          UNCHANGED the whole run. If the now-bonded crystal stays crystalline
          it is intrinsically stable; if it only held via the springs, it melts.

Geometry: the two-atom-basis monoclinic builder (generate_positions + compute_h
/ compute_staggered_h). basis-0 and basis-1 carry opposite (point-inverted) lobe
sets, reachable as the proper rotation R0 @ Rx180, so both share one rigid body
definition and differ only by a per-core orientation quaternion.

Interaction : P_Implicit_Dep (brush U_B + depletion U_AO), all sphere pairs.
Ensemble    : Langevin on the rigid centers (translational + rotational).
No FIRE, no walls: the spring stage replaces both.
"""
import os, math, time
import numpy as np
from scipy.spatial.transform import Rotation
import hoomd
import gsd.hoomd

KT       = 1.0
PHI      = 0.05
SIGMAP   = 0.02
RDEP     = 0.5 * SIGMAP
L_BRUSH  = 0.01
SIGMAG   = 0.09 * (1000)**2
LOBE_R   = 0.5
OUT_DIR  = "/home/akash/hoomd-results/stability"

NX, NY, NZ   = 6, 6, 6        # nx*ny*nz*4 bodies -> 256 at 4,4,4
DT           = 1e-4
WRITE_EVERY  = 10_00
N_TABLE      = 1000
SEED         = 35
USE_GPU      = False
SZ0, CR0     = 1.21, 0.60

# ---- spring / two-stage controls ----
K_SPRING      = 100.0        # spring constant (kT / length^2) on the core centers
SPRING_STEPS  = 100_000      # tethered thermalization (bump toward 10^7 for a real hold)
FREE_STEPS    = 1_000_000    # free run after the springs are removed
BOX_PAD       = 8.0          # vacuum on EACH side. Box is BUILT at this size and kept
                             # CONSTANT throughout. Set 0.0 for a tight bulk-PBC box.
GAMMA_R       = 1.0          # rotational Langevin drag (per axis). REQUIRED so the lobes
                             # rotate into the well and STAY (spring tethers translation only).


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
# Geometry  (monoclinic two-atom-basis builder)
# ===========================================================================
thetax = np.arccos(1.0 / np.sqrt(3.0))
Rx = np.array([[1.0, 0.0, 0.0],
               [0.0, np.cos(thetax), -np.sin(thetax)],
               [0.0, np.sin(thetax),  np.cos(thetax)]])
Rz90 = np.array([[0.0, -1.0, 0.0],
                 [1.0,  0.0, 0.0],
                 [0.0,  0.0, 1.0]])
Rx180 = np.diag([1.0, -1.0, -1.0])
TETRA = np.array([[-1/2, -1/(2*np.sqrt(3)), -1/(2*np.sqrt(6))],
                  [ 1/2, -1/(2*np.sqrt(3)), -1/(2*np.sqrt(6))],
                  [ 0.0,  1/np.sqrt(3),     -1/(2*np.sqrt(6))],
                  [ 0.0,  0.0,               np.sqrt(3/8)]])


def find_nndistance(sz, cr, L=L_BRUSH, rl=LOBE_R):
    """In-plane nearest-neighbour spacing `a` (diamond / mtab form)."""
    a = (4.0/np.sqrt(6.0)) * (2.0*rl*sz + 2.0*L)
    if sz < cr/np.sqrt(6.0) + np.sqrt(1.0 - cr**2/3.0):
        a = (4.0/np.sqrt(6.0)) * (2.0*cr*rl/np.sqrt(6.0)
                                  + 2.0*rl*np.sqrt((1.0 + L/rl)**2 - cr**2/3.0))
    return a


def compute_h(cr, rl=LOBE_R, L=L_BRUSH):
    return rl * (cr/np.sqrt(2.0) + np.sqrt(1.0 - cr*cr/2.0)) + L


def compute_staggered_h(sz, cr, rl=LOBE_R, L=L_BRUSH):
    a_bcd = find_nndistance(sz, cr, L, rl=rl) / 2.0
    dnn   = np.sqrt(6.0) * find_nndistance(sz, cr, L, rl=rl) / 4.0
    return np.sqrt(dnn*dnn - a_bcd**2)


def wrap_to_box(x, Lb):
    return (x + Lb/2.0) % Lb - Lb/2.0


def mat_to_quat(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [w, x, y, z]


def generate_positions(nx, ny, nz, a, h, hs):
    """Monoclinic two-atom-basis core centres + per-core basis index (0/1)."""
    Lx = nx * a
    Ly = ny * a
    Lz = nz * 2.0 * (h + hs)
    N = nx * ny * nz * 4
    centers   = np.zeros((N, 3), dtype=np.float64)
    basis_idx = np.zeros((N,),  dtype=np.int32)
    basis_atoms = np.array([[0.0, 0.0, 0.0],
                            [0.0, a/2.0, hs]])
    idx = 0
    for k_cell in range(nz):
        for z_step in range(2):
            k_total   = k_cell * 2 + z_step
            z_lattice = k_total * (h + hs)
            x_shift   = k_total * (a / 2.0)
            for i in range(nx):
                x_base = i * a + x_shift
                for j in range(ny):
                    y_base = j * a
                    for b in range(2):
                        x = x_base + basis_atoms[b, 0]
                        y = y_base + basis_atoms[b, 1]
                        z = z_lattice + basis_atoms[b, 2]
                        centers[idx] = [wrap_to_box(x - Lx/2.0 + 0.25*a, Lx),
                                        wrap_to_box(y - Ly/2.0 + 0.25*a, Ly),
                                        wrap_to_box(z - Lz/2.0 + 0.20*a, Lz)]
                        basis_idx[idx] = b
                        idx += 1
    return centers, basis_idx, (Lx, Ly, Lz)


def build_cores_snapshot(sz, cr):
    a  = find_nndistance(sz, cr)
    h  = compute_h(cr)
    hs = compute_staggered_h(sz, cr)
    centers, basis_idx, (Lx, Ly, Lz) = generate_positions(NX, NY, NZ, a, h, hs)
    Nc = len(centers)
    lobe_scaling = cr * 2.0 * LOBE_R
    body_pos = TETRA * lobe_scaling

    I = np.zeros((3, 3))
    for p in body_pos:
        I += (p @ p)*np.eye(3) - np.outer(p, p)
    moi = np.diag(I).copy()

    R0     = Rz90 @ Rx
    q_even = mat_to_quat(R0)
    q_odd  = mat_to_quat(R0 @ Rx180)
    orient = np.array([q_odd if b else q_even for b in basis_idx])

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
    """F_i = -k (r_i - r0_i) on central particles only (body == tag)."""
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
            d -= L * np.rint(d / L)
            f = -self.k * d
            f[body != tag] = 0.0
            arr.force[:] = f


# ===========================================================================
# One run
# ===========================================================================
def run_stability(sz, cr):
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = (f"sz_{sz:.2f}_cr_{cr:.1f}_dpl_fr_{PHI:.3f}_dpl_sz_{SIGMAP:.2f}_kT_{KT:.1f}"
           f"_spring_k{K_SPRING:.0f}_hold{SPRING_STEPS}_pad{BOX_PAD:.1f}")
    traj = os.path.join(OUT_DIR, f"stabSPRING_mtab_{tag}.gsd")

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

    Ntot = sim.state.N_particles
    Nc = len(centers)
    ref = np.zeros((Ntot, 3), dtype=np.float64)
    ref[:Nc] = centers
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
    logger.add(table, quantities=["energy"])

    hold_traj  = os.path.join(OUT_DIR, f"stabSPRING_mtab_{tag}_HOLD.gsd")
    hold_every = max(1, SPRING_STEPS // 50)
    hold_writer = hoomd.write.GSD(filename=hold_traj,
                                  trigger=hoomd.trigger.Periodic(hold_every, phase=0),
                                  mode="wb", logger=logger, dynamic=["property"])
    hold_writer.write_diameter = True
    sim.operations.writers.append(hold_writer)

    gsd_writer = hoomd.write.GSD(filename=traj, trigger=hoomd.trigger.Periodic(WRITE_EVERY),
                                 mode="wb", logger=logger, dynamic=["property"])
    gsd_writer.write_diameter = True

    sim.run(0)
    print(f"[{tag}] cores={Nc}  N_total={Ntot}  k_spring={K_SPRING}  "
          f"box=({Lx+2*BOX_PAD:.2f},{Ly+2*BOX_PAD:.2f},{Lz+2*BOX_PAD:.2f}) CONSTANT")
    t0 = time.perf_counter()

    # ================= STAGE 1: tethered =================
    sim.run(SPRING_STEPS)
    print(f"[{tag}] stage-1 done at step {sim.timestep}  "
          f"T={thermo.kinetic_temperature:.3f}  PE/N={thermo.potential_energy/Nc:.3f}  "
          f"table.E={table.energy:.1f}")

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
    # sweep:
    # for sz in np.arange(1.15, 1.1601, 0.01):
    #     for cr in np.arange(0.5, 1.01, 0.1):
    #         run_stability(round(float(sz), 2), round(float(cr), 2))
    
    
    
    
    
    
    
    
    
    
    