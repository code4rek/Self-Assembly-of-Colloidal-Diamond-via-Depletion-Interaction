# HOOMD cluster assembly simulation code
# Author: Akash Ghosh (NYU, Pine Group)
# For internal research, use within group.
# Email Id: agg7742@nyu.edu

import os
import math
import time
import hoomd
import gsd.hoomd
import numpy as np
from cores import create_initial_cluster_gsd

def P_Implicit_Dep(r1,r2,sigma_g,L,phi,rdep,rcc):

    # sigmaA is the particle size
    # sigmaA2 is another particle size
    # sigma_g is the brush density
    # L is the brush length
    # rcc is the centre to centre distance

    r1_eff = L + r1
    r2_eff = L + r2
    x = rcc - r1 - r2

    q1 = r1_eff/rdep
    q2 = r2_eff/rdep
    n = rcc/rdep
    r_Avg = 2/(1/r1+1/r2)
    if ((x <= 2*L and x >0) and (rcc <=(r1_eff + r2_eff + 2*rdep))):
        U_B = 16*math.pi*L**2*r_Avg*sigma_g**(3/2)/35*(28*((2*L/x)**(1/4)-1)+(20/11)*(1-(x/(2*L))**(11/4))+12*(x/(2*L)-1))
        U_AO = -phi/16*(q1+q2+2-n)**2*(n+2*(q1+q2+2)-3/n*(q1**2+q2**2-2*q1*q2))
    elif (( x > 2*L) and (rcc <= r1_eff + r2_eff + 2*rdep)):
        U_B = 0
        U_AO = -phi/16*(q1+q2+2-n)**2*(n+2*(q1+q2+2)-3/n*(q1**2+q2**2-2*q1*q2))
    else:
        U_AO = 0
        U_B = 0
    U = U_B + U_AO
    return U

def F_Implicit_Dep(r1,r2,sigma_g,L,phi,rdep,rcc):

    r1_eff = L + r1
    r2_eff = L + r2
    x = rcc - r1 - r2

    rmax = r1 + r2 + 2*(L+rdep)

    q1 = r1_eff/rdep
    q2 = r2_eff/rdep
    n = rcc/rdep
    r_Avg = 2/(1/r1+1/r2)
    if ((x <= 2*L and x >0) and (rcc <=(r1_eff + r2_eff + 2*rdep))):
        f = n + 2*(q1+q2+2) - 3/n*(q1**2+q2**2-2*q1*q2)
        g = (q1 + q2 + 2 - n)**2
        fprime = 1 + 3/(n**2)*(q1**2 + q2**2 - 2*q1*q2)
        gprime = -2*(q1+q2+2-n)
        FAO = phi/(16*rdep)*(gprime*f + fprime*g)
        FB = 16*math.pi*r_Avg*L**2*sigma_g**(3/2)/35*((7*(2*L)**(1/4))/(x**(5/4))+5*(x)**(7/4)/(2*L)**(11/4)-6/L)
    elif (( x> 2*L) and (rcc <= r1_eff + r2_eff + 2*rdep)):
        f = n + 2*(q1+q2+2) - 3/n*(q1**2+q2**2-2*q1*q2)
        g = (q1 + q2 + 2 - n)**2
        fprime = 1 + 3/(n**2)*(q1**2 + q2**2 - 2*q1*q2)
        gprime = -2*(q1+q2+2-n)
        FAO = phi/(16*rdep)*(gprime*f + fprime*g)
        FB = 0
    else:
        FB = 0
        FAO = 0
    F = FB + FAO
    return rmax, F

def tabulate_P_implicit(r1, r2, sigma_g, L, phi, rdep,
                        r_min, r_max, n_points=1000):
    r = np.linspace(r_min, r_max, n_points)
    U = np.array([P_Implicit_Dep(r1, r2, sigma_g, L, phi, rdep, ri) for ri in r])
    F = -np.gradient(U, r)
    return U, F

thetax = np.arccos(1/np.sqrt(3))
thetaz = np.radians(45)
Rx = np.array([
    [1.0, 0.0, 0.0],
    [0.0,  np.cos(thetax), -np.sin(thetax)],
    [0.0,  np.sin(thetax), np.cos(thetax)]
])

Rz = np.array([
    [np.cos(thetaz), -np.sin(thetaz), 0.0],
    [np.sin(thetaz),  np.cos(thetaz), 0.0],
    [0.0,             0.0,            1.0]
])

# for phi in np.arange(0.045,0.055,0.01):
gravity_strength = 0.05
L = 0.01


size_ratio = 1.19
compression_ratio = 0.6
phi = 0.046
sigmap = 0.02

rdep = 0.5*sigmap
sigmag = 0.09*(1000)**2

lobe_radius = 0.5
core_radius = size_ratio * lobe_radius

template = 111

init_file = create_initial_cluster_gsd(
    size_ratio,
    compression_ratio,
    gravity_strength,
    lobe_radius,
    n_cols=5,
    n_rows=5,
    spacing=3.0,
    template=template,
    output_dir="/home/akash/hoomd-results")

output_file = (
    "/home/akash/hoomd-results/crystallized_eao_gravity_"+
    f"{gravity_strength:1.3f}_template_" +
    f"{template}_sz_" +
    f"{size_ratio}_cr_" +
    f"{compression_ratio}_lr_"+
    f"{lobe_radius}_dpl_fr"+
    f"{phi}_dpl_sz"+
    f"{sigmap}.gsd"
)

dt = 1e-4

device = hoomd.device.GPU()
sim = hoomd.Simulation(device=device, seed=35)

if os.path.exists(output_file) and len(gsd.hoomd.open(output_file)) > 0:
    print(f"Resuming from {output_file}...")
    sim.create_state_from_gsd(filename=output_file, frame=-1)
    write_mode = 'ab'
    resume_status = True
else:
    print("Starting fresh...")
    sim.create_state_from_gsd(filename=init_file)
    write_mode = 'wb'
    resume_status = False

print(f"\nThe Simulation will be saved in output file: {output_file}")

core_radius = size_ratio * lobe_radius
lobe_scaling = compression_ratio * 2*lobe_radius
positions = np.array([
    [-1/2,          -1/(2*np.sqrt(3)),   -1/(2*np.sqrt(6))],
    [ 1/2,          -1/(2*np.sqrt(3)),   -1/(2*np.sqrt(6))],
    [ 0,             1/np.sqrt(3),       -1/(2*np.sqrt(6))],
    [ 0,             0,                   np.sqrt(3/8)    ]
]) * lobe_scaling

final_positions = positions @Rx.T  if template == 100 else positions

total_duration = 4e8

rigid = hoomd.md.constrain.Rigid()
types = sim.state.particle_types

rigid.body["core"] = {
    "constituent_types": ["lobe", "lobe", "lobe", "lobe"],
    "positions": final_positions.tolist(),
    "orientations": [(1.0, 0.0, 0.0, 0.0)]*4
}

rigid.body["frozen_core"] = {
    "constituent_types": ["frozen_lobe", "frozen_lobe", "frozen_lobe", "frozen_lobe"],
    "positions": final_positions.tolist(),
    "orientations": [(1.0, 0.0, 0.0, 0.0)]*4
}


if not resume_status:
    rigid.create_bodies(sim.state)
else:
    pass

# rigid.create_bodies(sim.state)
integrator = hoomd.md.Integrator(dt = dt, integrate_rotational_dof = True)

sim.operations.integrator = integrator
integrator.rigid = rigid

rigid_centers_and_free = hoomd.filter.Rigid(("center","free"))
mobile_cluster_centers = hoomd.filter.Type(["core"])

N = (sim.state.N_particles)/5

rcc_min = 2*core_radius
rcc = 2*core_radius+2*L+sigmap
U_cc, F_cc = tabulate_P_implicit(core_radius, core_radius, sigmag, L, phi, rdep,rcc_min, rcc, n_points=1000)

rcl_min = core_radius+lobe_radius
rcl = core_radius+lobe_radius+2*L+sigmap
U_cl, F_cl = tabulate_P_implicit(core_radius, lobe_radius, sigmag, L, phi, rdep,rcl_min, rcl, n_points=1000)

rll_min =  2*lobe_radius
rll = 2*lobe_radius+2*L+sigmap
U_ll, F_ll = tabulate_P_implicit(lobe_radius, lobe_radius, sigmag, L, phi, rdep,rll_min, rll, n_points=1000)

cell = hoomd.md.nlist.Cell(buffer=0, exclusions=["body"])

table = hoomd.md.pair.Table(nlist=cell, default_r_cut=None)
table.params[('core', 'core')] = dict(r_min = rcc_min, U=U_cc, F=F_cc)
table.r_cut[('core', 'core')] = rcc

table.params[('core', 'lobe')] = dict(r_min = rcl_min, U = U_cl, F=F_cl)
table.r_cut[('core', 'lobe')] = rcl

table.params[('lobe', 'lobe')] = dict(r_min = rll_min, U=U_ll, F=F_ll)
table.r_cut[('lobe', 'lobe')] = rll

table.params[('frozen_core', 'lobe')] = dict(r_min = rcl_min, U=U_cl, F=F_cl)
table.r_cut[('frozen_core', 'lobe')] = rcl

table.params[('frozen_core', 'core')] = dict(r_min = rcc_min, U=U_cc, F=F_cc)
table.r_cut[('frozen_core', 'core')] = rcc

table.params[('frozen_lobe', 'lobe')] = dict(r_min = rll_min, U=U_ll, F=F_ll)
table.r_cut[('frozen_lobe', 'lobe')] = rll

table.params[('frozen_lobe', 'core')] = dict(r_min = rcl_min, U=U_cl, F=F_cl)
table.r_cut[('frozen_lobe', 'core')] = rcl

table.params[('frozen_core', 'frozen_core')] = dict(r_min = rcc_min, U=U_cc, F=F_cc)
table.r_cut[('frozen_core', 'frozen_core')] = rcc

table.params[('frozen_lobe', 'frozen_lobe')] = dict(r_min = rll_min, U=U_ll, F=F_ll)
table.r_cut[('frozen_lobe', 'frozen_lobe')] = rll

table.params[('frozen_core', 'frozen_lobe')] = dict(r_min = rcl_min, U=U_cl, F=F_cl)
table.r_cut[('frozen_core', 'frozen_lobe')] = rcl

gravity = hoomd.md.force.Constant(filter = hoomd.filter.All())
gravity.constant_force["core"] = (0, 0, -gravity_strength)

langevin = hoomd.md.methods.Langevin(filter = mobile_cluster_centers, kT=1.0)
langevin.gamma["core"] = 1.0
langevin.gamma_r["core"] = [1.0, 1.0, 1.0]

thermo = hoomd.md.compute.ThermodynamicQuantities(filter=rigid_centers_and_free)
sim.operations.computes.append(thermo)

integrator.methods.append(langevin)
integrator.forces.append(gravity)
integrator.forces.append(table)

logger = hoomd.logging.Logger()
logger.add(sim, quantities=["timestep", "tps"])
logger.add(thermo, quantities=['kinetic_energy', 'potential_energy'])


# Write the trajectory to a GSD file every 1000 time steps
gsd_writer = hoomd.write.GSD(
    filename = output_file,
    trigger=hoomd.trigger.Periodic(100000),
    mode=write_mode,
    logger =logger,
    dynamic=['property']
)
gsd_writer.write_diameter = True
sim.operations.writers.append(gsd_writer)

# fire = hoomd.md.minimize.FIRE(dt=1e-5, 
#                               force_tol=1e-2, 
#                               angmom_tol=1e-2, 
#                               energy_tol=1e-5)
# fire.methods.append(langevin)
# fire.forces = [table, gravity]
# sim.operations.integrator = fire

# while not fire.converged:
#    sim.run(100)


start_time = time.perf_counter()
# Run the simulation
sim.run(total_duration)
end_time = time.perf_counter()

print(f"Simulation time: {end_time - start_time:.2f} seconds")
























