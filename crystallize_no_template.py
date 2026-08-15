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
import argparse
from cores_without_template import create_initial_cluster_gsd

def P_Implicit_Dep(r1,r2,sigma_g,L,phi,rdep,rcc):

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

def find_nndistance(size_ratio, compression_ratio, L, rl=0.5):

    cr = compression_ratio
    sz = size_ratio
    a = (4/np.sqrt(6))*(2*rl*sz+ 2*L)

    if size_ratio < cr/np.sqrt(6)+np.sqrt(1-cr**2/3):
        a = (4/np.sqrt(6))*(2*cr*rl/np.sqrt(6) + 2*rl*np.sqrt(1-cr**2/3)+2*L)
    return a

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

#gravity_strength = 0.05
L = 0.01

ap = argparse.ArgumentParser()
ap.add_argument("--row", type=int, required=True)
args = ap.parse_args()

data = np.genfromtxt("eao_params_no_template.csv", delimiter=",", skip_header=1, ndmin = 2)
row = data[args.row]

sz, cr, rl, g, sigmap, phi = row

size_ratio = sz
compression_ratio = cr
gravity_strength = g

phi = phi
sigmap = sigmap
rdep = 0.5*sigmap
sigmag = 0.09*(1000)**2

lobe_radius = 0.5
core_radius = size_ratio * lobe_radius

create_initial_cluster_gsd(
    size_ratio=size_ratio,
    compression_ratio=compression_ratio,
    gravity_strength=0.05,
    lobe_radius=0.5,
    n_cols=5,
    n_rows=5,
    nz = 20,
    spacing=3.0,
    output_dir="/home/agg7742/hoomd-results"
)

init_file = f"/home/agg7742/hoomd-results/initial_cluster_cores_sz_{size_ratio:1.2f}_cr_{compression_ratio:1.1f}_no_template.gsd"

output_file = (
    "/scratch/agg7742/hoomd-results/crystallized_eao_size_05_gravity_"+
    f"{gravity_strength:1.3f}_no_template_" +
    f"{size_ratio:1.2f}_cr_" +
    f"{compression_ratio:1.1f}_lr_"+
    f"{lobe_radius:1.1f}_dpl_fr_"+
    f"{phi:1.3f}_dpl_sz_"+
    f"{sigmap:1.3f}.gsd"
)

dt = 1e-4

device = hoomd.device.GPU()
sim = hoomd.Simulation(device=device, seed=35)

if os.path.exists(output_file) and len(gsd.hoomd.open(output_file)) > 0:
    print(f"Resuming from {output_file}...")
    sim.create_state_from_gsd(filename=output_file, frame=-1)
    write_mode = 'ab'
    resume_status = True  # Flag to skip body creation
else:
    print("Starting fresh...")
    sim.create_state_from_gsd(filename=init_file)
    write_mode = 'wb'
    resume_status = False

bottom_normal = np.array([0.0, 0.0, 1.0])
top_normal = np.array([0.0, 0.0, -1.0])
# a = find_nndistance(size_ratio, compression_ratio, L, rl=0.5)
Lz = sim.state.box.Lz
Ly = sim.state.box.Ly
# z_shift = -Lz/2 + 3*Ly*np.sin(thetax)/4 + 2*lobe_radius
# y_shift = 0.25 * a
# template_particles = np.array([0.0, y_shift, z_shift])
# buffer_distance = 2*lobe_radius
bottom_origin = np.array([0.0,0.0,-Lz/2])
# Place the ceiling just slightly below the absolute top of the box.
# (If Lz is massive, this just acts as a safety lid).
top_origin = np.array([0.0, 0.0, Lz/2])

top_plane = hoomd.wall.Plane(
    origin=tuple(top_origin),
    normal=tuple(top_normal)
)


bottom_plane = hoomd.wall.Plane(
    origin=tuple(bottom_origin),
    normal=tuple(bottom_normal)
)

core_radius = size_ratio * lobe_radius
lobe_scaling = compression_ratio * 2*lobe_radius
positions = np.array([
    [-0.5,   -0.288675, -0.204124],
    [ 0.5,   -0.288675, -0.204124],
    [ 0.0,    0.57735,  -0.204124],
    [ 0.0,    0.0,      0.612372]
  ])*lobe_scaling

total_duration = 4e8

rigid = hoomd.md.constrain.Rigid()
types = sim.state.particle_types

rigid.body["core"] = {
    "constituent_types": ["lobe", "lobe", "lobe", "lobe"],
    "positions": positions.tolist(),
    "orientations": [(1.0, 0.0, 0.0, 0.0)]*4
}

if not resume_status:
    rigid.create_bodies(sim.state)
else:
    pass

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

# table.params[('frozen_core', 'lobe')] = dict(r_min = rcl_min, U=U_cl, F=F_cl)
# table.r_cut[('frozen_core', 'lobe')] = rcl

# table.params[('frozen_core', 'core')] = dict(r_min = rcc_min, U=U_cc, F=F_cc)
# table.r_cut[('frozen_core', 'core')] = rcc

# table.params[('frozen_lobe', 'lobe')] = dict(r_min = rll_min, U=U_ll, F=F_ll)
# table.r_cut[('frozen_lobe', 'lobe')] = rll

# table.params[('frozen_lobe', 'core')] = dict(r_min = rcl_min, U=U_cl, F=F_cl)
# table.r_cut[('frozen_lobe', 'core')] = rcl

# table.params[('frozen_core', 'frozen_core')] = dict(r_min = rcc_min, U=U_cc, F=F_cc)
# table.r_cut[('frozen_core', 'frozen_core')] = rcc

# table.params[('frozen_lobe', 'frozen_lobe')] = dict(r_min = rll_min, U=U_ll, F=F_ll)
# table.r_cut[('frozen_lobe', 'frozen_lobe')] = rll

# table.params[('frozen_core', 'frozen_lobe')] = dict(r_min = rcl_min, U=U_cl, F=F_cl)
# table.r_cut[('frozen_core', 'frozen_lobe')] = rcl

wca_wall = hoomd.md.external.wall.LJ(walls=[bottom_plane, top_plane])
sigma_lobe = 2 * lobe_radius
sigma_core = 2 * core_radius

wca_wall.params['lobe'] = {
    "epsilon": 1.0, 
    "sigma": sigma_lobe, 
    "r_cut": sigma_lobe * (2.0**(1.0/6.0))
}

wca_wall.params['core'] = {
    "epsilon": 1.0, 
    "sigma": sigma_core, 
    "r_cut": sigma_core * (2.0**(1.0/6.0))
}


# Turn off wall interactions for frozen particles to prevent numerical explosions
# wca_wall.params[['frozen_core', 'frozen_lobe']] = {
#     "epsilon": 0.0, 
#     "sigma": 1.0, 
#     "r_cut": 0.0
# }


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
integrator.forces.append(wca_wall)

logger = hoomd.logging.Logger()
logger.add(sim, quantities=["timestep", "tps"])
logger.add(thermo, quantities=['kinetic_energy', 'potential_energy'])


# Write the trajectory to a GSD file every 1000 time steps
gsd_writer = hoomd.write.GSD(
    filename = output_file,
    trigger=hoomd.trigger.Periodic(4000000),
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
























