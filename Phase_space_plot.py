import gsd.hoomd
import freud
import numpy as np
import matplotlib.pyplot as plt
import os

template = 100

sz_ratios = [1.15, 1.16, 1.17, 1.18, 1.19, 1.20, 1.21,1.22]
cr_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

dpl_fr = 0.05
dpl_sz = 0.02
gravity = 0.01

size = 10
# myfolder = "/home/akash/hpc_hoomd-files/hoomd-results_100/nx_05_ny_05_gravity_0.05_ta"
myfolder = f"/home/akash/hpc_hoomd-files/hoomd-results_{template}/nx_{size:02d}_ny_{size:02d}_gravity_{gravity}_no_ta/"

# Initialize Steinhardt objects
q4_compute = freud.order.Steinhardt(l=4, average=True)
q6_compute = freud.order.Steinhardt(l=6, average=True)
query_args = dict(num_neighbors=4, exclude_ii=True)

# Lists to store phase diagram data
plot_cr = []
plot_sz = []
plot_frac = []

print("Extracting last frames for phase diagram...")

# --- Main Processing Loop ---
for sz in sz_ratios:
    for cr in cr_ratios:
        filename = os.path.join(myfolder, f"crystallized_eao_size_{size:02d}_gravity_{gravity:.3f}_template_{template}_sz_{sz:1.2f}_cr_{cr}_lr_0.5_dpl_fr_{dpl_fr:1.3f}_dpl_sz_{dpl_sz:1.3f}.gsd")        
        if not os.path.exists(filename):
            print(f"File missing: {filename}")
            continue
            
        with gsd.hoomd.open(filename, 'r') as traj:
            if len(traj) == 0:
                continue
            frame = traj[99]
            # Grab ONLY the last frame
            # frame = traj[-1]
            
            # Identify core particles
            core_type_indices = [idx for idx, name in enumerate(frame.particles.types) if name in ['core', 'frozen_core']]
            is_core = np.isin(frame.particles.typeid, core_type_indices)
            core_positions = frame.particles.position[is_core].astype(np.float32)
            core_positions = core_positions[core_positions[:,2]<-50]
            
            box = frame.configuration.box
            
            # Compute Steinhardt
            q4_compute.compute(system=(box, core_positions), neighbors=query_args)
            q6_compute.compute(system=(box, core_positions), neighbors=query_args)
            
            q4_vals = q4_compute.particle_order
            q6_vals = q6_compute.particle_order
            
            # Calculate diamond fraction
            is_diamond = (q4_vals > 0.46) & (q4_vals < 0.52) & \
                         (q6_vals > 0.58) & (q6_vals < 0.63)
            
            fraction = np.sum(is_diamond)/len(is_diamond)
            
            # Store data for scattering
            plot_cr.append(cr)
            plot_sz.append(sz)
            plot_frac.append(fraction)

print("Data extraction complete. Plotting phase diagram...")

# Convert lists to NumPy arrays for easy boolean masking
plot_cr = np.array(plot_cr)
plot_sz = np.array(plot_sz)
plot_frac = np.array(plot_frac)

# Create masks for the threshold condition
mask_dia = plot_frac >= 0.001
mask_not_dia = plot_frac < 0.001

# Determine global min and max for consistent color scaling across both shapes
vmin = np.min(plot_frac) if len(plot_frac) > 0 else 0
vmax = np.max(plot_frac) if len(plot_frac) > 0 else 1

# --- Plotting Phase Diagram ---
plt.figure(figsize=(9, 7))

# 1. Plot Diamond marker for fraction >= 0.1
if np.any(mask_dia):
    sc_dia = plt.scatter(plot_sz[mask_dia], plot_cr[mask_dia], c=plot_frac[mask_dia], 
                         cmap='jet', marker='D', s=150, edgecolor='black', linewidths=0.5, 
                         zorder=2, vmin=vmin, vmax=vmax, label='diamond')

# 2. Plot Cross marker ('X') for fraction < 0.1
if np.any(mask_not_dia):
    sc_not = plt.scatter(plot_sz[mask_not_dia], plot_cr[mask_not_dia], c='gray', 
                         marker='s', s=150, edgecolor='black', linewidths=0.2, 
                         zorder=2, vmin=vmin, vmax=vmax, label='not diamond')

# --- Plotting the Kissing Curve ---
cr_curve = np.linspace(min(cr_ratios), max(cr_ratios), 200)
sz_curve = (cr_curve / np.sqrt(6)) + np.sqrt(1 - (cr_curve**2 / 3))
plt.plot(sz_curve, cr_curve, color='black', linestyle='--', linewidth=2.5, zorder=1)
# ----------------------------------

# Add colorbar using the diamond scatter (or the cross scatter if there are no diamonds)
active_sc = sc_dia if np.any(mask_dia) else sc_not
cbar = plt.colorbar(active_sc)
cbar.set_label('Fraction of Cubic Diamond', rotation=270, labelpad=20)

# Formatting 
plt.title("Phase Diagram: Cubic Diamond Yield", fontsize=20)
plt.xlabel("Size Ratio", fontsize=20)
plt.ylabel("Compression Ratio", fontsize=20)

plt.xticks(sz_ratios,fontsize=15)
plt.yticks(cr_ratios,fontsize=15)

# Add Legend to distinguish the shapes (using custom handlers so the color isn't tied to a specific point)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='D', color='w', label='diamond', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='X', color='w', label='not diamond', markerfacecolor='gray', markersize=10, markeredgecolor='black')
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=20)

plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# Save figure
save_path = os.path.join(myfolder, f"phase_diagram_gravity_{gravity:1.3f}_size_{size:02d}_diamond_{template}.png")
plt.savefig(save_path, dpi=300)
# plt.close()

print(f"Phase diagram saved successfully to: {save_path}")