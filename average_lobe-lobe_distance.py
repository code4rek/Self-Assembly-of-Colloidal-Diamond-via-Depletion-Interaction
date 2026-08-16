import freud
import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt

def avg_bottom_layer_lobe_distance(gsd_path, frame=-1, cutoff=1.03):
    with gsd.hoomd.open(gsd_path, "r") as traj:
        snap = traj[frame] 

    box_params = snap.configuration.box
    Lz = box_params[2]
    z_bottom = -Lz / 2.0
    
    z_min = z_bottom
    z_max = z_bottom +4
    
    box = freud.box.Box.from_box(box_params)
    pos = snap.particles.position
    typeid = snap.particles.typeid
    body = snap.particles.body
    types = snap.particles.types

    core_tids = [i for i, t in enumerate(types) if "core" in t.lower()]
    core_mask = (np.isin(typeid, core_tids)) & (body >= 0) & (pos[:, 2] >= z_min) & (pos[:, 2] <= z_max)
    
    valid_bodies = np.unique(body[core_mask])
    
    if len(valid_bodies) == 0:
        print(f"No clusters found in height range [{z_min}, {z_max}]")
        return np.nan

    lobe_tids = [i for i, t in enumerate(types) if "lobe" in t.lower()]
    mask = np.isin(typeid, lobe_tids) & np.isin(body, valid_bodies)
    
    lobe_pos = pos[mask]
    lobe_body = body[mask]

    if len(lobe_pos) == 0:
        return np.nan

    aq = freud.locality.AABBQuery(box, lobe_pos)
    query_result = aq.query(lobe_pos, dict(r_max=cutoff, exclude_ii=True))
    
    distances = []
    for i, j, dist in query_result:
        if lobe_body[i] != lobe_body[j]:
            distances.append(dist)

    if not distances:
        return np.nan
    
    return np.mean(distances)

sz_values = [1.15, 1.16, 1.17, 1.18, 1.19, 1.20, 1.21, 1.22]
cr_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
base_dir  = "/home/akash/hoomd-results/diamond_vs_mtab"

# rows = cr (y), cols = sz (x)
data = np.full((len(cr_values), len(sz_values)), np.nan)
for j, sz in enumerate(sz_values):          # sz -> column (x)
    for i, cr in enumerate(cr_values):      # cr -> row (y)
        gsd_path = f"{base_dir}/mtab_sz_{sz:.2f}_cr_{cr:.1f}.gsd"
        try:
            data[i, j] = avg_bottom_layer_lobe_distance(gsd_path, cutoff=1.04)
        except Exception:
            pass

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(data, origin='lower', aspect='auto', cmap='viridis')
ax.set_xticks(range(len(sz_values))); ax.set_xticklabels(sz_values)
ax.set_yticks(range(len(cr_values))); ax.set_yticklabels(cr_values)
ax.set_xlabel('Size ratio (sz)')
ax.set_ylabel('Compression ratio (cr)')
for i in range(len(cr_values)):
    for j in range(len(sz_values)):
        if not np.isnan(data[i, j]):
            ax.text(j, i, f"{data[i,j]:.3f}", ha='center', va='center',
                    fontsize=8, color='white')
fig.colorbar(im, ax=ax, label='Avg lobe-lobe contact')
plt.tight_layout()
plt.savefig(f"{base_dir}/phase_diagram.png", dpi=150)
plt.show()









