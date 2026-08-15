import numpy as np
import gsd.hoomd

thetax = np.arccos(1/np.sqrt(3))
thetaz = np.radians(45.0)

Rx = np.array([
    [1.0, 0.0, 0.0],
    [0.0,  np.cos(thetax), -np.sin(thetax)],
    [0.0,  np.sin(thetax),  np.cos(thetax)]
], dtype=np.float64)

Rz = np.array([
    [np.cos(thetaz), -np.sin(thetaz), 0.0],
    [np.sin(thetaz),  np.cos(thetaz), 0.0],
    [0.0,             0.0,            1.0]
], dtype=np.float64)

Rz90 = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0]
], dtype=np.float64)

def find_nndistance(size_ratio, compression_ratio, L=0.01, rl=0.5):
    cr = compression_ratio
    sz = size_ratio
    # rl = lobe_radius

    a = (4 / np.sqrt(6)) * (2 * rl * sz + 2 * L)
    if sz < cr / np.sqrt(6) + np.sqrt(1 - cr**2 / 3):
        a = (4 / np.sqrt(6)) * (2 * cr * rl / np.sqrt(6) + 2 * rl * np.sqrt((1+L/rl)**2 - cr**2 / 3))

    return a

def compute_h(cr, rl=0.5):
    L = 0.01
    h = rl * (cr/np.sqrt(2) + np.sqrt(1 - cr*cr/2)) + L
    return h

def compute_staggered_h(sz, cr, rl=0.5):
    L = 0.01
    a_bcd = find_nndistance(sz, cr, L, rl=rl)/2
    dnn = np.sqrt(6) * find_nndistance(sz, cr, L, rl=rl) / 4
    hs = np.sqrt(dnn*dnn - a_bcd**2)
    return hs

def wrap_to_box(x, L):
    return (x + L/2) % L - L/2

def generate_positions(nx, ny, nz, a, h, hs):

    Lx = nx * a
    Ly = ny * a
    Lz = nz * 2 * (h + hs) 
    
    N = nx * ny * nz * 4
    centers = np.zeros((N, 3), dtype=np.float32)
    layer_k = np.zeros((N,), dtype=np.int32)
    

    basis_atoms = np.array([
        [0.0, 0.0, 0.0],
        [0.0, a/2, hs]
    ])
    
    idx = 0
    for k_cell in range(nz):
        for z_step in range(2): 
            k_total = k_cell * 2 + z_step
            
            z_lattice = k_total * (h + hs)
            x_shift_from_c = k_total * (a / 2.0)
            
            for i in range(nx):
                x_base = i * a + x_shift_from_c
                for j in range(ny):
                    y_base = j * a
                    
                    for b_idx in range(2):
                        x = x_base + basis_atoms[b_idx, 0]
                        y = y_base + basis_atoms[b_idx, 1]
                        z = z_lattice + basis_atoms[b_idx, 2]
                        
                        x_final = wrap_to_box(x - Lx/2 + 0.25*a, Lx)
                        y_final = wrap_to_box(y - Ly/2 + 0.25*a, Ly)
                        z_final = wrap_to_box(z - Lz/2 + 0.2*a, Lz)
                        
                        centers[idx, :] = [x_final, y_final, z_final]
                        
                        layer_k[idx] = k_total
                        idx += 1
                        
    return centers, layer_k, (Lx, Ly, Lz)

def add_tetra_lobes_to_centers(centers, layer_k, R_base, compression_ratio, lobe_radius, include_core=True, wrap_lobes=False, box_lengths=None):
    rl = lobe_radius
    cr = compression_ratio
    lobe_scaling = cr * 2.0 * rl
    
    # 1. Standard tetrahedron coordinates
    tetra = np.array([
        [-0.5,     -0.288675, -0.204124],
        [ 0.5,     -0.288675, -0.204124],
        [ 0.0,      0.57735,  -0.204124],
        [ 0.0,      0.0,       0.612372]
    ], dtype=np.float64) * lobe_scaling
    
    R_0 = Rz90@R_base
    
    Nc = centers.shape[0]
    n_per = 5 if include_core else 4
    N = Nc * n_per
    all_pos = np.zeros((N, 3), dtype=np.float32)
    typeid  = np.zeros((N,), dtype=np.uint32)
    body_ids = np.zeros((N,), dtype=np.int32)
    
    idx = 0
    for c_idx, (c, k) in enumerate(zip(centers.astype(np.float64), layer_k)):
        if include_core:
            all_pos[idx] = c
            typeid[idx] = 0
            body_ids[idx] = c_idx
            idx += 1
            
        basis_idx = c_idx % 2 
        lobes_rel = tetra @ R_0.T
        
        if basis_idx == 1:
            lobes_rel = -lobes_rel
            
        lobes_abs = c[None, :] + lobes_rel
        
        if wrap_lobes:
            Lx, Ly, Lz = box_lengths
            lobes_abs[:, 0] = (lobes_abs[:, 0] + Lx/2) % Lx - Lx/2
            lobes_abs[:, 1] = (lobes_abs[:, 1] + Ly/2) % Ly - Ly/2
            lobes_abs[:, 2] = (lobes_abs[:, 2] + Lz/2) % Lz - Lz/2
            
        all_pos[idx:idx+4] = lobes_abs.astype(np.float32)
        typeid[idx:idx+4] = 1
        body_ids[idx:idx+4] = c_idx 
        idx += 4
        
    return all_pos, typeid, body_ids

def write_cluster_gsd(filename, positions, typeid, body_ids, box_lengths, core_radius, lobe_radius):
    Lx, Ly, Lz = box_lengths
    N = positions.shape[0]
    
    fr = gsd.hoomd.Frame()
    fr.configuration.step = 0
    fr.configuration.box = [Lx, Ly, Lz, 0.0, 0.0, 0.0]
    
    fr.particles.N = N
    fr.particles.types = ["core", "lobe"]
    fr.particles.typeid = typeid.astype(np.uint32)
    fr.particles.position = positions.astype(np.float32)
    fr.particles.body = body_ids.astype(np.int32)
    
    diam = np.empty(N, dtype=np.float32)
    diam[typeid == 0] = 2.0 * core_radius
    diam[typeid == 1] = 2.0 * lobe_radius
    fr.particles.diameter = diam
    
    with gsd.hoomd.open(name=filename, mode="w") as f:
        f.append(fr)

nx, ny, nz = 8, 8, 8
L = 0.01
rl = 0.5

for sz in np.arange(1.15, 1.2201, 0.01):
    for cr in np.arange(0.5, 1.01, 0.1):
        
        out = f"/home/akash/hoomd-results/diamond_vs_mtab/mtab_sz_{sz:.2f}_cr_{cr:.1f}.gsd"
        
        a = find_nndistance(sz, cr, L=L, rl=rl)
        h = compute_h(cr=cr, rl=rl)

        hs = compute_staggered_h(sz, cr, rl=rl)

        centers, layer_k, box_lengths = generate_positions(nx, ny, nz, a, h, hs)
        
        cluster_pos, cluster_typeid, cluster_bodyid = add_tetra_lobes_to_centers(
            centers=centers,
            layer_k=layer_k,
            R_base=Rx,
            compression_ratio=cr,
            lobe_radius=rl,
            wrap_lobes=True,            
            box_lengths=box_lengths
        )
        
        write_cluster_gsd(
            filename=out,
            positions=cluster_pos,
            typeid=cluster_typeid,
            body_ids=cluster_bodyid,
            box_lengths=box_lengths,
            core_radius=sz * rl,
            lobe_radius=rl
        )
        
        print(f"Generated {out}")
        
        
        
        