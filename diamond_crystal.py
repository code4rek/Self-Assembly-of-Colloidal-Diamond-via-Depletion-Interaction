import numpy as np
import gsd.hoomd

# thetax = np.radians(54.725)

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

Rx45 = np.array([
    [1.0, 0.0, 0.0],
    [0.0,  np.cos(thetaz), -np.sin(thetaz)],
    [0.0,  np.sin(thetaz),  np.cos(thetaz)]
], dtype=np.float64)

Rz90 = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0]
], dtype=np.float64)

# def find_nndistance(size_ratio, compression_ratio, L=0.01, lobe_radius=0.5):
#     a = (4 / np.sqrt(6)) * (2 * rl * sz + 2 * L)
#     if sz < cr / np.sqrt(6) + np.sqrt(1 - cr**2 / 3):
#         a = (4 / np.sqrt(6)) * (2 * cr * rl / np.sqrt(6) + 2 * rl * np.sqrt(1 - cr**2 / 3) + 2 * L)
#     return a

def find_nndistance(size_ratio, compression_ratio, L=0.01, lobe_radius=0.5):
    cr = compression_ratio
    sz = size_ratio
    rl = lobe_radius

    a = (4 / np.sqrt(6)) * (2 * rl * sz + 2 * L)
    if sz < cr / np.sqrt(6) + np.sqrt(1 - cr**2 / 3):
        a = (4 / np.sqrt(6)) * (2 * cr * rl / np.sqrt(6) + 2 * rl * np.sqrt((1+L/rl)**2 - cr**2 / 3))
        # a1 = (4 / np.sqrt(6)) * (2 * cr * rl / np.sqrt(6) + 2 * rl * np.sqrt(1 - cr**2 / 3) + 2 * L)
        # print(f"err {(a1-a)*100/a}")
    return a


def wrap_to_box(x, L):
    return (x + L/2) % L - L/2


def generate_positions_custom_basis_with_layers(nx, ny, nz, a):

    a_lat = a*np.sqrt(2.0)
    
    ax = a_lat/np.sqrt(2.0)
    ay = a_lat/np.sqrt(2.0)
    az = a_lat

    Lx = nx * ax
    Ly = ny * ay
    Lz = nz * az

    basis_frac = np.array([
        [0.0,   0.0,   0.0],
        [1/2,  0.0,  1/(2*np.sqrt(2))],
        [1/2,  1/2,  1/np.sqrt(2)],
        [0.0,   1/2, 3/(2*np.sqrt(2))],
    ], dtype=np.float64)

    basis = a * basis_frac

    dz_layer = a/(2 * np.sqrt(2))

    centers = []
    layer_k = []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                T = np.array([i*ax, j*ay, k*az], dtype=np.float64)
                for b in basis:
                    p = T + b

                    p[0] = wrap_to_box(p[0] - Lx/2.0, Lx)
                    p[1] = wrap_to_box(p[1] - Ly/2.0, Ly)
                    p[2] = wrap_to_box(p[2] - Lz/2.0, Lz)

                    centers.append(tuple(p))

                    kz = int(np.floor((p[2] + Lz/2.0) / dz_layer + 1e-9))
                    layer_k.append(kz)

    return np.asarray(centers, np.float32), np.asarray(layer_k, np.int32), (Lx, Ly, Lz)


def add_tetra_lobes_to_centers(centers, layer_k, R_base, compression_ratio, lobe_radius,
                              include_core=True, wrap_lobes=False, box_lengths=None):

    rl = lobe_radius
    cr = compression_ratio
    lobe_scaling = cr * 2.0 * rl

    tetra = np.array([
        [-0.5,     -0.288675, -0.204124],
        [ 0.5,     -0.288675, -0.204124],
        [ 0.0,      0.57735,  -0.204124],
        [ 0.0,      0.0,       0.612372]
    ], dtype=np.float64) * lobe_scaling

    R_even =  R_base
    R_odd  =  Rz90@R_base

    Nc = centers.shape[0]
    n_per = 5 if include_core else 4
    N = Nc * n_per

    all_pos = np.zeros((N, 3), dtype=np.float32)
    typeid  = np.zeros((N,), dtype=np.uint32)
    body    = np.zeros((N,), dtype=np.int32)

    if wrap_lobes and box_lengths is None:
        raise ValueError("If wrap_lobes=True, provide box_lengths=(Lx,Ly,Lz)")

    idx = 0
    for c_idx, (c, k) in enumerate(zip(centers.astype(np.float64), layer_k)):
        if include_core:
            all_pos[idx] = c
            typeid[idx] = 0
            body[idx] = c_idx
            idx += 1

        R_use = R_even if (k % 2 == 0) else R_odd
        lobes_abs = c[None, :] + (tetra @ R_use.T)

        if wrap_lobes:
            Lx, Ly, Lz = box_lengths
            lobes_abs[:, 0] = wrap_to_box(lobes_abs[:, 0], Lx)
            lobes_abs[:, 1] = wrap_to_box(lobes_abs[:, 1], Ly)
            lobes_abs[:, 2] = wrap_to_box(lobes_abs[:, 2], Lz)

        all_pos[idx:idx+4] = lobes_abs.astype(np.float32)
        typeid[idx:idx+4] = 1
        body[idx:idx+4] = c_idx
        idx += 4

    return all_pos, typeid, body


def write_cluster_gsd(filename, positions, typeid, body, box_lengths, core_radius, lobe_radius):
    Lx, Ly, Lz = box_lengths
    N = positions.shape[0]

    fr = gsd.hoomd.Frame()
    fr.configuration.step = 0
    fr.configuration.box = [Lx, Ly, Lz, 0.0, 0.0, 0.0]

    fr.particles.N = N
    fr.particles.types = ["core", "lobe"]
    fr.particles.typeid = typeid.astype(np.uint32)
    fr.particles.position = positions.astype(np.float32)
    fr.particles.body = body.astype(np.int32)

    diam = np.empty(N, dtype=np.float32)
    diam[typeid == 0] = 2.0 * core_radius
    diam[typeid == 1] = 2.0 * lobe_radius
    fr.particles.diameter = diam

    with gsd.hoomd.open(name=filename, mode="w") as f:
        f.append(fr)

for sz in np.arange(1.15, 1.2201, 0.01):
    for cr in np.arange(0.5, 1.01, 0.1):

        L = 0.01
        rl = 0.5
        nx, ny, nz = 10,10,10
        out = f"/home/akash/hoomd-results/diamond_vs_mtab/diamond_100_sz_{sz:.2f}_cr_{cr:.1f}.gsd"
        
        a = find_nndistance(size_ratio=sz, compression_ratio=cr, L=L, lobe_radius=rl)
        centers, layer_k, box_lengths = generate_positions_custom_basis_with_layers(nx, ny, nz, a)

        core_radius = sz * rl
        R_base = Rx

        cluster_pos, cluster_typeid, cluster_body = add_tetra_lobes_to_centers(
            centers=centers,
            layer_k=layer_k,
            R_base=R_base,
            compression_ratio=cr,
            lobe_radius=rl,
            include_core=True,
            wrap_lobes=False,
            box_lengths=box_lengths
        )

        write_cluster_gsd(
            filename=out,
            positions=cluster_pos,
            typeid=cluster_typeid,
            body=cluster_body,
            box_lengths=box_lengths,
            core_radius=core_radius,
            lobe_radius=rl
        )
        









