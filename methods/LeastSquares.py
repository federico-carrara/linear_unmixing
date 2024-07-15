import numpy as np
from scipy.linalg import lstsq
from tqdm import tqdm 

def lstsq_fit_2D(mixed_img, fp_ref_matrix):
    _, n_rows, n_cols = mixed_img.shape
    n_fps = fp_ref_matrix.shape[0]

    fp_conc_img = np.zeros((n_fps, n_rows, n_cols))

    for r in range(n_rows):
        for c in range(n_cols):
            mixed_pixel = mixed_img[:, r, c]
            fp_conc, residuals, rank, sing_vals = lstsq(fp_ref_matrix.T, mixed_pixel)
            fp_conc_img[:, r, c] = fp_conc

    return fp_conc_img

def lstsq_fit_3D(mixed_img, fp_ref_matrix):
    _, nz, ny, nx = mixed_img.shape
    n_fps = fp_ref_matrix.shape[0]

    fp_conc_img = np.zeros((n_fps, nz, ny, nx))

    for z in tqdm(range(nz), desc="Solving LS on z slice"):
        for y in range(ny):
            for x in range(nx):
                mixed_voxel = mixed_img[:, z, y, x]
                fp_conc, residuals, rank, sing_vals = lstsq(fp_ref_matrix.T, mixed_voxel)
                fp_conc_img[:, z, y, x] = fp_conc

    return fp_conc_img

def lstsq_fit(mixed_img, fp_ref_matrix):
    if len(mixed_img.shape) == 4:
        return lstsq_fit_3D(mixed_img, fp_ref_matrix)
    elif len(mixed_img.shape) == 3:
        return lstsq_fit_2D(mixed_img, fp_ref_matrix)
    else:
        raise ValueError(f"Invalid image shape {mixed_img.shape}")