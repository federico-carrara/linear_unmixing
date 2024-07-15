import numpy as np
from math import log10

def pixel_wise_mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return (gt - pred)**2

def PSNR(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return 10 * (log10((gt.max() - gt.min())**2) - log10(pixel_wise_mse(gt, pred).mean()))

def spectral_PSNR(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute PSNR for spectral data.
    
    The PSNR is computed separately for each spectral band and then averaged.
    """
    PSNR_lst = []
    for i in range(gt.shape[0]):
        PSNR_lst.append(PSNR(gt[i, ...], pred[i, ...]))
    
    return np.mean(PSNR_lst)