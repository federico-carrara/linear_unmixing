import numpy as np

def PixelWiseMSE(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return (gt - pred)**2

def PixelWiseMAE(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return abs(gt - pred)

def PSNR(
    gt: np.ndarray, 
    pred: np.ndarray, 
    range_: np.ndarray | None =  None
) -> np.ndarray:
    if range_ is None:
        range_ = gt.max() - gt.min()
    return 10 * (np.log10(range_**2) - np.log10(PixelWiseMSE(gt, pred).mean()))

def RangeInvariantPSNR(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the Range Invariant PSNR.
    
    NOTE: input arrays can be either 2D or 3D.
    """
    # Compute the standardized range and ground truth
    std_range = (gt.max() - gt.min()) / gt.std()
    std_gt = (gt - gt.mean()) / gt.std()
                                                                                                                                              
    # Scale the prediction to match the range of the ground truth & standardize
    scaled_pred = fix_range(std_gt, pred)
    
    return PSNR(std_gt, scaled_pred, std_range)
    
def fix_range(gt: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fix the range of x to match the range of gt.
    
    NOTE: input arrays can be either 2D or 3D.
    """
    # Center input data on the mean
    x_ = x - x.mean()
    gt_ = gt - gt.mean()
    # Compute scaling parameter
    a = (gt_ * x_).sum(keepdims=True) / (x_ * x_).sum(keepdims=True)
    return a * x_

def SpectralPSNR(
    gt: np.ndarray, 
    pred: np.ndarray,
    range_inv: bool = True
) -> np.ndarray:
    """Compute PSNR for spectral data.
    
    The PSNR is computed separately for each spectral band and then averaged.
    """
    psnr_fun = RangeInvariantPSNR if range_inv else PSNR
    
    PSNR_lst = []
    for i in range(gt.shape[0]):
        PSNR_lst.append(psnr_fun(gt[i, ...], pred[i, ...]))
    
    return np.mean(PSNR_lst)