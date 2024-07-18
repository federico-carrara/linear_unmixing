import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .metrics import RangeInvariantPSNR, PixelWiseMAE


def plot_unmixed_vs_gt(
    gt_img: np.ndarray | list[np.ndarray], 
    unmixed_img: np.ndarray, 
    method: str
):
    if isinstance(gt_img, list):
        gt_img = np.concatenate(gt_img, axis=0)
    
    num_images = gt_img.shape[0]
    fig, ax = plt.subplots(num_images, 3, figsize=(13, 4 * num_images))
    fig.suptitle(f"{method} Unmixing Results", fontsize=20)
    
    is_3d = len(gt_img.shape) == 4
    if not is_3d:
        gt_img = gt_img[:, np.newaxis, ...]
        unmixed_img = unmixed_img[:, np.newaxis, ...]
    
    psnr_values = []
    for i in range(num_images):
        psnr = RangeInvariantPSNR(gt_img[i, ...], unmixed_img[i, ...])
        psnr_values.append(psnr)
        
        # Standardize imgs
        std_gt = (gt_img[i, ...] - gt_img[i, ...].mean()) / gt_img[i, ...].std()
        std_unmixed = (unmixed_img[i, ...] - unmixed_img[i, ...].mean()) / unmixed_img[i, ...].std()
        mae = PixelWiseMAE(std_gt, std_unmixed)
        
        if i == 0:
            ax[i, 0].set_title(f"Standardized GT {"- MIP" if is_3d else ""}")
        im0 = ax[i, 0].imshow(std_gt.max(axis=0))
        divider = make_axes_locatable(ax[i, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im0, cax=cax)
        
        if i == 0:
            ax[i, 1].set_title(f"Standardized Unmixing Result {"- MIP" if is_3d else ""}")
        im1 = ax[i, 1].imshow(std_unmixed.max(axis=0))
        divider = make_axes_locatable(ax[i, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax)
        
        if i == 0:
            ax[i, 2].set_title(f"Pixel-wise MAE {"- MIP" if is_3d else ""}")
        im2 = ax[i, 2].imshow(mae.max(axis=0), cmap="RdPu")
        divider = make_axes_locatable(ax[i, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax)
        
        ax[i, 2].text(
            0.66, 0.1, f'PSNR: {psnr:.2f}', transform=ax[i, 2].transAxes,
            fontsize=12, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5)
        )
    
    plt.tight_layout()
    plt.draw()

    # Quantitative results
    psnr_str = ", ".join([f"{psnr:2f}" for psnr in psnr_values])
    print(f"{method} PSNR: {psnr_str}")
