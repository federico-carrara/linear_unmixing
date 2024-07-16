import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .metrics import PixelWiseMSE


def plot_unmixed_vs_gt(
    gt_img: np.ndarray, 
    unmixed_img: np.ndarray, 
):
    num_images = gt_img.shape[0]
    fig, ax = plt.subplots(num_images, 3, figsize=(13, 4 * num_images))
    
    is_3d = len(gt_img.shape) == 4
    if not is_3d:
        gt_img = gt_img[:, np.newaxis, ...]
        unmixed_img = unmixed_img[:, np.newaxis, ...]
    
    for i in range(num_images):
        mse = PixelWiseMSE(gt_img[i, ...], unmixed_img[i, ...])
        
        if i == 0:
            ax[i, 0].set_title(f"GT (abundancy/concentration) {"- MIP" if is_3d else ""}")
        im0 = ax[i, 0].imshow(gt_img.max(axis=1)[i, :, :])
        divider = make_axes_locatable(ax[i, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im0, cax=cax)
        
        if i == 0:
            ax[i, 1].set_title(f"Unmixing Result (abundancy/concentration) {"- MIP" if is_3d else ""}")
        im1 = ax[i, 1].imshow(unmixed_img.max(axis=1)[i, :, :])
        divider = make_axes_locatable(ax[i, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax)
        
        if i == 0:
            ax[i, 2].set_title(f"Pixel-wise MSE (abundancy/concentration) {"- MIP" if is_3d else ""}")
        im2 = ax[i, 2].imshow(mse.max(axis=0), cmap="RdPu")
        divider = make_axes_locatable(ax[i, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax)
        
        ax[i, 2].text(
            0.6, 0.1, f'MSE: {mse.mean():.2e}', transform=ax[i, 2].transAxes,
            fontsize=12, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5)
        )
    
    plt.tight_layout()
    plt.draw()

    # Quantitative results
    mse_values = [PixelWiseMSE(gt_img[i, ...], unmixed_img[i, ...]).mean() for i in range(num_images)]
    mse_str = ", ".join([f"{mse:.2e}" for mse in mse_values])
    print(f"MSE: {mse_str}")
