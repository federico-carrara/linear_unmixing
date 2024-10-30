from typing import Sequence, Union

import xarray as xr 
import numpy as np


def coarsen_img(
    img: np.ndarray, 
    downscaling_factor: Union[int, Sequence[int]]
) -> np.ndarray:
    if isinstance(downscaling_factor, Sequence):
        assert len(downscaling_factor) == img.ndim - 1, (
            "Downscaling factor must be a single int or a sequence matching `img` spatial dimensions!"
        )
    else:
        downscaling_factor = [downscaling_factor] * (img.ndim - 1)
    
    if len(img.shape) == 4:
        dimnames = ["c", "z", "y", "x"]
    elif len(img.shape) == 3:
        dimnames = ["c", "y", "x"]
    
    img_array = xr.DataArray(img, dims=dimnames)
    coarsened_dims = {
        dim: fact 
        for dim, fact in zip(img_array.dims[1:], downscaling_factor)
    }
    return img_array.coarsen(coarsened_dims).sum().data

def channel_wise_norm(arr: np.ndarray) -> np.ndarray:
    spatial_dims = tuple(range(len(arr.shape))[1:])
    pixel_mins = arr.min(axis=spatial_dims, keepdims=True)
    pixel_maxs = arr.max(axis=spatial_dims, keepdims=True)
    
    return (arr - pixel_mins) / (pixel_maxs - pixel_mins + np.finfo(float).eps)

def pixel_wise_sum_to_one(arr: np.ndarray) -> np.ndarray:
    pw_total = np.sum(arr, axis=0, keepdims=True)
    return arr / (pw_total + np.finfo(float).eps)