import xarray as xr 
import numpy as np

def coarsen_img(
    img: np.ndarray, 
    downscaling_factor: int
) -> np.ndarray:
    
    assert isinstance(downscaling_factor, int), "Downscaling factor must be a single int!"
    
    if len(img.shape) == 4:
        dimnames = ["c", "z", "y", "x"]
    elif len(img.shape) == 3:
        dimnames = ["c", "y", "x"]
    
    img_array = xr.DataArray(img, dims=dimnames)
    coarsened_dims = {dim: downscaling_factor for dim in img_array.dims[1:]}
    return img_array.coarsen(coarsened_dims).sum().data

def channel_wise_norm(arr: np.ndarray) -> np.ndarray:
    spatial_dims = tuple(range(len(arr.shape))[1:])
    pixel_mins = arr.min(axis=spatial_dims, keepdims=True)
    pixel_maxs = arr.max(axis=spatial_dims, keepdims=True)
    
    return (arr - pixel_mins) / (pixel_maxs - pixel_mins + np.finfo(float).eps)

def pixel_wise_sum_to_one(arr: np.ndarray) -> np.ndarray:
    pw_total = np.sum(arr, axis=0, keepdims=True)
    return arr / (pw_total + np.finfo(float).eps)