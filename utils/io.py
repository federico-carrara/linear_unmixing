import glob
from pathlib import Path
import os
from typing import Callable, Literal, Optional, Union

import numpy as np
import tifffile as tiff


def read_multifile_tiff(
    path_to_dir: Union[str, Path], 
    sort_fn: Optional[Callable] = None
) -> np.ndarray:
    """Read a directory of tiff files and concatenates in a numpy array.
    
    Parameters
    ----------
    path_to_dir : Union[str, Path]
        Path to the directory containing tiff files.
    sort_fn : Optional[Callable], optional
        Function to sort the files before reading them, by default None.
    """
    files = glob.glob(os.path.join(path_to_dir, "*.tif"))
    assert len(files) > 0, f"No files found in {path_to_dir}"
    print(f"Reading {len(files)} files from {path_to_dir}...")
    if sort_fn:
        files = sorted(files, key=sort_fn)
    arrs = [np.array(tiff.imread(f)) for f in files]
    return np.stack(arrs)


def sorting_key(x: str) -> int:
    return int(x.split("/")[-1].split(".")[0].split("_")[-1])