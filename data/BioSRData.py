import os
import json
from pathlib import Path
from typing import Literal, Union

import tifffile as tiff
import numpy as np

from utils import coarsen_img


class BioSRData:
    def __init__(
        self,
        data_dir: str,
        idx: int,
        gt_type: Literal['digital', 'optical'] = 'optical',
    ):
        """Class to load BioSR simulated spectral data.
        
        Parameters:
        -----------
        data_dir: str
            Root directory containing the simulated data.
        idx: int
            Index of the image to load.
        gt_type: Literal['digital', 'optical']
            Type of ground truth image to load. Default is 'optical'.
            
        Attributes:
        -----------
        gt_img: np.ndarray
            Ground truth image representing fluorophore counts. Shape (F, Z, Y, X).
            F is the number of different fluorophores.
        mixed_clean_img: np.ndarray
            Mixed (spectral) clean image. Shape (W, Z, Y, X).
            W is the number of different spectral bands.
        mixed_noisy_img: np.ndarray
            Mixed (spectral) noisy image. Shape (W, Z, Y, X).
        coords_metadata: dict
            Metadata containing coordinates of the simulated image.
        sim_metadata: dict
            Metadata containing information about the simulated image.
        gt_img_downsc: np.ndarray
            Downscaled ground truth image. Shape (F, Z, Y, X).
        mixed_clean_img_downsc: np.ndarray
            Downscaled mixed clean image. Shape (W, Z, Y, X).
        micro_gt_data: np.ndarray
            Ground truth microscopy images. Shape (F, Z, Y, X).
        """
        assert gt_type in ['digital', 'optical'], \
            ValueError("Invalid ground truth type! Choose from 'digital', 'optical'.")
        
        self.data_dir = data_dir
        self.idx = idx
        self.gt_type = gt_type
        
        # Load Images
        imgs_dir = os.path.join(data_dir, "imgs")
        print("-------------------------")
        print("Loading images...")
        self.gt_img = self.read_BioSR_image(
            path_to_dir=os.path.join(imgs_dir, f"{gt_type}_pf"),
            img_type=f"{gt_type}_pf", 
            idx=idx
        )
        print(f"    Loaded ground truth image!") # Shape (F, [Z], Y, X)
        self.mixed_img = self.read_BioSR_image(
            path_to_dir=os.path.join(imgs_dir, f"digital"), 
            img_type="digital", 
            idx=idx
        )
        print(f"    Loaded noisy mixed image!") # Shape (W, [Z], Y, X)
        
        # Load Metadata
        print("-------------------------")
        print("Loading metadata...")
        self.coords_metadata = self._load_json(os.path.join(data_dir, "sim_coords.json"))
        self.sim_metadata = self._load_json(os.path.join(data_dir, "sim_metadata.json"))
        print("    Done!")
        
        # Downscale images (if needed)
        if gt_type == "optical":
            print("-------------------------")
            print("Computing downscaled version of GT images...")
            self.gt_img_downsc = self._downscale(self.gt_img)
            print("    Done!")
    
    def _read_BioSR_image(
        path_to_dir: Union[str, Path],
        img_type: Literal["digital", "digital_pf", "optical_pf"],
        idx: int
    ) -> np.ndarray:
        """Read a single tiff image.
        
        Parameters
        ----------
        path_to_dir : Union[str, Path]
            Path to the dir containing tiff images.
        img_type : Literal["digital", "digital_pf", "optical_pf"]
            Type of image to read.
        idx: int
            Index of the image to read.
            
        Returns
        -------
        np.ndarray
            Image as a numpy array.
            
        NOTE: each tiff file path has the format: {path_to_dir}/{img_type}_img_{idx}.tif
        """
        fname = f"{img_type}_img_{idx}.tif"
        fpath = os.path.join(path_to_dir, fname) 
        return tiff.imread(fpath)
    
    def _downscale(self, img: np.ndarray) -> np.ndarray:
        downscaling = self.sim_metadata["downscale"]
        return coarsen_img(img, downscaling)
        
    def _load_json(self, fpath: str) -> dict:
        try:    
            with open(fpath, "r") as f:
                json_dict = json.load(f)
        except FileNotFoundError as e:
            print(f"    {fpath} file not found!")
            json_dict = None
        return json_dict
    
    def __repr__(self) -> str:
        msg = "-------------------------\n"
        msg += f"GT Image Shape: {self.gt_img.shape}\n"
        msg += f"Mixed Image Shape: {self.mixed_img.shape}\n"
        msg += "-------------------------\n"
        msg += "Simulated Metadata:\n"
        msg += "\n".join([f"+ {k}: {v}" for k, v in self.sim_metadata.items()]) + "\n"
        msg += "-------------------------\n"
        msg += f"PSNR (noisy vs. clean): {self.PSRN:.2f}"
        return msg