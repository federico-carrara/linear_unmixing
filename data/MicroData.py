import os
import json
from typing import Literal

import tifffile as tiff
import numpy as np

from utils import coarsen_img
from utils.metrics import SpectralPSNR

class MicroData:
    def __init__(
        self,
        data_dir: str,
        use_mip: bool = False, 
        clean_img_name: Literal['optical', 'emission'] = "optical",
    ):
        """
        Parameters:
        -----------
        data_dir: str
            Root directory containing the simulated data.
        use_mip: bool
            Whether to load and use MIP images or not.
        clean_img_name: Literal['optical', 'digital']
            The clean image to be used. Choose from ['optical', 'emission'].
            
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
        assert clean_img_name in ["optical", "digital"], \
            "Invalid clean image name. Choose from ['optical', 'digital']"
        assert not use_mip, NotImplementedError("MIP images are not supported yet!") 
        
        self.data_dir = data_dir
        self.use_mip = use_mip
        
        # Load Images
        print("-------------------------")
        print("Loading images...")
        self.gt_img = self._load_image(os.path.join(data_dir, "ground_truth_img.tif"))
        print(f"    Loaded ground truth image!")
        self.mixed_clean_img = self._load_image(
            self._get_img_fpath(f"{clean_img_name}_mixed")
        )
        print(f"    Loaded {clean_img_name} mixed image!")
        self.mixed_noisy_img = self._load_image(self._get_img_fpath("digital_mixed"))
        print(f"    Loaded noisy mixed image!")
        micro_fnames = [f"{clean_img_name}_fluor{i+1}_gt.tif" for i in range(self.gt_img.shape[0])]
        micro_fnames = [os.path.join(data_dir, "imgs", micro_fname) for micro_fname in micro_fnames]
        self.micro_gt_data = self._load_multi_images(micro_fnames)
        print(f"    Loaded ground truth microscopy images!")
        
        # Load Metadata
        print("-------------------------")
        print("Loading metadata...")
        self.coords_metadata = self._load_json(os.path.join(data_dir, "sim_coords.json"))
        self.sim_metadata = self._load_json(os.path.join(data_dir, "sim_metadata.json"))
        print("    Done!")
        
        # Downscale images
        print("-------------------------")
        print("Computing downscaled version of GT images...")
        self.gt_img_downsc = self._downscale(self.gt_img)
        self.mixed_clean_img_downsc = self._downscale(self.mixed_clean_img)
        self.micro_gt_data_downsc = self._downscale(self.micro_gt_data)
        print("    Done!")
        
    def __repr__(self) -> str:
        msg = "-------------------------\n"
        msg += f"Clean Mixed Image Shape: {self.mixed_clean_img.shape}\n"
        msg += f"Noisy Mixed Image Shape: {self.mixed_noisy_img.shape}\n"
        msg += "-------------------------\n"
        msg += "Simulated Metadata:\n"
        msg += "\n".join([f"+ {k}: {v}" for k, v in self.sim_metadata.items()]) + "\n"
        msg += "-------------------------\n"
        msg += f"PSNR (noisy vs. clean): {self.PSRN:.2f}"
        return msg
        
    def _downscale(self, img: np.ndarray) -> np.ndarray:
        downscaling = self.sim_metadata["downscale"]
        return coarsen_img(img, downscaling)
    
    def _get_img_fpath(self, img_name: str) -> str:
        """Get image file path.
        
        It depends whether one wants to load the MIP or not.
        """
        dir_name = f"{"mips" if self.use_mip else "imgs"}"
        suffix = f"{"_mip" if self.use_mip else ""}.tif"
        return os.path.join(self.data_dir, dir_name, img_name + suffix)
            
    def _load_image(self, fpath: str) -> np.ndarray:
        return tiff.imread(fpath)
    
    def _load_multi_images(self, fpaths: list[str]) -> np.ndarray:
        """Load multiple images from a list of files.
        
        Load images as `np.ndarray`, then add an extra dimension to concat them.
        e.g., N original images are (Z, Y, X), result is (N, Z, Y, X). 
        """
        imgs = [self._load_image(fpath)[np.newaxis, ...] for fpath in fpaths]
        return np.concatenate(imgs, axis=0)
        
    def _load_json(self, fpath: str) -> dict:
        try:    
            with open(fpath, "r") as f:
                json_dict = json.load(f)
        except FileNotFoundError as e:
            print(f"    {fpath} file not found!")
            json_dict = None
        return json_dict
    
    @property
    def PSRN(self) -> float:
        assert self.mixed_clean_img_downsc.shape == self.mixed_noisy_img.shape, \
            "Images should be of same shape for PSNR computation."
        return SpectralPSNR(
            gt=self.mixed_clean_img_downsc, 
            pred=self.mixed_noisy_img, 
            range_inv=True
        )