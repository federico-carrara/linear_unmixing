import os
import json
from typing import Literal, Sequence

import tifffile as tiff
from careamics.dataset.dataset_utils.readers.astro_neurons import get_fnames


class AstroNeuronData:
    def __init__(
        self,
        data_dir: str,
        dset_type: Literal["astrocytes", "neurons"],
        groups: Sequence[Literal["control", "arsenite", "tharps"]],
        dim: Literal["2D", "3D"] = "2D",
        idx: int = 0,
    ):
        """Class to load Astrocytes and Neurons spectral data.
        
        Parameters:
        -----------
        data_dir: str
            Root directory containing the simulated data.
        dset_type : Literal["astrocytes", "neurons"]
            The type of dataset to load.
        groups : Sequence[Literal["control", "arsenite", "tharps"]]
            The groups of samples to load.
        idx: int
            Index of the image to load.
            
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
        self.data_dir = data_dir
        self.idx = idx
        
        # Load Images
        gt_fnames = get_fnames(
            data_path=data_dir,
            dset_type=dset_type,
            groups=groups,
            img_type="unmixed",
            dim=dim,
        )
        self.gt_img = tiff.imread(gt_fnames[idx]) # Shape (F+1, [Z], Y, X)
        mixed_fnames = get_fnames(
            data_path=data_dir,
            dset_type=dset_type,
            groups=groups,
            img_type="raw",
            dim=dim,
        )
        self.N = len(mixed_fnames)
        self.mixed_img = tiff.imread(mixed_fnames[idx]) # Shape (W, [Z], Y, X)
        
        # Load Metadata
        with open(os.path.join(data_dir, dset_type, "info/metadata.json")) as f:
            self.metadata = json.load(f)
    
    def __len__(self) -> int:
        return self.N
    
    def __repr__(self) -> str:
        msg = "-------------------------\n"
        msg += f"GT Image Shape: {self.gt_img.shape}\n"
        msg += f"Mixed Image Shape: {self.mixed_img.shape}\n"
        msg += "-------------------------\n"
        msg += "Metadata:\n"
        msg += "\n".join([f"+ {k}: {v}" for k, v in self.metadata.items()]) + "\n"
        return msg