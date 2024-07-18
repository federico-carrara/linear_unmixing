import xarray as xr
import numpy as np

from microsim.schema.sample import Fluorophore

class FPRefMatrix:
    def __init__(self, fp_names: list[str], w_bins: int = 32) -> None:
        self.fp_names = fp_names
        self.w_bins = w_bins  
        self.n = len(fp_names)
        self.p = len(w_bins)
        self.sbins = sorted(set([bins[0] for bins in w_bins] + [w_bins[-1][1]]))
        
        self.fp_list = None # list of Fluorophore objects
        self.fp_em_list = None # list of xr.DataArray's containing emission spectra

    def _fetch_FPs(self) -> list[Fluorophore]:
        return [Fluorophore.from_fpbase(name=fp_name) for fp_name in self.fp_names]
    
    def _normalize(self) -> np.ndarray:
        assert self.fp_em_list is not None
        return [
            (fp_em - fp_em.min()) / (fp_em.max() - fp_em.min())
            for fp_em in self.fp_em_list
        ]
    
    def _fill_NaNs(self, num: int = 0) -> list[xr.DataArray]:
        assert self.fp_em_list is not None
        return [
            fp_em.fillna(num)
            for fp_em in self.fp_em_list
        ]
    
    def _bin_spectra(self) -> list[xr.DataArray]:
        assert self.fp_em_list is not None
        return [
            fp_em.groupby_bins(fp_em["w"], self.sbins).sum()
            for fp_em in self.fp_em_list
        ]
        
    def create(self) -> np.ndarray:
        self.fp_list = self._fetch_FPs()
        self.fp_em_list = [
            xr.DataArray(
                fp.emission_spectrum.intensity, 
                coords=[fp.emission_spectrum.wavelength.magnitude], 
                dims=["w"]
            )
            for fp in self.fp_list
        ]
        self.fp_em_list = self._bin_spectra()
        self.fp_em_list = self._fill_NaNs()
        self.fp_em_list = self._normalize()
        return np.stack(
            [fp_em.values for fp_em in self.fp_em_list], 
            axis=1
        )