from typing import Optional, Union

import xarray as xr
import numpy as np

from microsim.schema.sample import Fluorophore


class FPRefMatrix:
    def __init__(
        self, 
        fp_names: list[str], 
        w_bins: Union[int, list[list[int, int]]] = 32, 
        w_range: Optional[list[int, int]] = None
    ) -> None:
        """Class to create a reference matrix of fluorophore emission spectra.
        
        NOTE: self.w_bins is a list of bin delimiters.
        """
        self.fp_names = fp_names
        
        if isinstance(w_bins, int):
            # need to create bins first
            assert w_range is not None, "Need wavelength range to create bins!"
            w_bins = self._get_bins(w_bins, w_range)
        elif isinstance(w_bins, list[list[int, int]]):
            w_bins = sorted(set([bins[0] for bins in w_bins] + [w_bins[-1][1]]))
        self.w_bins = w_bins
        
        self.fp_list = None # list of Fluorophore objects
        self.fp_em_list = None # list of xr.DataArray's containing emission spectra

    def _fetch_FPs(self) -> list[Fluorophore]:
        return [Fluorophore.from_fpbase(name=fp_name) for fp_name in self.fp_names]
    
    def _get_bins(
        self, num_bins: int, interval: list[int, int]
    ) -> list[list[int, int]]:
        """Get bin delimiters for the given interval.
        
        Parameters
        ----------
        num_bins : int
            The number of bins to create.
        interval : Sequence[int, int]
            The interval to create the bins for.
        
        Returns
        -------
        list[list[int, int]]
            The bin delimiters.
        """
        range_ = interval[1] - interval[0]
        min_bin_length = range_ // num_bins
        remainder = range_ % num_bins
        bins = [interval[0]]
        for i in range(num_bins):
            # add extra wavelengths (reminder) at the beginning
            curr_bin_length = min_bin_length if i >= remainder else min_bin_length + 1
            bins.append(bins[-1] + curr_bin_length)
        return bins
    
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
            fp_em.groupby_bins(fp_em["w"], self.w_bins).sum()
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