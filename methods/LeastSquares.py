import numpy as np
from scipy.linalg import lstsq
from tqdm import tqdm 

class LeastSquares:
    def __init__(
        self,
        mixed_img: np.ndarray,  
        ref_matrix: np.ndarray
    ) -> None:
        
        self.spatial_dims = mixed_img.shape[1:]
        mixed_img = self._normalize(mixed_img)
        self.Y = self._unroll_image(mixed_img) # shape: (n, N)
        self.E = ref_matrix # shape: (n, p)
        
    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg
    
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        return (img - img.min()) / (img.max() - img.min())
    
    @staticmethod
    def _unroll_image(img: np.ndarray) -> np.ndarray:
        return img.reshape(img.shape[0], -1)

    @staticmethod
    def _roll_image(img: np.ndarray, shape: tuple) -> np.ndarray:
        return img.reshape(shape)
    
    def solve(self) -> np.ndarray:
        """Perform least squares for each pixel in Y (n x N matrix) using the
        endmember signatures of E. 
               
        Returns:
        --------
            X: `nd.nparray`
                Concentration/abundance maps for different endmembers (p x H x W x D).
    
        Notes:
            1. Shapes of matrices and dimensions
                - H, W, D: spatial dimensions of the input image (2D or 3D).
                - n: number of spectral bands in the input image.
                - p: number of endmembers in the input image.
                - N: total number of pixels in the input image.
        """
        n, N = self.Y.shape # shape: (n, N)
        n, p = self.E.shape # shape: (n, p)
        
        X = np.zeros((N, p)) # shape: (N, p)
        for i in tqdm(range(N), desc="Solving LS for pixel"):
            sol, _, _, _ = lstsq(a=self.E, b=self.Y[:, i])
            X[i, :] = np.array(sol).squeeze()
        X = X.T # shape: (p, N)
        new_shape = (p, *self.spatial_dims)
        return self._roll_image(X, new_shape)