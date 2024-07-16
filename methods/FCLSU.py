import time

import numpy as np
import cvxopt as cvx
from tqdm import tqdm

class FCLSU:
    def __init__(
        self,
        mixed_img: np.ndarray,  
        ref_matrix: np.ndarray
    ) -> None:
        
        self.spatial_dims = mixed_img.shape[1:]
        self.Y = self._unroll_image(mixed_img) # shape: (n, N)
        self.E = ref_matrix # shape: (n, p)
        

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg
    
    @staticmethod
    def _unroll_image(img: np.ndarray) -> np.ndarray:
        return img.reshape(img.shape[0], -1)

    @staticmethod
    def _roll_image(img: np.ndarray, shape: tuple) -> np.ndarray:
        return img.reshape(shape)

    @staticmethod
    def _numpy_to_cvxopt_matrix(A: np.ndarray) -> cvx.matrix:
        A = np.array(A, dtype=np.float64)
        if A.ndim == 1:
            return cvx.matrix(A, (A.shape[0], 1), "d")
        else:
            return cvx.matrix(A, A.shape, "d")

    def solve(self) -> np.ndarray:
        """
        Performs fully constrained least squares of each pixel in Y (n x N matrix)
        using the endmember signatures of E. Fully constrained least squares
        is least squares with the abundance sum-to-one constraint (ASC) and the
        abundance nonnegative constraint (ANC).
               
        Returns:
            C: `numpy array`
                Concentration/abundance maps for different endmembers (p x H x W x D).
    
        References:
            Daniel Heinz, Chein-I Chang, and Mark L.G. Fully Constrained
            Least-Squares Based Linear Unmixing. Althouse. IEEE. 1999.
    
        Notes:
            1. These sources have been useful to build the algorithm:
                * The function hyperFclsMatlab, part of the Matlab Hyperspectral
                Toolbox of Isaac Gerg.
                * The Matlab (tm) help on lsqlin.
                * And the Python implementation of lsqlin by Valera Vishnevskiy, click:
                http://maggotroot.blogspot.ca/2013/11/constrained-linear-least-squares-in.html
                , it's great code.
                
            2. Shapes of matrices and dimensions
                - H, W, D: spatial dimensions of the input image (2D or 3D).
                - n: number of spectral bands in the input image.
                - p: number of endmembers in the input image.
                - N: total number of pixels in the input image.
                
            3. Constraints are in this form:
                - Gc <= h --> -Ic <= 0
                - Ac = b --> 1(_p)c = 1
        """
        # tic = time.time()
        # assert len(self.Y.shape) == 2
        # assert len(self.E.shape) == 2
        
        assert self.Y.shape[0] == self.E.shape[0]

        n, N = self.Y.shape # shape: (n, N)
        n, p = self.E.shape # shape: (n, p)

        cvx.solvers.options["show_progress"] = False

        # NOTE: cvxopt only accepts double dtype
        E = self._numpy_to_cvxopt_matrix(self.E.astype(np.double)) # shape: (n, p)
        Q = E.T * E # shape: (p, p)

        # Compute the constraints for the problem
        G = self._numpy_to_cvxopt_matrix(-np.eye(p, dtype=np.double)) # shape: (p, p)
        h = self._numpy_to_cvxopt_matrix(np.zeros(p, dtype=np.double)) # shape: (p,)
        A = self._numpy_to_cvxopt_matrix(np.ones((1, p), dtype=np.double)) # shape: (1, p)
        b = self._numpy_to_cvxopt_matrix(np.ones(1, dtype=np.double)) # shape: (1,)

        X = np.zeros((N, p)) # shape: (N, p)
        for i in tqdm(range(N), desc="FCLSU for pixel"):
            y = cvx.matrix(self.Y[:, i].astype(np.double), (n, 1), "d") # shape: (n, 1)
            r = -y.T * E # shape: (1, p)
            sol = cvx.solvers.qp(Q, r.T, G, h, A, b, None, None)["x"]
            X[i, :] = np.array(sol).squeeze()
        # tac = time.time()
        # print(f"{self} took {tac - tic:.2f}s")
        X = X.T # shape: (p, N)
        new_shape = (p, *self.spatial_dims)
        return self._roll_image(X, new_shape)