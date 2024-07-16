# linear_unmixing

In this repo we tackle the problem of linear unmixing in spectral data.

Data could belong to different domains, such as fluoroscence microscopy and satellite/remote sensing.

## Problem Statement

In the case of spectral data, for a given pixel we suppose to have a set of intensity measurements at different wavelengths, e.g., $y = [y(\lambda_1),y(\lambda_2),\dots,y(\lambda_n)]$. For each one of these spectral bands $\lambda_i$, with $i=1,\dots,n$, and for each endmember $e$, with $e=1,\dots,m$, we assume the reference spectra $R_e=[R_e(\lambda_1), R_e(\lambda_2), \dots, R_e(\lambda_n)]$ to be known. Therefore, $y$ is a column vector of size $n \times 1$ and $R_e$ are columns of a matrix $\mathbf{R}$ of size $n \times m$. In this context, we define the abundancy/concentration of the different endmembers in the sample as a vector $c = [c_1, c_2, \dots, c_m]$ of size $m \times 1$. Therefore, the problem of linear unmixing reads as follows: <br><br>

\begin{equation}
y = \mathbf{R}c
\end{equation}

or, by specifying the vectors and matrix:

\begin{align}
    \begin{bmatrix}
        y(\lambda_1) \\
        y(\lambda_2) \\
        \vdots \\
        y(\lambda_n)
    \end{bmatrix}
        =
    \begin{bmatrix}
        R_{1}(\lambda_1) & R_{2}(\lambda_1) & \dots & R_{m}(\lambda_1) \\
        R_{1}(\lambda_2) & R_{2}(\lambda_2) & \dots & R_{m}(\lambda_2) \\
        \vdots & \vdots & \ddots & \vdots \\
        R_{1}(\lambda_n) & R_{2}(\lambda_n) & \dots & R_{m}(\lambda_n) \\
    \end{bmatrix}
    \begin{bmatrix}
        c_{1} \\
        c_{2} \\
        \vdots \\
        c_{m}
    \end{bmatrix}
\end{align}

## Linear Spectral Unmixing with Least Squares

Least Squares is a method that allows to solve the system. It works as follows:

**Goal:** to compute $c$ such that it minimizes the objective $J(c)=||y - \mathbf{R}c||^2$.

**How:** by setting the derivative of $J(c)$ w.r.t. $c$ to $0$. Namely:
\begin{align}
\frac{\partial{J(c)}}{\partial{c}}=-2(y-\mathbf{R}c)\mathbf{R}^T=0 \Longrightarrow \mathbf{R}c\mathbf{R}^T=y\mathbf{R}^T \Longrightarrow c=y\mathbf{R}^T(\mathbf{R}\mathbf{R}^T)^{-1} 
\end{align}

**NOTES**

- Notice that in the context of a spectral image we need to repeat this procedure for every pixel.
- $y(\lambda_i)$ is the intensity for a given pixel in the mixed image at wavelength $\lambda_i$.
- $R_e(\lambda_i)$ is the intensity in the reference spectrum of endmember $e$ at wavelength $\lambda_i$. 
