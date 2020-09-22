from abc import ABC, abstractmethod
from sklearn.decomposition import NMF, TruncatedSVD
import numpy as np

from .imagereader import apply_mask


class AbstractDenoiser(ABC):
    @abstractmethod
    def denoise(self, data, center, radius):
        pass


class NMFDenoiser(AbstractDenoiser):
    def __init__(self, n_components=5, important_components=1):
        self.n_components = n_components
        self.important_components = important_components

    def denoise(self, data, center, radius=45):
        """
        nmf_denoise performs NMF-decomposition based denoising
        - (N, M, M) image series --> (N, M**2) flattened images
        - (N, M**2) = (N, n_components) @ (n_components, M**2) NMF decomposition
        - background: (n_components, M**2) --> (important_components, M**2)
        - scales: (N, n_components) --> (N, important_components)
        - scaled_background = scales @ background
        - return arr - scaled_background

        Parameters
        ----------
        data : np.ndarray
            Input data (series of 2D images, 3D total)
        center : tuple
            (corner_x, corner_y) tuple
        n_components : int, optional
            n_components for dimensionality reduction, by default 5
        important_components : int, optional
            number of components to account for, by default 1

        Returns
        -------
        np.ndarray
            Denoised data
        """
        img_shape = data.shape[1:]
        X = data.reshape(data.shape[0], -1)

        nmf = NMF(n_components=self.n_components)
        nmf.fit(X)
        coeffs = nmf.transform(X)

        bg_full = nmf.components_
        bg_scaled = (
            coeffs[:, : self.important_components]
            @ bg_full[: self.important_components, :]
        ).reshape(data.shape[0], *img_shape)

        return apply_mask(data - bg_scaled, radius=radius, center=center)


class SVDDenoiser(AbstractDenoiser):
    def __init__(
        self, n_components=5, important_components=1, n_iter=5, random_state=42
    ):
        self.n_components = n_components
        self.important_components = important_components
        self.n_iter = n_iter
        self.random_state = random_state

    def denoise(self, data, center, radius=45):
        """
        svd_denoise performs SVD-based denoising of input array
        - (N, M, M) image series --> (N, M**2) flattened images
        - (N, M**2) = (N, n_components) @ (n_components, M**2) SVD decomposition
        - background: (n_components, M**2) --> (important_components, M**2)
        - scales: (N, n_components) --> (N, important_components)
        - scaled_background = scales @ background
        - return arr - scaled_background

        Parameters
        ----------
        arr : np.ndarra
            3D numpy array (series of 2D images)
        center : tuple
            (corner_x, corner_y) tuple
        n_components : int, optional
            n_components for TruncatedSVD decomposition, by default 5
        important_components : int, optional
            number of components to account fo, by default 1
        n_iter : int, optional
            number of iterations in TruncatedSVD, by default 5

        Returns
        -------
        np.ndarray
            Denoised array of same shape
        """
        img_shape = data.shape[1:]
        X = data.reshape(data.shape[0], -1)

        svd = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state,
            n_iter=self.n_iter,
        )
        svd.fit(X)
        coeffs = svd.transform(X)

        bg_full = svd.components_
        bg_scaled = (
            coeffs[:, : self.important_components]
            @ bg_full[: self.important_components, :]
        ).reshape(data.shape[0], *img_shape)

        return apply_mask(data - bg_scaled, radius=radius, center=center)


class PercentileDenoiser(AbstractDenoiser):
    def __init__(self, percentile=45, alpha=5e-2):
        self.percentile = percentile
        self.alpha = alpha

    def denoise(self, data, center, radius=45):
        """
        percentile_denoise applies percentile denoising:
        - create percentile-based background profille
        - apply mask
        - subtract background with such scale that less thatn `alpha` resulting pixels are negative

        Parameters
        ----------
        data : np.ndarray
            Input data (series of 2D images, 3D total)
        center : tuple, optional
            (corner_x, corner_y), by default (720, 710)
        percentile : int, optional
            percentile to use, by default 45

        Returns
        -------
        np.ndarray
            Denoised images
        """
        data = apply_mask(data, center=center, radius=radius)
        bg = self._percentile_filter(data, q=self.percentile)
        scales = self._scalefactors_bin(data, bg, alpha=self.alpha)

        full_bg = np.dot(bg.reshape(*(bg.shape), 1), scales.reshape(1, -1))
        full_bg = np.moveaxis(full_bg, 2, 0)

        data = data - full_bg
        del full_bg
        return data

    def _percentile_filter(self, arr, q=45):
        """
        _percentile_filter creates background profile

        Parameters
        ----------
        arr : 3D np.ndarray (series of 2D images)
            input array
        q : int, optional
            percentile for filtering, by default 45

        Returns
        -------
        np.ndarray
            2D np.ndarray of background profile
        """

        return np.percentile(arr, q=q, axis=(0))

    def _bin_scale(self, arr, b, alpha=0.01, num_iterations=10):
        """
        _bin_scale binary search for proper scale factor

        Parameters
        ----------
        arr : np.ndarray
            Input 3-D array (N + 2D)
        b background
            Single image (backgorund profile)
        alpha : float, optional
            Share of pixels to be negative, by default 0.01
        num_iterations : int, optional
            Number of binary search iterations, by default 10

        Returns
        -------
        np.ndarray
            proper scalefactors
        """

        num_negative = alpha * arr.shape[0] * arr.shape[1]

        def count_negative(scale):
            return (arr - scale * b < 0).sum()

        l, r, m = 0, 1, 2

        for _ in range(num_iterations):
            m = (l + r) / 2
            mv = count_negative(m)

            if mv < num_negative:
                l, r = m, r
            else:
                l, r = l, m

        return l

    def _scalefactors_bin(self, arr, bg, alpha=0.01, num_iterations=10):
        """\
        Find proper scalefactor for an image given f _percentile_filter
        so that the share of negative pixels in resulting difference
        is less that threshold
        """
        # print("Start scalefactor estimation")
        return np.array(
            [
                self._bin_scale(arr[i], bg, alpha=alpha, num_iterations=num_iterations)
                for i in range(arr.shape[0])
            ]
        )
