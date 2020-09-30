from abc import ABC, abstractmethod
from sklearn.decomposition import NMF, TruncatedSVD
import numpy as np


class AbstractDenoiser(ABC):
    def __init__(self):
        self._scales = None
        self._bg = None

    @abstractmethod
    def fit(self, data, bg, center, radius=45):
        pass

    @abstractmethod
    def transform(self, data, bg, scales):
        pass

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

    def _scalefactors(self, arr, bg, alpha=0.01, num_iterations=10):
        """\
        Find proper scalefactor for an image
        so that the share of negative pixels in resulting difference
        is less than alpha
        """
        return np.array(
            [
                self._bin_scale(arr[i], bg, alpha=alpha, num_iterations=num_iterations)
                for i in range(arr.shape[0])
            ]
        )


class NMFDenoiser(AbstractDenoiser):
    def __init__(self, n_components=5, important_components=1):
        super().__init__()
        self.n_components = n_components
        self.important_components = important_components

    def fit(self, data, center, radius=45):
        """
        fit searches for the background profile using NMF decomposition
        - (N, M, M) image series --> (N, M**2) flattened images
        - (N, M**2) = (N, n_components) @ (n_components, M**2) NMF decomposition
        - background: (n_components, M**2) --> (important_components, M**2)

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
            Background profile
        """
        data_m = apply_mask(data, center=center, radius=radius)
        X = data_m.reshape(data_m.shape[0], -1)

        nmf = NMF(n_components=self.n_components)
        nmf.fit(X)
        coeffs = nmf.transform(X)

        # memorize scalefactors in order not to repeat factorization later
        self._scales = coeffs[: self.important_components]

        bg_full = (
            nmf.components_[: self.important_components]
            .sum(axis=0)
            .reshape(data_m.shape[1:])
        )
        self._bg = bg_full

        return bg_full

    def transform(self, data, center, radius=45, alpha=None):
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

        if self._bg is None:
            _ = self.fit(data=data, center=center, radius=radius)

        if alpha is None:
            coeffs = self._scales
        else:
            coeffs = self._scalefactors(arr=data, bg=self._bg, alpha=alpha)

        bg_scaled = (coeffs @ self._bg).reshape(data.shape[0], *img_shape)

        return data - bg_scaled


class SVDDenoiser(AbstractDenoiser):
    def __init__(
        self, n_components=5, important_components=1, n_iter=5, random_state=42
    ):
        super().__init__()
        self.n_components = n_components
        self.important_components = important_components
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, data, center, radius=45):
        """
        fit searches for the background profile using NMF decomposition
        - (N, M, M) image series --> (N, M**2) flattened images
        - (N, M**2) = (N, n_components) @ (n_components, M**2) NMF decomposition
        - background: (n_components, M**2) --> (important_components, M**2)

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
            Background profile
        """
        data_m = apply_mask(data, center=center, radius=radius)
        X = data_m.reshape(data_m.shape[0], -1)

        svd = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state,
            n_iter=self.n_iter,
        )

        svd.fit(X)
        coeffs = svd.transform(X)
        bg_full = (
            svd.components_[: self.important_components]
            .sum(axis=0)
            .reshape(data_m.shape[1:])
        )

        # memorize scalefactors and background
        self._scales = coeffs[: self.important_components]
        self._bg = bg_full

        return bg_full

    def transform(self, data, center, radius=45, alpha=None):
        """
        nmf_denoise performs SVD-decomposition based denoising
        - (N, M, M) image series --> (N, M**2) flattened images
        - (N, M**2) = (N, n_components) @ (n_components, M**2) SVD decomposition
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

        if self._bg is None:
            _ = self.fit(data=data, center=center, radius=radius)

        if alpha is None:
            coeffs = self._scales
        else:
            coeffs = self._scalefactors(arr=data, bg=self._bg, alpha=alpha)

        bg_scaled = (coeffs @ self._bg).reshape(data.shape[0], *img_shape)

        return data - bg_scaled


class PercentileDenoiser(AbstractDenoiser):
    def __init__(self, percentile=45, alpha=1e-2):
        super().__init__()
        self._percentile = percentile
        self._alpha = alpha

    def transform(self, data, center, radius=45):
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

        if self._bg is None:
            self.fit(data, q=self._percentile)
        scales = self._scalefactors(data, self._bg, alpha=self._alpha)

        full_bg = np.dot(self._bg.reshape(*(self._bg.shape), 1), scales.reshape(1, -1))
        full_bg = np.moveaxis(full_bg, 2, 0)

        data = data - full_bg
        del full_bg
        return data

    def fit(self, arr, q=45):
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
        bg = np.percentile(arr, q=q, axis=(0))
        self._bg = bg

        return bg


def apply_mask(np_arr, center=(719.9, 711.5), radius=45):
    """
    _apply_mask applies circular mask to a single image or image series

    Parameters
    ----------
    np_arr : np.ndarray
        Input array to apply mask to
    center : tuple
        (corner_x, corner_y) pair of floats
    r : int, optional
        radius of pixels to be zeroed, by default 45

    Returns
    -------
    np.ndarray
        Same shaped and dtype'd array as input
    """

    if len(np_arr.shape) == 3:
        shape = np_arr.shape[1:]
        shape_type = 3
    else:
        shape = np_arr.shape
        shape_type = 2
    mask = np.ones(shape)

    rx, ry = map(int, center)
    r = radius
    for x in range(rx - r, rx + r):
        for y in range(ry - r, ry + r):
            if (x - rx) ** 2 + (y - ry) ** 2 <= r ** 2:
                mask[x][y] = 0

    if shape_type == 2:
        return (np_arr * mask).astype(np_arr.dtype)
    else:
        mask = mask.reshape((*shape, 1))
        return (np_arr * mask.reshape(1, *shape)).astype(np_arr.dtype)
