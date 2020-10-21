from abc import ABC, abstractmethod

from sklearn.decomposition import NMF, TruncatedSVD
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist

import h5py
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
from typing import Union

from stilutils.utils.imagereader import ImageLoader


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

    def _bin_scale(self, arr, b, alpha=1e-3, num_iterations=100, mm=1e10):
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

        l, r, m = 0, mm, mm / 2.0

        for _ in range(num_iterations):
            # print(l,r,m)
            m = (l + r) / 2
            mv = count_negative(m)

            if mv < num_negative:
                l, r = m, r
            else:
                l, r = l, m

        return l

    def _scalefactors(self, arr, bg, alpha=0.01):
        """\
        Find proper scalefactor for an image
        so that the share of negative pixels in resulting difference
        is less than alpha
        """
        assert len(arr.shape) == 3, f"Shape of your array: {arr.shape} (should be 3D)"

        return np.array(
            [self._bin_scale(arr[i], bg, alpha=alpha) for i in range(arr.shape[0])]
        ).reshape(1, -1)


class NMFDenoiser(AbstractDenoiser):
    def __init__(self, n_components=5, important_components=1):
        super().__init__()
        self.n_components = n_components
        self.important_components = important_components

    def fit(self, data):
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
        X = data.reshape(data.shape[0], -1)

        nmf = NMF(
            n_components=self.n_components,
        )

        nmf.fit(X)
        coeffs = nmf.transform(X)
        bg_full = nmf.components_[: self.important_components, :].reshape(
            (-1, *data.shape[1:])
        )

        # memorize scalefactors and background
        self._scales = coeffs[:, : self.important_components].reshape(1, -1)
        self._bg = bg_full

        return bg_full

    def transform(self, data, alpha=None):
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
            _ = self.fit(data=data)

        if alpha is None:
            coeffs = self._scales
        else:
            # can not determine scalefactor with binary search, unless the background is 2D
            self._bg = self._bg.sum(axis=0).reshape((-1, *img_shape))
            coeffs = self._scalefactors(arr=data, bg=self._bg, alpha=alpha)

        bg_scaled = np.dot(self._bg.T, coeffs).T

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

    def fit(self, data):
        """
        fit searches for the background profile using NMF decomposition
        - (N, M, M) image series --> (N, M**2) flattened images
        - (N, M**2) = (N, n_components) @ (n_components, M**2) NMF decomposition
        - background: (n_components, M**2) --> (important_components, M**2)

        Parameters
        ----------
        data : np.ndarray
            Input data (series of 2D images, 3D total)
        n_components : int, optional
            n_components for dimensionality reduction, by default 5
        important_components : int, optional
            number of components to account for, by default 1

        Returns
        -------
        np.ndarray
            Background profile
        """
        X = data.reshape(data.shape[0], -1)

        svd = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state,
            n_iter=self.n_iter,
        )

        svd.fit(X)
        coeffs = svd.transform(X)
        bg_full = svd.components_[: self.important_components, :].reshape(
            (-1, *data.shape[1:])
        )

        # memorize scalefactors and background
        self._scales = coeffs[:, : self.important_components].reshape(1, -1)
        self._bg = bg_full

        return bg_full

    def transform(self, data, alpha=None):
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
            # can not determine scalefactor with binary search, unless the background is 2D
            self._bg = self._bg.sum(axis=0).reshape((-1, *img_shape))
            coeffs = self._scalefactors(arr=data, bg=self._bg, alpha=alpha)
            self._scales = coeffs

        bg_scaled = np.dot(self._bg.T, coeffs).T

        return apply_mask(data, center=center, radius=radius) - bg_scaled


class PercentileDenoiser(AbstractDenoiser):
    def __init__(self, percentile=45, alpha=1e-2):
        super().__init__()
        self._percentile = percentile
        self._alpha = alpha

    def transform(self, data):
        """
        percentile_denoise applies percentile denoising:
        - create percentile-based background profille
        - apply mask
        - subtract background with such scale that less thatn `alpha` resulting pixels are negative

        Parameters
        ----------
        data : np.ndarray
            Input data (series of 2D images, 3D total)

        Returns
        -------
        np.ndarray
            Denoised images
        """

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


def _radial_profile(data, center, rmin=0, rmax=float(1e9), normalize=True):
    """
    _radial_profile returns radial profile of a 2D image

    Parameters
    ----------
    data : np.ndarray
        (M,N)-shaped np.ndarray
    center : tuple
        two float numbers as a center
    rmin : int
        minimal resolution
    rmax : int
        maximum resolutionk
    normalize : bool, optional
        whether to normalize images, by default True

    Returns
    -------
    np.ndarray
        1D numpy array
    """
    # taken from here: https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    if normalize:
        radialprofile = radialprofile / radialprofile.sum()

    return radialprofile[rmin:rmax]


def _stride_profile(data, stride=1):
    """
    _stride_profile returns 2D-stride of an image, reshaped into 1D0

    Parameters
    ----------
    data : np.ndarray
        (M,N)-shaped np.ndarray
    stride : int, optional
        take each nth pixel, by default 0

    Returns
    -------
    np.ndarray
        1D numpy array
    """
    rv = data[::stride, ::stride].flatten()
    return rv


def lst2profiles_ndarray(
    input_lst: str,
    func=_stride_profile,
    cxi_path="/entry_1/data_1/data",
    h5_path="/data/rawdata0",
    chunksize=100,
    **func_kwargs,
) -> np.ndarray:
    """
    lst2profiles_ndarray converts CrystFEL list into 2D np.ndarray with following structure:
    np.array([filename, event_name, *profile])

    Parameters
    ----------
    input_lst : str
        input list filename
    cxi_path : str, optional
        datapath inside cxi/h5 file, by default "/entry_1/data_1/data"
    chunksize : int, optional
        size of chunk for reading, by default 100

    Returns
    -------
    List
        [filename, *profile]
    """

    loader = ImageLoader(
        input_lst, cxi_path=cxi_path, h5_path=h5_path, chunksize=chunksize
    )

    profiles = []

    for lst, data in tqdm(loader, desc="Converting images to radial profiles"):

        for elem in zip(lst, data):
            profile = func(elem[1], **func_kwargs)
            profiles.append([elem[0], profile])

    return profiles


def ndarray2index(
    profiles_arr,
    metric="correlation",
    criterion="distance",
    threshold=5e-3,
):
    """
    ndarray2index returns linkage and index for a given ndarray of profiles

    Parameters
    ----------
    profiles_arr : np.ndarray
        input ndarray
    singletone_threshold : int, optional
        minimum number of images in cluster to be considered a singletone, by default 25
    metric : str, optional
        metric (see scipy.spatial.pdist), by default 'correlation'
    criterion : str, optional
        criterion (see scipy.cluster.hierarchy.fcluster), by default 'distance'
    threshold : float, optional
        distance threshold for a given metric, by default 0.01
    """

    Z = ward(pdist(profiles_arr, metric=metric))
    index = fcluster(Z, t=threshold, criterion=criterion)

    return index


def index2lists(
    index,
    names,
    output_prefix="cluster",
    singletone_threshold=25,
):
    """
    index2lists writes lists to disk

    Parameters
    ----------
    index : np.ndarray
        return value of scipy.cluster.hierarchy.fcluster
    names : list
        list with event lines in order as in index
    output_prefix : str, optional
        output list prefix, by default "cluster"
    singletone_threshold : int, optional
        minimum number of frames in index to be considered non-singletone, by default 25
    """
    clusters = [
        np.array(names)[np.where(index == cluster_idx)] for cluster_idx in set(index)
    ]

    singletone = np.hstack(
        [cluster for cluster in clusters if len(cluster) < singletone_threshold]
    )
    clusters = [elem for elem in clusters if len(elem) >= singletone_threshold]

    with open(f"{output_prefix}_singletone.lst", "w") as fout:
        for elem in singletone:
            print(elem, file=fout)

    for cluster_idx, cluster in enumerate(clusters):
        with open(f"{output_prefix}_{cluster_idx}.lst", "w") as fout:
            print(*cluster, sep="\n", end="", file=fout)
