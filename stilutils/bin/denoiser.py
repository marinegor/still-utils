#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import os
import sys
from collections import defaultdict
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
from tqdm import tqdm
from typing import Union

from stilutils.utils.imagereader import CXIReader, CBFReader, H5Reader, ImageLoader
from stilutils.utils.denoisers import (
    NMFDenoiser,
    PercentileDenoiser,
    SVDDenoiser,
    _radial_profile,
    cluster_ndarray,
)


def lst2profiles_ndarray(
    input_lst: str,
    center,
    cxi_path="/entry_1/data_1/data",
    h5_path="/data/rawdata0",
    chunksize=100,
    radius=45,
) -> np.ndarray:
    """
    lst2profiles_ndarray converts CrystFEL list into 2D np.ndarray with following structure:
    np.array([filename, event_name, *profile])

    Parameters
    ----------
    input_lst : str
        input list filename
    center : [type]
        tuple of (center_x, center_y)
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
            profile = _radial_profile(elem[1], center=center)
            profiles.append([elem[0], profile])

    return profiles


def denoise_lst(
    input_lst: str,
    alpha=1e-2,
    center=None,
    radius=None,
    denoiser_type="nmf",
    cxi_path="/entry_1/data_1/data",
    h5_path="/data/rawdata0",
    output_cxi_prefix=None,
    output_lst=None,
    compression="gzip",
    chunks=True,
    chunksize=100,
    zero_negative=True,
    dtype=np.int16,
    **denoiser_kwargs,
) -> None:
    """
    denoise_lst applies denoiser to a list

    Parameters
    ----------
    input_lst : str
        input list in CrystFEL format
    denoiser_type : str, optional
        denoiser type, by default "nmf"
    cxi_path : str, optional
        path inside a cxi file, by default "/entry_1/data_1/data"
    output_cxi_prefix : [type], optional
        prefix for output cxi files, by default None
    output_lst : [type], optional
        output list filename, by default None
    compression : str, optional
        which losless compression to use, by default "gzip"
    chunks : bool, optional
        whether to output in chunks (saves RAM), by default True
    chunksize : int, optional
        chunksize for reading, by default 100
    zero_negative : bool, optional
        whether to convert negative values to 0, by default True

    Raises
    ------
    TypeError
        If denoiser is not in ('percentile', 'nmf','svd')
    """

    if denoiser_type == "percentile":
        denoiser = PercentileDenoiser(**denoiser_kwargs)
    elif denoiser_type == "nmf":
        denoiser = NMFDenoiser(**denoiser_kwargs)
    elif denoiser_type == "svd":
        denoiser = SVDDenoiser(**denoiser_kwargs)
    else:
        raise TypeError("Must provide correct denoiser")

    if output_cxi_prefix is None:
        output_cxi_prefix = ""

    loader = ImageLoader(
        input_lst, cxi_path=cxi_path, h5_path=h5_path, chunksize=chunksize
    )

    chunk_idx = 0

    for lst, data in loader:
        new_data = denoiser.transform(
            data, center=center, radius=radius, alpha=alpha, **denoiser_kwargs
        ).astype(dtype)

        if zero_negative:
            new_data[new_data < 0] = 0

        output_cxi = f'{output_cxi_prefix}_{input_lst.rsplit(".")[0]}_{chunk_idx}.cxi'
        shape = data.shape

        with h5py.File(output_cxi, "w") as h5fout:
            h5fout.create_dataset(
                cxi_path, shape, compression=None, data=new_data, chunks=chunks
            )
        chunk_idx += 1


def main(args):
    """The main function"""

    parser = argparse.ArgumentParser(
        description="Provides interface for image denoising",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(help="Mode selection", dest="mode")

    # adding subparsers for different modes

    clustering_parser = subparsers.add_parser("cluster", help="Clustering mode")
    clustering_parser.add_argument(
        "--min_num_images",
        type=int,
        help="Chunks less then that size will be sent to singletone",
        default=50,
    )
    clustering_parser.add_argument(
        "--clustering_distance",
        type=float,
        help="Clustering distance threshold",
        default=25.0,
    )

    clustering_parser.add_argument(
        "--criterion",
        type=str,
        help="Clustering criterion (google `scipy.fcluster`)",
        default="maxclust",
    )

    clustering_parser.add_argument(
        "--method",
        type=str,
        help="Clustering method",
        choices=["cc", "radial"],
        default="cc",
    )
    clustering_parser.add_argument(
        "--center",
        nargs=2,
        metavar=("center_x", "center_y"),
        type=float,
        help="Detector center",
    )
    clustering_parser.add_argument(
        "--reslim",
        nargs=2,
        metavar=("min_res", "max_res"),
        help="Minimum and maximum resolution for radial profile (in pixels)",
        type=float,
        default=(0, float("inf")),
    )
    clustering_parser.add_argument(
        "--stride",
        type=int,
        help="Stride (equal on fs and ss directions) to boost correlation calculation",
        default=0,
    )

    # ----------------------------------------
    extracting_parser = subparsers.add_parser(
        "extract", help="Background extraction mode"
    )
    DENOISERS = {
        "percentile": PercentileDenoiser,
        "nmf": NMFDenoiser,
        "svd": SVDDenoiser,
    }
    extracting_parser.add_argument(
        "--method", choices=list(DENOISERS.keys()), help="Type of denoiser to use"
    )
    extracting_parser.add_argument(
        "--output", type=str, default="", help="Output file prefix"
    )
    extracting_parser.add_argument(
        "--quantile",
        type=float,
        default=45,
        help="Quantile for percentile denoiser",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "uint16", "int32"],
        default="float32",
        help="Output numeric type",
    )

    # ----------------------------------------
    subtracting_parser = subparsers.add_parser(
        "subtract", help="Background subtraction mode"
    )
    subtracting_parser.add_argument(
        "--background", type=str, help="Background profile image"
    )
    subtracting_parser.add_argument(
        "--alpha",
        type=float,
        help="Share of negative pixels in resulting image",
        default=5e-3,
    )
    subtracting_parser.add_argument("--mask", action="store_true", default=False)
    subtracting_parser.add_argument(
        "--center",
        nargs=2,
        metavar=("center_x", "center_y"),
        type=float,
        help="Detector center",
    )
    subtracting_parser.add_argument(
        "--radius", type=float, help="Mask radius", default=0
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "uint16", "int32"],
        default="uint16",
        help="Output numeric type",
    )

    # add common arguments
    parser.add_argument(
        "input_lst",
        type=str,
        help="Input list in CrystFEL format (might be with or without events, or even combined)",
    )
    parser.add_argument(
        "--datapath", type=str, help="Path to your data inside a cxi file", default=None
    )
    parser.add_argument(
        "--store_negative_values",
        action="store_false",
        help="Whether to save negative values",
        default=True,
    )

    # parser.add_argument(
    # "--center", type=str, help="Center position", default="719.9 711.5"
    # )
    # parser.add_argument(
    # "--chunksize",
    # type=int,
    # help="Defines size of a single denoising chunk",
    # default=100,
    # )

    args = parser.parse_args()
    print(args)

    # compabibility checks
    if args.mode == "cluster":
        if args.method == "radial":
            assert hasattr(
                args, "center"
            ), f"Clustering mode is {args.method} but you have not provided center argument"
    elif args.mode == "extract":
        if args.method == "percentile":
            assert (
                hasattr(args, "quantile") and 0 < args.quantile < 100
            ), f"Percentile requires you provide a quantile {args.quantile}, and it must be 0 < {args.quantile} < 100"
    elif args.mode == "subtract":
        assert os.path.exists(
            args.background
        ), f"File you provided does not exist: {args.background}"
        if args.mask:
            assert hasattr(
                args, "center"
            ), "Should provide --center if you want to use mask"
            assert hasattr(
                args, "radius"
            ), "Should provide --radius if you want to use mask"
        if args.store_negative_values:
            assert (
                args.dtype != "uint16"
            ), f"You want to store negative values but have provided wrong dtype: {args.dtype}"

    # Main execution selector
    if args.mode == "cluster":
        ...
    elif args.mode == "extract":
        ...
    elif args.mode == "subtract":
        ...


if __name__ == "__main__":
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        main(sys.argv[1:])
