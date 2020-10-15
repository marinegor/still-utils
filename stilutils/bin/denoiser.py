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
        description="Denoises images using simple percentile filter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_lst",
        type=str,
        help="Input list in CrystFEL format (might be with or without events, or even combined)",
    )
    parser.add_argument(
        "--datapath", type=str, help="Path to your data inside a cxi file"
    )
    parser.add_argument(
        "--center", type=str, help="Center position", default="719.9 711.5"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        help="Defines size of a single denoising chunk",
        default=100,
    )
    parser.add_argument(
        "--min_num_images",
        type=int,
        help="Chunks less then that size will be sent to singletone",
        default=50,
    )
    parser.add_argument(
        "--zero_negative",
        type=int,
        help="Whether to zero out negative pixels or not",
        default=True,
    )
    parser.add_argument(
        "--out_prefix", type=str, help="Prefix for output cxi files", default="denoised"
    )
    parser.add_argument("--percentile", type=int, help="Percentile to use", default=45)
    parser.add_argument(
        "--alpha",
        type=float,
        help="Threshold for scale factor determination",
        default=0.01,
    )
    parser.add_argument(
        "--radius", type=int, default=45, help="Radius to apply mask in the center"
    )

    list_of_denoisers = ["percentile", "nmf", "svd"]
    parser.add_argument(
        "--denoiser", choices=list_of_denoisers, help="Type of denoiser to use"
    )
    parser.add_argument(
        "--output_lst_prefix",
        type=str,
        help="List prefix for lists after clustering",
        default="clustered",
    )
    parser.add_argument(
        "--clustering_distance",
        type=float,
        help="Clustering distance threshold",
        default=25.0,
    )

    parser.add_argument(
        "--criterion",
        type=str,
        help="Clustering criterion (google `scipy.fcluster`)",
        default="maxclust",
    )

    args = parser.parse_args()

    center = map(float, args.center.split())
    center = list(center)

    profiles_ndarray = lst2profiles_ndarray(
        args.input_lst,
        center=center,
        cxi_path=args.datapath,
        h5_path=args.datapath,
        chunksize=args.chunksize,
    )
    print("Clustering images using their radial profiles", file=sys.stderr)
    image_lists_list = cluster_ndarray(
        profiles_ndarray,
        output_prefix=args.output_lst_prefix,
        output_lists=True,
        threshold=args.clustering_distance,
        criterion=args.criterion,
        min_num_images=args.min_num_images,
    )

    for list_to_denoise in tqdm(image_lists_list, desc="Denoising each clustered list"):
        denoise_lst(
            input_lst=list_to_denoise,
            alpha=args.alpha,
            center=center,
            denoiser_type=args.denoiser,
            cxi_path=args.datapath,
            h5_path=args.datapath,
            output_cxi_prefix=args.out_prefix,
            chunksize=args.chunksize,
            zero_negative=args.zero_negative,
            radius=args.radius,
        )


if __name__ == "__main__":
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        main(sys.argv[1:])
