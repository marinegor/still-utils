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

from stilutils.utils.imagereader import ImageLoader
from stilutils.utils.denoisers import (
    NMFDenoiser,
    PercentileDenoiser,
    SVDDenoiser,
    _radial_profile,
    _stride_profile,
    lst2profiles_ndarray,
    ndarray2index,
    index2lists,
    apply_mask,
)


def main(args):
    """The main function"""

    parser = argparse.ArgumentParser(
        description="Provides interface for image denoising",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    subparsers = parser.add_subparsers(help="Mode selection", dest="mode")

    # adding subparsers for different modes
    clustering_parser = subparsers.add_parser("cluster", help="Clustering mode")
    clustering_parser.add_argument(
        "--singletone_threshold",
        type=int,
        help="Clusters less then that size will be sent to a single list",
        default=50,
    )
    clustering_parser.add_argument(
        "--clustering_distance",
        type=float,
        help="Clustering distance threshold",
        default=5e-3,
    )

    clustering_parser.add_argument(
        "--criterion",
        type=str,
        help="Clustering cutoff selection (see `scipy.fcluster`)",
        default="distance",
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
        "--metric",
        type=str,
        default="correlation",
        help="Metric to compare 1D-profiles of images between each other (see scipy.spacial.distance.pdist for full list of choices",
    )
    clustering_parser.add_argument(
        "--reslim",
        nargs=2,
        metavar=("min_res", "max_res"),
        help="Minimum and maximum resolution for radial profile (in pixels)",
        type=int,
        default=(0, int(1e10)),
    )
    clustering_parser.add_argument(
        "--stride",
        type=int,
        help="Stride (equal on fs and ss directions) to boost correlation calculation",
        default=0,
    )
    clustering_parser.add_argument(
        "--output_prefix", type=str, default="cluster", help="Output list file prefix"
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
        "--quantile", type=float, default=45, help="Quantile for percentile denoiser",
    )
    extracting_parser.add_argument(
        "--chunksize",
        type=int,
        default=100,
        help="Number of images in chunk for denoising",
    )
    extracting_parser.add_argument(
        "--bg_dtype",
        choices=["float32", "uint16", "int32"],
        default="float32",
        help="Output numeric type",
    )

    extracting_parser.add_argument(
        "--alpha",
        type=float,
        help="Share of negative pixels in resulting image",
        default=5e-3,
    )
    extracting_parser.add_argument("--mask", action="store_true", default=False)
    extracting_parser.add_argument(
        "--center",
        nargs=2,
        metavar=("center_x", "center_y"),
        type=float,
        help="Detector center",
    )
    extracting_parser.add_argument(
        "--radius", type=float, help="Mask radius", default=0
    )
    parser.add_argument(
        "--subtract_dtype",
        choices=["float32", "uint16", "int32"],
        default="uint16",
        help="Output numeric type",
    )
    extracting_parser.add_argument(
        "--patch_size", nargs=2, type=int, default=(None, None), help="Patch size"
    )

    # ---------------------------------
    args = parser.parse_args()
    print(args)

    # compabibility checks
    if args.mode == "cluster":
        if args.method == "radial":
            assert hasattr(
                args, "center"
            ), f"Clustering mode is {args.method} but you have not provided center argument"

        if args.method == "radial":
            func_ = _radial_profile
            func_kw = {
                "rmin": args.reslim[0],
                "rmax": args.reslim[1],
                "center": args.center,
            }
        elif args.method == "cc":
            func_ = _stride_profile
            func_kw = {"stride": args.stride}
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
                args.extract_dtype != "uint16"
            ), f"You want to store negative values but have provided wrong dtype: {args.dtype}"

    # Main execution selector
    if args.mode == "cluster":
        profiles_1d = lst2profiles_ndarray(
            args.input_lst,
            func=func_,
            cxi_path=args.datapath,
            h5_path=args.datapath,
            **func_kw,
        )
        names, profiles = (
            [elem[0] for elem in profiles_1d],
            [elem[1] for elem in profiles_1d],
        )

        index = ndarray2index(
            profiles,
            metric=args.metric,
            criterion=args.criterion,
            threshold=args.clustering_distance,
        )
        index2lists(
            index,
            names,
            output_prefix=args.output_prefix,
            singletone_threshold=args.singletone_threshold,
        )
    elif args.mode == "extract":
        loader = ImageLoader(
            input_list=args.input_lst,
            cxi_path=args.datapath,
            h5_path=args.datapath,
            chunksize=args.chunksize,
        )

        for chunk_idx, chunk in enumerate(loader):
            names, frames = chunk
            if args.method == "percentile":
                denoiser = DENOISERS[args.method](percentile=args.quantile)
            else:
                denoiser = DENOISERS[args.method]()

            if args.mask:
                frames = apply_mask(frames, center=args.center, radius=args.radius)

            px, py = args.patch_size
            if px is not None and py is not None:
                nx, ny = frames.shape[1:]
                rv = np.zeros(frames.shape)
                slices = [
                    (slice(ix, ix + px), slice(iy, iy + py))
                    for ix in range(0, nx, px)
                    for iy in range(0, ny, py)
                ]

                def denoise_batch(s):
                    sx, sy = s
    
                    patch = frames[:, sx, sy]
                    try:
                        patch_d = denoiser.transform(patch, alpha=args.alpha)
                    except:
                        patch_d = patch
    
                    return sx, sy, patch_d
                
                rv = np.zeros(frames.shape)
                answ = map(denoise_batch, slices)
                for idx, (sx, sy, patch_d) in enumerate(answ):
                    print(f"{100*idx/len(slices):2.2f}%", end='\r')
                    rv[:,sx, sy] += patch_d
                denoised = rv


            else:
                bg = denoiser.fit(frames)

                # write background profile to a file
                fout_bg = f"{args.input_lst.strip('.lst')}_bg.h5"
                with h5py.File(fout_bg, mode="w") as bg_fout:
                    bg_fout.create_dataset(
                        args.datapath, data=bg,
                    )

                # perform the subtraction itself
                denoised = denoiser.transform(frames, alpha=args.alpha)

            # do type conversion & set negative pixels to 0
            denoised = denoised.astype(args.subtract_dtype)
            if not args.store_negative_values:
                denoised[denoised < 0] = 0

            # write denoised data
            output_cxi = f"{args.method}_{args.output}_{chunk_idx}.cxi"
            with h5py.File(output_cxi, mode="w") as fout:
                fout.create_dataset(args.datapath, data=denoised)


if __name__ == "__main__":
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        main(sys.argv[1:])
