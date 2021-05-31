#!/usr/bin/env python3

import argparse
import numpy as np
from functools import partial
from scipy.spatial.distance import pdist, squareform

from stilutils.utils.imagereader import ImageLoader
from stilutils.utils.denoisers import _radial_profile, lst2profiles_ndarray

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Analyzes quality of clustering, performed by `denoiser.py cluster` step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "lists",
        metavar="INPUT LISTS",
        nargs="+",
        type=str,
        help="Input files (can be a mask, e.g. `cluster_*.lst`)",
    )
    parser.add_argument(
        "--datapath", default="/entry_1/data_1/data", type=str, nargs=1, help=""
    )
    parser.add_argument(
        "--center",
        metavar=("corner_x", "corner_y"),
        nargs=2,
        type=float,
        help="Coordinates of the detector center",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Whether to normalize the radial profiles",
        default=False,
    )
    parser.add_argument(
        "--reslim",
        metavar=("min_res", "max_res"),
        nargs=2,
        default=[0, int(1e10)],
        help="Resolution limit (in puxels) for radial profile",
    )

    args = parser.parse_args()

    func_ = partial(
        _radial_profile,
        normalize=args.normalize,
        center=args.center,
        rmin=args.reslim[0],
        rmax=args.reslim[1],
    )
    for fle in args.lists:
        profiles_1d = lst2profiles_ndarray(
            fle, cxi_path=args.datapath, h5_path=args.datapath, func=func_
        )
        names, profiles = (
            [elem[0] for elem in profiles_1d],
            np.array([elem[1] for elem in profiles_1d]),
        )

        pd = pdist(profiles)
        sf = squareform(pd)

        print("List\tmin\tmax\tmean")
        print(
            f"{fle}, min: {sf[sf>0].min():.2f}, max: {sf.max():.2f}, mean: {sf.mean():.2f}"
        )
        print("-" * 80)
