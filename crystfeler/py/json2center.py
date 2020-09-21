#!/usr/bin/env python3

import numpy as np
import json
import sys
import argparse
from tqdm import tqdm
from typing import List, Tuple
from itertools import combinations
import matplotlib.pyplot as plt
import warnings


def moving_average(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Return moving average of 1D array in numpy

    Arguments:
        arr {np.ndarray} -- input array
        window_size {int} -- window size

    Returns:
        ret {np.ndarray} -- smoothened array
    """
    ret = np.cumsum(arr, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1 :] / window_size


def rms_mask(arr: np.ndarray, window_size: int, percentile=0.1) -> np.ndarray:
    """
    Return mask where arr's deviation is smaller than it's mean*percentile:
    RMSD[arr]_window_size < MEAN[arr]*percentile

    Arguments:
        arr {np.ndarray} -- input array
        window_size {int} -- window size

    Returns:
        np.ndarray -- boolean mask
    """
    arr_mean = moving_average(arr, window_size)
    arr_rmsd = np.power(moving_average(np.power(arr, 2), window_size), 0.5)
    return np.where(np.abs(arr_rmsd) < np.abs(arr_mean) * percentile)


def radial_binning(
    fs: np.ndarray, ss: np.ndarray, rs: np.ndarray, N=1000
) -> np.ndarray:
    """
    Returns binned by radius table

    Arguments:
        fs {np.ndarray} -- np.ndarray of x values
        ss {np.ndarray} -- np.ndarray of y values
        rs {np.ndarray} -- np.ndarray of r values

    Keyword Arguments:
        N {int} -- number of bins (default: {1000})

    Returns:
        answ {np.ndarray} -- (len(fs), 4) shape: rmean, num, fsmean, ssmean
    """
    rmin, rmax = rs.min(), rs.max()
    step = (rmax - rmin) / N
    answ = []
    for rcur, _ in tqdm(
        enumerate(np.linspace(rmin, rmax, N)), desc="Binning values", total=N
    ):
        mask = (rs < rcur + step) & (rs >= rcur)
        if sum(mask) > 0:
            rmean = rs[mask].mean()
            num = mask.sum()
            fsmean = fs[mask].mean()
            ssmean = ss[mask].mean()
            answ.append([rmean, num, fsmean, ssmean])
    answ = np.array(answ)
    return answ


def ang(v1, v2):
    """
    Returns angle between two vectors
    """
    return np.abs(np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))


def is_inside(triangle: np.ndarray, point: np.ndarray) -> bool:
    """
    Checks whether point is inside a triangle

    Arguments:
        triangle {np.pdarray} -- triangle coordinates (3,2) shape
        point {np.ndarray} -- point to check (x,y)

    Returns:
        bool -- check value
    """

    a, b, c = triangle
    oa, ob, oc = a - point, b - point, c - point
    return (
        np.abs(
            sum([ang(v1, v2) for v1, v2 in combinations([oa, ob, oc], 2)]) - 2 * np.pi
        )
        < 1e-2
    )


def get_center_position_from_binning(
    binning: np.ndarray, rmin=None, rmax=None
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Estimate center position from binning results

    Arguments:
        binning {np.ndarray} -- input (N,4) shape array

    Keyword Arguments:
        rmin {[type]} -- lower radius threshold (default: {None})
        rmax {[type]} -- upper radius threshold (default: {None})

    Returns:
        Tuple[Tuple[float, float], Tuple[float, float]] -- (fs, ss) tuple
    """
    rs, num, fs, ss = binning.T

    if rmin is None:
        rmin = float("-inf")
    if rmax is None:
        rmax = float("inf")

    mask = (rs >= rmin) & (rs < rmax)
    fs_val, fs_std = fs[mask].mean(), fs[mask].std()
    ss_val, ss_std = ss[mask].mean(), ss[mask].std()

    return (fs_val, fs_std), (ss_val, ss_std)


def circle(points: np.ndarray, acute_angle=True, presumable_centre=None) -> np.ndarray:
    """
    Returns coordinates and radius of circumscribed circle for 3 points.

    Arguments:
        points {np.ndarray} -- (3,2)-shaped array

    Keyword Arguments:
        acute_angle {bool} -- whether the points should be an acute-anbled triangle (default: {True})
        presumable_centre {np.ndarr} -- approximate centre position to check whether it's inside triangle of points

    Returns:
        np.ndarray -- (x,y,R)
    """
    x1, y1, x2, y2, x3, y3 = points.reshape(-1)

    A = np.array([[x3 - x1, y3 - y1], [x3 - x2, y3 - y2]])
    Y = np.array(
        [
            (x3 ** 2 + y3 ** 2 - x1 ** 2 - y1 ** 2),
            (x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2),
        ]
    )

    if acute_angle:
        if not is_inside(points, points.mean(axis=0)):
            return None

    if presumable_centre is not None:
        if not is_inside(points, presumable_centre):
            return None

    if np.abs(np.linalg.det(A)) < 1e-3:
        return None

    Ainv = np.linalg.inv(A)
    X = 0.5 * np.dot(Ainv, Y)
    x, y = X[0], X[1]
    r = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    return (x, y, r)


def main(args: List[str]):
    """
    The main function

    Arguments:
        args {List[str]} -- arguments
    """

    parser = argparse.ArgumentParser(
        description="Detector center search, based on random selection of circumscribed circle centers"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of times algorithm will try to guess center choosing 3 random points",
    )
    parser.add_argument(
        "input_json", type=str, help="input json file, produced by streampeaks2json.py"
    )
    parser.add_argument(
        "--rmin", type=int, default=None, help="Minimum radius for center estimation"
    )
    parser.add_argument(
        "--rmax", type=int, default=None, help="Maximum radius for center estimation"
    )
    parser.add_argument(
        "--center_fs",
        type=int,
        default=None,
        help="Presumable center fs for acute angle rejection",
    )
    parser.add_argument(
        "--center_ss",
        type=int,
        default=None,
        help="Presumable center ss for acute angle rejection",
    )
    parser.add_argument(
        "--plot", help="Whether to save plots or not", action="store_true"
    )
    parser.add_argument(
        "--nbins", type=int, default=1000, help="Number of bins for radial binning"
    )

    args = parser.parse_args()

    with open(args.input_json) as f:
        peaks = json.load(f)
    number_of_panels = len(set([elem["panel"] for elem in peaks.values()]))
    assert number_of_panels == 1, f"Wrong number of panels: {number_of_panels}"

    points = np.array([(elem["fs"], elem["ss"]) for elem in peaks.values()])

    if args.center_fs is not None and args.center_ss is not None:
        presumable_centre = np.array([args.center_fs, args.center_ss])
    else:
        presumable_centre = None
    answ = []

    for _ in tqdm(range(args.iterations), desc="Sampling points"):
        idx = np.random.randint(points.shape[0], size=3)
        T = points[idx]
        cur = circle(T, acute_angle=True, presumable_centre=presumable_centre)
        if cur is not None:
            answ.append(cur)

    fs, ss, rs = np.array(answ).T
    bins = radial_binning(fs, ss, rs, N=args.nbins)
    if args.plot:
        plt.plot(bins.T[0], bins.T[2], label="fs")
        plt.plot(bins.T[0], bins.T[3], label="ss")
        plt.xlim(args.rmin, args.rmax)
        plt.savefig(f"{args.input_json}_plot.png", dpi=300)
        print(f"Saved {args.input_json}_plot.png")

    (fs, fs_s), (ss, ss_s) = get_center_position_from_binning(
        bins, rmin=args.rmin, rmax=args.rmax
    )
    print(f"fs: {fs} +- {fs_s}; ss: {ss} +- {ss_s}")


if __name__ == "__main__":
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    main(sys.argv[1:])
