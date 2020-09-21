#!/usr/bin/env python3

import argparse
from itertools import chain, product
from typing import Dict

import json
import h5py
import numpy as np
from tqdm import tqdm
import sys


def _peaks_from_ndarray(
    nparr: np.ndarray, size=1
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns positions of peaks -- pixels that are larger than any of their neighbours within a certain radius

    Arguments:
        ndarr {np.ndarray} -- input array. Last two coordinates are treated as `x` and `y`.
        size {int} -- size of square of neighbours will be (2n+1)

    Returns:
        [np.ndarr, np.ndarr, np.ndarr] -- return value of np.nonzero. Shape corresponds to input array shape
    """
    ...
    ret = np.zeros(nparr.shape).astype(np.int8)

    shifts_lst = product(range(-size, size + 1), repeat=2)
    num_neighbours = (2 * size + 1) ** 2 - 2

    for roll_combination in shifts_lst:
        ret += (np.roll(nparr, roll_combination, axis=(-2, -1)) < nparr).astype(np.int8)

    return np.nonzero(ret - num_neighbours > 0)


def peak_profiles(nparray: np.ndarray, size_compare=1, size_return=3) -> list:
    """
    Returns peak profiles as (n, x, y) tensor, with n=nparray.shape[0], and x=y=2*size+1

    Arguments:
        nparray {np.ndarray} -- input numpy array with full images

    Returns:
        dict -- peak profiles
    """

    pks = _peaks_from_ndarray(nparray, size=size_compare)
    answ = {}

    if len(nparray.shape) == 3:
        for n, x, y in zip(*pks):
            profile = nparray[n][
                x - size_return : x + size_return + 1,
                y - size_return : y + size_return + 1,
            ]
            answ[(x, y)] = profile
    else:
        for x, y in zip(*pks):
            profile = nparray[
                x - size_return : x + size_return + 1,
                y - size_return : y + size_return + 1,
            ]
            answ[(x, y)] = profile

    return answ


def extract_hits(
    cxi_filename: str,
    indices: list,
    cxi_path="entry_1/data_1/data",
    size_compare=1,
    size_return=5,
) -> Dict:
    """
    From single cxi file and list of indices within it (which are usually "Event" field in crystfel stream), returns np.ndarray representing each hit image

    Arguments:
        cxi_filename {str} -- input cxi filename
        cxi_path {str} -- path to data in hierarchical cxi structure
        indices {list} -- indices of hit images along time coordinate
        size_compare {int} -- peaks should be larger or equal to all peaks within (-size,+size) square
        size_return {int} -- will return (2*size_return-1)**2 sized square with peak in center

    Returns:
        Dict -- dictionary of f'{image_filename|event : {'x':x, 'y':y, 'profile':list}}
    """

    hits = {}
    answ = {}

    with h5py.File(cxi_filename, "r") as f:
        for idx in indices:
            current_key = (cxi_filename, idx)
            current_nparray = np.array(f[cxi_path][idx])
            hits[current_key] = current_nparray

            peaks = peak_profiles(
                current_nparray, size_compare=size_compare, size_return=size_return
            )

            size = 2 * size_return + 1
            for (x, y), profile in tqdm(peaks.items()):
                if profile.shape == (size, size):
                    joined_key = "|".join([cxi_filename, str(idx)])
                    single_peak = {
                        "x": str(x),
                        "y": str(y),
                        "profile": [int(i) for i in profile.flatten()],
                    }
                    if joined_key in answ:
                        answ[joined_key].append(single_peak)
                    else:
                        answ[joined_key] = [single_peak]

    return answ


def main(args):
    """
    The main function
    """

    parser = argparse.ArgumentParser(
        description="Extracts peaks (=larger-than-their-neighbours pixels) from cxi file"
    )

    parser.add_argument(
        "--cxi_path",
        type=str,
        default="entry_1/data_1/data",
        help="path to actual data within a cxi filename",
    )
    parser.add_argument(
        "input_lst",
        type=str,
        help="Input .lst filename, containing both filenames and event numbers",
    )
    parser.add_argument(
        "--size_compare",
        type=int,
        default=1,
        help="peaks should be larger or equal to all peaks within (-size,+size) square",
    )
    parser.add_argument(
        "--size_return",
        type=int,
        default=5,
        help="will return (2*size_return-1)**2 sized square with peak in center",
    )
    parser.add_argument("--fout", type=str, help="Output file")
    args = parser.parse_args()

    filenames = {}  # dict with {cxi_filename:{events_set}}
    with open(f"{args.input_lst}", mode="r") as fin:
        for line in fin:
            filename, eventnum = line.split(r"//")
            filename = filename.replace(" ", "")
            eventnum = int(eventnum)
            #  print(filename, eventnum)
            if filename in filenames:
                filenames[filename].add(eventnum)
            else:
                filenames[filename] = {eventnum}

    ret = {}
    for filename, events_list in tqdm(filenames.items()):
        #  print(filename, events_list)
        current_peaks = extract_hits(
            cxi_filename=filename,
            indices=events_list,
            cxi_path=args.cxi_path,
            size_compare=args.size_compare,
            size_return=args.size_return,
        )
        ret.update(current_peaks)

    #  print(ret)
    with open(args.fout, mode="w") as fout:
        json.dump(ret, fout, indent=4)


if __name__ == "__main__":
    main(sys.argv[1:])
