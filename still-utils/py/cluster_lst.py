#!/usr/bin/env python3

import argparse
import numpy as np
import h5py
import sys
from tqdm import tqdm
import os
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist


def lst2dict(
    input_lst: str, center, cxi_path="/entry_1/data_1/data", chunksize=100
) -> np.ndarray:

    events_dict = {}
    with open(input_lst, "r") as fin:
        for line in fin:
            if "//" in line:  # if line represents an event in cxi
                filename, event = line.split(" //")
                event = int(event)
                if filename[0] != "/":  # if path is not absolute
                    filename = f"{os.getcwd()}/{filename}"
                if filename in events_dict:
                    if None not in events_dict[filename]:
                        events_dict[filename].add(event)
                    else:
                        pass
                else:
                    events_dict[filename] = {event}
            else:  # if line represents both event and its absence
                filename = line.rstrip()
                if filename[0] != "/":  # if path is not absolute
                    filename = os.getcwd() + filename
                events_dict[line] = {None}

    cxi_cnt = 0

    profiles_dict = {}
    for input_cxi, events in tqdm(
        events_dict.items(), desc=f"Processing files in {input_lst} one by one"
    ):
        cxi_cnt += 1

        # create events list
        if None in events:
            with h5py.File(input_cxi, "r") as fin:
                cxi_fist_dimension = fin.shape[0]
                events = list(range(cxi_fist_dimension))
        else:
            events = sorted(list(events))

        num_images = len(events)

        event_profile_pairs = []
        with h5py.File(input_cxi, "r") as h5fin:
            data = h5fin[cxi_path]

            for chunk_start in tqdm(
                range(0, num_images, chunksize), desc="Reading images in chunks"
            ):
                start, stop = chunk_start, chunk_start + chunksize

                events_idx = np.array(events)[start:stop]
                current_data = data[events_idx]
                current_data = apply_mask(current_data, center=center)

                for event, image in zip(events_idx, current_data):
                    profile = radial_profile(image, center=center)
                    event_profile_pairs.append([event, profile])

        profiles_dict[input_cxi] = event_profile_pairs

    ret = []
    for cxi_name, arr in tqdm(
        profiles_dict.items(), desc="Iterating over profiles_dict"
    ):
        for event, profile in tqdm(
            arr, desc="Iterate over each single cxi image filename"
        ):
            ret.append(np.array([cxi_name, event, profile]))
    ret = np.array(ret)
    return ret


def dict2clustered_lists(
    profiles_arr: np.ndarray,
    output_prefix="clustered",
    threshold=25,
    criterion="maxclust",
    min_num_images=50,
):
    profiles = np.array([elem[2] for elem in profiles_arr])
    names_and_events = profiles_arr[:, :2]

    Z = ward(pdist(profiles))
    idx = fcluster(Z, t=threshold, criterion=criterion)

    # output lists
    for list_idx in tqdm(list(set(idx)), desc="Output lists"):
        belong_to_this_idx = np.where(idx == list_idx)[0]
        if len(belong_to_this_idx) < min_num_images:
            fout_name = f"{output_prefix}_singletone.lst"
        else:
            fout_name = f"{output_prefix}_{list_idx}.lst"
        with open(fout_name, "a") as fout:
            for line in names_and_events[belong_to_this_idx]:
                cxi_name, event = line
                print(f"{cxi_name} //{event}", file=fout)


def radial_profile(data, center, normalize=True):
    # taken from here: https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    if normalize:
        radialprofile = radialprofile / radialprofile.sum()

    return radialprofile


def apply_mask(np_arr, center, r=45):

    shape = np_arr.shape[1:]
    mask = np.ones(shape)

    rx, ry = map(int, center)
    for x in range(rx - r, rx + r):
        for y in range(ry - r, ry + r):
            if (x - rx) ** 2 + (y - ry) ** 2 <= r ** 2:
                mask[x][y] = 0

    mask = mask.reshape((*shape, 1))
    return np_arr * mask.reshape(1, *shape)


def main(args):
    """The main function"""

    parser = argparse.ArgumentParser(
        description="Splits images from list into separate lists clusterd by their radial profile similarities",
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
    parser.add_argument("--center", type=str, help="Center position")
    parser.add_argument(
        "--threshold", type=float, help="Clustering threshold", default=25
    )
    parser.add_argument(
        "--criterion", type=str, help="Clustering criterion", default="maxclust"
    )
    parser.add_argument(
        "--min_num_images",
        type=int,
        help="Minimum number of images in list; all below this will be thrown in single file",
        default=50,
    )
    parser.add_argument(
        "--output_prefix", type=str, help="Output prefix", default="clustered"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        help="Defines size of a single denoising chunk",
        default=100,
    )

    args = parser.parse_args()

    center = map(float, args.center.split())
    center = list(center)
    profiles_for_clustering = lst2dict(
        args.input_lst, center=center, cxi_path=args.datapath, chunksize=args.chunksize
    )
    dict2clustered_lists(
        profiles_for_clustering,
        output_prefix=args.output_prefix,
        threshold=args.threshold,
        criterion=args.criterion,
        min_num_images=args.min_num_images,
    )


if __name__ == "__main__":
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        main(sys.argv[1:])
