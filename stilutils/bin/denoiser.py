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

from stilutils.utils.imagereader import CXIReader, CBFReader, H5Reader
from stilutils.utils.denoisers import NMFDenoiser, PercentileDenoiser, SVDDenoiser


class ImageLoader:
    """
    ImageLoader class that abstracts the image loading process,
    loading images one by one and returning a handle to write them
    """

    def __init__(self, input_list, chunksize=100, cxi_path=None, h5_path=None):
        self.input_list = input_list
        self.chunksize = chunksize

        # initialize chain of image readers
        self.cxi_reader = CXIReader(path_to_data=cxi_path)
        self.cbf_reader = CBFReader()
        self.h5_reader = H5Reader(path_to_data=h5_path)
        self.cxi_reader.next_reader(self.cbf_reader).next_reader(self.h5_reader)
        self.image_reader = self.cxi_reader

        # load all frames from input list
        data = set()
        with open(input_list, mode="r") as fin:
            for line in fin:
                if line.rstrip().endswith(".cxi"):
                    num_events = self.cxi_reader.get_events_number(line.rstrip())
                    for i in range(num_events):
                        data.add(line.rstrip() + " //" + str(i))
                else:
                    data.add(line.rstrip())

        self._data = list(data)

    def __iter__(self):
        return self

    def __next__(self, mode="r"):
        """Here the magic happens that helps to iterate"""
        """
        Pseudocode:

        data_to_return, handles_to_return = self.data[:chunksize]
        self.data = self.data - data_to_return
        return data_to_return, handles_to_return
        """
        current_chunk_list = self._data[: self.chunksize]
        if len(current_chunk_list) == 0:
            raise StopIteration
        result = []
        for event in current_chunk_list:
            result.append(self.image_reader.get_image(event))
        self._data = self._data[self.chunksize :]
        result = np.stack(result, axis=0)
        return current_chunk_list, result


def cluster_ndarray(
    profiles_arr,
    output_prefix="clustered",
    output_lists=False,
    threshold=25,
    criterion="maxclust",
    min_num_images=50,
):
    """
    cluster_ndarray clusters images based on their radial profiles

    Parameters
    ----------
    profiles_arr : np.ndarray
        radial profiles (or any other profiles, honestly) 2D np.ndarray
    output_prefix : str, optional
        output prefix for image lists0, by default "clustered"
    output_lists : bool, optional
        whether to output lists as text fiels, by default False
    threshold : int, optional
        distance according to criterion, by default 25
    criterion : str, optional
        criterion for clustering, by default "maxclust"
    min_num_images : int, optional
        minimal number of images in single cluster, others will go to singletone, by default 50

    Returns
    -------
    Union[dict, list]
        Either:
           - Dictionary {cluster_num:[*image_and_event_lines]} -- if output_lists == False
           - List [output_list_1.lst, output_list_2.lst, ...] -- if output_lists == True
    """
    profiles = np.array([elem[1] for elem in profiles_arr])
    names = np.array([elem[0] for elem in profiles_arr])

    # this actually does clustering
    Z = ward(pdist(profiles))
    idx = fcluster(Z, t=threshold, criterion=criterion)

    # output lists
    clusters = defaultdict(lambda: set())
    out_lists = set()
    for list_idx in tqdm(list(set(idx)), desc="Output lists"):
        belong_to_this_idx = np.where(idx == list_idx)[0]
        if len(belong_to_this_idx) < min_num_images:
            fout_name = f"{output_prefix}_singletone.lst"
            out_cluster_idx = -1
        else:
            fout_name = f"{output_prefix}_{list_idx}.lst"
            out_cluster_idx = list_idx
        out_lists.add(fout_name)
        try:
            os.remove(fout_name)
        except OSError:
            pass

        # print output lists if you want to
        for name in names[belong_to_this_idx]:
            clusters[out_cluster_idx].add(name)
        if output_lists:
            with open(fout_name, "a") as fout:
                print(*clusters[out_cluster_idx], sep="\n", file=fout)

    if output_lists:
        return list(out_lists)
    else:
        return clusters


def _radial_profile(data, center, normalize=True):
    """
    _radial_profile returns radial profile of a 2D image

    Parameters
    ----------
    data : np.ndarray
        (M,N)-shaped np.ndarray
    center : tuple
        two float numbers as a center
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

    return radialprofile


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
        new_data = denoiser.denoise(
            data, center=center, radius=radius, **denoiser_kwargs
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
        description="Denoises images using simple percentile filter"
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
        default=0.05,
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
