#!/usr/bin/env python3

import argparse
import h5py
import os
from tqdm import tqdm
import numpy as np
import sys


def lst2cxi(
    input_lst: str,
    cxi_path="/entry_1/data_1/data",
    output_cxi_prefix=None,
    output_lst=None,
    compression="gzip",
    chunks=True,
) -> str:
    """
    Saves all images from input cxi file that are present in input list.

    Arguments:
        input_lst {str} -- input list (might be both with and without events)
        cxi_path {str} -- path of data inside cxi file to be copied
        output_cxi_prefix {None} -- output cxi file prefix. If 'None', will be <input_list>.cxi for <input_list>_{1,2,3,...}.lst

    Returns:
        str -- filename of cxi file
    """

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

    if output_cxi_prefix is None:
        output_cxi_prefix = ""

    cxi_cnt = 0
    output_cxi_list = []
    for input_cxi, events in tqdm(
        events_dict.items(), desc=f"Processing files in {input_lst} one by one"
    ):
        cxi_cnt += 1
        with h5py.File(input_cxi, "r") as h5fin:
            data = h5fin[cxi_path]
            if None in events:  # meaning we must dump whole cxi
                data = h5fin[cxi_path]
            else:
                data = np.array(data[np.array(sorted(list(events)))])
            shape = data.shape

            output_cxi = f'{input_lst.rsplit(".")[0]}_{cxi_cnt}.cxi'
            output_cxi_list.append(output_cxi)

            with h5py.File(output_cxi, "w") as h5fout:
                h5fout.create_dataset(
                    cxi_path, shape, compression="gzip", data=data, chunks=chunks
                )

    # here we print updated lst
    if output_lst is None:
        output_lst = f'{input_lst.rsplit(".")[0]}_upd.lst'
    with open(output_lst, "w") as fout:
        print(*output_cxi_list, sep="\n", file=fout)
    return output_lst


def main(args):
    """The main function"""

    parser = argparse.ArgumentParser(
        description="Will save only necessary files from input list"
    )
    parser.add_argument(
        "input_lst",
        type=str,
        help="Input list in CrystFEL format (might be with or without events, or even combined)",
    )
    parser.add_argument(
        "--cxi_path", type=str, help="Path to your data inside a cxi file"
    )
    parser.add_argument(
        "--output_cxi_prefix",
        type=str,
        help="Output cxi file (if not present, will replace input *.lst to *.h5) ",
        default=None,
    )

    args = parser.parse_args()

    if args.cxi_path is None:
        ret = lst2cxi(
            input_lst=args.input_lst, output_cxi_prefix=args.output_cxi_prefix
        )
    else:
        ret = lst2cxi(
            input_lst=args.input_lst,
            cxi_path=args.cxi_path,
            output_cxi_prefix=args.output_cxi_prefix,
        )

    print(f"Updated list successfully written to {ret}")


if __name__ == "__main__":

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        main(sys.argv[1:])
