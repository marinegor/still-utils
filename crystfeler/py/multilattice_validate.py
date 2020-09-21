#!/usr/bin/env python3

import numpy as np
import rmsd
import sys
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import Dict, List
import argparse
import warnings


def read_stream(filename: str, print_multiples=None) -> Dict:
    """\
    Read stream and return dictionary {image_tag:[unit_cell_1, unit_cell_2,...]}
    """

    def begin_chunk(s):
        return "Begin chunk" in s

    def end_chunk(s):
        return "End chunk" in s

    def begin_crystal(s):
        return "Begin crystal" in s

    def end_crystal(s):
        return "End crystal" in s

    def image_filename(s):
        return s.startswith("Image filename")

    def event(s):
        return s.startswith("Event")

    def has_inverse(s):
        return "star" in s and (
            s.startswith("astar") or s.startswith("bstar") or s.startswith("cstar")
        )

    def has_cellparams(s):
        return s.startswith("Cell parameters")

    def peaksfromlist(s):
        return "Peaks from peak search" in s

    def cell_params(line: str):
        """Return inverse lattice params extracted from the line"""
        return np.array([float(i) for i in line.split()[2:5]])

    active = False
    current_event = None  # to handle non-event streams
    current_chunk_crystals = []
    database = {}
    list_of_multiples_to_print = []

    with open(filename, "r") as fin:
        for line_num, line in tqdm(enumerate(fin), desc="Reading stream"):
            if begin_chunk(line) or begin_crystal(line) or end_crystal(line):
                active = True
            if not active:
                continue
            else:
                if image_filename(line):
                    current_filename = line.split()[-1]
                elif event(line):
                    current_event = line.split()[-1]
                elif peaksfromlist(line):
                    active = False
                    continue
                elif has_cellparams(line):
                    current_cell_params = cell_params(line)
                elif has_inverse(line):
                    if line.startswith("astar"):
                        astar = cell_params(line)
                    elif line.startswith("bstar"):
                        bstar = cell_params(line)
                    elif line.startswith("cstar"):
                        cstar = cell_params(line)
                        active = False
                    else:
                        raise TypeError(
                            f"Please check line {line}, num {line_num} -- it matches inverse param regex, but can not be parsed"
                        )
                elif end_crystal(line):
                    current_chunk_crystals.append(
                        (np.array([astar, bstar, cstar]), current_cell_params)
                    )
                    continue
                elif end_chunk(line):
                    active = False
                    if len(current_chunk_crystals) > 0:
                        image_id = (
                            f"{current_filename} {current_event}"
                            if current_event is not None
                            else current_filename
                        )
                        database[image_id] = current_chunk_crystals
                    if print_multiples is not None and len(current_chunk_crystals) > 1:
                        list_of_multiples_to_print.append(image_id)
                    current_chunk_crystals = []
                    continue

    if print_multiples is not None and len(list_of_multiples_to_print) > 0:
        with open(print_multiples, "w") as fout:
            print(f"Full list of multiple hits is here: {print_multiples}")
            print(*list_of_multiples_to_print, sep="\n", file=fout)

    return database


def get_angle(vecs1, vecs2):
    """Returns rotation angle between two vector sets, representing inverse lattices"""

    # https://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_angle
    return np.rad2deg(np.arccos((np.trace(rmsd.kabsch(vecs1, vecs2)) - 1) / 2))


def number_of_different_inverse_cells(vectors_set, rmsd_threshold):
    """\
    Returns number of actually different cells in a given set,
    where cells that differ on rotation of less than `rmsd` radians are
    considered same
    """
    rmsd_matrix = [
        [1 if get_angle(v1, v2) < rmsd_threshold else 0 for v1 in vectors_set]
        for v2 in vectors_set
    ]
    graph = csr_matrix(rmsd_matrix)
    n_components = connected_components(
        csgraph=graph, directed=False, return_labels=False
    )
    return n_components


def main(args: List[str]):
    parser = argparse.ArgumentParser(
        description="Estimate number of falsely-multiple lattices"
    )
    parser.add_argument("stream", help="Input stream")
    parser.add_argument(
        "--out", default=None, help="Out list of multiple hits in a separate file"
    )
    parser.add_argument(
        "--rmsd",
        type=float,
        default=5.0,
        help="RMSD (in deg) between 2 inverse lattices to be treated as separate ones",
    )
    args = parser.parse_args()
    parsed_stream = read_stream(args.stream, print_multiples=args.out)
    parsed_lattices = [[elem[0] for elem in chunk] for chunk in parsed_stream.values()]

    # check if there are any crystals in the input stream
    if len(parsed_lattices) == 0:
        print("No lattices found in input stream")
        return

    max_lattices_on_one_crystal = max([len(i) for i in parsed_lattices])

    acc_total_images, acc_total_crystals, acc_true_crystals, acc_false_crystals = (
        0,
        0,
        0,
        0,
    )

    multiples_found = False
    for num_lattices in range(2, max_lattices_on_one_crystal + 1):
        multiples_found = True
        multiple_lattices = [
            elem for elem in parsed_lattices if len(elem) == num_lattices
        ]
        unique_cells = [
            number_of_different_inverse_cells(elem, rmsd_threshold=args.rmsd)
            for elem in multiple_lattices
        ]
        total_images = len(multiple_lattices)
        total_crystals = total_images * num_lattices
        true_crystals = sum(unique_cells)
        false_crystals = total_crystals - true_crystals
        if not total_images:
            continue
        print(
            f"For {num_lattices} crystals on image "
            f"total images: {total_images}, "
            f"total crystals: {total_crystals}, "
            f"true crystals: {true_crystals}, "
            f"false_crystals: {false_crystals}"
        )  # this will print everythin on one line -- notice commas are abscent

        acc_total_images += total_images
        acc_total_crystals += total_crystals
        acc_true_crystals += true_crystals
        acc_false_crystals += false_crystals

    if multiples_found:
        print("-" * 80)
        print(
            f"For n > 1 crystals on image "
            f"total images: {acc_total_images}, "
            f"total crystals: {acc_total_crystals}, "
            f"true crystals: {acc_true_crystals}, "
            f"false_crystals: {acc_false_crystals}"
        )  # this will print everythin on one line -- notice commas are abscent
        false_crystal_ratio = acc_false_crystals / acc_total_crystals
        print(f"False crystal ratio: {false_crystal_ratio:1.2f}")
        if false_crystal_ratio > 0.5:
            print("Warning: more than a half crystals are false multiples")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(sys.argv[1:])
