#!/usr/bin/env python3

import argparse
import sys
import numpy as np
from typing import List


def plot_hist(
    input_array: list,
    bins: int,
    width: int,
    symbol: str,
    xmin: np.float,
    xmax: np.float,
) -> None:
    """
    Plots histogram in stdout, accepting list of floats or ints on input
    Source: https://gist.github.com/tammoippen/4474e838e969bf177155231ebba52386

    Arguments:
        input_array {list} -- input list of numbers

    Keyword Arguments:
        bins {int} -- number of bins (=lines on output) (default: {50})
        width {int} -- histogram width(=maximum line length) (default: {140})
        symbol {str} -- symbol to draw histogram with (default: {"#"})
        xmin {np.float} -- minimum for histogram plotting
        xmax {np.float} -- maximum for histogram plotting
    """
    input_array = np.array(input_array)
    mask = (input_array <= xmax) & (input_array >= xmin)
    input_array = input_array[mask]
    h, b = np.histogram(input_array, bins)

    for i in range(0, bins):
        print(
            "{:12.5f}  | {:{width}s} {}".format(
                b[i], symbol * int(width * h[i] / np.amax(h)), h[i], width=width
            )
        )
    print("{:12.5f}  |".format(b[bins]))


def main(args):
    """
    The main function

    Arguments:
        args {List[str]} -- input arguments for main
    """

    parser = argparse.ArgumentParser(description="Simple histogram drawer")
    parser.add_argument(
        "stdin", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    parser.add_argument(
        "--bins", type=int, help="number of bins (=lines on output)", default=15
    )
    parser.add_argument(
        "--width", type=int, help="histogram width (=number of columns)", default=50
    )
    parser.add_argument(
        "--symbol", type=str, help="symbol to draw histogram with", default="#"
    )
    parser.add_argument(
        "--column",
        type=int,
        help="column of input data to draw histogram with",
        default=-1,
    )
    parser.add_argument(
        "--xmin", help="minimum for plotting", default=-np.inf, type=float
    )
    parser.add_argument(
        "--xmax", help="maximum for plotting", default=np.inf, type=float
    )
    args = parser.parse_args()

    def _extract_float(elem, column):
        try:
            return float(elem.split()[column])
        except:
            return None

    data = [
        _extract_float(elem, args.column)
        for elem in args.stdin.read().split("\n")
        if _extract_float(elem, args.column) is not None
    ]

    if not data:
        print("No input data found by hist.py")
        return None
    plot_hist(
        data,
        bins=args.bins,
        width=args.width,
        symbol=args.symbol,
        xmin=args.xmin,
        xmax=args.xmax,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
