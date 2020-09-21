#!/usr/bin/env python3


import argparse
import os
import numpy as np
import sys
from typing import Union
from collections import defaultdict
from scipy.spatial.transform import Rotation as R


def angles2matrix(ss_line: str, fs_line: str):

    fs_line = f"{fs_line} + 0.0z" if "z" not in fs_line else fs_line
    ss_line = f"{ss_line} + 0.0z" if "z" not in ss_line else ss_line

    std = lambda s: s.replace(" ", "")

    assert list(sorted(filter(str.isalpha, std(fs_line)))) == [
        "x",
        "y",
        "z",
    ], "Please change fs line to follow alphabetic x, y, z order"
    assert list(sorted(filter(str.isalpha, std(ss_line)))) == [
        "x",
        "y",
        "z",
    ], "Please change ss line to follow alphabetic x, y, z order"

    x1, *rest1 = std(fs_line).split("x")
    y1, *rest1 = rest1[0].split("y")
    z1, *rest1 = rest1[0].split("z")

    x2, *rest2 = std(ss_line).split("x")
    y2, *rest2 = rest2[0].split("y")
    z2, *rest2 = rest2[0].split("z")

    M = np.array([[x1, y1, z1], [x2, y2, z2]]).astype(np.float)
    return M


def read_geom_to_dict(input_geom) -> dict:
    """
    read_geom reads the geometry from file and returns corner_x, corner_y and matrix as dictionary

    Parameters
    ----------
    input_geom : str
        Input file name

    Returns
    -------
    dict
        {'corner_x':corner_x, 'corner_y':corner_y, 'coffset':coffset, 'M':M} dictionary, where M is the (fs,ss) <-- (x,y) matrix
    """

    important_keys = ["fs", "ss", "corner_x", "corner_y", "coffset", "clen", "res"]
    float_keys = ["corner_x", "corner_y", "coffset", "clen", "res"]
    ret = {}
    for key in important_keys:
        ret[key] = 0.0

    with open(input_geom, "r") as fin:
        for line in fin:
            for key in important_keys:
                if key == "clen":
                    if line.replace(" ", "").split("=")[0] == key:
                        ret[key] = np.float(line.replace(" ", "").split("=")[1])
                elif line[0] != ";" and line.replace(" ", "").split("=")[0].endswith(
                    f"/{key}"
                ):
                    _, value = line[:-1].replace(" ", "").split("=")
                    if key in float_keys:
                        value = np.float(value)
                    ret[key] = value

    M = angles2matrix(ret["ss"], ret["fs"])
    ret["M"] = M

    return ret


def update_geom_params_dict(
    initial_geom: dict,
    alpha: 0,
    beta: 0,
    coffset: None,
    corner_x: None,
    corner_y: None,
    relative=True,
) -> dict:
    """
    Given an initial dict of values,
    generates dict of geometry params,
    given certain alpha, beta, coffset, corner_x and corner_y

    Arguments:
        initial_geom {dict} -- initial geometry file
        alpha {np.float} -- angle of rotation along vertical axis (deg)
        beta {np.float} -- angle of rotation along horizontal axis (deg)
        coffset {np.float} -- crystal-to-detector length offset
        corner_x {Union[None, np.float]} -- x coordinate of detector corner
        corner_x {Union[None, np.float]} -- x coordinate of detector corner
        relative{bool} -- whether corner_x, corner_y are relative shifts to previous position or absolute new values

    Returns:
        dict -- updated params in a dict
    """

    alpha, beta = np.deg2rad(alpha), np.deg2rad(beta)

    M = initial_geom["M"]
    Mhat = np.hstack((-M.T, np.array([0, 0, 1]).reshape(3, 1))).T

    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])

    rotation_x = R.from_rotvec(alpha * x_axis)
    rotation_y = R.from_rotvec(beta * y_axis)
    Mhat_upd = rotation_y.apply(rotation_x.apply(Mhat))
    M_upd = -1 * Mhat_upd[:-1, :]

    corner_x_rot, corner_y_rot, clen_rot = (
        Mhat_upd
        @ np.linalg.inv(Mhat)
        @ np.array(
            [
                initial_geom["corner_x"],
                initial_geom["corner_y"],
                initial_geom["clen"] * initial_geom["res"],  # converts clen to pixels
            ]
        )
    )
    if corner_x is None:  # keep the centre unmoved
        corner_x = corner_x_rot
    if corner_y is None:  # keep the centre unmoved
        corner_y = corner_y_rot
    if coffset is None:  # keep the centre unmoved
        coffset = (
            clen_rot - initial_geom["clen"] * initial_geom["res"]
        ) / initial_geom[
            "res"
        ]  # back to meters
    if relative:
        if corner_x is not None:
            corner_x = initial_geom["corner_x"] + corner_x
        if corner_x is not None:
            corner_y = initial_geom["corner_y"] + corner_y
        if coffset is not None:
            coffset = initial_geom["coffset"] + coffset

    ret = {
        "corner_x": corner_x,
        "corner_y": corner_y,
        "coffset": coffset,
        "M": M_upd,
        "alpha": alpha,
        "beta": beta,
    }

    return ret


def update_geom_file_from_dict(
    input_file: str, dict_to_apply: dict, inplace=False
) -> str:
    alpha, beta, corner_x, corner_y, coffset, M = (
        dict_to_apply["alpha"],
        dict_to_apply["beta"],
        dict_to_apply["corner_x"],
        dict_to_apply["corner_y"],
        dict_to_apply["coffset"],
        dict_to_apply["M"],
    )
    if inplace:
        geom_upd = input_file
    else:
        geom_upd = (
            f"{input_file[:-4]}__"
            f"a_{np.rad2deg(alpha):.4f}__"
            f"b_{np.rad2deg(beta):.4f}__"
            f"coff_{coffset:.4f}__"
            f"corX_{corner_x:.2f}__"
            f"corY_{corner_y:.2f}.geom"
        )

    # assemble the fs/ss lines into crystfel format
    coordinates = ("x", "y", "z")
    fs_line = " ".join([f"{elem:+f}{name}" for elem, name in zip(M[0], coordinates)])
    ss_line = " ".join([f"{elem:+f}{name}" for elem, name in zip(M[1], coordinates)])
    dict_to_apply["fs"] = fs_line
    dict_to_apply["ss"] = ss_line

    pr = []  # lines to print into output file
    with open(input_file, "r") as fin:
        for line in fin:
            pr.append(line[:-1])
            if "=" not in line or "/" not in line:
                continue
            for key in dict_to_apply:
                if key in line:
                    key_g = line[:-1].replace(" ", "").split("=")[0].split("/")[1]
                    if line and key_g == key:
                        pr = pr[:-1]  # un-append the line
                        upd_line = line.split("=")[0] + "=" + f" {dict_to_apply[key]}"
                        pr.append(upd_line)
                        break  # don't search the dict anymore

    with open(geom_upd, "w") as fout:
        print(*pr, sep="\n", file=fout)

    return geom_upd


def main(args):
    """\
    The main function.

    Remember following:

    (x,y).T = M @ (fs, ss).T + (corner_x, corner_y).T,

    where M is the matrix from fs=..., ss=... in geometry,
    corner_x and corner_y are also from there,
    and x, y are the resulting physical coordinates
    """

    parser = argparse.ArgumentParser(
        description="""Applies angles to given geometry and saves updated geometry in a separate file"""
    )
    parser.add_argument("input_file", type=str, help="Input geometry file")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        help="Alpha angle (rotation along x axis -- completes celling and beam to orthonormal)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0,
        help="Beta angle (rotation along y axis -- along the beam)",
    )
    parser.add_argument(
        "--coffset",
        type=float,
        default=None,
        help="New coffset value (keep old by default)",
    )
    parser.add_argument(
        "--corner_x",
        type=float,
        default=None,
        help="New corner_x value (keep old by default)",
    )
    parser.add_argument(
        "--corner_y",
        type=float,
        default=None,
        help="New corner_y value (keep old by default)",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        default=False,
        help="Whether to invoke relative corner_x, corner_y update",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        default=False,
        help="Whether to update geometry in-place",
    )

    args = parser.parse_args()

    initial_geom_dict = read_geom_to_dict(args.input_file)
    new_geom_dict = update_geom_params_dict(
        initial_geom=initial_geom_dict,
        alpha=args.alpha,
        beta=args.beta,
        coffset=args.coffset,
        corner_x=args.corner_x,
        corner_y=args.corner_y,
        relative=args.relative,
    )

    new_geom_file = update_geom_file_from_dict(
        args.input_file, dict_to_apply=new_geom_dict, inplace=False
    )

    print(f"Saved geometry to {new_geom_file}")


if __name__ == "__main__":
    main(sys.argv[1:])
