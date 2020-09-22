#!/usr/bin/env python3

import yaml
import sys
import argparse
import contextlib


@contextlib.contextmanager
def _smart_open_for_writing(filename=None):
    if filename and filename != "-":
        fh = open(filename, "w")
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()


def geom2yaml(filename: str, output="-") -> None:
    """
    Converts geom file to dictionary

    Arguments:
        filename {str} -- input filename

    Keyword Arguments:
        output {str} -- output filename ('-' stands for sys.stdout)

    Returns:
        dict -- dict representation of crystfel geometry
    """

    ret = {}
    with open(filename, mode="r") as fin:
        for line in fin:
            if line.strip().startswith(";"):
                continue
            if line.strip():
                bigkey, value = line.replace(" ", "").split("=", maxsplit=1)
                value = value[:-1]
                assert bigkey.count("/") < 2, f'{line}, {bigkey.count("/")}'
                if "/" in bigkey:
                    k1, k2 = bigkey.split("/")
                    if k1 in ret:
                        ret[k1][k2] = value
                    else:
                        ret[k1] = {k2: value}
                else:
                    ret[bigkey] = value

    with _smart_open_for_writing(output) as stream:
        yaml.safe_dump(
            {"GEOM": ret}, stream, default_style=":", indent=4, default_flow_style=False
        )


def yaml2geom(filename: str, output="-") -> None:
    if output is None:
        output = filename + ".geom"

    with open(filename, "r") as stream:
        d = yaml.safe_load(stream)

    d = d["GEOM"]

    with _smart_open_for_writing(output) as fout:
        for key, value in d.items():
            if not isinstance(value, dict):
                print(" = ".join((key, value)), end="\n", file=fout)
            else:
                for subkey, subvalue in value.items():
                    print(f"{key}/{subkey} = {subvalue}", end="\n", file=fout)


def main(args):
    """
    The main function
    """

    parser = argparse.ArgumentParser(
        description="Converter between yaml and geom formats"
    )
    parser.add_argument("input_file", type=str, help="Input file")
    parser.add_argument(
        "--inf",
        type=str,
        choices=("geom", "yaml", None),
        help="Input file format",
        default=None,
    )
    parser.add_argument(
        "--outf",
        type=str,
        choices=("geom", "yaml", None),
        help="Output file format",
        default=None,
    )
    parser.add_argument("-fout", type=str, help="Output filen name", default=sys.stdout)

    args = parser.parse_args()

    if args.inf is None or args.outf is None:
        ext = args.input_file.rsplit(".")[-1]
        assert ext in (
            "yaml",
            "geom",
        ), f"Failed to detect input filename extension: you provided {ext}, should be geom or yaml"
        args.inf = ext
        args.outf = "yaml" if ext == "geom" else "geom"

    if args.inf == args.outf and args.inf is not None:
        raise TypeError(
            f"Input and output formats should be different, yours are both {args.inf}"
        )

    if args.inf == "yaml":
        yaml2geom(args.input_file, output="-")
    else:
        geom2yaml(args.input_file, output="-")


if __name__ == "__main__":
    main(sys.argv[1:])
