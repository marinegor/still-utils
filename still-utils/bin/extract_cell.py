#!/usr/bin/env python3

import sys
import os

helpstring = """\
Usage: ./extract_cell.py my.stream"""

if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
    print(helpstring)
    exit(0)
elif not os.path.exists(sys.argv[1]):
    print(f"File {sys.argv[1]} does not exist, check path or access rights")
else:
    cell_start = False
    for line in open(sys.argv[1]):
        if "Begin unit cell" in line:
            cell_start = True
            continue
        elif "End unit cell" in line:
            break
        else:
            if cell_start and line[0] != ";":
                print(line, end="")
