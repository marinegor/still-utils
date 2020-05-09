#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Determine mean detector shift based on prediction refinement results
#
# Copyright © 2015-2018 Deutsches Elektronen-Synchrotron DESY,
#                       a research centre of the Helmholtz Association.
#
# Author:
#    2015-2018 Thomas White <taw@physics.org>
#    2016      Mamoru Suzuki <mamoru.suzuki@protein.osaka-u.ac.jp>
#    2018      Chun Hong Yoon
#

import sys
import os
import re
import numpy as np

if sys.argv[1] == "-":
    f = sys.stdin
else:
    f = open(sys.argv[1], "r")

if len(sys.argv) > 2:
    geom = sys.argv[2]
    have_geom = 1
else:
    have_geom = 0

# Determine the mean shifts
x_shifts = []
y_shifts = []
z_shifts = []

prog1 = re.compile(
    "^predict_refine/det_shift\sx\s=\s([0-9\.\-]+)\sy\s=\s([0-9\.\-]+)\smm$"
)
prog2 = re.compile("^predict_refine/clen_shift\s=\s([0-9\.\-]+)\smm$")

while True:

    fline = f.readline()
    if not fline:
        break

    match = prog1.match(fline)
    if match:
        xshift = float(match.group(1))
        yshift = float(match.group(2))
        x_shifts.append(xshift)
        y_shifts.append(yshift)

    match = prog2.match(fline)
    if match:
        zshift = float(match.group(1))
        z_shifts.append(zshift)

f.close()

mean_x = sum(x_shifts) / len(x_shifts)
mean_y = sum(y_shifts) / len(y_shifts)
print("Mean shifts: dx = {:.2} mm,  dy = {:.2} mm".format(mean_x, mean_y))

nbins = 200
H, xedges, yedges = np.histogram2d(x_shifts, y_shifts, bins=nbins)
H = np.rot90(H)
H = np.flipud(H)
Hmasked = np.ma.masked_where(H == 0, H)


# Apply shifts to geometry
if have_geom:

    out = os.path.splitext(geom)[0] + "-predrefine.geom"
    print("Applying corrections to {}, output filename {}".format(geom, out))
    g = open(geom, "r")
    h = open(out, "w")
    panel_resolutions = {}

    prog1 = re.compile("^\s*res\s+=\s+([0-9\.]+)\s")
    prog2 = re.compile("^\s*(.*)\/res\s+=\s+([0-9\.]+)\s")
    prog3 = re.compile("^\s*(.*)\/corner_x\s+=\s+([0-9\.\-]+)\s")
    prog4 = re.compile("^\s*(.*)\/corner_y\s+=\s+([0-9\.\-]+)\s")
    default_res = 0
    while True:

        fline = g.readline()
        if not fline:
            break

        match = prog1.match(fline)
        if match:
            default_res = float(match.group(1))
            h.write(fline)
            continue

        match = prog2.match(fline)
        if match:
            panel = match.group(1)
            panel_res = float(match.group(2))
            default_res = panel_res
            panel_resolutions[panel] = panel_res
            h.write(fline)
            continue

        match = prog3.match(fline)
        if match:
            panel = match.group(1)
            panel_cnx = float(match.group(2))
            if panel in panel_resolutions:
                res = panel_resolutions[panel]
            else:
                res = default_res
                print(
                    "Using default resolution ({} px/m) for panel {}".format(res, panel)
                )
            h.write("%s/corner_x = %f\n" % (panel, panel_cnx + (mean_x * res * 1e-3)))
            continue

        match = prog4.match(fline)
        if match:
            panel = match.group(1)
            panel_cny = float(match.group(2))
            if panel in panel_resolutions:
                res = panel_resolutions[panel]
            else:
                res = default_res
                print(
                    "Using default resolution ({} px/m) for panel {}".format(res, panel)
                )
            h.write("%s/corner_y = %f\n" % (panel, panel_cny + (mean_y * res * 1e-3)))
            continue

        h.write(fline)

    g.close()
    h.close()
