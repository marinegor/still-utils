#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

input_filename = sys.argv[1]
df = pd.read_csv(
    input_filename, header=None, names=["name", "crystals", "snr", "threshold"]
)

M = np.zeros((df["snr"].unique().shape[0], df["threshold"].unique().shape[0]))

for x, snr in enumerate(df["snr"].unique()):
    for y, threshold in enumerate(df["threshold"].unique()):
        cur = df[(df["snr"] == snr) & (df["threshold"] == threshold)]
        if len(cur) == 0:
            M[x][y] = 0
        else:
            M[x][y] = cur["crystals"].iloc[0]

sns.heatmap(
    M, xticklabels=df["threshold"].unique(), yticklabels=df["snr"].unique(), annot=True
)
plt.yticks(rotation=0)

plt.rcParams["figure.figsize"] = (20, 20)
plt.show()
