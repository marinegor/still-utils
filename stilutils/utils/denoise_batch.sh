#!/bin/bash

for cxi in ./*.cxi; do
	srun --comment 'denoiser' denoiser.py --datapath '/entry_1/data_1/data' "$cxi" --center "720 711" --top 1000
done
