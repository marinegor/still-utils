#!/bin/bash

time=$(date "+%Y_%m_%d_%H_%M")
output_folder="streams/search_$time"
mkdir -p "$output_folder" # creates search subfolder in the stream

PROJECT_NAME="GPCR"
NPROC="32"
#SNR='4.0'
#THRESHOLD='10'
#HIGHRES='3.0'
NIMAGES=1000 # number of images to run optinization on
LST='yourlist.lst'
shuf "$LST" | head -n "$NIMAGES" > input.lst

GEOM='yourgeom.geom'
CELL="yourcell.cell"


## parameters to loop over
declare -a SNR=("3.0" "3.5" "4.0" "4.5" "5.0")
declare -a THRESHOLD=("15" "30" "50" "80" "100")
declare -a HIGHRES=("2.2" "2.5" "2.8" "3.1" "3.6")

## now loop through the above array
for snr in "${SNR[@]}"; do
	for threshold in "${THRESHOLD[@]}"; do
		for highres in "${HIGHRES[@]}"; do
    		string="indexamajig -i input.lst -g "$GEOM" \
    		-o "$output_folder""/"$PROJECT_NAME"_${snr}_${threshold}_${highres}_${median}.stream" \
    		--peaks=peakfinder8 \
    		-j "$NPROC" \
    		-p "$CELL" \
    		--min-snr="$snr" \
    		--threshold="$threshold" \
    		--highres="$highres" \
    		--indexing=mosflm,dirax,xds"
			echo EXECUTING:
			echo "$string"
			eval "$string"
		done
	done
done
