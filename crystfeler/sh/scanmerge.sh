#!/bin/bash

INPUT=$1
SYMMETRY='mmm'
HIGHRES="2.3"

for pushres in $(python3 -c "print(*[i/10 for i in range(1,40,5)])"); do
	for method in process_hkl; do
		for min_cc in -1 -0.1 0.0 0.1 0.2; do
			for string_params in ' ' '--scale'; do
				analyzer -i ${INPUT} --merging "$method" --model "$model" --symmetry ${SYMMETRY} --iterations "$iterations" --lowres 30.0 --highres ${HIGHRES} --pushres "$pushres" --min_cc "$min_cc" --string_params "$string_params"
			done
		done
	done
done

for pushres in $(python3 -c "print(*[i/10 for i in range(1,40,5)])"); do
	for method in partialator; do
		for iterations in 0 1 2; do
			for model in xsphere unity; do
				analyzer -i ${INPUT} --merging "$method" --model "$model" --symmetry ${SYMMETRY} --iterations "$iterations" --lowres 30.0 --highres ${HIGHRES} --pushres "$pushres" --min_cc "$min_cc" --string_params "$string_params"
			done
		done
	done
done

# srun -c 8 --comment "..." scanmerge.sh
