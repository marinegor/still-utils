#!/bin/bash

for inputstream in "$@"; do
	grep -Hc "Begin crystal" "$inputstream" | tr ":" "," | tr '\n' ','
	head -n 3 $inputstream | tail -n 1 | awk '{print $6, $7}' | tr "=" " " | awk 'BEGIN {OFS=","}; {print $2, $4}'
	echo ""
done | grep -E -v ",$" | grep -E -v "^$"
