#!/bin/bash


RESTRAINTS="merged.cif"
MODEL="refine.pdb"
MTZ="refine.mtz"

LOWLIM=3.0
HIGHLIM=2.3
STEP=0.1
LOWRES=30.0
TIME="$(date +%F-%H-%M-%S)"
MODE=$1


if [[ "${MODE}" == "run" ]]; then
	for highres in $(seq ${HIGHLIM} ${STEP} ${LOWLIM}); do
	
		refine \
			-p "$MODEL" \
			-m "$MTZ" \
			-l "$RESTRAINTS" \
			-nthreads 2 \
			-autoncs -TLS \
			-d "./buster__pairef_${TIME}_${highres}" \
			-R ${LOWRES} "${highres}" \
			-nbig 1 &
	done
	wait
fi
 

if [[ "${MODE}" == "scan" ]]; then
	TEMPLATE=$2
	
	if [[ "${TEMPLATE}" == "" ]]; then
		echo "Please provide output folder template!";
		exit 1; fi

	for highres in $(seq ${HIGHLIM} ${STEP} ${LOWLIM}); do
		echo "$highres"	
	
		folder="./${TEMPLATE}_${highres}"
		refine \
			-p "${folder}/refine.pdb" \
			-m "$MTZ" \
			-l "$RESTRAINTS" \
			-l "$CCP4"/lib/data/monomers/o/OLC.cif \
			-nthreads 1 \
			-d "${folder}_answ" \
			-R ${LOWRES} "${LOWLIM}" \
			-nbig 1 \
			-M MapOnly &
	done
	wait
fi

if [[ "${MODE}" == "table" ]]; then
	TEMPLATE=$2
	
	if [[ "${TEMPLATE}" == "" ]]; then
		echo "Please provide output folder template!";
		exit 1; fi
	
	echo "Template: ${TEMPLATE}"
	echo -e "Highres\tRwork\tRfree"
	grep -m 1 "best refinement for F,SIGF with R/Rfree"  "${TEMPLATE}"_*_answ/refine.pdb | 
		tr "_" " " | \
		awk '{print $5,$13}' | \
		tr '/' ' ' | \
		sort -k 3 | \
		column -t

fi
