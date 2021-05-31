#!/bin/bash

pdb="$1"
grep -m 1 "best refinement for F,SIGF" "$pdb"  | awk '{print $7,$8}'
grep CRYST1 "$pdb"
grep -m 1 '\s-R\s' "$pdb" | awk '{print $6,$7,$8}'

pdb_selchain -A "$pdb" | grep "^ATOM" | pdb_selres -1:999,1107:9999 > protein_chainA.pdb
pdb_selchain -B "$pdb" | grep "^ATOM" | pdb_selres -1:999,1107:9999 > protein_chainB.pdb
pdb_selchain -A "$pdb" | grep "^ATOM" | pdb_selres -1000:1106 > bril_chainA.pdb
pdb_selchain -B "$pdb" | grep "^ATOM" | pdb_selres -1000:1106 > bril_chainB.pdb

pdb_selchain -A "$pdb" | grep "^HETATM" | grep "XXX" > lig_chainA.pdb
pdb_selchain -B "$pdb" | grep "^HETATM" | grep "XXX" > lig_chainB.pdb
pdb_selchain -A "$pdb" | grep "^HETATM" | grep -v "XXX" > het_chainA.pdb
pdb_selchain -B "$pdb" | grep "^HETATM" | grep -v "XXX" > het_chainB.pdb

for elem in ./*.pdb; do
	if [[ "$elem" == ./${pdb} ]]; then
		:; fi

	echo "${elem%.pdb}" 
	echo "  atoms: " $(cat "$elem" | wc -l)
	echo "  bfact: " $(cat "$elem" | awk '{print substr($0,62,5)}' | awk '{ total += $1; count++ } END { print total/count }')
done

# echo "----------------------"
# echo "Protein atoms:"
# echo "Count:"
# printf "  Chain A:   "; pdb_selchain -A "$pdb" | grep -c "^ATOM"
# printf "  Chain B:   "; pdb_selchain -B "$pdb" | grep -c "^ATOM"
# echo "B-factors:"
# printf "  Chain A:   "; pdb_selchain -A refine.pdb  | grep "^ATOM" | awk '{print substr($0,62,5)}' | awk '{ total += $1; count++ } END { print total/count }'
# printf "  Chain A:   "; pdb_selchain -B refine.pdb  | grep "^ATOM" | awk '{print substr($0,62,5)}' | awk '{ total += $1; count++ } END { print total/count }'
#
# echo "----------------------"
# echo "Ligand atoms:"
# printf "  Chain A:   "; pdb_selchain -A "$pdb" | grep "^HETATM" | grep -c "XXX"
# printf "  Chain B:   "; pdb_selchain -B "$pdb" | grep "^HETATM" | grep -c "XXX"
# echo "B-factors:"
# printf "  Chain A:   "; pdb_selchain -A "$pdb" | grep "^HETATM" | grep "XXX" | awk '{print substr($0,62,5)}' | awk '{ total += $1; count++ } END { print total/count }'
# printf "  Chain B:   "; pdb_selchain -B "$pdb" | grep "^HETATM" | grep "XXX" | awk '{print substr($0,62,5)}' | awk '{ total += $1; count++ } END { print total/count }'
#
#
