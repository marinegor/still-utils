#!/bin/bash

output="overall_stats_$(date "+%Y_%m_%d_%H_%M")"
basename="tmp"
symmetry='-1'
highres='3.0'
lowres='30.0'
iterations='0'
model='unity'
pushres="inf"
j="96"

while [[ $# -gt 1 ]]
do
key="$1"
cell_set=0

case $key in
    -h|--help)
    echo "Usage:"
    echo "      ./rate.sh -i basename --highres 2.2 --cell yourcell.cell -s 222"
    exit 0;;
    -i|--basename)
    basename="$2"
    shift # past argument
    ;;
    -s|--symmetry)
    symmetry="$2"
    shift # past argument
    ;;
    --highres)
    highres="$2"
    shift # past argument
    ;;
    --lowres)
    lowres="$2"
    shift # past argument
    ;;
    -c|--cell)
    cell="$2"
    cell_set=1
    shift # past argument
    ;;
    --default)
    DEFAULT=YES
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done

#----------------------------

echo "$basename"
#compare_hkl "$basename".hkl1 "$basename".hkl2 -y "$symmetry" -p "$cell" --fom rsplit --nshells=10 --lowres "$lowres" --highres "$highres"
#exit

# outputs to overall_stats.log statistics, obtained with check_hkl (SNR, multiplicity, N of refl, etc), and also Rsplit, CC and CC*.
function rate {
    rm stats[0-9].dat &>/dev/null
        compare_hkl "$basename".hkl1 "$basename".hkl2 -y "$symmetry" -p "$cell" --fom rsplit --nshells=10 --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; cat shells.dat >  stats1.dat
        compare_hkl "$basename".hkl1 "$basename".hkl2 -y "$symmetry" -p "$cell" --fom cc     --nshells=10 --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; grep -a -v "shitcentre" shells.dat > stats2.dat
        compare_hkl "$basename".hkl1 "$basename".hkl2 -y "$symmetry" -p "$cell" --fom ccstar --nshells=10 --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; grep -a -v "shitcentre" shells.dat > stats3.dat
        check_hkl "$basename".hkl -y "$symmetry" -p "$cell"                                       --lowres="$lowres" --highres "$highres" &> compare_hkl.log ; cat shells.dat > stats4.dat

        compare_hkl "$basename".hkl1 "$basename".hkl2 -y "$symmetry" -p "$cell" --fom rsplit --nshells=1 --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; cat shells.dat >  stats5.dat
        compare_hkl "$basename".hkl1 "$basename".hkl2 -y "$symmetry" -p "$cell" --fom cc     --nshells=1 --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; grep -a -v "shitcentre" shells.dat > stats6.dat
        compare_hkl "$basename".hkl1 "$basename".hkl2 -y "$symmetry" -p "$cell" --fom ccstar --nshells=1 --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; grep -a -v "shitcentre" shells.dat >> stats7.dat
        check_hkl "$basename".hkl --nshells 1 -y "$symmetry" -p "$cell"                          --lowres "$lowres" --highres "$highres" &> compare_hkl.log ; cat shells.dat >> stats8.dat
    paste stats4.dat <(awk '{print $3'} stats1.dat) <(awk '{print $3'} stats2.dat) <(awk '{print $3'} stats3.dat) | head -1        > overall_stats.csv
    paste stats4.dat  <(awk '{print $2}' stats1.dat)  <(awk '{print $2}' stats2.dat)  <(awk '{print $2}' stats3.dat) | tail -n +2 >> overall_stats.csv

    echo "   -------------------------------------------------------------------------------------------------------------------------------------------------------" >> overall_stats.csv
    paste stats8.dat  <(awk '{print $2}' stats5.dat)  <(awk '{print $2}' stats6.dat)  <(awk '{print $2}' stats7.dat) | tail -n +2 >> overall_stats.csv
}


rate

echo "================"
echo "Merging summary:"
echo "================"
echo "Merging stats backup file: $output"
echo "Merging stats with partialator --iterations "$iterations" --lowres "$lowres" --highres "$highres" --push-res "$pushres":"


echo "================" >>  "$output"
echo "Merging summary:" >> "$output"
echo "================" >> "$output"
echo "Merging stats with partialator --iterations "$iterations" --lowres "$lowres" --highres "$highres" --push-res "$pushres":" >> "$output"
cat overall_stats.csv >> "$output"

rm stats[0-9].dat
cat overall_stats.csv
