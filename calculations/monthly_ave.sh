for moy in {1..12}; do 
    mm=$(printf "%02d" ${moy})
    ncra $2/ANHA12_EXH006_5d_${1}0101_${1}1231_ptrc_T_${1}${mm}??-${1}${mm}??.nc $2/ANHA12_EXH006_${1}${mm}.nc 
done
ncecat $2/ANHA12_EXH006_${1}??.nc $2/ANHA12_EXH006_${1}_monthly.nc
rm -f $2/ANHA12_EXH006_${1}??.nc
