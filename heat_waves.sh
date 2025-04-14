#!/bin/bash

# Declare an array of strings with the models
declare -a StringArray=("NorESM2-MM" "MIROC6" "MRI-ESM2-0" 
                        "GFDL-ESM4" "INM-CM4-8" "INM-CM5-0"
                        "EC-Earth3-Veg-LR" "FGOALS-g3" "GFDL-CM4"
                        "ACCESS-ESM1-5" "CMCC-CM2-SR5" "EC-Earth3")

pathERA=$1
pathTarget=$2
pathDat=$3
pathlsmcmip6=$4
pathlsmera5=$5


for ((year=1980;year<=2018;year++)); 
do
    echo "ERA5" $scenario $year
    time python src/heat_waves.py \
        --model=ERA5 \
        --scenario=$scenario \
        --year=$year \
        --pathERA=$pathERA \
        --pathTarget=$pathTarget \
        --pathDat=$pathDat \
        --lsm_CMIP6=$pathlsmcmip6 \
        --lsm_ERA5=$pathlsmera5
done

for val in ${StringArray[@]}; 
do
    for scenario in ssp245 ssp585
    do
        for ((year=2020;year<=2100;year++)); 
        do
            echo $val $scenario $year
            time python src/heat_waves.py \
                --model=$val \
                --scenario=$scenario \
                --year=$year \
                --pathERA=$pathERA \
                --pathTarget=$pathTarget \
                --pathDat=$pathDat \
                --lsm_CMIP6=$pathlsmcmip6 \
                --lsm_ERA5=$pathlsmera5
        done
    done
done