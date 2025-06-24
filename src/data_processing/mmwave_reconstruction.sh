#!/bin/bash
### 
# This file can be used to process a list of one or more experiments at different angles. 
# You should only need to change the parameters within the "Parameters to Change" block. 
# All other code shouldn't be changed unless more advanced functionality is needed. 
###

###################  Parameters to Change ###################
# For names/ids, these should be lists of the names/ids of the objects to process.
#       One of these two lists can be left empty, and it will be filled in accordingly
# For is_los_s, these should be:
#       - a list with the same length as names/ids (e.g., ("y" "n")) OR
#       - a single string which will be applied for all objects (e.g., "n")
# If overwrite_existing is n, then this code will only run the computation if there is not already a valid output file. If overwerite_existing is y, then this code will run the computation regardless, and any existing files may be overwritten
names=("spoon") 
ids=("031") 
is_los_s="y"
overwrite_existing="y"
##############################################################

num_names=${#names[@]}
num_ids=${#ids[@]}
num_is_los=${#is_los_s[@]}
num_exps=${#exp_num[@]}
# Check the name/id lists are valid
if [[ $num_names -ne $num_ids  && $num_names -ne 0 && $num_ids -ne 0 ]]; then
    echo "The names and ids list should either have the same length, or one should be empty"
    exit 1
fi

num_objs=$(( $num_names > $num_ids ? $num_names : $num_ids ))

if [[ $num_is_los -ne $num_objs && $num_is_los -ne 0 && $num_is_los -ne 1 ]]; then
    echo "is_los_s should either be a list with the same length as ids/names, be a single string, or be an empty list."
    exit 1
fi

# Iterate over the list and run the Python script with each argument
for ((i=0; i<$num_objs; i++)); do
    if test $i -lt $num_names; then
        name="${names[i]}"
    else 
        name="None"
    fi
    if test $i -lt $num_ids; then
        id="${ids[i]}"
    else 
        id="None"
    fi

    if test $num_is_los -eq 1; then
        is_los=$is_los_s
    elif test $num_is_los -ne 0; then
        is_los="${is_los_s[i]}"
    else
        is_los= "y"
    fi

    # You can use the ext parameter to add additional strings to the end of the save file names. This allows you to process the same experiment multiple times without overwriting your prior results
    echo "=============================================================="
    echo "Starting mmWave surface normal estimation"
    python3 mmwave_normal_estimation.py --name $name --id $id --is_los $is_los --ext "" --overwrite_existing $overwrite_existing

    echo "=============================================================="
    echo "Starting RSDF computation"
    python3 sdf.py --name $name --id $id --is_los $is_los --ext "" --overwrite_existing $overwrite_existing

    echo "=============================================================="
    echo "Starting isosurface optimization"
    python3 isosurface_optimization.py --name $name --id $id --is_los $is_los --ext "" --overwrite_existing $overwrite_existing
done