#!/bin/bash

gpu=0 
data_path="/devb/WHU-OPT-SAR/jointpatch224full_txt"
condition_way="SARMAPTXT"
output_root="/devb/sar2opt_diff_sub/sarmaptxt_predict"
batch_size=10
map_swinunet="/devb/sar2opt_diff/mapping_test2/model_statedict_best.tor"
diffusion_trained_model="/devb/sar2opt_diff_sub/sarmaptxt/model/model002520.pt"
subset=100

python mainpredict_s2on.py --gpu $gpu --data_path $data_path  --condition_way $condition_way \
         --output_root $output_root --map_swinunet $map_swinunet\
        --batch_size $batch_size  --diffusion_trained_model $diffusion_trained_model\
        --subset $subset 
