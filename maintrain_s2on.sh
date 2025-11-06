#!/bin/bash

gpu=1
data_path="/devb/WHU-OPT-SAR/jointpatch224full_txt"
condition_way="SARMAPTXT"
num_epochs=3
output_root="/devb/sar2opt_diff_sub/sarmaptxt"
save_interval=30
log_interval=30
batch_size=30
subset=100
map_swinunet="/devb/sar2opt_diff/mapping_test2/model_statedict_best.tor"
###diffusion_trained_model="/devb/sar2opt_diff_txt/sar2opt_diff_sarmap_Oct2_test1/model/model122400.pt"

python maintrain_s2on.py --gpu $gpu --data_path $data_path --condition_way $condition_way \
        --num_epochs $num_epochs --output_root $output_root \
        --save_interval $save_interval --log_interval $log_interval \
        --batch_size $batch_size  --map_swinunet $map_swinunet 