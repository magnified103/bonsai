#!/bin/zsh

## TODO put arguments correct and test distributed ray

trial_number=${1}
num_trials=${2}
data_file=${3}
shift 3

mkdir tuning_results

#rm -rf /tmp/checkpoint*
#rm -rf ~/ray_results/*

# Run the tuning script
python3 -m training.tune_ffnn_blocks --num_trials ${num_trials}  --output_dir tuning_results/trial_${trial_number} --data_file ${data_file} $@