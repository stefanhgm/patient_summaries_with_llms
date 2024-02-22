#!/bin/bash
# This script wraps the parameter tuning script and submits it to the cluster.
# * Only iterate over all tuning parameters
# * Only create experiment folder if not run already
# -> Rest is done in the main script

# General
model_name_dir="led-large-16384" # led-base-16384 led-large-16384
run_dir="mimic-iv-note-di-bhc_led-large-16384_4000_600_chars_100_valid" # 4000_600_chars_100_valid long_data_100_valid

# Cluster
project="/home/s/s_hegs02/scratch/mimic-iv-note-di-bhc"
code="/home/s/s_hegs02/patient_summaries_with_llms"
# Local
# project="/home/s_hegs02/mimic-iv-note-di-bhc"
# code="/home/s_hegs02/patient_summaries_with_llms"
data_path="${project}/dataset"
output_path="${project}/models/${model_name_dir}/${run_dir}"

# Parameters
for dropout in 0.05 0.1 0.2; do  # 0.05 0.1 0.2
    for learning_rate in "5e-4" "1e-5" "5e-5" "1e-6" "5e-6"; do  # "5e-4" "1e-5" "5e-5" "1e-6" "5e-6"
        # Define run folder
        # folder_name="debug"
        folder_name="dropout_${dropout}_learning_rate_${learning_rate}"
        experiment_path="${output_path}/${folder_name}"

        if [ ! -d "$experiment_path" ]; then
            echo "Starting experiment: $experiment_path"
            mkdir -p "$experiment_path"
            # Cluster
            sbatch ${code}/scripts/parameter_tuning_led.sh ${experiment_path} ${dropout} ${learning_rate}
            # Local
            # bash ${code}/scripts/parameter_tuning_led.sh ${experiment_path} ${dropout} ${learning_rate}
        else
            echo "X Experiment already exists: $experiment_path"
        fi
done done
