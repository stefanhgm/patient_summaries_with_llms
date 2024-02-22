#!/bin/bash
# This script wraps the parameter tuning script and submits it to the cluster.
# * Only iterate over all tuning parameters
# * Only create experiment folder if not run already
# -> Rest is done in the main script

# General
model_name_dir="Llama-2-70b-hf" # Llama-2-7b-hf Llama-7-70b-hf
run_dir="mimic-iv-note-di-bhc_Llama-2-70b-hf_4000_600_chars_100_valid" # 4000_600_chars_100_valid long_data_100_valid

# Cluster
project="/home/s/s_hegs02/scratch/mimic-iv-note-di-bhc"
code="/home/s/s_hegs02/patient_summaries_with_llms"
# Local
# project="/home/s_hegs02/mimic-iv-note-di-bhc"
# code="/home/s_hegs02/patient_summaries_with_llms"
data_path="${project}/dataset"
output_path="${project}/models/${model_name_dir}/${run_dir}"

# Parameters
for lora_rank in 8 32; do  # 8 32, if possible might also check 4, 16
    for lora_alpha in 8 32; do  # 8 32
        for lora_dropout in 0.05 0.1; do # 0.05 0.1
            for num_target_modules in 2 4; do # 2 4
                for learning_rate in "2e-5" "2e-4"; do # First round only "2e-5" "2e-4", full: "5e-6" "2e-5" "5e-4" "2e-4" "5e-5"
                    # Define run folder
                    # folder_name="debug"
                    folder_name="lora_rank_${lora_rank}_lora_alpha_${lora_alpha}_lora_dropout_${lora_dropout}_num_target_modules_${num_target_modules}_learning_rate_${learning_rate}"
                    experiment_path="${output_path}/${folder_name}"

                    if [ ! -d "$experiment_path" ]; then
                        echo "Starting experiment: $experiment_path"
                        mkdir -p "$experiment_path"
                        # Cluster
                        sbatch ${code}/scripts/parameter_tuning_llama.sh ${experiment_path} ${lora_rank} ${lora_alpha} ${lora_dropout} ${num_target_modules} ${learning_rate}
                        # Local
                        # bash ${code}/scripts/parameter_tuning_llama.sh ${experiment_path} ${lora_rank} ${lora_alpha} ${lora_dropout} ${num_target_modules} ${learning_rate}
                    else
                        echo "X Experiment already exists: $experiment_path"
                    fi
done done done done done
