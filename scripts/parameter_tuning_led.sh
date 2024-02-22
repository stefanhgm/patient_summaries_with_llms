#!/bin/bash

#SBATCH --job-name=parameter_tuning
#SBATCH --output=/home/s/s_hegs02/logs/parameter_tuning-%J.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# GPU - LED-base
# --time=0-13:00:00
# --partition=gpuv100,gpu3090
# --cpus-per-task=2
# --gres=gpu:1
# --mem=50GB

# GPU - LED-large
#SBATCH --time=0-40:00:00
#SBATCH --partition=gpu3090,gputitanrtx,gpua100
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

# Load the JupyterLab module
ml palma/2022a
ml GCCcore/11.3.0
# ml CUDA/11.7.0

# Load conda
source /home/s/s_hegs02/.bashrc-slurm
 
# Load environment
conda activate ps_llm

# Change path
# cd /home/s/s_hegs02/patient_summaries_with_llms

# Run the application
echo "Running script"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# Set device with CUDA_VISIBLE_DEVICES
device="cuda"
echo "Device: $device"

# General
model="allenai/led-large-16384"
# Cluster
project="/home/s/s_hegs02/scratch/mimic-iv-note-di-bhc"
# Local
# project="/home/s_hegs02/mimic-iv-note-di-bhc"
data_path="${project}/dataset"
output_path=$1

# Experiment
max_steps="200000"
save_and_logging_steps="20000"

# General
batch_size="1"

# Parameters
# Default parameters
# dropout="0.1"
# learning_rate="5e-5"
dropout=$2
learning_rate=$3

python summarization/run_summarization_large_long.py \
	--model_name_or_path ${model} \
	--do_train --do_eval --do_predict \
	--train_file ${data_path}/train.json \
	--validation_file ${data_path}/valid_last_100.json \
	--test_file ${data_path}/valid_last_100.json \
	--output_dir ${output_path} \
	--max_steps ${max_steps} \
	--evaluation_strategy steps \
	--eval_steps ${save_and_logging_steps} \
	--save_steps ${save_and_logging_steps} \
	--load_best_model_at_end \
	--per_device_train_batch_size=${batch_size} \
	--per_device_eval_batch_size=${batch_size} \
	--dropout ${dropout} \
	--learning_rate ${learning_rate} \
	--predict_with_generate \
	--max_source_length 4096 \
	--max_target_length 350