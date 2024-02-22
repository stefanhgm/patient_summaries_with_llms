#!/bin/bash

#SBATCH --job-name=parameter_tuning
#SBATCH --output=/home/s/s_hegs02/logs/parameter_tuning-%J.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# GPU - LLAMA 7B
# --time=0-12:00:00
# --partition=gpuhgx
# --cpus-per-task=4
# --gres=gpu:1
# --mem=50GB

# GPU - LLAMA 70B
#SBATCH --time=0-24:00:00
#SBATCH --partition=gpuhgx
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80GB

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
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Change path
# cd /home/s/s_hegs02/patient_summaries_with_llms

# Run the application
echo "Running script"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# echo ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS} - if SLURM_STEP_GPUS is not set, use SLURM_JOB_GPUS
echo "SLURM GPUs: $SLURM_JOB_GPUS"
# echo "Set CUDA_VISIBLE_DEVICES: $SLURM_JOB_GPUS"

# Set device with CUDA_VISIBLE_DEVICES
device="cuda"
echo "Device: $device"

# General
model="meta-llama/Llama-2-70b-hf"
# Cluster
project="/home/s/s_hegs02/scratch/mimic-iv-note-di-bhc"
# Local
# project="/home/s_hegs02/mimic-iv-note-di-bhc"
data_path="${project}/dataset"
output_path=$1

# Experiment
num_train_examples="100"
num_val_examples="100"
num_test_examples="100"
max_steps="100"
save_and_logging_steps="10"

# General
batch_size="1"
gradient_accumulation_steps="16"

# Parameters
# Default parameters
# lora_rank="8"
# lora_alpha="8"
# lora_dropout="0.1"
# num_target_modules="4"
# learning_rate="5e-4"
lora_rank=$2
lora_alpha=$3
lora_dropout=$4
num_target_modules=$5
learning_rate=$6

python summarization/fine_tune_llama.py \
    --model_name_or_path ${model} \
	--data_path ${data_path} \
	--output_path ${output_path} \
	--device ${device} \
	--max_steps ${max_steps} \
	--save_and_logging_steps ${save_and_logging_steps} \
	--batch_size ${batch_size} \
	--gradient_accumulation_steps ${gradient_accumulation_steps} \
	--lora_rank ${lora_rank} \
	--lora_alpha ${lora_alpha} \
	--lora_dropout ${lora_dropout} \
	--num_target_modules ${num_target_modules} \
	--learning_rate ${learning_rate} \
	--num_train_examples ${num_train_examples} \
	--num_val_examples ${num_val_examples} \
	--num_test_examples ${num_test_examples} \
