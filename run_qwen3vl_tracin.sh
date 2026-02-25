#!/bin/bash
#SBATCH --job-name=qwen3vl_tracin
#SBATCH --account=bgek-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=qwen3vl_tracin_%j.out
#SBATCH --error=qwen3vl_tracin_%j.err

cd /u/yhuang48/if_4_icl/tda_4_mllm_hallucination

export DEBUG_TRAIN_SIZE=50
export DEBUG_EVAL_SIZE=10

python qwen3vl_hallucination_tracin.py
