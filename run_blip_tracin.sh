#!/bin/bash
#SBATCH --job-name=blip_tracin
#SBATCH --account=bgek-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=blip_tracin_%j.out
#SBATCH --error=blip_tracin_%j.err

cd /u/yhuang48/if_4_icl/tda_4_mllm_hallucination
python blip_hallucination_tracin.py
