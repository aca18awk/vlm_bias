#!/bin/bash

#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --nodelist loki
#SBATCH --output=run_logs/train_slurm.%N.%j.log

source /vol/biomedic3/awk24/miniconda3/bin/activate

export NCCL_P2P_LEVEL=LOC
export RDMAV_FORK_SAFE=1
export OPENAI_LOG_FORMAT="stdout,log,csv,tensorboard"
export OPENAI_LOGDIR="/vol/biomedic3/awk24/code/conditional-flow-matching/examples/images/outputs"
export GEMINI_API_KEY="AIzaSyDy_Fp-vGhcn5N0UqLNuKToE9ijDLVwTFU"

# uv run collect_gemini_responses.py
uv run evaluate_responses.py
