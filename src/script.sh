#!/bin/bash

#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --nodelist luna
#SBATCH --output=run_logs/train_slurm.%N.%j.log

source /vol/biomedic3/awk24/miniconda3/bin/activate


export NCCL_P2P_LEVEL=LOC
export RDMAV_FORK_SAFE=1
export OPENAI_LOG_FORMAT="stdout,log,csv,tensorboard"
export OPENAI_LOGDIR="/vol/biomedic3/awk24/code/conditional-flow-matching/examples/images/outputs"



# Redirect HuggingFace cache away from home dir (quota too small) to biomedic3
export HF_HOME="/vol/biomedic3/awk24/.cache/huggingface"
export PYTHONUNBUFFERED=1


# uv run collect_gemini_responses.py
# /homes/awk24/.local/bin/uv run collect_local_responses.py llama
# /homes/awk24/.local/bin/uv run collect_local_responses.py qwen
# /homes/awk24/.local/bin/uv run evaluate_responses_local.py qwen
/homes/awk24/.local/bin/uv run create_control_dataset.py
# /homes/awk24/.local/bin/uv run extend_master_with_ee.py

