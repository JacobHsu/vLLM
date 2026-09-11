#!/bin/bash
set -e

export PATH="$HOME/.local/bin:$PATH"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

cd ~/vllm-quickstart
source .venv/bin/activate

CUDA_DIR="$HOME/vllm-quickstart/.venv/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CUDA_DIR/bin:$PATH"

vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name qwen2.5-7b-bf16
