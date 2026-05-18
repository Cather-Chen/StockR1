#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

export WANDB_API_KEY=[WANDB_API_KEY]
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3,4,5,6,7

NUM_PROCESSES=5

GROUP_NAME="sft_v1"
SAVE_DIR="checkpoints/sft_v1"
LOG_DIR="logs/sft_v1"

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --deepspeed_config_file ds_config.json \
  sftv1.py \
  --task sft_v1 \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --special_tokens "<analysis>" "<answer>" "</answer>" \
  --per_device_batch_size 1 \
  --grad_accum_steps 8 \
  --epochs 15 \
  --eval_every_steps -1 \
  --lora_enable \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lr 2e-5 \
  --group_name "${GROUP_NAME}" \
  --save_dir "${SAVE_DIR}" \
  --ts_soft_tokens \
  --ts_encoder_checkpoint checkpoints/ts_encoder/dm512_heads8_layers10_patch5_specify_dateTrue_start2015-01-01_end2025-01-01 \
  --run_name "qwen3-4b-s-sft-v1" \
  > "${LOG_DIR}/qwen3-4b-s-sft-v1.log" 2>&1

# nohup bash sft_v1.sh > /dev/null 2>&1 &
