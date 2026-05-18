#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

export WANDB_API_KEY=[WANDB_API_KEY]
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3,4,5,6,7

NUM_PROCESSES=5

GROUP_NAME="sft_new"
SAVE_DIR="checkpoints/sft_new"
LOG_DIR="logs/sft_new"


mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

# accelerate launch \
#   --num_processes "${NUM_PROCESSES}" \
#   --num_machines 1 \
#   --mixed_precision bf16 \
#   --dynamo_backend no \
#   --deepspeed_config_file ds_config.json \
#   sft_new.py \
#   --task sft_new \
#   --model_name Qwen/Qwen3-4B-Instruct-2507 \
#   --special_tokens "<answer>" "</answer>" "<forecast_ts>" "</forecast_ts>" "<forecast_router>" "<forecast_hint>" "</forecast_hint>" \
#   --per_device_batch_size 1 \
#   --grad_accum_steps 8 \
#   --sft_epochs 5 \
#   --hint_only_epochs 100 \
#   --eval_every_steps -1 \
#   --lora_enable \
#   --lora_r 8 \
#   --lora_alpha 16 \
#   --lora_dropout 0.05 \
#   --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
#   --lr 2e-5 \
#   --group_name "${GROUP_NAME}" \
#   --save_dir "${SAVE_DIR}" \
#   --ts_soft_tokens \
#   --ts_encoder_checkpoint checkpoints/ts_encoder/dm512_heads8_layers10_patch5_specify_dateTrue_start2015-01-01_end2025-01-01 \
#   --run_name "qwen3-4b-s" \
#   > "${LOG_DIR}/qwen3-4b-s.log" 2>&1


accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --deepspeed_config_file ds_config.json \
  sft_new.py \
  --task sft_new \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --special_tokens "<answer>" "</answer>" "<forecast_ts>" "</forecast_ts>" "<forecast_router>" "<forecast_hint>" "</forecast_hint>" \
  --per_device_batch_size 1 \
  --grad_accum_steps 8 \
  --sft_epochs 5 \
  --hint_only_epochs 100 \
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
  --ts_encoder_checkpoint checkpoints/ts_encoder/dm768_heads12_layers16_patch5_specify_dateTrue_start2015-01-01_end2025-01-01 \
  --run_name "qwen3-4b-m" \
  > "${LOG_DIR}/qwen3-4b-m.log" 2>&1


  # accelerate launch \
  # --num_processes "${NUM_PROCESSES}" \
  # --num_machines 1 \
  # --mixed_precision bf16 \
  # --dynamo_backend no \
  # --deepspeed_config_file ds_config.json \
  # sft_new.py \
  # --task sft_new \
  # --model_name Qwen/Qwen3-4B-Instruct-2507 \
  # --special_tokens "<answer>" "</answer>" "<forecast_ts>" "</forecast_ts>" "<forecast_router>" "<forecast_hint>" "</forecast_hint>" \
  # --per_device_batch_size 1 \
  # --grad_accum_steps 8 \
  # --sft_epochs 5 \
  # --hint_only_epochs 100 \
  # --eval_every_steps -1 \
  # --lora_enable \
  # --lora_r 8 \
  # --lora_alpha 16 \
  # --lora_dropout 0.05 \
  # --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  # --lr 2e-5 \
  # --group_name "${GROUP_NAME}" \
  # --save_dir "${SAVE_DIR}" \
  # --ts_soft_tokens \
  # --ts_encoder_checkpoint checkpoints/ts_encoder/dm1024_heads16_layers16_patch5_specify_dateTrue_start2015-01-01_end2025-01-01 \
  # --run_name "qwen3-4b-l" \
  # > "${LOG_DIR}/qwen3-4b-l.log" 2>&1


# accelerate launch \
#   --num_processes "${NUM_PROCESSES}" \
#   --num_machines 1 \
#   --mixed_precision bf16 \
#   --dynamo_backend no \
#   --deepspeed_config_file ds_config.json \
#   sft_new.py \
#   --task sft_new \
#   --model_name Qwen/Qwen3-8B \
#   --special_tokens "<answer>" "</answer>" "<forecast_ts>" "</forecast_ts>" "<forecast_router>" "<forecast_hint>" "</forecast_hint>" \
#   --per_device_batch_size 1 \
#   --grad_accum_steps 8 \
#   --sft_epochs 15 \
#   --hint_only_epochs 100 \
#   --eval_every_steps -1 \
#   --lora_enable \
#   --lora_r 8 \
#   --lora_alpha 16 \
#   --lora_dropout 0.05 \
#   --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
#   --lr 2e-5 \
#   --group_name "${GROUP_NAME}" \
#   --save_dir "${SAVE_DIR}" \
#   --ts_soft_tokens \
#   --ts_encoder_checkpoint checkpoints/ts_encoder/dm512_heads8_layers10_patch5_specify_dateTrue_start2015-01-01_end2025-01-01 \
#   --run_name "qwen3-8b-s" \
#   > "${LOG_DIR}/qwen3-8b-s.log" 2>&1



# accelerate launch \
#   --num_processes "${NUM_PROCESSES}" \
#   --num_machines 1 \
#   --mixed_precision bf16 \
#   --dynamo_backend no \
#   --deepspeed_config_file ds_config.json \
#   sft_new.py \
#   --task sft_new \
#   --model_name Qwen/Qwen3-8B \
#   --special_tokens "<answer>" "</answer>" "<forecast_ts>" "</forecast_ts>" "<forecast_router>" "<forecast_hint>" "</forecast_hint>" \
#   --per_device_batch_size 1 \
#   --grad_accum_steps 8 \
#   --sft_epochs 10 \
#   --hint_only_epochs 100 \
#   --eval_every_steps -1 \
#   --lora_enable \
#   --lora_r 8 \
#   --lora_alpha 16 \
#   --lora_dropout 0.05 \
#   --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
#   --lr 2e-5 \
#   --group_name "${GROUP_NAME}" \
#   --save_dir "${SAVE_DIR}" \
#   --ts_soft_tokens \
#   --ts_encoder_checkpoint checkpoints/ts_encoder/dm768_heads12_layers16_patch5_specify_dateTrue_start2015-01-01_end2025-01-01 \
#   --run_name "qwen3-8b-m" \
#   > "${LOG_DIR}/qwen3-8b-m.log" 2>&1

# nohup bash sft_new.sh > /dev/null 2>&1 &
