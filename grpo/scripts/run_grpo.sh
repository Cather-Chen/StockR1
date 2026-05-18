#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

# Use FlashAttention2 when available.
export USE_FLASH_ATTN=1
unset TRANSFORMERS_NO_FLASH_ATTN
WANDB_API_KEY="[WANDB_API_KEY]"

find_latest_checkpoint() {
  local run_dir="$1"
  local latest
  latest="$(find "${run_dir}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' \
    | awk -F/ '{print $NF}' \
    | awk -F- 'NF>=3 && $NF ~ /^[0-9]+$/ {print $0, $NF}' \
    | sort -k2,2n \
    | tail -n1 \
    | awk '{print $1}')"

  if [ -z "${latest}" ]; then
    echo "No checkpoint-* found under ${run_dir}" >&2
    exit 1
  fi
  echo "${run_dir}/${latest}"
}

resolve_base_model_from_adapter() {
  local adapter_dir="$1"
  local adapter_cfg="${adapter_dir}/adapter_config.json"
  if [ ! -f "${adapter_cfg}" ]; then
    echo "Qwen/Qwen3-4B-Instruct-2507"
    return
  fi

  python3 - <<'PY' "${adapter_cfg}"
import json
import sys

cfg_path = sys.argv[1]
try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(data.get("base_model_name_or_path") or "Qwen/Qwen3-4B-Instruct-2507")
except Exception:
    print("Qwen/Qwen3-4B-Instruct-2507")
PY
}
# Paths
TRAIN_PARQUET="${TRAIN_PARQUET:-${PROJECT_DIR}/data/nqsp_stock/verl_grpo/train.parquet}"
SFT_RUN_DIR="${SFT_RUN_DIR:-checkpoints/sft/qwen3-4b-m}"
LATEST_SFT_CHECKPOINT="${LATEST_SFT_CHECKPOINT:-$(find_latest_checkpoint "${SFT_RUN_DIR}")}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-${LATEST_SFT_CHECKPOINT}/tfmr}"
BASE_MODEL_PATH_FROM_ADAPTER="$(resolve_base_model_from_adapter "${SFT_ADAPTER_PATH}")"
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL_PATH_FROM_ADAPTER}}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-${SFT_ADAPTER_PATH}}"
TS_ENCODER_CHECKPOINT="${TS_ENCODER_CHECKPOINT:-checkpoints/ts_encoder/dm768_heads12_layers16_patch5_specify_dateTrue_start2015-01-01_end2025-01-01}"
HINT_CONDITIONER_CHECKPOINT="${HINT_CONDITIONER_CHECKPOINT:-${LATEST_SFT_CHECKPOINT}/heads.pt}"
SAVE_DIR="${SAVE_DIR:-checkpoints/grpo_verl/qwen3-4b-m}"
# Logging
PROJECT_NAME="${PROJECT_NAME:-stockr1_update}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_m_verl_vllm}"
LOGGER="${LOGGER:-wandb}"
USE_UNCERTAINTY_REWEIGHT="${USE_UNCERTAINTY_REWEIGHT:-0}"
LORA_MERGE_FOR_ROLLOUT="${LORA_MERGE_FOR_ROLLOUT:-1}"



# Safety guard: if MODEL_PATH is mistakenly set to a PEFT adapter dir (e.g. */tfmr),
# switch to its base model and treat it as lora_adapter_path.
if [ -f "${MODEL_PATH}/adapter_config.json" ]; then
  echo "[GRPO][warn] MODEL_PATH points to a LoRA adapter dir: ${MODEL_PATH}"
  if [ -z "${LORA_ADAPTER_PATH:-}" ]; then
    LORA_ADAPTER_PATH="${MODEL_PATH}"
  fi
  MODEL_PATH="$(resolve_base_model_from_adapter "${MODEL_PATH}")"
  echo "[GRPO][warn] reset base MODEL_PATH to: ${MODEL_PATH}"
  echo "[GRPO][warn] keep LORA_ADAPTER_PATH as: ${LORA_ADAPTER_PATH}"
fi

# Training
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}"
GROUP_SIZE="${GROUP_SIZE:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
TEST_FREQ="${TEST_FREQ:-0}"
LR="${LR:-1e-6}"
LORA_RANK="${LORA_RANK:-8}"
KL_COEF="${KL_COEF:-0.001}"

# Rollout
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
STAGE1_MAX_NEW_TOKENS="${STAGE1_MAX_NEW_TOKENS:-2048}"
STAGE2_MAX_NEW_TOKENS="${STAGE2_MAX_NEW_TOKENS:-1024}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.45}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}"



# Fill your wandb key here directly (as requested)

EXTRA_FLAGS=()
EXTRA_OVERRIDES=()
if [ "${USE_UNCERTAINTY_REWEIGHT}" = "1" ]; then
  EXTRA_FLAGS+=(--use_uncertainty_reweight)
fi
if [ "${LORA_MERGE_FOR_ROLLOUT}" = "1" ]; then
  EXTRA_OVERRIDES+=(actor_rollout_ref.model.lora.merge=true)
fi
if [ -d "${LORA_ADAPTER_PATH}" ]; then
  EXTRA_OVERRIDES+=(actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}")
fi

echo "[GRPO] SFT run dir: ${SFT_RUN_DIR}"
echo "[GRPO] latest SFT checkpoint: ${LATEST_SFT_CHECKPOINT}"
echo "[GRPO] base model_path (Qwen load): ${MODEL_PATH}"
echo "[GRPO] LoRA adapter path (SFT): ${LORA_ADAPTER_PATH}"
echo "[GRPO] hint_conditioner_checkpoint: ${HINT_CONDITIONER_CHECKPOINT}"
echo "[GRPO] lora merge for rollout sync: ${LORA_MERGE_FOR_ROLLOUT}"

python3 grpo/grpo.py \
  --train_parquet "${TRAIN_PARQUET}" \
  --model_path "${MODEL_PATH}" \
  --ts_encoder_checkpoint "${TS_ENCODER_CHECKPOINT}" \
  --hint_conditioner_checkpoint "${HINT_CONDITIONER_CHECKPOINT}" \
  --save_dir "${SAVE_DIR}" \
  --n_gpus_per_node "${N_GPUS_PER_NODE}" \
  --nnodes "${NNODES}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --ppo_mini_batch_size "${PPO_MINI_BATCH_SIZE}" \
  --ppo_micro_batch_size_per_gpu "${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  --rollout_log_prob_micro_batch_size_per_gpu "${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  --group_size "${GROUP_SIZE}" \
  --total_epochs "${TOTAL_EPOCHS}" \
  --test_freq "${TEST_FREQ}" \
  --lr "${LR}" \
  --lora_rank "${LORA_RANK}" \
  --kl_coef "${KL_COEF}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_response_length "${MAX_RESPONSE_LENGTH}" \
  --stage1_max_new_tokens "${STAGE1_MAX_NEW_TOKENS}" \
  --stage2_max_new_tokens "${STAGE2_MAX_NEW_TOKENS}" \
  --rollout_tp "${ROLLOUT_TP}" \
  --rollout_gpu_mem_util "${ROLLOUT_GPU_MEM_UTIL}" \
  --rollout_max_model_len "${ROLLOUT_MAX_MODEL_LEN}" \
  --project_name "${PROJECT_NAME}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --logger "${LOGGER}" \
  "${EXTRA_FLAGS[@]}" \
  --extra "${EXTRA_OVERRIDES[@]}" "$@"
