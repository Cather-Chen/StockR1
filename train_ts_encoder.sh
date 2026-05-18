#! /bin/bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

SAVE_DIR="checkpoints/ts_encoder_test"

mkdir -p "${SAVE_DIR}" logs


CUDA_VISIBLE_DEVICES=0 python -u train_ts_encoder.py \
  --ts_patch_len 5 \
  --ts_num_layers 10 \
  --ts_heads 8 \
  --ts_d_model 512 \
  --batch_size 512 \
  --specify_date True \
  --start_date "2015-01-01" \
  --end_date "2025-01-01" \
  --save_dir "${SAVE_DIR}" \
  > "logs/ts_encoder_rm_p5_l10_h8_dm512_2015-01-01.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=1 python -u train_ts_encoder.py \
  --ts_patch_len 5 \
  --ts_num_layers 16 \
  --ts_heads 12 \
  --ts_d_model 768 \
  --batch_size 256 \
  --specify_date True \
  --start_date "2015-01-01" \
  --end_date "2025-01-01" \
  --save_dir "${SAVE_DIR}" \
  > "logs/ts_encoder_rm_p5_l16_h12_dm768_2015-01-01.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=2 python -u train_ts_encoder.py \
  --ts_patch_len 5 \
  --ts_num_layers 16 \
  --ts_heads 16 \
  --ts_d_model 1024 \
  --batch_size 128 \
  --specify_date True \
  --start_date "2015-01-01" \
  --end_date "2025-01-01" \
  --save_dir "${SAVE_DIR}" \
  > "logs/ts_encoder_rm_p5_l16_h16_dm1024_2010-01-01.log" 2>&1 &



# nohup bash train_ts_encoder.sh > /dev/null 2>&1 &
