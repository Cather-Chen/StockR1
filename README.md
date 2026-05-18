# StockR1

This repository is for short-horizon stock forecasting and question answering. 

## Data and Checkpoints

Please download the following files first:

- Data file: <https://drive.google.com/file/d/1409me6JQ9UxVULub1XBiKXdaR6-gZt5I/view?usp=sharing>
  Put it under `data/`
- TS encoder checkpoint: <https://drive.google.com/file/d/1pBk4ogED58bpdQfhUW4Atx6qjllERBkq/view?usp=drive_link>
  Put it under `checkpoints/ts_encoder/`

Other `v1` and `v2` model checkpoints are hosted on Hugging Face.

Currently used Hugging Face model paths in this repo:

- `v1`: `catherpker/stockr1-8b-v1`
- `v2`: `catherpker/stockr1-4b-v2`, `catherpker/stockr1-8b-v2`

## Recommended Workflow

### 1. Train the time-series encoder

Main entry points:

- `train_ts_encoder.py`: trains the multichannel time-series encoder to predict future 10-day price/volume related returns
- `train_ts_encoder.sh`: example training commands

This stage produces the `ts_encoder` checkpoint used later by both SFT and RL.

### 2. SFT

There are two main SFT scripts:

- `sftv1.py`: earlier SFT training pipeline
- `sft_new.py`: newer SFT pipeline with structured formats such as `forecast_hint` and `forecast_ts`
- `sft_v1.sh`: launch script for `sftv1.py`
- `sft_new.sh`: launch script for `sft_new.py`

If you want to follow the current main path of this repo, start with `sft_new.py` / `sft_new.sh`.

### 3. RL / GRPO

Reinforcement learning related code lives under `grpo/`:

- `grpo/grpo.py`: main GRPO launcher that builds the VERL + vLLM training command
- `grpo/scripts/run_grpo.sh`: recommended launch script
- `grpo/reward.py`: reward logic
- `grpo/verl_reward_adapter.py`: adapter that connects custom rewards to VERL
- `grpo/verl_stock_agent_loop.py`: two-stage agent loop
- `grpo/ts_forecast_tool.py`: forecast tool used during training
- `grpo/prepare_verl_dataset.py`: prepares parquet data for VERL
- `grpo/config/agent_loop_stockr1.yaml`: agent loop configuration

The RL stage is intended to continue from an SFT checkpoint.

### 4. Evaluate

- `evaluate.py`: main evaluation script for loading models, generating answers, running the judge model, and saving results
- `evaluate_hf_v1.py`: an alternative HF-based evaluation script

## Repository Structure

### `dataloader/`

- `dataset.py`: defines `TSPretrainDataset` and `SFTDataset`, and converts raw stock samples into training samples
- `loader.py`: builds dataloaders and `collate_fn` logic for `ts_pretrain`, `sft_v1`, and `sft_new`
- `prompt.py`: prompt templates, mainly used for training data generation and output formatting
- `llm_prompt_generator.py`: assembles stock, news, fundamentals, and macro context into LLM input text

### `model/`

- `multichannel_ts_enc.py`: core multichannel time-series encoder used by `train_ts_encoder.py`
- `ts_enc_v1.py`: older TS encoder implementation
- `ts_llm_v1.py`: TS-LLM model used by SFT v1
- `ts_llm_v2.py`: TS-LLM model mainly used by SFT new and evaluation
- `loss.py`: loss functions
- `metrics.py`: evaluation metrics

### `layers/`

- `self_attention.py`, `attn_projection.py`: low-level attention modules

### `utils/`

- `forecast_utils.py`: utilities for forecast formatting, sampling, and post-processing
- `hint_extract.py`: utilities for handling `forecast_hint` and related intermediate structures
- `precompute.py`: precomputation helpers

### Other files

- `ds_config.json`: DeepSpeed configuration

## Suggested Reading Order

If you want to understand the repo quickly, read these files in order:

1. `train_ts_encoder.py`
2. `sft_new.py`
3. `grpo/grpo.py`
4. `evaluate.py`

If you mainly want to run experiments, start with:

1. `train_ts_encoder.sh`
2. `sft_new.sh`
3. `grpo/scripts/run_grpo.sh`
4. `evaluate.py`

## Notes

- The training stack depends on `wandb`, `accelerate`, `deepspeed`, and `verl`
- Scripts contain placeholder `WANDB_API_KEY` values and should be updated before running
