"""Prepare StockR1 GRPO parquet data for VERL.

Outputs parquet rows with keys expected by RLHFDataset + reward manager + custom agent loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataset import SFTDataset
from dataloader.llm_prompt_generator import batch_build_prompts


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _build_user_prompt(raw_sample: Dict[str, Any], question: str) -> str:
    prompt_text = batch_build_prompts([raw_sample], task="sft_new")[0]                
    template = f"""<|im_start|>user\n{prompt_text}
                Question: {question}<|im_end|>\n
                <|im_start|>assistant\n"""
    return template


def _convert_split(split: str, file_name: str | None = None) -> list[dict]:
    if split == "train":
        ds = SFTDataset(split="train", file_name=file_name)
    else:
        ds = SFTDataset(split="test")

    rows = []
    for idx in range(len(ds)):
        sample = ds[idx]
        raw_sample = ds.samples[idx]

        input_features = sample["input_features"].tolist()
        output_features = sample["output_features"].tolist()
        input_raw_features = sample["input_raw_features"].tolist()
        output_raw_features = sample["output_raw_features"].tolist()

        last_close = _to_float(sample["input_raw_features"][-1, 3])
        last_volume = _to_float(sample["input_raw_features"][-1, 4])

        question = sample.get("question", "")
        answer = sample.get("answer", "")
        forecast_hint = sample.get("forecast_hint", "")
        question_type = sample.get("question_type", sample.get("task", "pure_forecast"))

        row = {
            "data_source": "financeqa/stock",
            "agent_name": "stockr1_two_stage_agent",
            "prompt": [
                {
                    "role": "user",
                    "content": _build_user_prompt(raw_sample, question),
                }
            ],
            "ability": "finance",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "answer": answer,
                    "forecast_hint": forecast_hint,
                    "output_features": output_features,
                    "output_raw_features": output_raw_features,
                    "question_type": question_type,
                    "ticker": sample.get("ticker", ""),
                },
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "question": question,
                "answer": answer,
                "ticker": sample.get("ticker", ""),
                "date": sample.get("date_obj", ""),
                "question_type": question_type,
                "tools_kwargs": {
                    "ts_forecast": {
                        "input_features": input_features,
                        "last_close": last_close,
                        "last_volume": last_volume,
                    }
                },
            },
        }
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VERL dataset (jsonl/parquet) for StockR1 GRPO.")
    parser.add_argument("--train_file", type=str, default="rl.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--with_test", action="store_true")
    parser.add_argument("--output_format", type=str, default="jsonl", choices=["jsonl", "parquet"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_rows = _convert_split(split="train", file_name=args.train_file)
    train_ext = "jsonl" if args.output_format == "jsonl" else "parquet"
    train_path = os.path.join(args.output_dir, f"train.{train_ext}")
    _save_rows(train_rows, train_path, args.output_format)
    print(f"Saved train data: {train_path} ({len(train_rows)} rows)")

    if args.with_test:
        test_rows = _convert_split(split="test")
        test_path = os.path.join(args.output_dir, f"test.{train_ext}")
        _save_rows(test_rows, test_path, args.output_format)
        print(f"Saved test data: {test_path} ({len(test_rows)} rows)")


def _save_rows(rows: list[dict], out_path: str, output_format: str) -> None:
    if output_format == "jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    # optional parquet path (requires pandas+pyarrow/fastparquet)
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Parquet export requires pandas (and parquet backend). "
            "Install dependencies or use --output_format jsonl."
        ) from e

    pd.DataFrame(rows).to_parquet(out_path, index=False)


if __name__ == "__main__":
    main()


# python grpo/prepare_verl_dataset.py --train_file rl.jsonl --output_dir data/nqsp_stock/verl_grpo/ --output_format parquet
