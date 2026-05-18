import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import random
from dataloader.dataset import SFTDataset
from model.ts_llm_v2 import TS2LLM_v2Config, TS2QwenModel_v2
from utils.forecast_utils import (
    cut_to_forecast_hint_end,
    extract_hint_block,
    format_forecast_ts,
    returns_to_ohlcv,
    sample_returns_from_pred,
)
from utils.hint_extract import prepare_batch
from tqdm import tqdm

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge. Determine whether the model prediction is correct given the question and "
    "ground-truth answer.\n"
    "Rules:\n"
    "1) Semantic equivalence is allowed for non-numeric answers (e.g., yes/no phrasing, equivalent wording).\n"
    "2) If both ground truth and prediction are scalar numbers (including percentages), mark correct when absolute relative error is less than 3%.\n"
    "3) If prediction is empty, refusal, or unrelated, mark incorrect.\n"
    "Output strictly one character: 1 for correct, 0 for incorrect. No extra words."
)

ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

def build_test_prompt(sample: Dict[str, torch.Tensor]) -> str:
    return (
        f"<|im_start|>user\n{sample['text']}\n"
        f"                Question: {sample['question']}<|im_end|>\n\n"
        f"                <|im_start|>assistant\n"
    )

class StopOnSubsequence(StoppingCriteria):
    def __init__(self, stop_ids: List[int]):
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        n = len(self.stop_ids)
        if n == 0 or input_ids.shape[1] < n:
            return False
        return input_ids[0, -n:].tolist() == self.stop_ids


def _clean_answer_text(text: str) -> str:
    text = text.replace("<|endoftext|>", "")
    text = text.replace("<|im_end|>", "")
    text = text.replace("</think>", "")
    text = text.replace("<answer>", "").replace("</answer>", "")
    text = text.strip()
    return re.sub(r"^[\s\.\:\;]+", "", text)


def parse_prediction_from_generated_text(generated_text: str) -> str:
    text = generated_text or ""
    matches = list(ANSWER_TAG_PATTERN.finditer(text))
    if matches:
        return _clean_answer_text(matches[-1].group(1)).strip()

    forecast_end = text.rfind("</forecast_ts>")
    if forecast_end != -1:
        start = forecast_end + len("</forecast_ts>")
        im_end = text.find("<|im_end|>", start)
        tail = text[start:] if im_end == -1 else text[start:im_end]
        return _clean_answer_text(tail).strip()

    im_end = text.find("<|im_end|>")
    fallback = text if im_end == -1 else text[:im_end]
    return _clean_answer_text(fallback).strip()


def build_judge_prompt(date: str, question: str, ground_truth: str, prediction: str) -> str:
    return (
        "Given the following today's date, question, groundtruth answer and the model prediction, evaluate if the model prediction is correct.\n"
        f"Today's Date: {date}\n"
        f"Question: {question}\n"
        f"Groundtruth Answer: {ground_truth}\n"
        f"Model Prediction: {prediction}\n\n"
        "Evaluate if the model prediction is correct based on the question and the answer.\n"
        "If answer is a scalar and the absolute difference between the prediction and the answer is less than 3%, then take it as correct.\n"
        "Return 1 if correct, 0 if incorrect, without any other words."
    )


@dataclass
class JudgeResult:
    judge: int
    raw_output: str


class HFJudge:
    def __init__(self, model_name: str, temperature: float, max_new_tokens: int):
        import transformers

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.terminators = [self.pipeline.tokenizer.eos_token_id]
        eot_id = self.pipeline.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(eot_id, int) and eot_id >= 0:
            self.terminators.append(eot_id)

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def judge_once(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            top_p=0.9,
        )
        return (outputs[0]["generated_text"][-1]["content"] or "").strip()


def parse_binary_judge(text: str) -> Optional[int]:
    stripped = (text or "").strip()
    if stripped in {"0", "1"}:
        return int(stripped)

    match = re.search(r"\b([01])\b", stripped)
    if match:
        return int(match.group(1))
    return None


def compute_judge_metrics_by_type(judged_rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    overall = {"total": 0, "judged_valid": 0, "correct": 0}
    by_type: Dict[str, Dict[str, float]] = {}

    for row in judged_rows:
        qtype = row.get("question_type") or "unknown"
        judge = row.get("judge")
        overall["total"] += 1
        by_type.setdefault(qtype, {"total": 0, "judged_valid": 0, "correct": 0})
        by_type[qtype]["total"] += 1

        if judge in (0, 1):
            overall["judged_valid"] += 1
            overall["correct"] += int(judge)
            by_type[qtype]["judged_valid"] += 1
            by_type[qtype]["correct"] += int(judge)

    for stats in by_type.values():
        valid = stats["judged_valid"]
        stats["accuracy"] = (stats["correct"] / valid) if valid > 0 else 0.0

    return {
        "overall": {
            "total": float(overall["total"]),
            "judged_valid": float(overall["judged_valid"]),
            "accuracy": (overall["correct"] / overall["judged_valid"]) if overall["judged_valid"] > 0 else 0.0,
        },
        "by_type": by_type,
    }


def judge_generated_records(args, records_path: Path, save_dir: Path) -> Dict[str, object]:
    judge_model = HFJudge(args.judge_model, args.judge_temperature, args.judge_max_new_tokens)
    judged_path = save_dir / "judged_records.jsonl"
    judged_rows: List[Dict] = []

    with records_path.open("r", encoding="utf-8") as in_f, judged_path.open("w", encoding="utf-8") as out_f:
        for line in tqdm(in_f, desc="Judging"):
            if not line.strip():
                continue
            item = json.loads(line)
            prediction = parse_prediction_from_generated_text(str(item.get("generated_text", "")))
            prompt = build_judge_prompt(
                date=str(item.get("date", "")),
                question=str(item.get("question", "")).strip(),
                ground_truth=str(item.get("ground_truth_answer", "")).strip(),
                prediction=prediction,
            )

            judge_value: Optional[int] = None
            raw_output = ""
            for _ in range(args.judge_retries):
                raw_output = judge_model.judge_once(prompt)
                judge_value = parse_binary_judge(raw_output)
                if judge_value is not None:
                    break
            if judge_value is None:
                judge_value = -1

            item["parsed_prediction"] = prediction
            item["judge"] = judge_value
            item["judge_raw_output"] = raw_output
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            judged_rows.append(item)

    metrics_by_type = compute_judge_metrics_by_type(judged_rows)
    metrics_path = save_dir / "judged_records_metrics.json"
    metrics_by_type_path = save_dir / "judged_records_metrics_by_type.json"
    metrics_path.write_text(json.dumps(metrics_by_type["overall"], ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_by_type_path.write_text(json.dumps(metrics_by_type, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[save] judged_records={judged_path}")
    print(f"[save] judge_metrics={metrics_path}")
    print(f"[save] judge_metrics_by_type={metrics_by_type_path}")
    return {
        "judged_records_path": str(judged_path),
        "judge_metrics_path": str(metrics_path),
        "judge_metrics_by_type_path": str(metrics_by_type_path),
        "judge_metrics": metrics_by_type["overall"],
        "judge_metrics_by_type": metrics_by_type,
    }


def load_model_and_tokenizer(
    qwen_checkpoint_dir: str,
    heads_path: str,
    ts_encoder_checkpoint: str,
    device: torch.device,
) -> Tuple[TS2QwenModel_v2, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(qwen_checkpoint_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg_kwargs = dict(
        tokenizer=tokenizer,
        qwen_model_name=qwen_checkpoint_dir,
        llm_dim=2560,
        ts_encoder_checkpoint=ts_encoder_checkpoint,
        ts_soft_tokens=True,
        freeze_qwen=False,
        lora_enable=False,
        is_eval=True,
    )

    cfg = TS2LLM_v2Config(**cfg_kwargs)
    model = TS2QwenModel_v2(cfg)
    heads_state = torch.load(heads_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(heads_state, strict=False)
    print(f"[load] qwen_checkpoint_dir={qwen_checkpoint_dir}")
    print(f"[load] heads_path={heads_path}")
    print(f"[load] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    model.to(device)
    model.eval()
    return model, tokenizer


def resolve_model_sources(args) -> Dict[str, object]:
    model_name_or_path = args.model_name_or_path
    local_model_path = Path(model_name_or_path)
    if local_model_path.exists():
        heads_path = local_model_path / "stockr1_heads.pt"
        ts_encoder_checkpoint = local_model_path / "stockr1_ts_encoder"
        if not heads_path.exists():
            raise FileNotFoundError(f"stockr1_heads.pt not found: {heads_path}")
        if not (ts_encoder_checkpoint / "best_val.pt").exists():
            raise FileNotFoundError(f"stockr1_ts_encoder/best_val.pt not found under: {local_model_path}")
        qwen_checkpoint_dir = str(local_model_path)
    else:
        heads_path = Path(hf_hub_download(repo_id=model_name_or_path, filename="stockr1_heads.pt"))
        ts_best = Path(hf_hub_download(repo_id=model_name_or_path, filename="stockr1_ts_encoder/best_val.pt"))
        ts_encoder_checkpoint = ts_best.parent
        qwen_checkpoint_dir = model_name_or_path

    return {
        "model_name_or_path": model_name_or_path,
        "qwen_checkpoint_dir": qwen_checkpoint_dir,
        "heads_path": str(heads_path),
        "ts_encoder_checkpoint": str(ts_encoder_checkpoint),
    }


def build_dataset_with_test_file(qa_file: str) -> SFTDataset:
    # Reuse SFTDataset feature pipeline; replace QA source with the requested file.
    ds = SFTDataset(split="test")
    ds.qa_data = ds._load_qa_data(qa_file)
    ds.samples = ds._create_samples()
    return ds


@torch.no_grad()
def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_sources = resolve_model_sources(args)

    model, tokenizer = load_model_and_tokenizer(
        qwen_checkpoint_dir=str(model_sources["qwen_checkpoint_dir"]),
        heads_path=str(model_sources["heads_path"]),
        ts_encoder_checkpoint=str(model_sources["ts_encoder_checkpoint"]),
        device=device,
    )

    dataset = build_dataset_with_test_file(args.qa_file)
    samples = [dataset[i] for i in range(len(dataset)) if dataset.samples[i]['task'] != 'analysis']
    sample_count = len(samples) if args.max_samples <= 0 else min(args.max_samples, len(samples))
    random_indices = random.sample(range(len(samples)), sample_count)
    print(f"[data] qa_file={args.qa_file} use={sample_count}")

    close_hint_ids = tokenizer("</forecast_hint>")["input_ids"]
    stop_criteria = StoppingCriteriaList([StopOnSubsequence(close_hint_ids)])
    save_text_dir = args.save_text_dir
    save_dir = Path(save_text_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    records_path = save_dir / "generated_records.jsonl"
    if records_path.exists():
        records_path.unlink()

    hint_used = 0

    for idx in tqdm(random_indices, desc="Inference"):
        sample = samples[idx]
        base_prompt = build_test_prompt(sample)

        ts_inputs = sample["input_features"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        last_close = sample["input_raw_features"][-1, 3].view(1).to(device=device, dtype=torch.float32)
        last_volume = sample["input_raw_features"][-1, 4].view(1).to(device=device, dtype=torch.float32)

        forecast_wo = model.llm_forecasting(ts_inputs, hint_batch=None)["mean_returns"]
        raw_wo = returns_to_ohlcv(forecast_wo, last_close, last_volume, force_float32=True)

        inputs = tokenizer(base_prompt, return_tensors="pt").to(device)
        gen_stage1 = model.qwen.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens_stage1,
            do_sample=False,
            use_cache=True,
            stopping_criteria=stop_criteria,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        stage1_text = tokenizer.decode(gen_stage1[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

        hint_block = extract_hint_block(stage1_text)
        if hint_block is not None:
            try:
                hint_batch = prepare_batch(
                    hint_strings=[hint_block],
                    current_prices=[float(last_close.item())],
                )
                hint_batch = {
                    "numerical": hint_batch["numerical"].to(device=device, dtype=torch.bfloat16),
                    "categorical_ids": hint_batch["categorical_ids"].to(device=device, dtype=torch.long),
                    "price_relative": hint_batch["price_relative"].to(device=device, dtype=torch.bfloat16),
                }
                if args.sample_forecast:
                    pred_with = model.llm_forecasting(ts_inputs, hint_batch=hint_batch)
                    forecast_with = sample_returns_from_pred(pred_with)
                else:
                    pred_with = model.llm_forecasting(ts_inputs, hint_batch=hint_batch)
                    forecast_with = pred_with["mean_returns"]
                stage1_prefix = cut_to_forecast_hint_end(stage1_text)
                hint_used += 1
            except Exception as exc:
                print(f"[warn] malformed forecast_hint at sample={idx}: {exc}")
                forecast_with = forecast_wo
                stage1_prefix = stage1_text
        else:
            forecast_with = forecast_wo
            stage1_prefix = stage1_text

        raw_with = returns_to_ohlcv(forecast_with, last_close, last_volume, force_float32=True)
        forecast_text = format_forecast_ts(raw_with[0].detach().cpu().numpy())
        prompt_stage2 = base_prompt + stage1_prefix + "\n" + forecast_text + "\n<think>"
        inputs2 = tokenizer(prompt_stage2, return_tensors="pt").to(device)
        gen_stage2 = model.qwen.generate(
            **inputs2,
            max_new_tokens=args.max_new_tokens_stage2,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        stage2_text = tokenizer.decode(gen_stage2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=False)

        print(f"\n===== SAMPLE {idx} | {sample['ticker']} | {sample['date_obj'].strftime('%Y-%m-%d')} =====")
        print(f"Question: {sample['question']}")
        print("[assistant text]")
        full_generated_text = stage1_prefix + "\n" + forecast_text + "\n" + stage2_text
        print(full_generated_text)

        raw_wo_np = raw_wo[0].detach().cpu().numpy()
        raw_with_np = raw_with[0].detach().cpu().numpy()
        record = {
            "sample_id": idx,
            "ticker": sample["ticker"],
            "date": sample["date_obj"].strftime("%Y-%m-%d"),
            "question": sample["question"],
            "question_type": sample["task"],
            "generated_text": full_generated_text,
            "without_forecast_hint_time_series": {
                "open": np.round(raw_wo_np[:, 0], 4).tolist(),
                "high": np.round(raw_wo_np[:, 1], 4).tolist(),
                "low": np.round(raw_wo_np[:, 2], 4).tolist(),
                "close": np.round(raw_wo_np[:, 3], 4).tolist(),
                "volume": np.round(raw_wo_np[:, 4], 4).tolist(),
            },
            "with_forecast_hint_time_series": {
                "open": np.round(raw_with_np[:, 0], 4).tolist(),
                "high": np.round(raw_with_np[:, 1], 4).tolist(),
                "low": np.round(raw_with_np[:, 2], 4).tolist(),
                "close": np.round(raw_with_np[:, 3], 4).tolist(),
                "volume": np.round(raw_with_np[:, 4], 4).tolist(),
            },
            "ground_truth_answer": sample.get("answer", ""),
        }
        with records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = {
        "meta": {
            **model_sources,
            "qa_file": args.qa_file,
            "num_samples": sample_count,
            "hint_used_count": hint_used,
            "hint_used_ratio": float(hint_used / max(sample_count, 1)),
        },
    }

    if args.run_judge:
        result["judge"] = judge_generated_records(args, records_path, save_dir)

    print("\n===== QA EVAL SUMMARY =====")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[save] records_jsonl={records_path}")
    summary_path = save_dir / "qa_eval_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="catherpker/stockr1-4b-v2")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--qa_file", type=str, default="data/nqsp_stock/processed/test_qa_data.jsonl")
    parser.add_argument("--max_samples", type=int, default=1000, help="<=0 means all samples")
    parser.add_argument("--max_new_tokens_stage1", type=int, default=2048)
    parser.add_argument("--max_new_tokens_stage2", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cuda_visible_devices", type=str, default="0")
    parser.add_argument("--sample_forecast", default=False)
    parser.add_argument("--save_text_dir", type=str, default="")
    parser.add_argument("--run_judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judge_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--judge_max_new_tokens", type=int, default=8)
    parser.add_argument("--judge_retries", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = parse_args()
    model_name = args.run_name or args.model_name_or_path.rstrip("/").split("/")[-1]
    print("evaluating model:", model_name)
    if not args.save_text_dir:
        args.save_text_dir = f"tmp/inference/{model_name}"

    if args.cuda_visible_devices != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    run(args)


# nohup python inference_demo.py > logs/inference_1000.log 2>&1 &
