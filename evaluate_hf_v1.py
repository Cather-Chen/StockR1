import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer

from dataloader.dataset import SFTDataset
from dataloader.llm_prompt_generator import build_llm_prompt
from model.ts_llm_v1 import TS2LLM_v2Config, TS2QwenModel_v2
import re

PROJECT_PATH = os.environ.get("PROJECT_PATH", str(Path(__file__).resolve().parent))
print(f"Using PROJECT_PATH: {PROJECT_PATH}")
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
DATA_DIR = os.path.join(PROJECT_PATH, "data/nqsp_stock/processed")
MAX_SAMPLES = 1000
RESULTS_DIR = os.path.join(PROJECT_PATH, "tmp")
JUDGE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 4000
VERBOSE_OUTPUT = False
RUN_JUDGE = True
TS_ENCODER_CHECKPOINT = (
    os.path.join(PROJECT_PATH, "checkpoints/ts_encoder/")
    +
    "dm1024_heads16_layers16_patch5_specify_dateTrue_start2015-01-01_end2025-01-01"
)

# print(f"Using TS_ENCODER_CHECKPOINT: {TS_ENCODER_CHECKPOINT}")

def load_ts2qwen_v2_from_checkpoint(
    hf_path: str,
    ts_encoder_checkpoint: str,
    ts_soft_tokens: bool = False,
) -> Tuple[TS2QwenModel_v2, AutoTokenizer]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=True, padding_side="left")
    cfg = TS2LLM_v2Config(
        tokenizer=tokenizer,
        qwen_model_name=hf_path,
        llm_dim=2560,
        ts_encoder_checkpoint=ts_encoder_checkpoint,
        freeze_qwen=False,
        lora_enable=False,
        lora_r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        is_eval=True,
        ts_soft_tokens=ts_soft_tokens,
    )

    model = TS2QwenModel_v2(cfg)
    model.out_days = getattr(model.ts_encoder, "out_days", cfg.out_days)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def returns_to_ohlcv(
    returns: torch.Tensor,
    last_close: torch.Tensor,
    last_volume: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    def _inv_safe_log_ratio(log_ratio: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return torch.exp(log_ratio) * (denominator + eps) - eps

    B, H, _ = returns.shape
    device, dtype = returns.device, returns.dtype
    prev_close = last_close.view(B, 1).to(device=device, dtype=dtype)
    prev_volume = last_volume.view(B, 1).to(device=device, dtype=dtype)

    outputs = []
    for k in range(H):
        r_on = returns[:, k, 0]
        r_c = returns[:, k, 1]
        r_v = returns[:, k, 2]
        delta_h = returns[:, k, 3]
        delta_l = returns[:, k, 4]

        next_close = _inv_safe_log_ratio(r_c, prev_close[:, 0])
        open_price = _inv_safe_log_ratio(r_on, prev_close[:, 0])
        high_price = _inv_safe_log_ratio(delta_h, next_close)
        low_price = _inv_safe_log_ratio(delta_l, next_close)
        volume = _inv_safe_log_ratio(r_v, prev_volume[:, 0])

        outputs.append(torch.stack([open_price, high_price, low_price, next_close, volume], dim=-1))
        prev_close = next_close.view(B, 1)
        prev_volume = volume.view(B, 1)
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def get_analysis_hidden_batch(
    model: TS2QwenModel_v2,
    tokenizer: AutoTokenizer,
    prompts: List[str],
) -> torch.Tensor:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.qwen.device)
    outputs = model.qwen(**inputs, output_hidden_states=True, return_dict=True)

    analysis_id = tokenizer.convert_tokens_to_ids("<analysis>")
    input_ids = inputs["input_ids"]
    positions = input_ids == analysis_id
    last_pos = []
    for i in range(input_ids.size(0)):
        pos = torch.nonzero(positions[i], as_tuple=False).flatten()
        if pos.numel() == 0:
            last_pos.append(input_ids.size(1) - 1)
        else:
            last_pos.append(pos[-1].item())
    last_pos = torch.tensor(last_pos, device=outputs.hidden_states[-1].device)
    return outputs.hidden_states[-1][torch.arange(input_ids.size(0), device=last_pos.device), last_pos]


@torch.no_grad()
def forecast_ohlcv_batch(
    model: TS2QwenModel_v2,
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    prompt_texts: List[str],
) -> List[Dict[str, List[float]]]:
    analysis_prompts = [
        f"<|im_start|>user\n{prompt_text}<analysis>\n"
        f"Question: {sample['question']}<|im_end|>\n<|im_start|>assistant\n"
        for sample, prompt_text in zip(samples, prompt_texts)
    ]
    analysis_hidden = get_analysis_hidden_batch(model, tokenizer, analysis_prompts)
    # analysis_hidden = None
    device = model.qwen.device
    ts_inputs = torch.stack([s["input_features"] for s in samples], dim=0).to(
        device=device, dtype=torch.bfloat16
    )
    forecasting_dict = model.llm_forecasting(ts_inputs, analysis_hidden)
    pred_returns = forecasting_dict["mean_returns"]
    uncertainty = forecasting_dict["uncertainty"]
    uncertainty = uncertainty.float().cpu().tolist()
    last_close = torch.stack([s["input_raw_features"][-1, 3] for s in samples], dim=0).to(
        device=device, dtype=torch.bfloat16
    )
    last_volume = torch.stack([s["input_raw_features"][-1, 4] for s in samples], dim=0).to(
        device=device, dtype=torch.bfloat16
    )
    pred_ohlcv = returns_to_ohlcv(pred_returns, last_close, last_volume).float().cpu().tolist()

    forecasts = []
    for row in pred_ohlcv:
        open_price = [float(v[0]) for v in row]
        high_price = [float(v[1]) for v in row]
        low_price = [float(v[2]) for v in row]
        close_price = [float(v[3]) for v in row]
        volume = [float(v[4]) for v in row]
        forecasts.append(
            {
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": volume,
            }
        )
    return forecasts, uncertainty


@torch.no_grad()
def run_inference_batch(
    model: TS2QwenModel_v2,
    tokenizer: AutoTokenizer,
    prompts: List[str],
) -> List[str]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.qwen.device)
    output_ids = model.qwen.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    outputs = []
    for i, out in enumerate(output_ids):
        gen_ids = out[int(input_lengths[i]):].tolist()
        outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=False))
    return outputs


def extract_answer(text: str) -> str:
    text = text.replace("<|endoftext|>", "")
    text = text.replace("<|im_end|>", "")
    assistant_marker = "<|im_start|>assistant"
    if assistant_marker in text:
        text = text.rsplit(assistant_marker, 1)[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    answer_matches = list(re.finditer(answer_pattern, text, re.DOTALL))
    if answer_matches:
        return answer_matches[-1].group(1).strip()

    think_end_matches = list(re.finditer(r"</think>", text))
    if think_end_matches:
        return text[think_end_matches[-1].end():].strip()

    return text.strip()




def format_forecast_text(forecast: Dict[str, List[float]]) -> str:
    return (
        "The forecasted time series for future 10 days' prices and volume for your reference during reasoning:\n"
        f" Close price: {forecast['Close']}\n"
        f" Low price: {forecast['Low']}\n"
        f" High price: {forecast['High']}\n"
        f" Open price: {forecast['Open']}\n"
        f" Volume: {forecast['Volume']}"
    )

def format_ground_text(sample: Dict) -> str:
    gt_close_price = sample['stock_data']['output']['Prices']['Close']
    gt_low_price = sample['stock_data']['output']['Prices']['Low']
    gt_high_price = sample['stock_data']['output']['Prices']['High']
    gt_open_price = sample['stock_data']['output']['Prices']['Open']
    gt_volume = sample['stock_data']['output']['Prices']['Volume']
    return (
        "The forecasted time series for future 10 days' prices and volume for your reference during reasoning:\n"
        f" Close price: {gt_close_price}\n"
        f" Low price: {gt_low_price}\n"
        f" High price: {gt_high_price}\n"
        f" Open price: {gt_open_price}\n"
        f" Volume: {gt_volume}"
    )


def generate_completion_hf_model(prompt: str, pipe, terminators) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=2000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][-1]["content"]


def run_judge(results: List[Dict]) -> Tuple[List[Dict], Dict[str, float]]:
    pipe = transformers.pipeline(
        "text-generation",
        model=JUDGE_MODEL_NAME,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    tasks = sorted({item.get("task", "unknown") for item in results})
    results_metrics = {task: [] for task in tasks}

    for item in tqdm(results, desc="Judging"):
        task = item.get("task", "unknown")
        if task == "analysis":
            item["judge"] = None
            continue

        prompt = (
            "Given the following question, groundtruth answer and the model prediction, evaluate if the model prediction is correct.\n"
            f"Question: {item['question']}\n"
            f"Groundtruth Answer: {item['ground_truth']}\n"
            f"Model Prediction: {item['answer']}\n\n"
            "Evaluate if the model prediction is correct based on the question and the answer.\n"
            "If answer is a scalar and the absolute difference between the prediction and the answer is less than 1%, "
            "then take it as correct.\n"
            "Return 1 if correct, 0 if incorrect, without any other words."
        )

        judge_value = -1
        for _ in range(3):
            judge_text = generate_completion_hf_model(prompt, pipe, terminators).strip()
            if judge_text in ("0", "1"):
                judge_value = int(judge_text)
                break
        item["judge"] = judge_value
        if judge_value in (0, 1) and task in results_metrics:
            results_metrics[task].append(judge_value)

    metrics: Dict[str, float] = {}
    overall_scores = []
    for task, scores in results_metrics.items():
        if scores:
            metrics[task] = sum(scores) / len(scores)
            overall_scores.append(metrics[task])
        else:
            metrics[task] = 0.0
    metrics["overall"] = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    return results, metrics


def resolve_results_path(results_dir: str, hf_path: str) -> str:
    model_name = hf_path.split("/")[-1]
    path = os.path.join(results_dir, f"{model_name}_qa.json")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TS2Qwen from an HF checkpoint path.")
    parser.add_argument("--cuda_visible_devices", type=str, default="0")
    parser.add_argument("--hf_path", type=str, default="catherpker/stockr1-8B-v1")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--ts_encoder_checkpoint", type=str, default=TS_ENCODER_CHECKPOINT)
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--no_judge", action="store_true")
    parser.add_argument("--judge_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    global MAX_NEW_TOKENS
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    MAX_NEW_TOKENS = args.max_new_tokens
    run_judge_enabled = RUN_JUDGE and not args.no_judge

    if args.judge_only:
        if not args.results_path:
            raise ValueError("--judge_only requires --results_path")

        results_path = resolve_results_path(args.results_dir, args.hf_path)
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        for item in results:
            raw_prediction = item.get("raw_prediction", "")
            item["answer"] = extract_answer(raw_prediction)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if run_judge_enabled:
            judged_results, metrics = run_judge(results)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(judged_results, f, ensure_ascii=False, indent=2)

            metrics_path = results_path.replace("_qa.json", "_qa_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        return

    if not args.hf_path:
        raise ValueError("--hf_path is required unless --judge_only is set")

    hf_path = args.hf_path
    run_tag = hf_path.split("/")[-1]

    model, tokenizer = load_ts2qwen_v2_from_checkpoint(
        hf_path=hf_path,
        ts_encoder_checkpoint=args.ts_encoder_checkpoint,
        ts_soft_tokens=False,
    )
    dataset = SFTDataset(data_dir=args.data_dir, split="test")
    valid_indices = [i for i in range(len(dataset)) if dataset.samples[i]["task"] != "analysis"]
    sample_count = min(args.max_samples, len(valid_indices))
    sample_indices = random.sample(valid_indices, sample_count)

    os.makedirs(args.results_dir, exist_ok=True)

    results = []
    for start in tqdm(range(0, len(sample_indices), args.batch_size), desc="Inference"):
        try:
            batch_indices = sample_indices[start:start + args.batch_size]
            raw_samples = [dataset.samples[i] for i in batch_indices]
            samples = [dataset[i] for i in batch_indices]
            prompt_texts = [build_llm_prompt(s, task="sftv1")[0] for s in raw_samples]
            forecasts, uncertainty_list = forecast_ohlcv_batch(model, tokenizer, samples, prompt_texts)
            prompts = []
            for sample, prompt_text, forecast in zip(samples, prompt_texts, forecasts):
                forecast_text = format_forecast_text(forecast)
                prompts.append(
                    (
                    f"<|im_start|>user\n{prompt_text}<analysis>\n"
                    f"{forecast_text}\n"
                    "Please reason step by step. Place your reasoning trace between <think> and </think>.\n"
                    "Then, provide your answer between <answer> and </answer>\n"
                    f"Question: {sample['question']}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                    )
                )
            raw_outputs = run_inference_batch(model, tokenizer, prompts)
            for sample, raw_output, uncertainty in zip(samples, raw_outputs, uncertainty_list):
                answer = extract_answer(raw_output)
                date_str = sample["date_obj"].strftime("%Y-%m-%d")
                custom_id = f"{sample['ticker']}-{date_str}-{sample['task']}-{run_tag}"

                # score = compute_score(answer, sample["answer"], extra_info={"question_type": sample["task"], "uncertainty": uncertainty})
                
                results.append(
                    {
                        "custom_id": custom_id,
                        "ticker": sample["ticker"],
                        "date": date_str,
                        "task": sample["task"],
                        "question": sample["question"],
                        "ground_truth": sample["answer"],
                        "answer": answer,
                        "raw_prediction": raw_output,
                    }
                )
                if VERBOSE_OUTPUT:
                    print(f"answer: {raw_output}")
        except Exception as exc:
            print(f"Batch starting at index {start} failed: {exc}")
            traceback.print_exc()
            continue

    try:
        results_path = os.path.join(args.results_dir, f"{run_tag}_qa.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"Failed to save results: {exc}")


    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if run_judge_enabled:
        judged_results, metrics = run_judge(results)
        # judged_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME_SHORT}_qa_judge.json")
        # with open(judged_path, "w", encoding="utf-8") as f:
        #     json.dump(judged_results, f, ensure_ascii=False, indent=2)

        metrics_path = os.path.join(args.results_dir, f"{run_tag}_qa_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
