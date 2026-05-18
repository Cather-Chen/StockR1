import re
import sys
from datetime import datetime
from functools import partial
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.append(".")

from dataloader.dataset import SFTDataset, TSPretrainDataset
from utils.hint_extract import prepare_batch


SUPPORTED_TASKS = ("ts_pretrain", "sft_v1", "sft_new")


@torch.no_grad()
def collate_fn_ts_pretrain(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "input_features": torch.stack([b["input_features"] for b in batch], dim=0),
        "output_features": torch.stack([b["output_features"] for b in batch], dim=0),
        "tickers": [b["ticker"] for b in batch],
        "dates": [b["date_list"] for b in batch],
        "last_close": torch.stack([b["last_close"] for b in batch], dim=0),
        "last_volume": torch.stack([b["last_volume"] for b in batch], dim=0),
    }


def _ensure_pad_token(tokenizer):
    if tokenizer is None:
        raise ValueError("Tokenizer is required for SFT dataloading.")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id or eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token


def _pad_to_batch_max(seqs: List[List[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    return torch.stack(
        [torch.tensor(seq + [pad_value] * (max_len - len(seq)), dtype=torch.long) for seq in seqs],
        dim=0,
    )


def _raw_feature_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw_stat_mu = torch.stack([b["input_raw_stats"]["mu"] for b in batch], dim=0)
    raw_stat_sigma = torch.stack([b["input_raw_stats"]["sigma"] for b in batch], dim=0)
    return {
        "input_features": torch.stack([b["input_features"] for b in batch], dim=0),
        "output_features": torch.stack([b["output_features"] for b in batch], dim=0),
        "tickers": [b["ticker"] for b in batch],
        "dates": [b["date_list"] for b in batch],
        "input_raw_stats": torch.cat([raw_stat_mu, raw_stat_sigma], dim=1),
        "input_raw_features": torch.stack([b["input_raw_features"] for b in batch], dim=0),
        "output_raw_features": torch.stack([b["output_raw_features"] for b in batch], dim=0),
    }


@torch.no_grad()
def collate_fn_sftv1(
    batch: List[Dict[str, Any]],
    tokenizer,
    max_text_len: int | None,
    split: str = "train",
    simple_text: bool = False,
) -> Dict[str, torch.Tensor]:
    _ensure_pad_token(tokenizer)
    input_id_lists = []
    attention_lists = []

    for sample in batch:
        if split == "train" and not simple_text:
            gt_close_price = sample["output_raw_features"][:, 3].tolist()
            gt_low_price = sample["output_raw_features"][:, 2].tolist()
            gt_high_price = sample["output_raw_features"][:, 1].tolist()
            gt_open_price = sample["output_raw_features"][:, 0].tolist()
            gt_volume = sample["output_raw_features"][:, 4].tolist()
            gt_ts_text = (
                "The forecasted time series for future 10 days' prices and volume for your reference during reasoning: \n"
                f" Close price: {gt_close_price} \n"
                f" Low price: {gt_low_price} \n"
                f" High price: {gt_high_price} \n"
                f" Open price: {gt_open_price} \n"
                f" Volume: {gt_volume}"
            )
            template = f"""<|im_start|>user
{sample["text"]}<analysis>
{gt_ts_text}
Please reason step by step. Place your reasoning trace between <think> and </think>.
Then, provide your answer between <answer> and </answer>
Question: {sample["question"]}<|im_end|>
<|im_start|>assistant
<think>{sample["thinking"]}</think><answer>{sample["answer"]}</answer><|im_end|>
"""
        else:
            template = f"""<|im_start|>user
{sample["text"]}<analysis>
"""

        seq_ids = tokenizer(template, add_special_tokens=False)["input_ids"]
        if max_text_len is not None and len(seq_ids) > max_text_len:
            seq_ids = seq_ids[-max_text_len:]
        input_id_lists.append(seq_ids)
        attention_lists.append([1] * len(seq_ids))

    out = {
        "text_ids": _pad_to_batch_max(input_id_lists, tokenizer.pad_token_id),
        "text_attention_mask": _pad_to_batch_max(attention_lists, 0),
    }
    out.update(_raw_feature_batch(batch))
    return out


@torch.no_grad()
def collate_fn_sft_new(
    batch: List[Dict[str, Any]],
    tokenizer,
    max_text_len: int | None,
    split: str = "train",
) -> Dict[str, torch.Tensor]:
    _ensure_pad_token(tokenizer)
    input_id_lists = []
    attention_lists = []
    loss_mask_lists = []
    hint_list = []
    current_prices = []

    think_open_ids = tokenizer("<think>", add_special_tokens=False)["input_ids"]
    forecast_open_ids = tokenizer("<forecast_ts>", add_special_tokens=False)["input_ids"]
    forecast_close_ids = tokenizer("</forecast_ts>", add_special_tokens=False)["input_ids"]

    def find_subseq(seq, subseq):
        for i in range(len(seq) - len(subseq) + 1):
            if seq[i : i + len(subseq)] == subseq:
                return i
        return -1

    def extract_hint_text(sample: Dict[str, Any]) -> str:
        hint_text = sample.get("forecast_hint", "")
        if hint_text and ("<forecast_hint>" in hint_text or "{" in hint_text):
            return hint_text
        match = re.search(r"<forecast_hint>\s*(.*?)\s*</forecast_hint>", sample.get("thinking", ""), re.DOTALL)
        if match:
            return f"<forecast_hint>{match.group(1)}</forecast_hint>"
        return "{}"

    for sample in batch:
        current_prices.append(float(sample["input_raw_features"][-1, 3].item()))
        hint_list.append(extract_hint_text(sample))

        if split == "train":
            gt_close_price = sample["output_raw_features"][:, 3].tolist()
            gt_low_price = sample["output_raw_features"][:, 2].tolist()
            gt_high_price = sample["output_raw_features"][:, 1].tolist()
            gt_open_price = sample["output_raw_features"][:, 0].tolist()
            gt_volume = sample["output_raw_features"][:, 4].tolist()
            gt_ts_text = (
                "<forecast_ts>The time-series decoder forecasts the future 10 days' prices and volume: \n"
                f" Close price: {gt_close_price} \n"
                f" Low price: {gt_low_price} \n"
                f" High price: {gt_high_price} \n"
                f" Open price: {gt_open_price} \n"
                f" Volume: {gt_volume}. Carefully analyze this forecast and previous forecast hints if they contradicts</forecast_ts>."
            )

            if sample["task"] == "new_sft_forecast":
                template = f"""<|im_start|>user
                            {sample["text"]}
                            Question: {sample["question"]}<|im_end|>
                            <|im_start|>assistant
                            {sample["thinking"]}<|im_end|>
                            """
            else:
                template = f"""<|im_start|>user
                            {sample["text"]}
                            Question: {sample["question"]}<|im_end|>
                            <|im_start|>assistant
                            {sample["forecast_hint"]}
                            {gt_ts_text}
                            <think>{sample["thinking"]}</think><answer>{sample["answer"]}</answer><|im_end|>
                            """
            seq_ids = tokenizer(template, add_special_tokens=False)["input_ids"]
            loss_mask = [0] * len(seq_ids)
            think_start = find_subseq(seq_ids, think_open_ids)
            if think_start != -1:
                loss_mask[think_start:] = [1] * (len(seq_ids) - think_start)

            if sample.get("task") == "new_sft_qa":
                forecast_start = find_subseq(seq_ids, forecast_open_ids)
                forecast_end = find_subseq(seq_ids, forecast_close_ids)
                if forecast_start != -1 and forecast_end != -1 and forecast_end >= forecast_start:
                    forecast_end += len(forecast_close_ids)
                    for idx in range(forecast_start, min(forecast_end, len(loss_mask))):
                        loss_mask[idx] = 0
        else:
            template = f"""<|im_start|>user
                        {sample["text"]}
                        Question: {sample["question"]}<|im_end|>
                        <|im_start|>assistant
                        """
            seq_ids = tokenizer(template, add_special_tokens=False)["input_ids"]
            loss_mask = [0] * len(seq_ids)

        if max_text_len is not None and len(seq_ids) > max_text_len:
            seq_ids = seq_ids[-max_text_len:]
            loss_mask = loss_mask[-max_text_len:]

        input_id_lists.append(seq_ids)
        attention_lists.append([1] * len(seq_ids))
        loss_mask_lists.append(loss_mask)

    out = {
        "text_ids": _pad_to_batch_max(input_id_lists, tokenizer.pad_token_id),
        "text_attention_mask": _pad_to_batch_max(attention_lists, 0),
        "text_loss_mask": _pad_to_batch_max(loss_mask_lists, 0),
        "tasks": [b["task"] for b in batch],
        "hint_feature_batch": prepare_batch(hint_strings=hint_list, current_prices=current_prices),
    }
    out.update(_raw_feature_batch(batch))
    return out


class StockDataLoader:
    def __init__(
        self,
        tokenizer,
        data_dir: str = "data/nqsp_stock/processed",
        batch_size: int = 32,
        num_workers: int = 4,
        task: str = "ts_pretrain",
    ):
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"task must be one of {SUPPORTED_TASKS}; got {task}")

        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task

        if task == "ts_pretrain":
            self.train_dataset = TSPretrainDataset(data_dir=data_dir, split="train")
            self.test_dataset = TSPretrainDataset(data_dir=data_dir, split="test")
        else:
            file_name = "train_qa_data.jsonl" if task == "sft_v1" else "sft.jsonl"
            prompt_task = "sftv1" if task == "sft_v1" else "sft_new"
            self.train_dataset = SFTDataset(data_dir=data_dir, split="train", file_name=file_name, prompt_task=prompt_task)
            self.test_dataset = SFTDataset(data_dir=data_dir, split="test", file_name=file_name, prompt_task=prompt_task)

    def _collate(self, split: str, simple_text: bool = False):
        if self.task == "ts_pretrain":
            return collate_fn_ts_pretrain
        if self.task == "sft_v1":
            return partial(collate_fn_sftv1, tokenizer=self.tokenizer, max_text_len=4096, split=split, simple_text=simple_text)
        return partial(collate_fn_sft_new, tokenizer=self.tokenizer, max_text_len=4096, split=split)

    def _dataset_for_split(self, split: str):
        if split == "train":
            return self.train_dataset
        if split == "test":
            return self.test_dataset
        raise ValueError("split must be 'train' or 'test'")

    def create_loader(self, split: str = "train", simple_text: bool = False, use_distributed_sampler: bool = True, **kwargs) -> DataLoader:
        dataset = self._dataset_for_split(split)
        sampler = DistributedSampler(dataset) if (use_distributed_sampler and dist.is_initialized()) else None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self._collate(split=split, simple_text=simple_text),
            **kwargs,
        )

    def get_ticker_specific_loader(self, ticker: str, split: str = "train", **kwargs) -> DataLoader:
        dataset = self._dataset_for_split(split)
        indices = [idx for idx in range(len(dataset)) if dataset[idx]["ticker"] == ticker]
        if not indices:
            raise ValueError(f"No samples found for ticker {ticker} in {split} split")
        return self._subset_loader(dataset, indices, split=split, **kwargs)

    def get_date_range_loader(self, start_date: str, end_date: str, split: str = "train", **kwargs) -> DataLoader:
        dataset = self._dataset_for_split(split)
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        indices = [idx for idx in range(len(dataset)) if start <= dataset[idx]["date_obj"] <= end]
        if not indices:
            raise ValueError(f"No samples found in date range {start_date} to {end_date} in {split} split")
        return self._subset_loader(dataset, indices, split=split, **kwargs)

    def _subset_loader(self, dataset, indices: List[int], split: str, **kwargs) -> DataLoader:
        subset = torch.utils.data.Subset(dataset, indices)
        sampler = DistributedSampler(subset) if dist.is_initialized() else None
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self._collate(split=split),
            **kwargs,
        )
