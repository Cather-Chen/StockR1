import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.llm_prompt_generator import batch_build_prompts


DEFAULT_DATA_DIR = "data/nqsp_stock/processed"


def _safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.log((numerator + eps) / (denominator + eps))


def _shift_with_first(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    return np.concatenate([arr[:1], arr[:-1]])


def _calc_input_log_returns(prices: Dict[str, List[float]], eps: float = 1e-8) -> np.ndarray:
    open_p = np.asarray(prices["Open"], dtype=np.float32)
    high_p = np.asarray(prices["High"], dtype=np.float32)
    low_p = np.asarray(prices["Low"], dtype=np.float32)
    close_p = np.asarray(prices["Close"], dtype=np.float32)
    volume_p = np.asarray(prices["Volume"], dtype=np.float32)

    prev_close = _shift_with_first(close_p)
    prev_volume = _shift_with_first(volume_p)

    return np.stack(
        [
            _safe_log_ratio(open_p, prev_close, eps),
            _safe_log_ratio(high_p, prev_close, eps),
            _safe_log_ratio(low_p, prev_close, eps),
            _safe_log_ratio(close_p, prev_close, eps),
            _safe_log_ratio(volume_p, prev_volume, eps),
        ],
        axis=1,
    )


def _calc_target_returns(
    input_prices: Dict[str, List[float]],
    output_prices: Dict[str, List[float]],
    eps: float = 1e-8,
) -> np.ndarray:
    open_out = np.asarray(output_prices["Open"], dtype=np.float32)
    close_out = np.asarray(output_prices["Close"], dtype=np.float32)
    high_out = np.asarray(output_prices["High"], dtype=np.float32)
    low_out = np.asarray(output_prices["Low"], dtype=np.float32)
    volume_out = np.asarray(output_prices["Volume"], dtype=np.float32)

    last_input_close = np.asarray(input_prices["Close"], dtype=np.float32)[-1]
    last_input_volume = np.asarray(input_prices["Volume"], dtype=np.float32)[-1]
    prev_close = np.concatenate([[last_input_close], close_out[:-1]])
    prev_volume = np.concatenate([[last_input_volume], volume_out[:-1]])

    return np.stack(
        [
            _safe_log_ratio(open_out, prev_close, eps),
            _safe_log_ratio(close_out, prev_close, eps),
            _safe_log_ratio(volume_out, prev_volume, eps),
            _safe_log_ratio(high_out, close_out, eps),
            _safe_log_ratio(low_out, close_out, eps),
        ],
        axis=1,
    )


def _ohlcv_tensor(prices: Dict[str, List[float]]) -> torch.Tensor:
    return torch.stack(
        [
            torch.tensor(prices["Open"], dtype=torch.float32),
            torch.tensor(prices["High"], dtype=torch.float32),
            torch.tensor(prices["Low"], dtype=torch.float32),
            torch.tensor(prices["Close"], dtype=torch.float32),
            torch.tensor(prices["Volume"], dtype=torch.float32),
        ],
        dim=1,
    )


class ReturnNormalizer:
    """
    Per-ticker normalizer for input log-return channels:
    [r_open, r_high, r_low, r_close, r_volume].
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, filename: str = "normalizer.json"):
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, filename)
        self.stats = {}
        if os.path.exists(self.path):
            self.load(self.path)

    def fit(self, stock_data: Dict[str, Dict]):
        for ticker, data in stock_data.items():
            all_returns = [_calc_input_log_returns(d["input"]["Prices"]) for d in data.values()]
            if not all_returns:
                continue
            ret_mat = np.concatenate(all_returns, axis=0)
            self.stats[ticker] = {
                "mu": ret_mat.mean(axis=0),
                "sigma": ret_mat.std(axis=0) + 1e-6,
            }
        self.save(self.path)
        print("Return normalizer saved to ", self.path)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({k: {"mu": v["mu"].tolist(), "sigma": v["sigma"].tolist()} for k, v in self.stats.items()}, f)

    def load(self, path: str):
        with open(path, "r") as f:
            stats = json.load(f)
        self.stats = {k: {"mu": np.array(v["mu"]), "sigma": np.array(v["sigma"])} for k, v in stats.items()}

    def normalize_raw(self, raw: np.ndarray, ticker: str) -> torch.Tensor:
        stats = self.stats[ticker]
        return torch.tensor((raw - stats["mu"]) / stats["sigma"], dtype=torch.float32)


class StockNormalizer:
    """
    Legacy OHLCV level normalizer kept for older model utilities.

    The active dataloader tasks use ReturnNormalizer.
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "normalizer.json")
        self.price_stats = {}
        self.volume_stats = {}
        if os.path.exists(self.path):
            self.load(self.path)

    def fit(self, stock_data: Dict):
        for ticker, data in stock_data.items():
            all_low, delta_o, delta_c, delta_h, all_vol = [], [], [], [], []
            for d in data.values():
                prices = d["input"]["Prices"]
                open_p = np.asarray(prices["Open"])
                high_p = np.asarray(prices["High"])
                low_p = np.asarray(prices["Low"])
                close_p = np.asarray(prices["Close"])
                all_low.extend(low_p)
                delta_o.extend(open_p - low_p)
                delta_c.extend(close_p - low_p)
                delta_h.extend(high_p - np.maximum.reduce([open_p, close_p, low_p]))
                all_vol.extend(prices["Volume"])

            price_mat = np.stack([all_low, delta_o, delta_c, delta_h], axis=1)
            self.price_stats[ticker] = {
                "mu": np.mean(price_mat, axis=0),
                "sigma": np.std(price_mat, axis=0) + 1e-6,
            }
            log_vol = np.log1p(all_vol)
            self.volume_stats[ticker] = {
                "mu": float(np.mean(log_vol)),
                "sigma": float(np.std(log_vol) + 1e-6),
            }
        self.save(self.path)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {
                    "price_stats": {
                        k: {"mu": v["mu"].tolist(), "sigma": v["sigma"].tolist()}
                        for k, v in self.price_stats.items()
                    },
                    "volume_stats": self.volume_stats,
                },
                f,
            )

    def load(self, path: str):
        with open(path, "r") as f:
            stats = json.load(f)
        self.price_stats = {
            k: {"mu": np.array(v["mu"]), "sigma": np.array(v["sigma"])}
            for k, v in stats["price_stats"].items()
        }
        self.volume_stats = stats["volume_stats"]

    def normalize(self, data: Dict, ticker: str) -> torch.Tensor:
        open_p = np.asarray(data["Open"])
        high_p = np.asarray(data["High"])
        low_p = np.asarray(data["Low"])
        close_p = np.asarray(data["Close"])
        volume = np.asarray(data["Volume"])

        price_feat = np.stack(
            [
                low_p,
                open_p - low_p,
                close_p - low_p,
                high_p - np.maximum.reduce([open_p, close_p, low_p]),
            ],
            axis=1,
        )
        price_stats = self.price_stats[ticker]
        price_feat = (price_feat - price_stats["mu"]) / price_stats["sigma"]

        vol_stats = self.volume_stats[ticker]
        volume_feat = ((np.log1p(volume) - vol_stats["mu"]) / vol_stats["sigma"])[:, None]
        return torch.tensor(np.concatenate([price_feat, volume_feat], axis=1), dtype=torch.float32)

    def denormalize(self, features: torch.Tensor, ticker: str) -> torch.Tensor:
        price_stats = self.price_stats[ticker]
        vol_stats = self.volume_stats[ticker]
        sigma = torch.as_tensor(price_stats["sigma"], device=features.device, dtype=features.dtype)
        mu = torch.as_tensor(price_stats["mu"], device=features.device, dtype=features.dtype)
        price_feat = features[:, :4] * sigma + mu
        volume_feat = features[:, 4:] * vol_stats["sigma"] + vol_stats["mu"]
        volume_feat = torch.exp(volume_feat) - 1
        return torch.cat([price_feat, volume_feat], dim=1)


class _BaseReturnDataset(Dataset):
    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        split: str = "train",
        normalizer_filename: str = "normalizer.json",
    ):
        self.data_dir = data_dir
        self.split = split
        self.normalizer = ReturnNormalizer(data_dir=data_dir, filename=normalizer_filename)
        self._load_macro_data()
        self._load_basic_info()
        self._load_fundamental_data()
        self._load_stock_data()
        if not self.normalizer.stats:
            self.normalizer.fit(self.stock_data)

    def _load_macro_data(self):
        with open(os.path.join(self.data_dir, "additional", "macro_150101_250815.json"), "r") as f:
            self.macro_data = json.load(f)

    def _load_basic_info(self):
        with open(os.path.join(self.data_dir, "additional", "basic_info.json"), "r") as f:
            self.basic_info = json.load(f)

    def _load_fundamental_data(self):
        fundamental_dir = os.path.join(self.data_dir, "additional", "fundamental")
        self.fundamental_data = {}
        for filename in os.listdir(fundamental_dir):
            if filename.endswith(".json"):
                ticker = filename.replace(".json", "")
                with open(os.path.join(fundamental_dir, filename), "r") as f:
                    self.fundamental_data[ticker] = json.load(f)

    def _load_stock_data(self):
        stock_dir = os.path.join(self.data_dir, self.split)
        self.stock_data = {}
        for filename in os.listdir(stock_dir):
            if filename.endswith(".json"):
                ticker = filename.replace(".json", "")
                with open(os.path.join(stock_dir, filename), "r") as f:
                    self.stock_data[ticker] = json.load(f)

    def _get_fundamental_data_for_date(self, date_str: str, fundamental_data: Dict) -> Dict:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        nearest_date = None
        nearest_data = {}
        for fund_date_str, data_dict in fundamental_data.items():
            fund_date = datetime.strptime(fund_date_str, "%Y-%m-%d")
            if nearest_date is None or nearest_date < fund_date <= target_date:
                nearest_date = fund_date
                nearest_data = data_dict
        return nearest_data

    def _get_macro_data_for_date(self, date_str: str, macro_data: Dict) -> Dict:
        data_dict = macro_data.get(date_str, {})
        nearest_data = {
            "yield_1_month": data_dict.get("yield_1_month"),
            "yield_3_month": data_dict.get("yield_3_month"),
            "yield_2_year": data_dict.get("yield_2_year"),
            "yield_10_year": data_dict.get("yield_10_year"),
            "yield_30_year": data_dict.get("yield_30_year"),
        }

        if data_dict.get("cpi") is not None:
            cpi_data = data_dict
        else:
            cpi_data = macro_data.get(date_str[:7] + "-01", {})

        nearest_data.update(
            {
                "cpi": cpi_data.get("cpi"),
                "cpi_core": cpi_data.get("cpi_core"),
                "pce_core": cpi_data.get("pce_core"),
                "pce_spending": cpi_data.get("pce_spending"),
                "market_5_year": cpi_data.get("market_5_year"),
                "market_10_year": cpi_data.get("market_10_year"),
            }
        )
        return nearest_data

    def _base_sample(self, ticker: str, date: str, stock_data: Dict, qa_data: Dict | None = None) -> Dict:
        ticker_upper = ticker.upper()
        fundamental_data = self.fundamental_data.get(ticker_upper, {})
        sample = {
            "ticker": ticker,
            "date_list": stock_data["input"]["Date"] + stock_data["output"]["Date"],
            "date_obj": datetime.strptime(date, "%Y-%m-%d"),
            "stock_data": stock_data,
            "macro_data": self._get_macro_data_for_date(date, self.macro_data),
            "fundamental_data": self._get_fundamental_data_for_date(date, fundamental_data),
            "basic_info": self.basic_info.get(ticker_upper, {}),
        }
        if qa_data:
            sample.update(
                {
                    "question": qa_data["question"],
                    "thinking": qa_data.get("thinking", ""),
                    "answer": qa_data.get("answer", ""),
                    "forecast_hint": qa_data.get("forecast_hint", ""),
                    "question_type": qa_data.get("question_type", qa_data.get("task", "pure_forecast")),
                    "task": qa_data["task"],
                }
            )
        return sample

    def _extract_return_features(self, sample: Dict):
        ticker = sample["ticker"]
        input_prices = sample["stock_data"]["input"]["Prices"]
        output_prices = sample["stock_data"]["output"]["Prices"]

        input_raw_returns = _calc_input_log_returns(input_prices)
        input_features = self.normalizer.normalize_raw(input_raw_returns, ticker)
        output_features = torch.tensor(_calc_target_returns(input_prices, output_prices), dtype=torch.float32)
        input_raw_features = _ohlcv_tensor(input_prices)
        output_raw_features = _ohlcv_tensor(output_prices)
        return input_features, output_features, input_raw_features, output_raw_features

    def __len__(self) -> int:
        return len(self.samples)


class TSPretrainDataset(_BaseReturnDataset):
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, split: str = "train"):
        super().__init__(data_dir=data_dir, split=split, normalizer_filename="normalizer_return.json")
        self.samples = self._create_samples()

    def _create_samples(self) -> List[Dict]:
        samples = []
        for ticker, stock_data_by_date in tqdm(self.stock_data.items(), desc="Creating TS samples"):
            for date, stock_data in stock_data_by_date.items():
                samples.append(self._base_sample(ticker=ticker, date=date, stock_data=stock_data))
        return samples

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        input_features, output_features, _, _ = self._extract_return_features(sample)
        input_prices = sample["stock_data"]["input"]["Prices"]
        return {
            "input_features": input_features,
            "output_features": output_features,
            "ticker": sample["ticker"],
            "date_list": sample["date_list"],
            "date_obj": sample["date_obj"],
            "last_close": torch.tensor(input_prices["Close"][-1], dtype=torch.float32),
            "last_volume": torch.tensor(input_prices["Volume"][-1], dtype=torch.float32),
            "output_prices": sample["stock_data"]["output"]["Prices"],
        }


class SFTDataset(_BaseReturnDataset):
    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        split: str = "train",
        file_name: str = "sft.jsonl",
        prompt_task: str = "sft_new",
    ):
        super().__init__(data_dir=data_dir, split=split, normalizer_filename="normalizer_rm.json")
        self.prompt_task = prompt_task
        qa_file = file_name if split == "train" else f"{split}_qa_data.jsonl"
        self.qa_data = self._load_qa_data(os.path.join(data_dir, qa_file))
        self.samples = self._create_samples()

    def _load_qa_data(self, path: str):
        with open(path, "r") as f:
            return [json.loads(line) for line in f]

    def _create_samples(self) -> List[Dict]:
        samples = []
        for qa_data in tqdm(self.qa_data, desc="Creating SFT samples"):
            ticker = qa_data["ticker"]
            date = qa_data["date"]
            stock_data = self.stock_data.get(ticker, {}).get(date, {})
            if not stock_data:
                continue
            samples.append(self._base_sample(ticker=ticker, date=date, stock_data=stock_data, qa_data=qa_data))
        return samples

    def _input_raw_stats(self, input_raw_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "mu": input_raw_features.mean(dim=0),
            "sigma": input_raw_features.std(dim=0),
        }

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        input_features, output_features, input_raw_features, output_raw_features = self._extract_return_features(sample)
        return {
            "input_features": input_features,
            "output_features": output_features,
            "ticker": sample["ticker"],
            "date_obj": sample["date_obj"],
            "date_list": sample["date_list"],
            "text": batch_build_prompts([sample], task=self.prompt_task)[0],
            "question": sample["question"],
            "thinking": sample.get("thinking", ""),
            "answer": sample.get("answer", ""),
            "forecast_hint": sample.get("forecast_hint", ""),
            "question_type": sample.get("question_type", sample.get("task", "pure_forecast")),
            "task": sample["task"],
            "input_raw_stats": self._input_raw_stats(input_raw_features),
            "input_raw_features": input_raw_features,
            "output_raw_features": output_raw_features,
        }
