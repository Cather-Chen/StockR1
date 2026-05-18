"""
GRPO Reward Function for StockR1

Reward components in RL stage:
1. answer_reward: answer correctness
2. format_reward: required tag coverage
3. forecast_reward: direction accuracy on close returns (da3/da5/da10)
4. hint_accuracy: generated hint vs ground-truth hint accuracy
5. consistency_reward: generated hint vs TS-decoder-implied hint accuracy
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Import existing reward functions
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grpo.answer_reward import compute_score as compute_answer_score
from utils.forecast_utils import returns_to_ohlcv


# ==============================================================================
# 1. FORMAT REWARD
# ==============================================================================

REQUIRED_TOKENS = [
    "<forecast_router>",
    "<forecast_hint>",
    "</forecast_hint>",
    "<forecast_ts>",
    "</forecast_ts>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
]


def compute_format_reward(
    generated_text: str,
    required_tokens: Optional[List[str]] = None,
    partial_credit: bool = True,
) -> Dict[str, float]:
    if required_tokens is None:
        required_tokens = REQUIRED_TOKENS

    result = {
        "score": 0.0,
        "total_tokens": len(required_tokens),
        "present_tokens": 0,
    }

    for token in required_tokens:
        key = f"has_{token.replace('<', '').replace('>', '').replace('/', 'close_')}"
        is_present = token in generated_text
        result[key] = 1.0 if is_present else 0.0
        if is_present:
            result["present_tokens"] += 1

    if partial_credit:
        result["score"] = result["present_tokens"] / max(1, result["total_tokens"])
    else:
        result["score"] = 1.0 if result["present_tokens"] == result["total_tokens"] else 0.0

    return result


# ==============================================================================
# 2. HINT ACCURACY / CONSISTENCY HELPERS
# ==============================================================================

CATEGORICAL_FIELDS = [
    "direction",
    "volatility_level",
    "range_width_bin",
    "max_drawdown_bin",
    "peak_timing_bin",
    "trough_timing_bin",
    "monotonicity",
    "trendline_fit",
    "tail_risk_level",
]

NUMERICAL_FIELDS = [
    "start_value",
    "end_value",
    "max_value",
    "min_value",
    "mean_close",
    "end_change_pct",
    "range_pct",
    "turning_point_count",
]

_EPS = 1e-8


def extract_forecast_hint_json(text: str) -> Optional[Any]:
    match = re.search(r"<forecast_hint>\s*(.*?)\s*</forecast_hint>", text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", json_str, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    return None
    return None


def _coerce_hint_to_dict(hint: Any) -> Optional[Dict[str, Any]]:
    if isinstance(hint, dict):
        return hint
    if isinstance(hint, list):
        for item in hint:
            if isinstance(item, dict):
                return item
    return None


def _coerce_text_or_dict_hint(hint: Any) -> Optional[Dict[str, Any]]:
    if isinstance(hint, dict):
        return hint
    if isinstance(hint, str):
        parsed = extract_forecast_hint_json(hint)
        if parsed is not None:
            return _coerce_hint_to_dict(parsed)
        try:
            return _coerce_hint_to_dict(json.loads(hint))
        except Exception:
            return None
    return _coerce_hint_to_dict(hint)


def _hint_accuracy(
    gen_hint: Dict[str, Any],
    ref_hint: Dict[str, Any],
    categorical_weight: float = 0.6,
    numerical_weight: float = 0.4,
    value_rel_tolerance: float = 0.01,
    pct_abs_tolerance: float = 0.5,
) -> Dict[str, float]:
    result = {
        "score": 0.0,
        "categorical_score": 0.0,
        "numerical_score": 0.0,
        "hint_parsed": 1.0,
    }

    cat_matches = 0
    cat_total = 0
    for field in CATEGORICAL_FIELDS:
        if field in ref_hint:
            cat_total += 1
            ok = gen_hint.get(field, "") == ref_hint.get(field, "")
            result[f"cat_{field}_match"] = 1.0 if ok else 0.0
            if ok:
                cat_matches += 1
    if cat_total > 0:
        result["categorical_score"] = cat_matches / cat_total

    num_scores: List[float] = []
    value_fields = {"start_value", "end_value", "max_value", "min_value", "mean_close"}
    pct_fields = {"end_change_pct", "range_pct"}

    for field in NUMERICAL_FIELDS:
        if field not in ref_hint or ref_hint[field] is None:
            continue
        gen_val = gen_hint.get(field)
        if gen_val is None:
            result[f"num_{field}_score"] = 0.0
            num_scores.append(0.0)
            continue

        try:
            gv = float(gen_val)
            rv = float(ref_hint[field])
        except (ValueError, TypeError):
            result[f"num_{field}_score"] = 0.0
            num_scores.append(0.0)
            continue

        if field == "turning_point_count":
            score = 1.0 if int(round(gv)) == int(round(rv)) else 0.0
        elif field in pct_fields:
            abs_error = abs(gv - rv)
            score = 1.0 if abs_error <= pct_abs_tolerance else math.exp(-abs_error / (5 * pct_abs_tolerance))
        else:
            denom = max(abs(rv), 1e-6)
            rel_error = abs(gv - rv) / denom
            tol = value_rel_tolerance if field in value_fields else value_rel_tolerance
            score = 1.0 if rel_error <= tol else math.exp(-rel_error / (5 * tol))

        result[f"num_{field}_score"] = score
        num_scores.append(score)

    if num_scores:
        result["numerical_score"] = sum(num_scores) / len(num_scores)

    result["score"] = categorical_weight * result["categorical_score"] + numerical_weight * result["numerical_score"]
    return result


def compute_hint_accuracy_reward(
    generated_text: str,
    ground_truth_hint: Union[str, Dict[str, Any]],
) -> Dict[str, float]:
    result = {
        "score": 0.0,
        "categorical_score": 0.0,
        "numerical_score": 0.0,
        "hint_parsed": 0.0,
    }
    gen_hint = _coerce_hint_to_dict(extract_forecast_hint_json(generated_text))
    if gen_hint is None:
        return result

    gt_hint = _coerce_text_or_dict_hint(ground_truth_hint)
    if gt_hint is None:
        return result

    acc = _hint_accuracy(gen_hint, gt_hint)
    result.update(acc)
    return result


# ==============================================================================
# 3. FORECAST REWARD (DA3/DA5/DA10 ONLY)
# ==============================================================================


def compute_forecast_reward(
    pred_returns: torch.Tensor,
    target_returns: torch.Tensor,
    horizons: Optional[List[int]] = None,
) -> Dict[str, float]:
    result = {
        "score": 0.0,
        "da3": 0.0,
        "da5": 0.0,
        "da10": 0.0,
        "da_sum": 0.0,
        "da_count": 0.0,
    }
    if horizons is None:
        horizons = [3, 5, 10]

    if not isinstance(pred_returns, torch.Tensor):
        pred_returns = torch.tensor(pred_returns, dtype=torch.float32)
    if not isinstance(target_returns, torch.Tensor):
        target_returns = torch.tensor(target_returns, dtype=torch.float32)

    if pred_returns.dim() == 2:
        pred_returns = pred_returns.unsqueeze(0)
    if target_returns.dim() == 2:
        target_returns = target_returns.unsqueeze(0)

    close_pred = pred_returns[:, :, 1]
    close_tgt = target_returns[:, :, 1]
    horizon_scores = []
    total_h = close_pred.shape[1]

    for h in horizons:
        if total_h < h:
            continue
        pred_dir = torch.sign(close_pred[:, :h].sum(dim=1))
        tgt_dir = torch.sign(close_tgt[:, :h].sum(dim=1))
        da_h = (pred_dir == tgt_dir).to(torch.float32).mean().item()
        result[f"da{h}"] = da_h
        horizon_scores.append(da_h)

    if horizon_scores:
        result["da_sum"] = float(sum(horizon_scores))
        result["da_count"] = float(len(horizon_scores))
        result["score"] = result["da_sum"] / result["da_count"]

    return result


# ==============================================================================
# 4. CONSISTENCY REWARD (generated hint vs ts_hint from pred_returns)
# ==============================================================================


def _timing_bin(day_1_based: int) -> str:
    return f"t+{day_1_based}_t+{day_1_based}"


def _build_ts_hint_from_ohlcv(ohlcv: np.ndarray) -> Dict[str, Any]:
    close = ohlcv[:, 3].astype(np.float64)
    high = ohlcv[:, 1].astype(np.float64)
    low = ohlcv[:, 2].astype(np.float64)

    start_value = float(close[0])
    end_value = float(close[-1])
    max_value = float(np.max(high))
    min_value = float(np.min(low))
    mean_close = float(np.mean(close))

    end_change_pct = float((end_value - start_value) / (start_value + _EPS) * 100.0)
    range_pct = float((max_value - min_value) / (start_value + _EPS) * 100.0)

    if end_change_pct > 1.0:
        direction = "up"
    elif end_change_pct < -1.0:
        direction = "down"
    else:
        direction = "sideways"

    close_returns = np.diff(close) / np.maximum(close[:-1], _EPS)
    vol_pct = float(np.std(close_returns) * 100.0) if close_returns.size > 0 else 0.0
    if vol_pct > 3.0:
        volatility_level = "high"
    elif vol_pct > 1.5:
        volatility_level = "medium"
    else:
        volatility_level = "low"

    if range_pct < 3.0:
        range_width_bin = "narrow"
    elif range_pct < 8.0:
        range_width_bin = "moderate"
    else:
        range_width_bin = "wide"

    drawdown_from_start_pct = float((np.min(close) - start_value) / (start_value + _EPS) * 100.0)
    if drawdown_from_start_pct >= -1.0:
        max_drawdown_bin = "negligible"
    elif drawdown_from_start_pct >= -3.0:
        max_drawdown_bin = "small"
    elif drawdown_from_start_pct >= -7.0:
        max_drawdown_bin = "moderate"
    else:
        max_drawdown_bin = "large"

    if close_returns.size == 0:
        turning_point_count = 0
        monotonicity = "non_monotonic"
        tail_risk_level = "low"
    else:
        sign = np.sign(close_returns)
        sign[sign == 0] = 1
        turning_point_count = int(np.sum(sign[1:] * sign[:-1] < 0))
        if np.all(close_returns >= 0):
            monotonicity = "monotonic_up"
        elif np.all(close_returns <= 0):
            monotonicity = "monotonic_down"
        else:
            monotonicity = "non_monotonic"

        worst_ret = float(np.min(close_returns))
        if worst_ret < -0.03:
            tail_risk_level = "high"
        elif worst_ret < -0.015:
            tail_risk_level = "medium"
        else:
            tail_risk_level = "low"

    peak_day = int(np.argmax(high)) + 1
    trough_day = int(np.argmin(low)) + 1

    x = np.arange(len(close), dtype=np.float64)
    if len(close) >= 3:
        p = np.polyfit(x, close, deg=1)
        y_hat = p[0] * x + p[1]
        ss_res = float(np.sum((close - y_hat) ** 2))
        ss_tot = float(np.sum((close - np.mean(close)) ** 2)) + _EPS
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = 0.0

    if r2 > 0.70:
        trendline_fit = "strong"
    elif r2 > 0.40:
        trendline_fit = "moderate"
    else:
        trendline_fit = "weak"

    return {
        "future_window": f"t+1_t+{len(close)}",
        "start_value": start_value,
        "end_value": end_value,
        "max_value": max_value,
        "min_value": min_value,
        "mean_close": mean_close,
        "direction": direction,
        "end_change_pct": end_change_pct,
        "range_pct": range_pct,
        "volatility_level": volatility_level,
        "range_width_bin": range_width_bin,
        "max_drawdown_bin": max_drawdown_bin,
        "turning_point_count": turning_point_count,
        "peak_timing_bin": _timing_bin(peak_day),
        "trough_timing_bin": _timing_bin(trough_day),
        "monotonicity": monotonicity,
        "trendline_fit": trendline_fit,
        "tail_risk_level": tail_risk_level,
    }


def compute_consistency_reward(
    generated_text: str,
    pred_returns: Optional[torch.Tensor],
    last_close: Optional[float],
    last_volume: Optional[float],
) -> Dict[str, float]:
    result = {
        "score": 0.0,
        "categorical_score": 0.0,
        "numerical_score": 0.0,
        "hint_parsed": 0.0,
    }

    gen_hint = _coerce_hint_to_dict(extract_forecast_hint_json(generated_text))
    if gen_hint is None:
        return result
    result["hint_parsed"] = 1.0

    if pred_returns is None or last_close is None or last_volume is None:
        return result

    if not isinstance(pred_returns, torch.Tensor):
        pred_returns = torch.tensor(pred_returns, dtype=torch.float32)
    if pred_returns.dim() == 2:
        pred_returns = pred_returns.unsqueeze(0)

    pred_ohlcv = returns_to_ohlcv(
        pred_returns,
        last_close=torch.tensor([float(last_close)], dtype=torch.float32),
        last_volume=torch.tensor([float(last_volume)], dtype=torch.float32),
        force_float32=True,
    )
    ts_hint = _build_ts_hint_from_ohlcv(pred_ohlcv[0].detach().cpu().numpy())

    acc = _hint_accuracy(gen_hint, ts_hint)
    result.update(acc)
    return result


# ==============================================================================
# 5. COMBINED REWARD FUNCTION
# ==============================================================================


@dataclass
class RewardWeights:
    answer: float = 0.4
    consistency: float = 0.2
    format: float = 0.1
    forecast: float = 0.3
    hint_accuracy: float = 0.2


def compute_grpo_reward(
    generated_text: str,
    ground_truth_answer: str,
    ground_truth_hint: Optional[Union[str, Dict]] = None,
    pred_returns: Optional[torch.Tensor] = None,
    target_returns: Optional[torch.Tensor] = None,
    question_type: str = "pure_forecast",
    weights: Optional[RewardWeights] = None,
    return_breakdown: bool = True,
    last_close: Optional[float] = None,
    last_volume: Optional[float] = None,
) -> Dict[str, Any]:
    if weights is None:
        weights = RewardWeights()

    result: Dict[str, Any] = {"reward": 0.0}

    answer_result = compute_answer_score(
        data_source="financeqa/stock",
        solution_str=generated_text,
        ground_truth=ground_truth_answer,
        extra_info={"question_type": question_type},
    )
    answer_reward = float(answer_result["score"])

    format_result = compute_format_reward(generated_text)
    format_reward = float(format_result["score"])

    if ground_truth_hint is not None:
        hint_acc_result = compute_hint_accuracy_reward(generated_text, ground_truth_hint)
        hint_accuracy_reward = float(hint_acc_result["score"])
    else:
        hint_acc_result = {"score": 0.0}
        hint_accuracy_reward = 0.0

    consistency_result = compute_consistency_reward(
        generated_text=generated_text,
        pred_returns=pred_returns,
        last_close=last_close,
        last_volume=last_volume,
    )
    consistency_reward = float(consistency_result["score"])

    if pred_returns is not None and target_returns is not None:
        forecast_result = compute_forecast_reward(pred_returns, target_returns)
        forecast_reward = float(forecast_result["score"])
    else:
        forecast_result = {"score": 0.0, "da3": 0.0, "da5": 0.0, "da10": 0.0}
        forecast_reward = 0.0

    total = 0.0
    total_w = 0.0

    total += weights.answer * answer_reward
    total_w += weights.answer

    total += weights.format * format_reward
    total_w += weights.format

    total += weights.forecast * forecast_reward
    total_w += weights.forecast

    total += weights.hint_accuracy * hint_accuracy_reward
    total_w += weights.hint_accuracy

    total += weights.consistency * consistency_reward
    total_w += weights.consistency

    result["reward"] = total / total_w if total_w > 0 else 0.0

    if return_breakdown:
        result["answer_reward"] = answer_reward
        result["format_reward"] = format_reward
        result["forecast_reward"] = forecast_reward
        result["hint_accuracy"] = hint_accuracy_reward
        result["consistency_reward"] = consistency_reward
        result["da3"] = float(forecast_result.get("da3", 0.0))
        result["da5"] = float(forecast_result.get("da5", 0.0))
        result["da10"] = float(forecast_result.get("da10", 0.0))
        result["answer_details"] = answer_result
        result["format_details"] = format_result
        result["forecast_details"] = forecast_result
        result["hint_accuracy_details"] = hint_acc_result
        result["consistency_details"] = consistency_result

    return result


# ==============================================================================
# 6. BATCH REWARD COMPUTATION
# ==============================================================================


def compute_batch_rewards(
    generated_texts: List[str],
    ground_truth_answers: List[str],
    ground_truth_hints: Optional[List[Union[str, Dict]]] = None,
    pred_returns_list: Optional[List[torch.Tensor]] = None,
    target_returns_list: Optional[List[torch.Tensor]] = None,
    question_types: Optional[List[str]] = None,
    weights: Optional[RewardWeights] = None,
    last_close_list: Optional[List[float]] = None,
    last_volume_list: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, List[Dict]]:
    B = len(generated_texts)

    if ground_truth_hints is None:
        ground_truth_hints = [None] * B
    if pred_returns_list is None:
        pred_returns_list = [None] * B
    if target_returns_list is None:
        target_returns_list = [None] * B
    if question_types is None:
        question_types = ["pure_forecast"] * B
    if last_close_list is None:
        last_close_list = [None] * B
    if last_volume_list is None:
        last_volume_list = [None] * B

    rewards: List[float] = []
    details: List[Dict] = []

    for i in range(B):
        out = compute_grpo_reward(
            generated_text=generated_texts[i],
            ground_truth_answer=ground_truth_answers[i],
            ground_truth_hint=ground_truth_hints[i],
            pred_returns=pred_returns_list[i],
            target_returns=target_returns_list[i],
            question_type=question_types[i],
            weights=weights,
            return_breakdown=True,
            last_close=last_close_list[i],
            last_volume=last_volume_list[i],
        )
        rewards.append(float(out["reward"]))
        details.append(out)

    return torch.tensor(rewards, dtype=torch.float32), details


# ==============================================================================
# 7. REWARD NORMALIZATION
# ==============================================================================


class RewardNormalizer:
    def __init__(self, beta: float = 0.99, eps: float = 1e-8):
        self.beta = beta
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, rewards: torch.Tensor):
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item() if rewards.numel() > 1 else 0.0

        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.mean = self.beta * self.mean + (1 - self.beta) * batch_mean
            self.var = self.beta * self.var + (1 - self.beta) * batch_var

        self.count += 1

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        std = math.sqrt(self.var + self.eps)
        return (rewards - self.mean) / std

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
            "beta": self.beta,
            "eps": self.eps,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.mean = state.get("mean", 0.0)
        self.var = state.get("var", 1.0)
        self.count = state.get("count", 0)
        self.beta = state.get("beta", 0.99)
        self.eps = state.get("eps", 1e-8)
