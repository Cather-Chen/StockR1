"""Custom reward function adapter for VERL.

Expected signature by VERL reward manager:
    compute_score(data_source, solution_str, ground_truth, extra_info)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from grpo.reward import RewardWeights, compute_grpo_reward


def uncertainty_weight(
    u: float,
    u0: float = 0.30,
    u_hi: float = 0.50,
    w_hi: float = 0.50,
    u_cap: float = 1.0,
) -> float:
    if u < 0.0:
        u = 0.0
    if u > u_cap:
        u = u_cap
    if u <= u0:
        return 1.0

    denom = max(1e-8, (u_hi - u0))
    w_hi = min(max(w_hi, 1e-6), 1.0)
    alpha = -math.log(w_hi)
    t = (u - u0) / denom
    w = math.exp(-alpha * t)
    return min(1.0, max(0.0, w))


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
    answer_reward_weight: float = 0.4,
    consistency_reward_weight: float = 0.2,
    format_reward_weight: float = 0.1,
    forecast_reward_weight: float = 0.3,
    hint_accuracy_weight: float = 0.2,
    use_uncertainty_reweight: bool = False,
    uncertainty_u0: float = 0.30,
    uncertainty_u_hi: float = 0.50,
    uncertainty_w_hi: float = 0.50,
    uncertainty_u_cap: float = 1.0,
) -> Dict[str, Any]:
    del data_source
    extra_info = extra_info or {}

    if not isinstance(ground_truth, dict):
        ground_truth = {"answer": str(ground_truth)}

    weights = RewardWeights(
        answer=answer_reward_weight,
        consistency=consistency_reward_weight,
        format=format_reward_weight,
        forecast=forecast_reward_weight,
        hint_accuracy=hint_accuracy_weight,
    )

    pred_returns = None
    if isinstance(extra_info, dict) and extra_info.get("pred_returns") is not None:
        pred_returns = extra_info.get("pred_returns")

    target_returns = ground_truth.get("output_features")

    tools_kwargs = extra_info.get("tools_kwargs", {}) if isinstance(extra_info, dict) else {}
    tool_payload = tools_kwargs.get("ts_forecast", tools_kwargs) if isinstance(tools_kwargs, dict) else {}

    last_close = _safe_float(tool_payload.get("last_close")) if isinstance(tool_payload, dict) else None
    last_volume = _safe_float(tool_payload.get("last_volume")) if isinstance(tool_payload, dict) else None

    if last_close is None:
        last_close = _safe_float(extra_info.get("last_close")) if isinstance(extra_info, dict) else None
    if last_volume is None:
        last_volume = _safe_float(extra_info.get("last_volume")) if isinstance(extra_info, dict) else None

    base = compute_grpo_reward(
        generated_text=solution_str,
        ground_truth_answer=ground_truth.get("answer", ""),
        ground_truth_hint=ground_truth.get("forecast_hint"),
        pred_returns=pred_returns,
        target_returns=target_returns,
        question_type=ground_truth.get("question_type", extra_info.get("question_type", "pure_forecast")),
        weights=weights,
        return_breakdown=True,
        last_close=last_close,
        last_volume=last_volume,
    )

    reward = float(base.get("reward", 0.0))
    original_reward = reward

    uncertainty = None
    rollout_scores = extra_info.get("rollout_reward_scores", {}) if isinstance(extra_info, dict) else {}
    if isinstance(rollout_scores, dict) and "uncertainty" in rollout_scores:
        uncertainty = _safe_float(rollout_scores.get("uncertainty"))
    if uncertainty is None and isinstance(extra_info, dict) and "uncertainty" in extra_info:
        uncertainty = _safe_float(extra_info.get("uncertainty"))

    uncertainty_w = 1.0
    if use_uncertainty_reweight and uncertainty is not None:
        uncertainty_w = uncertainty_weight(
            uncertainty,
            u0=uncertainty_u0,
            u_hi=uncertainty_u_hi,
            w_hi=uncertainty_w_hi,
            u_cap=uncertainty_u_cap,
        )
        reward = reward * uncertainty_w

    out = {
        "score": float(min(1.0, max(0.0, reward))),
        "original_reward": float(min(1.0, max(0.0, original_reward))),
        "answer_reward": float(base.get("answer_reward", 0.0)),
        "format_reward": float(base.get("format_reward", 0.0)),
        "forecast_reward": float(base.get("forecast_reward", 0.0)),
        "hint_accuracy": float(base.get("hint_accuracy", 0.0)),
        "consistency_reward": float(base.get("consistency_reward", 0.0)),
        "da3": float(base.get("da3", 0.0)),
        "da5": float(base.get("da5", 0.0)),
        "da10": float(base.get("da10", 0.0)),
        "uncertainty": None if uncertainty is None else float(uncertainty),
        "uncertainty_weight": float(uncertainty_w),
    }
    return out
