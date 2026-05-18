import re
from typing import Dict, Optional, Union

import numpy as np
import torch


ArrayLikeOHLCV = Union[np.ndarray, torch.Tensor]


def returns_to_ohlcv(
    returns: torch.Tensor,
    last_close: torch.Tensor,
    last_volume: torch.Tensor,
    eps: float = 1e-8,
    force_float32: bool = False,
) -> torch.Tensor:
    """
    Convert forecasted return features to raw OHLCV.

    returns: [B, H, 5] in order [r_on, r_c, r_v, delta_h, delta_l]
    output:  [B, H, 5] in order [Open, High, Low, Close, Volume]
    """
    if force_float32:
        returns = returns.to(torch.float32)
        last_close = last_close.to(torch.float32)
        last_volume = last_volume.to(torch.float32)

    bsz, horizon, _ = returns.shape
    device, dtype = returns.device, returns.dtype
    prev_close = last_close.view(bsz, 1).to(device=device, dtype=dtype)
    prev_volume = last_volume.view(bsz, 1).to(device=device, dtype=dtype)

    def _inv_safe_log_ratio(log_ratio: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return torch.exp(log_ratio) * (denominator + eps) - eps

    outputs = []
    for k in range(horizon):
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
        prev_close = next_close.view(bsz, 1)
        prev_volume = volume.view(bsz, 1)

    return torch.stack(outputs, dim=1)


def sample_returns_from_pred(pred_params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Sample return trajectories from forecasting distribution parameters.

    r_on/r_c/r_v use Gaussian samples; delta_h/delta_l use predicted quantiles.
    """
    eps_on = torch.randn_like(pred_params["mu_on"])
    eps_c = torch.randn_like(pred_params["mu_c"])
    eps_v = torch.randn_like(pred_params["mu_v"])

    r_on = pred_params["mu_on"] + pred_params["sigma_on"] * eps_on
    r_c = pred_params["mu_c"] + pred_params["sigma_c"] * eps_c
    r_v = pred_params["mu_v"] + pred_params["sigma_v"] * eps_v
    return torch.stack([r_on, r_c, r_v, pred_params["q_high"], pred_params["q_low"]], dim=-1)


def format_forecast_ts(
    raw_ohlcv: ArrayLikeOHLCV,
    decimals: int = 4,
    close_tag_period: bool = True,
    contradiction_word: str = "contradicts",
) -> str:
    if isinstance(raw_ohlcv, torch.Tensor):
        raw_ohlcv = raw_ohlcv.detach().cpu().numpy()

    def _fmt(arr: np.ndarray) -> str:
        return "[" + ", ".join(f"{float(x):.{decimals}f}" for x in arr) + "]"

    close_price = _fmt(raw_ohlcv[:, 3])
    low_price = _fmt(raw_ohlcv[:, 2])
    high_price = _fmt(raw_ohlcv[:, 1])
    open_price = _fmt(raw_ohlcv[:, 0])
    volume = _fmt(raw_ohlcv[:, 4])
    suffix = "</forecast_ts>." if close_tag_period else "</forecast_ts>"
    return (
        "<forecast_ts>The time-series decoder forecasts the future 10 days' prices and volume: \n"
        f" Close price: {close_price} \n"
        f" Low price: {low_price} \n"
        f" High price: {high_price} \n"
        f" Open price: {open_price} \n"
        f" Volume: {volume}. Carefully analyze this forecast and previous forecast hints if they {contradiction_word}{suffix}"
    )


def extract_hint_block(text: str, tolerate_missing_close: bool = False) -> Optional[str]:
    match = re.search(r"(<forecast_hint>\s*.*?\s*</forecast_hint>)", text, re.DOTALL)
    if match:
        return match.group(1)

    if tolerate_missing_close:
        partial = re.search(r"(<forecast_hint>\s*.*)$", text, re.DOTALL)
        if partial:
            return partial.group(1) + "</forecast_hint>"
    return None


def cut_to_forecast_hint_end(text: str) -> str:
    match = re.search(r"(.*?</forecast_hint>)", text, re.DOTALL)
    return match.group(1) if match else text
