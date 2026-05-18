"""External TS forecasting tool for VERL agent-loop rollout.

This module keeps non-HF TS components outside vLLM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from model.multichannel_ts_enc import MultichannelTSEncoder_rm
from utils.forecast_utils import cut_to_forecast_hint_end, extract_hint_block, format_forecast_ts, returns_to_ohlcv
from utils.hint_extract import ForecastHintConditioner, prepare_batch as prepare_hint_batch


@dataclass
class TSForecastToolConfig:
    ts_encoder_checkpoint: str
    hint_conditioner_checkpoint: Optional[str] = None
    hint_residual_scale: float = 0.05
    out_days: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TSForecastTool:
    def __init__(self, config: TSForecastToolConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.ts_encoder, self.out_days = self._load_ts_encoder(config.ts_encoder_checkpoint)
        self.hint_conditioner = ForecastHintConditioner(output_dim=self.out_days * 8)
        self.hint_residual_scale = config.hint_residual_scale

        self.ts_encoder.to(self.device, dtype=torch.bfloat16)
        self.hint_conditioner.to(self.device, dtype=torch.bfloat16)

        self.ts_encoder.eval()
        self.hint_conditioner.eval()
        for p in self.ts_encoder.parameters():
            p.requires_grad = False
        for p in self.hint_conditioner.parameters():
            p.requires_grad = False

        if config.hint_conditioner_checkpoint:
            self._load_hint_conditioner(config.hint_conditioner_checkpoint)

    def _load_ts_encoder(self, ts_encoder_dir: str):
        ckpt_path = os.path.join(ts_encoder_dir, "best_val.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"TS encoder checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        saved_args = checkpoint["args"]
        out_days = int(self.config.out_days or saved_args["out_days"])

        model = MultichannelTSEncoder_rm(
            input_len=saved_args["input_len"],
            in_channels=saved_args["ts_in_channels"],
            d_model=saved_args["ts_d_model"],
            nhead=saved_args["ts_heads"],
            num_layers=saved_args["ts_num_layers"],
            patch_len=saved_args["ts_patch_len"],
            dropout=saved_args["ts_dropout"],
            out_days=out_days,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
        return model, out_days

    def _load_hint_conditioner(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"hint_conditioner checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu")
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        cleaned = {}
        for k, v in state.items():
            if k.startswith("hint_conditioner."):
                cleaned[k[len("hint_conditioner.") :]] = v
            elif k.startswith("model.hint_conditioner."):
                cleaned[k[len("model.hint_conditioner.") :]] = v

        if not cleaned:
            cleaned = state

        self.hint_conditioner.load_state_dict(cleaned, strict=False)

    def _prepare_hint_batch(self, stage1_text: str, current_price: float) -> Optional[Dict[str, torch.Tensor]]:
        hint_block = extract_hint_block(stage1_text, tolerate_missing_close=True)
        if hint_block is None:
            return None

        try:
            batch = prepare_hint_batch(
                hint_strings=[hint_block],
                current_prices=[current_price],
            )
        except Exception:
            return None

        return {
            "numerical": batch["numerical"].to(self.device, dtype=torch.bfloat16),
            "categorical_ids": batch["categorical_ids"].to(self.device, dtype=torch.long),
            "price_relative": batch["price_relative"].to(self.device, dtype=torch.bfloat16),
        }

    @torch.no_grad()
    def _forecast_returns(
        self, ts_inputs: torch.Tensor, hint_batch: Optional[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        x = self.ts_encoder.embedding(ts_inputs)  # [B, C, N, D]
        bsz = x.shape[0]
        x = x.reshape(bsz, x.shape[1], -1)
        x = F.gelu(x)
        x_agg = x.mean(dim=1)

        raw_params = self.ts_encoder.param_head(x_agg).view(bsz, self.out_days, 8)
        if hint_batch is not None:
            cond_out = self.hint_conditioner(
                numerical=hint_batch["numerical"],
                categorical_ids=hint_batch["categorical_ids"],
                price_relative=hint_batch["price_relative"],
            )
            hint_params = cond_out["vector"].view(bsz, self.out_days, 8)
            raw_params = raw_params + self.hint_residual_scale * torch.tanh(hint_params)

        mu_on = raw_params[..., 0]
        log_sigma_on = raw_params[..., 1]
        mu_c = raw_params[..., 2]
        log_sigma_c = raw_params[..., 3]
        mu_v = raw_params[..., 4]
        log_sigma_v = raw_params[..., 5]
        q_high = raw_params[..., 6]
        q_low = raw_params[..., 7]
        mean_returns = torch.stack([mu_on, mu_c, mu_v, q_high, q_low], dim=-1)

        sigma_on = torch.exp(log_sigma_on)
        sigma_c = torch.exp(log_sigma_c)
        sigma_v = torch.exp(log_sigma_v)
        uncertainty = torch.sqrt(sigma_on**2 + sigma_c**2 + sigma_v**2).mean(dim=-1) / (3.0**0.5)
        uncertainty = torch.clamp(uncertainty, 0.0, 1.0)

        return {
            "mean_returns": mean_returns,
            "uncertainty": uncertainty,
        }

    @torch.no_grad()
    def forecast_from_stage1(
        self,
        stage1_text: str,
        ts_inputs: Any,
        last_close: float,
        last_volume: float,
    ) -> Dict[str, Any]:
        if not isinstance(ts_inputs, torch.Tensor):
            ts_inputs = torch.tensor(ts_inputs, dtype=torch.float32)
        if ts_inputs.dim() == 2:
            ts_inputs = ts_inputs.unsqueeze(0)

        ts_inputs = ts_inputs.to(self.device, dtype=torch.bfloat16)
        hint_batch = self._prepare_hint_batch(stage1_text, float(last_close))

        forecast_out = self._forecast_returns(ts_inputs, hint_batch=hint_batch)
        pred_returns = forecast_out["mean_returns"]
        uncertainty = forecast_out["uncertainty"]
        pred_ohlcv = returns_to_ohlcv(
            pred_returns,
            last_close=torch.tensor([float(last_close)], device=self.device),
            last_volume=torch.tensor([float(last_volume)], device=self.device),
            force_float32=True,
        )
        forecast_text = format_forecast_ts(pred_ohlcv[0], close_tag_period=False, contradiction_word="contradict")

        return {
            "forecast_text": forecast_text,
            "pred_returns": pred_returns[0].detach().cpu().tolist(),
            "uncertainty": float(uncertainty[0].detach().cpu().item()),
            "hint_used": hint_batch is not None,
            "stage1_prefix_text": cut_to_forecast_hint_end(stage1_text),
        }
