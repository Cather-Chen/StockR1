"""Custom VERL agent-loop for StockR1 two-stage generation.

Flow:
1) vLLM decode stage-1 to forecast hint region
2) external TS tool call to produce <forecast_ts>
3) vLLM decode stage-2 answer continuation
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from grpo.ts_forecast_tool import TSForecastTool, TSForecastToolConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register("stockr1_two_stage_agent")
class StockR1TwoStageAgentLoop(AgentLoopBase):
    _tool: Optional[TSForecastTool] = None

    def __init__(
        self,
        *args,
        stage1_max_new_tokens: int = 512,
        stage2_max_new_tokens: int = 1024,
        stop_string_stage1: str = "</forecast_hint>",
        tool_name: str = "ts_forecast",
        ts_encoder_checkpoint: Optional[str] = None,
        hint_conditioner_checkpoint: Optional[str] = None,
        hint_residual_scale: float = 0.05,
        out_days: Optional[int] = None,
        tool_device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stage1_max_new_tokens = stage1_max_new_tokens
        self.stage2_max_new_tokens = stage2_max_new_tokens
        self.stop_string_stage1 = stop_string_stage1
        self.tool_name = tool_name
        self.response_length = self.rollout_config.response_length

        if StockR1TwoStageAgentLoop._tool is None:
            if ts_encoder_checkpoint is None:
                raise ValueError("ts_encoder_checkpoint is required for stockr1_two_stage_agent")
            cfg = TSForecastToolConfig(
                ts_encoder_checkpoint=ts_encoder_checkpoint,
                hint_conditioner_checkpoint=hint_conditioner_checkpoint,
                hint_residual_scale=hint_residual_scale,
                out_days=out_days,
                device=tool_device or ("cuda" if torch.cuda.is_available() else "cpu"),
            )
            StockR1TwoStageAgentLoop._tool = TSForecastTool(cfg)

    @property
    def tool(self) -> TSForecastTool:
        assert StockR1TwoStageAgentLoop._tool is not None
        return StockR1TwoStageAgentLoop._tool

    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # apply chat template once for stage-1 prefix prompt
        prompt_ids = await self.apply_chat_template(messages)
        metrics: Dict[str, Any] = {"generate_sequences": 0.0, "tool_calls": 0.0, "num_preempted": 0}

        stage1_params = dict(sampling_params)
        stage1_params["max_new_tokens"] = self.stage1_max_new_tokens
        stage1_params["stop"] = [self.stop_string_stage1]

        t = {}
        with simple_timer("generate_sequences", t):
            out1: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=stage1_params,
            )
        metrics["generate_sequences"] += t["generate_sequences"]
        if out1.num_preempted is not None:
            metrics["num_preempted"] += out1.num_preempted

        stage1_ids = out1.token_ids
        stage1_text = self.tokenizer.decode(stage1_ids, skip_special_tokens=False)

        tools_kwargs = kwargs.get("tools_kwargs", {}) or {}
        tool_payload = tools_kwargs.get(self.tool_name, tools_kwargs)
        ts_inputs = tool_payload.get("input_features")
        last_close = tool_payload.get("last_close")
        last_volume = tool_payload.get("last_volume")

        if ts_inputs is None or last_close is None or last_volume is None:
            raise ValueError(
                "tools_kwargs must contain input_features, last_close, last_volume "
                f"(under key '{self.tool_name}' or at top-level)."
            )

        t = {}
        with simple_timer("tool_calls", t):
            tool_out = self.tool.forecast_from_stage1(
                stage1_text=stage1_text,
                ts_inputs=ts_inputs,
                last_close=float(last_close),
                last_volume=float(last_volume),
            )
        metrics["tool_calls"] += t["tool_calls"]

        stage1_prefix_text = tool_out["stage1_prefix_text"]
        stage1_prefix_ids = self.tokenizer.encode(stage1_prefix_text, add_special_tokens=False)

        forecast_text = tool_out["forecast_text"]
        injected_text = "\n" + forecast_text + "\n<think>"
        injected_ids = self.tokenizer.encode(injected_text, add_special_tokens=False)

        stage2_prompt_ids = prompt_ids + stage1_prefix_ids + injected_ids
        stage2_params = dict(sampling_params)
        stage2_params["max_new_tokens"] = self.stage2_max_new_tokens

        t = {}
        with simple_timer("generate_sequences", t):
            out2: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=stage2_prompt_ids,
                sampling_params=stage2_params,
            )
        metrics["generate_sequences"] += t["generate_sequences"]
        if out2.num_preempted is not None:
            metrics["num_preempted"] += out2.num_preempted

        stage2_ids = out2.token_ids

        # response ids include tool-injected tokens; mask them out for policy optimization.
        response_ids = stage1_prefix_ids + injected_ids + stage2_ids
        response_ids = response_ids[: self.response_length]

        response_mask = (
            [1] * len(stage1_prefix_ids) + [0] * len(injected_ids) + [1] * len(stage2_ids)
        )[: self.response_length]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            num_turns=2,
            metrics=metrics,
            extra_fields={
                "hint_used": bool(tool_out.get("hint_used", False)),
                "pred_returns": tool_out.get("pred_returns"),
                "last_close": float(last_close),
                "last_volume": float(last_volume),
                # This key is consumed by NaiveRewardManager and forwarded into extra_info.
                "reward_scores": {"uncertainty": float(tool_out.get("uncertainty", 0.0))},
                "uncertainty": float(tool_out.get("uncertainty", 0.0)),
                "forecast_text": forecast_text,
                "stage1_text": stage1_text,
            },
        )
        return output
