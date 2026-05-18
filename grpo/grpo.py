"""Launch StockR1 GRPO training on top of VERL + vLLM.

This replaces the legacy custom GRPO trainer implementation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="StockR1 VERL+vLLM GRPO launcher")

    p.add_argument("--train_parquet", type=str, required=True)
    p.add_argument("--val_parquet", type=str, default=None)

    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--ts_encoder_checkpoint", type=str, required=True)
    p.add_argument("--hint_conditioner_checkpoint", type=str, default=None)

    p.add_argument("--save_dir", type=str, default="./checkpoints/grpo_verl")
    p.add_argument("--project_name", type=str, default="stockr1_update")
    p.add_argument("--experiment_name", type=str, default="qwen3_4b_verl_vllm")
    p.add_argument("--logger", type=str, default="wandb")

    p.add_argument("--n_gpus_per_node", type=int, default=8)
    p.add_argument("--nnodes", type=int, default=1)

    p.add_argument("--train_batch_size", type=int, default=128)
    p.add_argument("--ppo_mini_batch_size", type=int, default=128)
    p.add_argument("--ppo_micro_batch_size_per_gpu", type=int, default=8)
    p.add_argument("--rollout_log_prob_micro_batch_size_per_gpu", type=int, default=64)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--total_epochs", type=int, default=5)
    p.add_argument("--test_freq", type=int, default=0)

    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--max_response_length", type=int, default=2048)
    p.add_argument("--stage1_max_new_tokens", type=int, default=512)
    p.add_argument("--stage2_max_new_tokens", type=int, default=1024)

    p.add_argument("--rollout_tp", type=int, default=2)
    p.add_argument("--rollout_gpu_mem_util", type=float, default=0.65)
    p.add_argument("--rollout_max_model_len", type=int, default=8192)

    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--kl_coef", type=float, default=0.001)

    p.add_argument("--answer_reward_weight", type=float, default=0.4)
    p.add_argument("--consistency_reward_weight", type=float, default=0.2)
    p.add_argument("--format_reward_weight", type=float, default=0.1)
    p.add_argument("--forecast_reward_weight", type=float, default=0.3)
    p.add_argument("--hint_accuracy_weight", type=float, default=0.2)
    p.add_argument("--use_uncertainty_reweight", action="store_true")

    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--extra", type=str, nargs="*", default=[])
    return p.parse_args()


def build_agent_loop_yaml(args: argparse.Namespace, out_path: Path) -> None:
    cfg = [
        {
            "name": "stockr1_two_stage_agent",
            "_target_": "grpo.verl_stock_agent_loop.StockR1TwoStageAgentLoop",
            "stage1_max_new_tokens": int(args.stage1_max_new_tokens),
            "stage2_max_new_tokens": int(args.stage2_max_new_tokens),
            "stop_string_stage1": "</forecast_hint>",
            "tool_name": "ts_forecast",
            "ts_encoder_checkpoint": str(args.ts_encoder_checkpoint),
            "hint_conditioner_checkpoint": str(args.hint_conditioner_checkpoint)
            if args.hint_conditioner_checkpoint
            else None,
            "hint_residual_scale": 0.05,
            "out_days": 10,
        }
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    args = parse_args()

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    agent_loop_cfg = save_dir / "agent_loop_stockr1.yaml"
    build_agent_loop_yaml(args, agent_loop_cfg)

    reward_path = str((Path(__file__).resolve().parent / "verl_reward_adapter.py").resolve())
    val_files = args.val_parquet if args.val_parquet else args.train_parquet

    cmd = [
        args.python,
        "-m",
        "grpo.verl_main_ppo_patched",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={args.train_parquet}",
        f"data.val_files={val_files}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        "data.return_raw_chat=True",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={args.model_path}",
        "+actor_rollout_ref.model.override_config._attn_implementation=flash_attention_2",
        f"actor_rollout_ref.actor.optim.lr={args.lr}",
        "actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.model.lora_rank={args.lora_rank}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.ppo_micro_batch_size_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={args.kl_coef}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.load_format=hf",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.rollout_gpu_mem_util}",
        f"actor_rollout_ref.rollout.max_model_len={args.rollout_max_model_len}",
        "actor_rollout_ref.rollout.layered_summon=True",
        f"actor_rollout_ref.rollout.n={args.group_size}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.rollout_log_prob_micro_batch_size_per_gpu}",
        "actor_rollout_ref.rollout.multi_turn.enable=False",
        "actor_rollout_ref.rollout.agent.default_agent_loop=stockr1_two_stage_agent",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_loop_cfg}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        f"trainer.logger={args.logger}",
        f"trainer.default_local_dir={save_dir}",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
        f"trainer.nnodes={args.nnodes}",
        "trainer.save_freq=50",
        f"trainer.test_freq={args.test_freq}",
        f"trainer.total_epochs={args.total_epochs}",
        f"reward.custom_reward_function.path={reward_path}",
        "reward.custom_reward_function.name=compute_score",
        f"+reward.custom_reward_function.reward_kwargs.answer_reward_weight={args.answer_reward_weight}",
        f"+reward.custom_reward_function.reward_kwargs.consistency_reward_weight={args.consistency_reward_weight}",
        f"+reward.custom_reward_function.reward_kwargs.format_reward_weight={args.format_reward_weight}",
        f"+reward.custom_reward_function.reward_kwargs.forecast_reward_weight={args.forecast_reward_weight}",
        f"+reward.custom_reward_function.reward_kwargs.hint_accuracy_weight={args.hint_accuracy_weight}",
        "+reward.custom_reward_function.reward_kwargs.use_uncertainty_reweight=true"
        if args.use_uncertainty_reweight
        else "+reward.custom_reward_function.reward_kwargs.use_uncertainty_reweight=false",
    ]

    cmd.extend(args.extra)

    print("Launching VERL GRPO command:\n")
    print(" ".join(map(str, cmd)))
    print("")

    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[1]
    py_paths = [str(project_root / "vllm"), str(project_root)]
    old_pp = env.get("PYTHONPATH", "")
    if old_pp:
        py_paths.append(old_pp)
    env["PYTHONPATH"] = ":".join(py_paths)

    rc = subprocess.call(cmd, env=env, cwd=str(project_root / "verl"))
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
