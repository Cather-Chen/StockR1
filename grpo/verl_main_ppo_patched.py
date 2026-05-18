"""Patched PPO entrypoint for GRPO.

Only modifies behavior from `grpo/` by injecting training-time reward metrics
into wandb logs (train-aux/*), without editing VERL source files.
"""

from __future__ import annotations

import os
from typing import Any

import hydra
import numpy as np
import ray

import verl.trainer.ppo.ray_trainer as ray_trainer
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device


def _is_tracked_reward_key(key: str) -> bool:
    if key in {
        "score",
        "original_reward",
        "reward",
        "acc",
        "answer_reward",
        "format_reward",
        "hint_accuracy",
        "consistency_reward",
        "forecast_reward",
        "da3",
        "da5",
        "da10",
        # "forecast_rmse",
        # "uncertainty",
        # "uncertainty_weight",
    }:
        return True
    return key.endswith("_reward") or key.startswith("forecast_")


def _to_numeric_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        return np.asarray([], dtype=np.float32)

    if np.issubdtype(arr.dtype, np.number):
        out = arr.astype(np.float32, copy=False)
        return out[np.isfinite(out)]

    parsed = []
    raw_vals = np.ravel(arr).tolist()
    if not isinstance(raw_vals, list):
        raw_vals = [raw_vals]
    for v in raw_vals:
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            parsed.append(fv)
    return np.asarray(parsed, dtype=np.float32)


def install_train_aux_patch() -> None:
    # Idempotent to avoid re-patching in retries/restarts.
    if getattr(ray_trainer, "_grpo_train_aux_patch_installed", False):
        return

    orig_compute_data_metrics = ray_trainer.compute_data_metrics

    def _compute_data_metrics_with_train_aux(batch, use_critic: bool = True):
        metrics = orig_compute_data_metrics(batch=batch, use_critic=use_critic)
        non_tensor_batch = getattr(batch, "non_tensor_batch", {}) or {}
        original_reward_vals = _to_numeric_array(non_tensor_batch.get("original_reward", []))

        for key, vals in non_tensor_batch.items():
            if not _is_tracked_reward_key(str(key)):
                continue
            if str(key) in {"score", "reward"} and original_reward_vals.size > 0:
                numeric_vals = original_reward_vals
            else:
                numeric_vals = _to_numeric_array(vals)
            if numeric_vals.size == 0:
                continue
            metrics[f"train-aux/{key}/mean"] = float(np.mean(numeric_vals))
            # metrics[f"train-aux/{key}/std"] = float(np.std(numeric_vals))
            # metrics[f"train-aux/{key}/max"] = float(np.max(numeric_vals))
            # metrics[f"train-aux/{key}/min"] = float(np.min(numeric_vals))

        return metrics

    ray_trainer.compute_data_metrics = _compute_data_metrics_with_train_aux
    ray_trainer._grpo_train_aux_patch_installed = True
    print(f"[GRPO_PATCH] train-aux metrics patch installed in PID={os.getpid()}")


class PatchedTaskRunner(TaskRunner):
    def run(self, config):
        install_train_aux_patch()
        return super().run(config)


@hydra.main(config_path="../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    install_train_aux_patch()
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(PatchedTaskRunner))


if __name__ == "__main__":
    main()
