import os
os.environ["DEEPSPEED_COMM"] = "nccl"
os.environ["DEEPSPEED_DISABLE_MPIOP"] = "1" 
os.environ["WANDB_API_KEY"] = "[WANDB_API_KEY]"
import wandb
wandb.login(key="[WANDB_API_KEY]")
import shutil
from pathlib import Path
import math
import time
import json
import pickle
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedDataParallelKwargs, set_seed as acc_set_seed

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# ---- your code (keep these imports consistent with your project) ----
from dataloader.loader import StockDataLoader
from model.ts_llm_v1 import TS2LLM_v2Config, TS2QwenModel_v2
from model.metrics import evaluate_metrics_after_denorm
from utils.forecast_utils import returns_to_ohlcv
from deepspeed.runtime.zero import GatheredParameters

def count_params(model: nn.Module) -> int:
    added = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.numel())
            added += p.numel()
    return added

def count_all_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def all_reduce_scalar(value: float, op: str = "mean", device: torch.device = None) -> float:
    if not is_dist_avail_and_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if op == "mean":
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    elif op == "sum":
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    elif op == "max":
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    elif op == "min":
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
    else:
        raise ValueError(f"Unsupported op for all_reduce_scalar: {op}")
    return float(t.item())

def gather_object_fallback(obj, accelerator):
    """
    gather Python objects (list/dict/str/..) across processes.
    returns a list: [rank0_obj, rank1_obj, ...]
    """
    if not dist.is_initialized():
        return [obj]

    world_size = dist.get_world_size()
    out_list = [None for _ in range(world_size)]
    dist.all_gather_object(out_list, obj)
    return out_list
# ---------------------------
# evaluation (DDP-aware)
# ---------------------------
@torch.no_grad()
def run_evaluation(
    accelerator: Accelerator,
    model: nn.Module,
    val_loader: DataLoader,
    autocast_dtype=None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    was_training = model.training
    model.eval()
    accelerator.print(f" 🔍 Start evaluation | batches={len(val_loader)} | dtype={autocast_dtype}")

    pred_returns_local: List[np.ndarray] = []
    target_returns_local: List[np.ndarray] = []
    pred_raw_local: List[np.ndarray] = []
    target_raw_local: List[np.ndarray] = []
    last_close_local: List[np.ndarray] = []
    last_volume_local: List[np.ndarray] = []

    tickers_gather_accum: List[List[str]] = []
    dates_gather_accum: List[List[str]] = []

    local_count = 0

    try:
        for step, batch in enumerate(val_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(accelerator.device, non_blocking=True)

            with accelerator.autocast() if autocast_dtype is not None else torch.cuda.amp.autocast(enabled=False):
                loss_dict = model(batch)

            pred_returns = loss_dict["forecasting"]  # [B, out_days, 5]
            target_returns = batch["output_features"]  # [B, out_days, 5]

            if "last_close" in batch and "last_volume" in batch:
                last_close = batch["last_close"]
                last_volume = batch["last_volume"]
            else:
                if "input_raw_features" not in batch:
                    raise ValueError("Missing last_close/last_volume and input_raw_features for return evaluation.")
                last_close = batch["input_raw_features"][:, -1, 3]
                last_volume = batch["input_raw_features"][:, -1, 4]

            pred_raw = returns_to_ohlcv(pred_returns, last_close, last_volume)
            target_raw = returns_to_ohlcv(target_returns, last_close, last_volume)

            pred_returns_g = accelerator.gather_for_metrics(pred_returns)
            target_returns_g = accelerator.gather_for_metrics(target_returns)
            pred_raw_g = accelerator.gather_for_metrics(pred_raw)
            target_raw_g = accelerator.gather_for_metrics(target_raw)
            last_close_g = accelerator.gather_for_metrics(last_close)
            last_volume_g = accelerator.gather_for_metrics(last_volume)

            tickers_g = gather_object_fallback(list(batch["tickers"]), accelerator)
            dates_g = gather_object_fallback(list(batch["dates"]), accelerator)

            if accelerator.is_main_process:
                pred_returns_local.append(pred_returns_g.detach().cpu().numpy())
                target_returns_local.append(target_returns_g.detach().cpu().numpy())
                pred_raw_local.append(pred_raw_g.detach().cpu().numpy())
                target_raw_local.append(target_raw_g.detach().cpu().numpy())
                last_close_local.append(last_close_g.detach().cpu().numpy())
                last_volume_local.append(last_volume_g.detach().cpu().numpy())
                tickers_gather_accum.extend(tickers_g)
                dates_gather_accum.extend(dates_g)

            local_count += pred_returns.shape[0]
    finally:
        if was_training:
            model.train()


    global_count = all_reduce_scalar(local_count, op="sum", device=accelerator.device)

    metrics: Dict[str, float] = {}
    to_save: Dict[str, Any] = {}

    if accelerator.is_main_process:
        pred_returns_all = np.concatenate(pred_returns_local, axis=0)  # [N, H, 5]
        target_returns_all = np.concatenate(target_returns_local, axis=0)  # [N, H, 5]
        pred_raw_all = np.concatenate(pred_raw_local, axis=0)  # [N, H, 5]
        target_raw_all = np.concatenate(target_raw_local, axis=0)  # [N, H, 5]
        last_close_all = np.concatenate(last_close_local, axis=0)  # [N]
        last_volume_all = np.concatenate(last_volume_local, axis=0)  # [N]
        tickers_list = [t for sub in tickers_gather_accum for t in sub]
        date_list = [d for sub in dates_gather_accum for d in sub]

        accelerator.print(
            f" Eval concat done: N={pred_raw_all.shape[0]}, H={pred_raw_all.shape[1]} | global_eval_size={int(global_count)}"
        )

        metrics = evaluate_metrics_after_denorm(
            pred_raw_all,
            target_raw_all,
            last_close_all,
            last_volume_all,
        )

        to_save = {
            "pred_returns": pred_returns_all,
            "target_returns": target_returns_all,
            "pred_raw_ohlcv": pred_raw_all,
            "target_raw_ohlcv": target_raw_all,
            "date_list": date_list,
            "tickers_list": tickers_list,
            "last_close_all": last_close_all,
            "last_volume_all": last_volume_all,
            "global_eval_size": int(global_count),
        }

        metrics_json = json.dumps(metrics)
        da3 = metrics.get("return/da_3", None)
        da5 = metrics.get("return/da_5", None)
        da10 = metrics.get("return/da_10", None)
        accelerator.print(f" ✅ Eval metrics: "
                          f"da3={da3}, da5={da5}, da10={da10}, keys={list(metrics.keys())[:6]}...")
    else:
        metrics_json = ""

    if is_dist_avail_and_initialized():
        obj_list = [metrics_json]
        dist.broadcast_object_list(obj_list, src=0)
        metrics_json = obj_list[0]

    metrics = json.loads(metrics_json) if metrics_json else {}
    accelerator.print(f" 🔁 Evaluation finished.")
    return metrics, (to_save if accelerator.is_main_process else {})




def save_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    tokenizer,
    args,
    epoch: int,
    step: int,
    global_step: int,
    job_dir: str,
):
    import shutil, os, torch

    save_dir = os.path.join(job_dir, f"checkpoint-{epoch}-{global_step}")
    tfmr_dir = os.path.join(save_dir, "tfmr")

    if accelerator.is_main_process:
        os.makedirs(tfmr_dir, exist_ok=True)
        if args.max_ckpts and args.max_ckpts > 0:
            ckpts = [
                p for p in os.listdir(job_dir)
                if p.startswith("checkpoint-") and os.path.isdir(os.path.join(job_dir, p))
            ]
            if len(ckpts) >= args.max_ckpts:
                ckpts.sort(key=lambda x: os.path.getctime(os.path.join(job_dir, x)))
                oldest = ckpts[0]
                shutil.rmtree(os.path.join(job_dir, oldest), ignore_errors=True)

    accelerator.wait_for_everyone()

    if args.zero_stage == 3:
        unwrap_model = accelerator.unwrap_model(model)
        with GatheredParameters(list(unwrap_model.qwen.parameters()), modifier_rank=0):
            if accelerator.is_main_process:
                unwrap_model.qwen.save_pretrained(
                    tfmr_dir,
                    is_main_process=True,
                    save_function=accelerator.save,
                    save_embedding_layers=True
                )
                unwrap_model.qwen.config.save_pretrained(tfmr_dir)

        head_params = [p for n, p in unwrap_model.named_parameters() if not n.startswith("qwen.")]
        with GatheredParameters(head_params, modifier_rank=0):
            if accelerator.is_main_process:
                state_dict = {n: p.cpu() for n, p in unwrap_model.named_parameters() if not n.startswith("qwen.")}
                torch.save(state_dict, os.path.join(save_dir, "heads.pt"))

        if accelerator.is_main_process and tokenizer is not None:
            tokenizer.save_pretrained(tfmr_dir)

    else:
        unwrap_model = model
        state_dict = unwrap_model.state_dict()

        unwrap_model.qwen.save_pretrained(
            tfmr_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save if accelerator.is_main_process else None,
            state_dict={k.replace("qwen.", ""): v for k, v in state_dict.items() if k.startswith("qwen.")},
            save_embedding_layers=True
        )

        if accelerator.is_main_process:
            unwrap_model.qwen.config.save_pretrained(tfmr_dir)
            torch.save(
                {k: v for k, v in state_dict.items() if not k.startswith("qwen.")},
                os.path.join(save_dir, "heads.pt"),
            )
            if tokenizer is not None:
                tokenizer.save_pretrained(tfmr_dir)

    # -------- barrier & log --------
    accelerator.wait_for_everyone()
    accelerator.print(f"💾 Checkpoint saved at {save_dir}")





# ---------------------------
# main train loop
# ---------------------------
def build_argparser():
    parser = argparse.ArgumentParser()

    # data / model
    parser.add_argument("--project", type=str, default="stockr1_update")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--special_tokens", nargs="*", default=["<analysis>", "<answer>", "</answer>"])
    parser.add_argument("--per_device_batch_size", type=int, default=1, help="Per-device batch size. Global batch = per_device * world_size * grad_accum.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--llm_dim", type=int, default=2560)
    parser.add_argument("--task", type=str, default="sft_v1", choices=["sft_v1"], help="Pass through to StockDataLoader")
    parser.add_argument("--ts_soft_tokens", action="store_true", default=True, help="Enable soft tokens for the TS encoder.")
    parser.add_argument("--ts_encoder_checkpoint", type=str, default="checkpoints/ts_encoder/dm512_heads8_layers10_patch5_specify_dateTrue_start2015-01-01_end2025-01-01")
    # LoRA
    parser.add_argument("--lora_enable", action="store_true", help="Enable LoRA adapters on Qwen.")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default=None, help="Comma-separated module names, e.g. q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # optimization
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # precision / accelerate
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Accelerate mixed precision mode.")
    parser.add_argument("--max_ckpts", type=int, default=5, help="Keep at most N checkpoints under save_dir (0=unlimited).")
    
    # eval / logging / checkpoint
    parser.add_argument("--eval_every_steps", type=int, default=-1, help="If >0, run eval every N optimizer steps; otherwise eval each epoch.")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--group_name", type=str, default="sft_new")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="checkpoints/sft_v1")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main():
    with open("ds_config.json") as f:
        ds_cfg = json.load(f)
    parser = build_argparser()
    args = parser.parse_args()
    args.zero_stage = ds_cfg["zero_optimization"]["stage"]
    acc_set_seed(args.seed)

    # ====== PRINT: launch args (subset) ======
    print(f" 🚀 Launch with args: "
          f"model={args.model_name}, mp={args.mixed_precision}, bs={args.per_device_batch_size}, "
          f"accum={args.grad_accum_steps}, epochs={args.epochs}, lora={args.lora_enable}")

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        static_graph=True,
    )
    accelerator = Accelerator(
        log_with="wandb" if not args.no_wandb else None,
        mixed_precision=(None if args.mixed_precision == "no" else args.mixed_precision),
        kwargs_handlers=[ddp_kwargs],
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = accelerator.num_processes
    device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)

    accelerator.init_trackers(
        project_name=args.project,
        config=vars(args),
        init_kwargs={"wandb": {"name": args.run_name, "group": args.group_name}},
    )
    # ---- PRINT: device/env ----
    accelerator.print(f" 📝 Accelerator init trackers done. Wandb project={args.project}, name={args.run_name}, group={args.group_name}")
    accelerator.print(f" 🧩 Accelerator ready | world_size={world_size}, local_rank={local_rank}, device={device}, mp={accelerator.mixed_precision}")


    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    added = 0
    if args.special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": args.special_tokens})
        added = len(args.special_tokens)
    accelerator.print(f" 🔤 Tokenizer loaded | vocab_size={len(tokenizer)} (+{added} special tokens)")

    # data
    loader = StockDataLoader(tokenizer, batch_size=args.per_device_batch_size, num_workers=args.num_workers, task=args.task)
    train_loader = loader.create_loader(split="train")
    val_loader   = loader.create_loader(split="test")
    accelerator.print(f" 📦 Data loaders ready | train_batches/epoch={len(train_loader)}, val_batches={len(val_loader)}")

    # model
    cfg = TS2LLM_v2Config(
        tokenizer=tokenizer,
        qwen_model_name=args.model_name,
        llm_dim=args.llm_dim,
        ts_encoder_checkpoint=args.ts_encoder_checkpoint,
        ts_soft_tokens=args.ts_soft_tokens,
        out_days=10,
        freeze_qwen=False,
        lora_enable=args.lora_enable,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=(args.lora_target_modules.split(',') if args.lora_target_modules else None),
        is_eval=False,
    )
    model = TS2QwenModel_v2(cfg)
    #model.qwen.config.vocab_size = 151936
    # len(tokenizer)= 151672
    #model.qwen.get_input_embeddings().weight.shape= torch.Size([151936, 2560])


    if args.lora_enable:
        accelerator.print(f" 🧩 LoRA enabled | r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, targets={args.lora_target_modules or 'DEFAULT'}")


    # optimizer / scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_train_steps = args.epochs * math.ceil(len(train_loader) / max(1, 1))
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps, num_training_steps=total_train_steps
    )


    if accelerator.is_main_process:
        total = count_all_params(model)
        trainable = count_params(model)
        accelerator.print(f" 📊 Model params | trainable={trainable/1e6:.2f}M / total={total/1e6:.2f}M")
    autocast_dtype = None
    if accelerator.mixed_precision == "fp16":
        autocast_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        autocast_dtype = torch.bfloat16

    accelerator.print(f" 🏁 Train: {len(train_loader)} batches/epoch | Val: {len(val_loader)} | autocast={autocast_dtype}")
        

    model, optim, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optim, train_loader, val_loader, scheduler
    )
    accelerator.print(f" ✅ accelerate.prepare() done.")


    global_step = 0
    best_val_score = -float("inf") 
    job_dir = os.path.join(args.save_dir, (args.run_name or "run"))
    if accelerator.is_main_process:
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir) 
        os.makedirs(job_dir, exist_ok=True)
        accelerator.print(f" 💾 Save dir: {job_dir}")
        
    # ====== TRAIN LOOP ======
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_sft_loss = 0.0
        epoch_ts_loss = 0.0
        t0 = time.time()

        accelerator.print(f" ▶️ Epoch {epoch}/{args.epochs} started.")

        for step, batch in enumerate(train_loader, start=1):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
                    
            with accelerator.autocast():
                out = model(batch)
                loss = out["loss"]
            accelerator.backward(loss)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

            loss_val = accelerator.gather_for_metrics(loss.detach()).mean().item()
            epoch_loss += loss_val
            sft_val = accelerator.gather_for_metrics(out["sft_loss"].detach()).mean().item()
            epoch_sft_loss += sft_val
            ts_val = accelerator.gather_for_metrics(out["ts_loss"].detach()).mean().item()
            epoch_ts_loss += ts_val

            global_step += 1

            # logging + console print
            if (global_step % args.log_interval == 0):
                lr = scheduler.get_last_lr()[0]
                accelerator.print(f" step {step}/{len(train_loader)} "
                                  f"(global {global_step}) | loss={loss_val:.4f} "
                                  f"{'(sft=' + f'{sft_val:.4f}' + ')' if 'sft_loss' in out else ''} "
                                  f"{'(ts=' + f'{ts_val:.4f}' + ')' if 'ts_loss' in out else ''} "
                                  f"| lr={lr:.2e}")
                if accelerator.is_main_process and not args.no_wandb:
                    accelerator.log({"train/total_loss": loss_val, 
                                     "train/sft_loss": sft_val, 
                                     "train/ts_loss": ts_val, 
                                     "train/lr": lr}, step=global_step)

            # optional per-N-steps eval
            if args.eval_every_steps > 0 and (global_step % args.eval_every_steps == 0):
                accelerator.wait_for_everyone()
                # save_checkpoint(accelerator, model, tokenizer, args, epoch, step, global_step, job_dir)
                val_metrics, to_save = run_evaluation(accelerator, model, val_loader, autocast_dtype=autocast_dtype)

                if accelerator.is_main_process:
                    if not args.no_wandb:
                        accelerator.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

                    da3 = float(val_metrics.get("return/da_3", 0.0))
                    da5 = float(val_metrics.get("return/da_5", 0.0))
                    da10 = float(val_metrics.get("return/da_10", 0.0))
                    val_score = da3 + da5 + da10 
                    accelerator.print(f" 📈 Val score (da3+da5+da10)={val_score:.4f}")

                    if val_score > best_val_score:  # save best checkpoints
                        best_val_score = val_score
                        best_pkl = os.path.join(job_dir, "results_best.pkl")
                        with open(best_pkl, "wb") as f:
                            pickle.dump(to_save, f)
                        accelerator.print(f" 🏅 New best saved: {best_pkl} (score={best_val_score:.4f})")
                        is_best = True
                    else:
                        is_best = False
                else:
                    is_best = False

                if is_dist_avail_and_initialized():
                    best_list = [is_best]
                    dist.broadcast_object_list(best_list, src=0)
                    is_best = bool(best_list[0])

                if is_best:
                    save_checkpoint(accelerator, model, tokenizer, args, epoch, step, global_step, job_dir)
                accelerator.wait_for_everyone()

        epoch_time = time.time() - t0
        avg_epoch_loss = all_reduce_scalar(epoch_loss / max(1, len(train_loader)), op="mean", device=device)
        avg_sft_loss = all_reduce_scalar(epoch_sft_loss / max(1, len(train_loader)), op="mean", device=device)
        avg_ts_loss = all_reduce_scalar(epoch_ts_loss / max(1, len(train_loader)), op="mean", device=device)

        accelerator.print(f" ⏹ Epoch {epoch} done. avg loss={avg_epoch_loss:.6f} | sft={avg_sft_loss:.6f} | ts={avg_ts_loss:.6f} | time={epoch_time:.1f}s")
        if accelerator.is_main_process and not args.no_wandb:
            accelerator.log(
                {
                    "train/total_loss": avg_epoch_loss,
                    "train/sft_loss": avg_sft_loss,
                    "train/ts_loss": avg_ts_loss,
                    "time/epoch_sec": epoch_time,
                },
                step=global_step,
            )
        
        if args.eval_every_steps <= 0 or epoch == args.epochs:  # save last checkpoints
            accelerator.wait_for_everyone()
            save_checkpoint(accelerator, model, tokenizer, args, epoch, step, global_step, job_dir)
            val_metrics, to_save = run_evaluation(accelerator, model, val_loader, autocast_dtype=autocast_dtype)

            if accelerator.is_main_process:
                if not args.no_wandb:
                    accelerator.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

                da3 = float(val_metrics.get("return/da_3", 0.0))
                da5 = float(val_metrics.get("return/da_5", 0.0))
                da10 = float(val_metrics.get("return/da_10", 0.0))
                val_score = da3 + da5 + da10
                accelerator.print(f" 📈 Val score (da3+da5+da10)={val_score:.4f}")

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_pkl = os.path.join(job_dir, "results_best.pkl")
                    with open(best_pkl, "wb") as f:
                        pickle.dump(to_save, f)
                    accelerator.print(f" 🏅 New best saved: {best_pkl} (score={best_val_score:.4f})")
            accelerator.wait_for_everyone()
                    
            # save_checkpoint(accelerator, model, tokenizer, args, epoch, step, global_step, job_dir)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(f" ✅ Training completed.")

    accelerator.end_training()


if __name__ == "__main__":
    main()
