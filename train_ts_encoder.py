import os
import time
import argparse
from typing import Dict, Any, List
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

import wandb
os.environ["WANDB_API_KEY"] = "[WANDB_API_KEY]"
wandb.login(key="[WANDB_API_KEY]")
from dataloader.loader import StockDataLoader
from model.multichannel_ts_enc import MultichannelTSEncoder_rm
from model.metrics import evaluate_metrics_after_denorm
from utils.forecast_utils import returns_to_ohlcv

def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
    diff = target - pred
    return torch.maximum(q * diff, (q - 1.0) * diff)


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    return 0.5 * torch.log(sigma.pow(2) + eps) + 0.5 * (target - mu).pow(2) / (sigma.pow(2) + eps)


def compute_return_loss(
    pred_params: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    r_on = targets[..., 0]
    r_c = targets[..., 1]
    r_v = targets[..., 2]
    delta_h = targets[..., 3]
    delta_l = targets[..., 4]

    loss_on = gaussian_nll(pred_params["mu_on"], pred_params["sigma_on"], r_on).mean()
    loss_c = gaussian_nll(pred_params["mu_c"], pred_params["sigma_c"], r_c).mean()
    loss_v = gaussian_nll(pred_params["mu_v"], pred_params["sigma_v"], r_v).mean()
    loss_h = pinball_loss(pred_params["q_high"], delta_h, q=0.95).mean()
    loss_l = pinball_loss(pred_params["q_low"], delta_l, q=0.05).mean()

    total = (
        weights["lambda_on"] * loss_on
        + weights["lambda_c"] * loss_c
        + weights["lambda_v"] * loss_v
        + weights["lambda_h"] * loss_h
        + weights["lambda_l"] * loss_l
    )
    return {
        "loss": total,
        "loss_on": loss_on.detach(),
        "loss_c": loss_c.detach(),
        "loss_v": loss_v.detach(),
        "loss_h": loss_h.detach(),
        "loss_l": loss_l.detach(),
    }


def forward_loss(model, batch, device, autocast_dtype=None, loss_weights=None) -> Dict[str, Any]:
    input_features = batch["input_features"].to(device, non_blocking=True)
    targets = batch["output_features"].to(device, non_blocking=True)
    loss_weights = loss_weights or {
        "lambda_on": 1.0,
        "lambda_c": 1.0,
        "lambda_v": 1.0,
        "lambda_h": 1.0,
        "lambda_l": 1.0,
    }

    if autocast_dtype is None:
        pred_params = model.forecasting(input_features)
    else:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            pred_params = model.forecasting(input_features)

    loss_dict = compute_return_loss(pred_params, targets, weights=loss_weights)
    loss_dict["pred_params"] = pred_params
    return loss_dict


@torch.no_grad()
def run_evaluation(
    model,
    val_loader: DataLoader,
    device: torch.device,
    loss_weights: Dict[str, float],
    autocast_dtype=None,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    preds_list, target_list = [], []
    pred_ohlcv_list, target_ohlcv_list = [], []
    tickers_list = []
    date_list = []
    last_close_list = []
    last_volume_list = []
    for batch in val_loader:
        input_features = batch["input_features"].to(device, non_blocking=True)
        targets = batch["output_features"].to(device, non_blocking=True)

        if autocast_dtype is None:
            pred_params = model.forecasting(input_features)
        else:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                pred_params = model.forecasting(input_features)

        loss_dict = compute_return_loss(pred_params, targets, weights=loss_weights)
        losses.append(float(loss_dict["loss"].detach().cpu().item()))

        mean_pred = pred_params["mean_returns"]  # [B, H, 5]
        preds_list.append(mean_pred.detach().cpu())
        target_list.append(targets.detach().cpu())

        last_close = batch["last_close"].to(device)
        last_volume = batch["last_volume"].to(device)
        pred_raw = returns_to_ohlcv(mean_pred, last_close, last_volume).detach().cpu()
        tgt_raw = returns_to_ohlcv(targets, last_close, last_volume).detach().cpu()
        pred_ohlcv_list.append(pred_raw)
        target_ohlcv_list.append(tgt_raw)
        last_close_list.append(last_close.detach().cpu())
        last_volume_list.append(last_volume.detach().cpu())
        tickers_list.extend(batch["tickers"])
        date_list.extend(batch["dates"])

    avg_loss = float(np.mean(losses))
    preds_all = torch.cat(preds_list, dim=0).numpy()
    targets_all = torch.cat(target_list, dim=0).numpy()
    pred_raw_all = torch.cat(pred_ohlcv_list, dim=0).numpy()
    target_raw_all = torch.cat(target_ohlcv_list, dim=0).numpy()
    last_close_all = torch.cat(last_close_list, dim=0).numpy()
    last_volume_all = torch.cat(last_volume_list, dim=0).numpy()
    
    val_metrics = evaluate_metrics_after_denorm(pred_raw_all, target_raw_all, last_close_all, last_volume_all)
    
    to_save = {
        "pred_returns": preds_all,  # [B, H, 5]
        "target_returns": targets_all, 
        "pred_raw_ohlcv": pred_raw_all, 
        "target_raw_ohlcv": target_raw_all, 
        "tickers_list": tickers_list,
        "date_list": date_list,
        "last_close_all": last_close_all,
        "last_volume_all": last_volume_all,
    }
    return {"val_loss": avg_loss, **val_metrics}, to_save


def main():
    parser = argparse.ArgumentParser()
    # Model architecture
    parser.add_argument("--ts_in_channels", type=int, default=5)
    parser.add_argument("--ts_d_model", type=int, default=512)
    parser.add_argument("--ts_heads", type=int, default=8)
    parser.add_argument("--ts_num_layers", type=int, default=8)
    parser.add_argument("--ts_patch_len", type=int, default=5)
    parser.add_argument("--ts_dropout", type=float, default=0.1)
    parser.add_argument("--out_days", type=int, default=10)
    parser.add_argument("--input_len", type=int, default=90)

    # Data
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--specify_date", type=bool, default=False)
    parser.add_argument("--start_date", type=str, default="2020-01-01")
    parser.add_argument("--end_date", type=str, default="2025-01-01")
    # Optimization
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Loss weights
    parser.add_argument("--lambda_on", type=float, default=1.0)
    parser.add_argument("--lambda_c", type=float, default=1.0)
    parser.add_argument("--lambda_v", type=float, default=1.0)
    parser.add_argument("--lambda_h", type=float, default=1.0)
    parser.add_argument("--lambda_l", type=float, default=1.0)

    # Logging/checkpoint
    parser.add_argument("--project", type=str, default="stockr1_update")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints/ts_encoder")

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    loss_weights = {
        "lambda_on": args.lambda_on,
        "lambda_c": args.lambda_c,
        "lambda_v": args.lambda_v,
        "lambda_h": args.lambda_h,
        "lambda_l": args.lambda_l,
    }

    loader = StockDataLoader(
        tokenizer=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task="ts_pretrain",
    )

    if args.specify_date:
        train_loader = loader.get_date_range_loader(start_date=args.start_date, end_date=args.end_date, split="train")
        val_loader = loader.get_date_range_loader(start_date=args.end_date, end_date="2025-08-31", split="test")
    else:
        train_loader = loader.create_loader(split="train")
        val_loader = loader.create_loader(split="test")

    model = MultichannelTSEncoder_rm(
        input_len=args.input_len,
        in_channels=args.ts_in_channels,
        d_model=args.ts_d_model,
        nhead=args.ts_heads,
        num_layers=args.ts_num_layers,
        patch_len=args.ts_patch_len,
        dropout=args.ts_dropout,
        out_days=args.out_days,
    )
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")
    autocast_dtype = torch.float16 if (args.use_amp and device.type == "cuda") else None

    wandb.init(project=args.project, name=args.run_name, config=vars(args), group="ts_encoder_pretrain")
    job_name = f"dm{args.ts_d_model}_heads{args.ts_heads}_layers{args.ts_num_layers}_patch{args.ts_patch_len}_specify_date{args.specify_date}_start{args.start_date}_end{args.end_date}"
    print(f"Train batches/epoch: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Trainable params: {count_params(model)/1e6:.2f}M")

    global_step = 0
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {"loss": 0.0, "on": 0.0, "c": 0.0, "v": 0.0, "h": 0.0, "l": 0.0}
        t0 = time.time()
        for step, batch in enumerate(train_loader, start=1):
            loss_dict = forward_loss(model, batch, device, autocast_dtype=autocast_dtype, loss_weights=loss_weights)
            loss = loss_dict["loss"]

            if args.use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if args.max_grad_norm and args.max_grad_norm > 0:
                if args.use_amp and device.type == "cuda":
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if args.use_amp and device.type == "cuda":
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()

            optim.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            epoch_losses["loss"] += float(loss.detach().cpu().item())
            epoch_losses["on"] += float(loss_dict["loss_on"].detach().cpu().item())
            epoch_losses["c"] += float(loss_dict["loss_c"].detach().cpu().item())
            epoch_losses["v"] += float(loss_dict["loss_v"].detach().cpu().item())
            epoch_losses["h"] += float(loss_dict["loss_h"].detach().cpu().item())
            epoch_losses["l"] += float(loss_dict["loss_l"].detach().cpu().item())

            if step == 10:
                print("estimate time per epoch: ", (time.time() - t0) / step * len(train_loader))

        num_batches = len(train_loader)
        epoch_time = time.time() - t0
        wandb.log(
            {
                "train/loss": epoch_losses["loss"] / num_batches,
                "train/loss_on": epoch_losses["on"] / num_batches,
                "train/loss_c": epoch_losses["c"] / num_batches,
                "train/loss_v": epoch_losses["v"] / num_batches,
                "train/loss_h": epoch_losses["h"] / num_batches,
                "train/loss_l": epoch_losses["l"] / num_batches,
                "time/epoch_sec": epoch_time,
            },
            step=global_step,
        )

        val_metrics, to_save = run_evaluation(
            model,
            val_loader,
            device,
            loss_weights=loss_weights,
            autocast_dtype=autocast_dtype,
        )
        wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

        save_path = os.path.join(args.save_dir, job_name, "results_last.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(to_save, f)

        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            ckpt_path = os.path.join(args.save_dir, job_name, "best_val.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "sched": scheduler.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_loss": val_metrics["val_loss"],
                },
                ckpt_path,
            )
            best_res_path = os.path.join(args.save_dir, job_name, "results_best.pkl")
            with open(best_res_path, "wb") as f:
                pickle.dump(to_save, f)

    print("Training completed.")


if __name__ == "__main__":
    main()
