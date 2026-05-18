
import torch
import sys
sys.path.append(".")
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataloader.dataset import StockNormalizer

def pinball_loss(pred: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
    diff = target - pred
    return torch.maximum(q * diff, (q - 1.0) * diff)

def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    sigma = torch.clamp(sigma, min=1e-3, max=50.0)
    var = sigma.pow(2)
    return 0.5 * torch.log(var + eps) + 0.5 * (target - mu).pow(2) / (var + eps)

def compute_return_loss(
    pred_params: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    weights = weights or {
        "lambda_on": 1.0,
        "lambda_c": 1.0,
        "lambda_v": 1.0,
        "lambda_h": 1.0,
        "lambda_l": 1.0,
    }

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


def weighted_mse_loss(preds, targets, channel_weights=torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0]), w_max=5, w_min=1):
    B, C, L = preds.shape
    
    diff = preds - targets
    mse = diff.pow(2)  # [B, C, L]

    if channel_weights is not None:
        channel_weights = channel_weights.view(1, C, 1).to(preds.device)
        mse = mse * channel_weights

    base = (w_max / w_min) ** (1.0 / (L - 1))
    idx = torch.arange(L, device=preds.device)
    weights = w_min * (base ** idx).to(preds.dtype)
    weights = weights.view(1, 1, L)
    mse = mse * weights

    loss = mse.mean()
    return loss



def _stack_stats_for_batch(
    tickers: List[str],
    normalizer,
    device,
    dtype
) -> Tuple[torch.Tensor, torch.Tensor]:

    p_mu_list, p_si_list = [], []
    for t in tickers:
        ps = normalizer.price_stats[t]  # {'mu': ..., 'sigma': ...}
        mu = torch.as_tensor(ps['mu'], device=device, dtype=dtype)
        si = torch.as_tensor(ps['sigma'], device=device, dtype=dtype)
        p_mu_list.append(mu.view(4))
        p_si_list.append(si.view(4))
    price_mu    = torch.stack(p_mu_list, dim=0).unsqueeze(-1)  # [B,4,1]
    price_sigma = torch.stack(p_si_list, dim=0).unsqueeze(-1)  # [B,4,1]
    return price_mu, price_sigma

def _denorm_close_from_norm_feat(
    feat_norm: torch.Tensor,        # [B, 5, L]; 0:low,1:ΔO,2:ΔH,3:ΔC,4:log-vol(z)  (仅前4维用 price_stats)
    price_mu: torch.Tensor,         # [B, 4, 1]
    price_sigma: torch.Tensor,      # [B, 4, 1]
    eps: float = 1e-6
) -> torch.Tensor:

    price_norm = feat_norm[:, :4, :]                                 # [B,4,L]
    price_real = price_norm * price_sigma + price_mu                  # [B,4,L]
    low_real   = price_real[:, 0, :]                                  # [B,L]
    dC_real    = price_real[:, 2, :]                                  # [B,L]
    close_real = (low_real + dC_real).clamp_min(eps)                  # ensure positive for log-return
    return close_real                                                 # [B,L]

def _horizon_log_returns(close: torch.Tensor, horizons: Tuple[int, ...]) -> torch.Tensor:
    B, L = close.shape
    Hs = [h for h in horizons if h < L]
    if len(Hs) == 0:
        return torch.zeros((B, 0), device=close.device, dtype=close.dtype)
    base = close[:, 0:1]                                      # [B,1]
    tgt  = torch.stack([close[:, h] for h in Hs], dim=1)      # [B,H']
    r = torch.log(tgt / base)                                 # log-returns
    return r                                                  # [B,H']

def _ordinal_trend_loss_from_scalar_return(
    r_pred: torch.Tensor,      # [B,H]
    r_true: torch.Tensor,      # [B,H]
    bins: List[float],        
    tau: float = 0.02,         
) -> torch.Tensor:
    if r_pred.numel() == 0:
        return r_pred.sum() * 0 

    device, dtype = r_pred.device, r_pred.dtype
    t = torch.tensor(bins, device=device, dtype=dtype)  # [K]
    logits = (t.view(1, 1, -1) - r_pred.unsqueeze(-1)) / tau  # [B,H,K]
    y_cdf  = (r_true.unsqueeze(-1) <= t.view(1,1,-1)).to(dtype)  # [B,H,K]
    loss = nn.BCEWithLogitsLoss()(logits, y_cdf)
    return loss

def compute_training_loss(
    preds,          # [B, L, 5] 
    targets,        # [B, L, 5]
    tickers,           # [B]
    normalizer,                  
    horizons = (3, 5, 10),
    bins = None,  # Will be computed as log-return equivalents
    weights = dict(norm_mse=1, ret_mse=1, trend_ord=1, trend_pin=1),
    pinball_q = 0.5,      
    tau = 0.02            
) -> Dict[str, torch.Tensor]:
    # Convert percentage bins to log-return bins if not provided
    if bins is None:
        # Original percentage values: [-8%, -4%, -2%, -1%, 0%, 1%, 2%, 4%, 8%]
        percentage_bins = [-0.08, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.08]
        # Convert to log-returns: log(1 + percentage)
        bins = [torch.log(torch.tensor(1 + p)) for p in percentage_bins]
    
    preds = preds.transpose(1, 2)  # [B, 5, L]
    targets = targets.transpose(1, 2)  # [B, 5, L]
    device, dtype = preds.device, preds.dtype
    L_norm_mse = weighted_mse_loss(preds, targets)


    price_mu, price_sigma = _stack_stats_for_batch(tickers, normalizer, device, dtype)
    close_pred  = _denorm_close_from_norm_feat(preds,   price_mu, price_sigma)
    close_true  = _denorm_close_from_norm_feat(targets, price_mu, price_sigma)

    r_pred = _horizon_log_returns(close_pred, horizons)
    r_true = _horizon_log_returns(close_true, horizons)
    L_ret_mse = ((r_pred - r_true).pow(2).mean(dim=1)).mean()

    L_trend_ord = _ordinal_trend_loss_from_scalar_return(r_pred, r_true, bins, tau=tau)
    L_trend_pin = pinball_loss(r_pred, r_true, q=pinball_q).mean()


    total = (
        weights['norm_mse'] * L_norm_mse +
        weights['ret_mse']  * L_ret_mse  +
        weights['trend_ord']* L_trend_ord+
        weights['trend_pin']* L_trend_pin
    )

    return {
        "loss": total,
        "loss_norm_mse": L_norm_mse.detach(),
        "loss_return_mse":  L_ret_mse.detach(),
        "loss_trend_ord":L_trend_ord.detach(),
        "loss_trend_pin":L_trend_pin.detach(),
    }

if __name__ == "__main__":
    # generate normalized random feature with 0 mean and 1 std
    preds = torch.randn(5, 5, 11) * 0.05  # [B, C, L]
    targets = torch.randn(5, 5, 11) * 0.05  # [B, C, L]
    tickers = ["aapl", "msft", "goog", "amzn", "tsla"]
    normalizer = StockNormalizer()
    loss = compute_training_loss(preds, targets, tickers, normalizer)
    print("Loss with default bins (log-return equivalents):")
    print(loss)
    
    
