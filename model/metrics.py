import numpy as np
from typing import Dict, Tuple, List, Literal
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
import ipdb
import matplotlib.pyplot as plt

def instance_normalize(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-8):
    return (x - mu.reshape(-1, 1)) / (sigma.reshape(-1, 1) + eps)

# ---------- denorm: from [low, delta_o, delta_c, delta_h, vol_norm] -> OHLCV ----------
def _denorm_one_to_ohlcv(
    x_norm: np.ndarray,                             # (5, L)
    price_stats: Dict[str, np.ndarray],             # {'mu': (4,), 'sigma': (4,)}
    volume_stats: Dict[str, float],                 # {'mu': float, 'sigma': float}
) -> np.ndarray:
    x_norm = np.asarray(x_norm, dtype=float)
    price_mu = price_stats["mu"].reshape(-1, 1)
    price_sd = np.maximum(price_stats["sigma"].reshape(-1, 1), 1e-12)
    vol_mu   = float(volume_stats["mu"])
    vol_sd   = max(float(volume_stats["sigma"]), 1e-12)

    price_feat = x_norm[0:4, :] * price_sd + price_mu
    low      = price_feat[0, :]
    delta_o  = price_feat[1, :]
    delta_c  = price_feat[2, :]
    delta_h  = price_feat[3, :]

    L = low
    O = L + delta_o
    C = L + delta_c
    base_max = np.maximum(np.maximum(O, C), L)
    H = delta_h + base_max

    vol = np.expm1(x_norm[4, :] * vol_sd + vol_mu)
    ohlcv = np.stack([O, H, L, C, vol], axis=0)
    return ohlcv

def denormalize_batch_to_ohlcv(
    x_norm: np.ndarray,
    tickers: List[str],
    price_stats: Dict[str, Dict[str, np.ndarray]],
    volume_stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    assert x_norm.ndim == 3 and x_norm.shape[1] == 5, "Expected shape (N, 5, L)."
    assert len(tickers) == x_norm.shape[0], "tickers length must match batch size."
    out = np.empty((x_norm.shape[0], 5, x_norm.shape[2]), dtype=float)
    for i, t in enumerate(tickers):
        if t not in price_stats or t not in volume_stats:
            raise KeyError(f"Missing stats for ticker {t}.")
        out[i] = _denorm_one_to_ohlcv(
            x_norm[i],
            price_stats[t],
            volume_stats[t],
        )
    return out  # (N, 5, L) with [O,H,L,C,V]

# ---------- top-level ----------
def evaluate_metrics_with_denorm(
    pred_norm: np.ndarray,
    gt_norm: np.ndarray,
    tickers: List[str],
    price_stats: Dict[str, Dict[str, np.ndarray]],
    volume_stats: Dict[str, Dict[str, float]],
    input_raw_stats: np.ndarray,
    channel_index = {'open':0, 'high':1, 'low':2, 'close':3, 'volume':4},
    trend_bins = [-0.08, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.08],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
        Evaluate:
        Price:   MAPE / RMSE / IC / RankIC (compute correlation along time per stock, then average over stocks)
        Return:  RMSE (daily simple returns) / 3-day & 10-day Directional Accuracy / 3-day & 10-day Trend F1
                (10-class trend bins = [-0.08, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.08])
        Volatility: MAE / R^2 (based on high/close ratio)

        Time dimension = 11: historical last day + next 10 days.
        """
   
    pred = denormalize_batch_to_ohlcv(pred_norm, tickers, price_stats, volume_stats)
    target   = denormalize_batch_to_ohlcv(gt_norm,  tickers, price_stats, volume_stats)

    t_hist = 0
    t_future = slice(1, 11)  # next 10 days

    # ---------- Price metrics (instance-wise normalized) ----------

    p_pred = pred[:, channel_index['close'], t_future]    # [N,10]
    p_true = target[:, channel_index['close'], t_future]  # [N,10]

    # instance-wise normalize again to unify the magnitude
    close_raw_mu = input_raw_stats[:, channel_index['close']]
    close_raw_sigma = input_raw_stats[:, channel_index['close']+4]
    p_pred_norm = instance_normalize(p_pred, close_raw_mu, close_raw_sigma)
    p_true_norm = instance_normalize(p_true, close_raw_mu, close_raw_sigma)
    
    # Price MAPE (mask near-zero denominator)
    mape_price = np.mean(np.abs((p_pred - p_true) / p_true))
    

    # Price RMSE
    rmse_price = np.sqrt(np.mean((p_pred_norm - p_true_norm) ** 2))


    # ---------- Return metrics ----------
    c_true_hist = target[:, channel_index['close'], t_hist]   # [N]
    c_true_fut  = target[:, channel_index['close'], t_future] # [N,10]
    c_pred_fut  = p_pred                                     # [N,10]

    r_true = (c_true_fut / np.clip(c_true_hist[:, None], eps, None)) - 1  # [N,10]

    r_pred = (c_pred_fut / np.clip(c_true_hist[:, None], eps, None)) - 1  # [N,10]

    # Return RMSE across all horizons and samples
    rmse_return = np.sqrt(np.mean((r_pred - r_true) ** 2))

    # Directional Accuracy (3-day / 10-day): based on cumulative simple return R = P_{t+h}/P_t - 1
    def cum_return(close_series, horizon):
        base = c_true_hist[:, None]  # [N,1]
        idx = horizon - 1       
        idx = min(max(idx, 0), close_series.shape[1]-1)
        return (close_series[:, idx] / np.clip(base[:, 0], eps, None)) - 1.0  # [N]

    # Cumulative returns (3-day and 10-day) for true & predicted prices
    r_true_3  = cum_return(c_true_fut, 3)
    r_pred_3  = cum_return(c_pred_fut, 3)
    r_true_5  = cum_return(c_true_fut, 5)
    r_pred_5  = cum_return(c_pred_fut, 5)
    r_true_10 = cum_return(c_true_fut, 10)
    r_pred_10 = cum_return(c_pred_fut, 10)

    def DA(pred_cum, true_cum):
        return np.mean(np.sign(pred_cum) == np.sign(true_cum))

    da_3  = DA(r_pred_3,  r_true_3)
    da_5  = DA(r_pred_5,  r_true_5)
    da_10 = DA(r_pred_10, r_true_10)

    # Trend Classification (3-day/10-day): use bins to discretize cumulative returns into 10 classes
    bins = np.array(sorted(trend_bins), dtype=float)

    def to_trend_class(x):
        # returns [0..9] class index
        return np.digitize(x, bins, right=False)  # [-inf,b0), [b0,b1), ...

    y_true_3  = to_trend_class(r_true_3)
    y_pred_3  = to_trend_class(r_pred_3)
    y_true_5  = to_trend_class(r_true_5)
    y_pred_5  = to_trend_class(r_pred_5)
    y_true_10 = to_trend_class(r_true_10)
    y_pred_10 = to_trend_class(r_pred_10)

    f1_3  = f1_score(y_true_3, y_pred_3, average='weighted', zero_division=0)
    f1_5  = f1_score(y_true_5, y_pred_5, average='weighted', zero_division=0)
    f1_10 = f1_score(y_true_10, y_pred_10, average='weighted', zero_division=0)

    # ---------- Realized Volatility (sum of squared log-returns) ----------
    # Horizon H = number of future days (here 10). We compute:
    #   RV = sum_{i=1}^{H-1} ( log(p_{i+1}) - log(p_i) )^2
    # for both predicted and true closing-price paths, then report MAE and R^2.

    # close prices over the future horizon
    p_pred = pred[:, channel_index['close'], t_future]   # [N, H]
    p_true = target[:, channel_index['close'], t_future] # [N, H]

    def calculate_rv(p_pred, p_true):
        log_p_pred = np.log(np.clip(p_pred, eps, None))
        log_p_true = np.log(np.clip(p_true, eps, None))

        lr_pred = log_p_pred[:, 1:] - log_p_pred[:, :-1]  # [N, H-1]
        lr_true = log_p_true[:, 1:] - log_p_true[:, :-1]  # [N, H-1]

        # realized volatility as sum of squared log-returns (variance-like)
        rv_pred = np.sum(lr_pred ** 2, axis=1)  # [N]
        rv_true = np.sum(lr_true ** 2, axis=1)  # [N]

        # MAE
        mae_rv = float(np.mean(np.abs(rv_pred - rv_true)))

        # R^2 = 1 - SSE/SST
        sse = np.sum((rv_pred - rv_true) ** 2)
        sst = np.sum((rv_true - np.mean(rv_true)) ** 2) + eps
        r2_rv = float(1.0 - sse / sst)
        return mae_rv, r2_rv
    
    mae_rv, r2_rv = calculate_rv(p_pred, p_true)
    
    
    # ---------- Volume metrics (MAPE / RMSE) ----------
    v_pred_norm = pred_norm[:, channel_index['volume'], t_future]
    v_true_norm = gt_norm[:, channel_index['volume'], t_future]
    mape_volume = np.mean(np.abs((v_pred_norm - v_true_norm) / v_true_norm))
    rmse_volume = np.sqrt(np.mean((v_pred_norm - v_true_norm) ** 2))
    
    
    # collect metrics (rename keys for clarity)
    # ---------- Aggregate results ----------
    metrics = {
        # Price
        'price/mape':    float(mape_price),
        'price/rmse':    float(rmse_price),
        # 'price/ic':      float(ic_price),
        # 'price/rankic':  float(rankic_price),

        # Return
        'return/rmse_daily': float(rmse_return),
        'return/da_3':    float(da_3),
        'return/da_5':   float(da_5),
        'return/da_10':   float(da_10),
        'return/f1_3':    float(f1_3),
        'return/f1_5':    float(f1_5),
        'return/f1_10':   float(f1_10),

        # Volatility
        'vol/mae': float(mae_rv),
        'vol/r2':  float(r2_rv),


        # Volume
        'amount/mape': float(mape_volume),
        'amount/rmse': float(rmse_volume),
    }
    
    return metrics, pred, target



def evaluate_metrics_after_denorm(
    pred: np.ndarray,
    target: np.ndarray,
    last_close: np.ndarray,
    last_volume: np.ndarray,
    channel_index = {'open':0, 'high':1, 'low':2, 'close':3, 'volume':4},
    trend_bins = [-0.08, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.08],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
        pred: [N, 5, L]
        target: [N, 5, L]
        last_close: [N] last observed close (used as history base)
        last_volume: [N] last observed volume (used for volume-relative metrics)
        Evaluate:
        Price:   MAPE / RMSE / IC / RankIC (compute correlation along time per stock, then average over stocks)
        Return:  RMSE (daily simple returns) / 3-day & 10-day Directional Accuracy / 3-day & 10-day Trend F1
                (10-class trend bins = [-0.08, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.08])
        Volatility: MAE / R^2 (based on high/close ratio)

        Time dimension = 11: historical last day + next 10 days.
        """
    def _instance_normalize(x: np.ndarray):
        if x.ndim == 2:
            mean = x.mean(axis=1, keepdims=True)   # per instance
            std = x.std(axis=1, keepdims=True)
            return (x - mean) / (std + 1e-8)
        elif x.ndim == 3:
            mean = x.mean(axis=-1, keepdims=True)  # per instance per channel
            std = x.std(axis=-1, keepdims=True)
            return (x - mean) / (std + 1e-8)
        else:
            raise ValueError(f"Unsupported input shape {x.shape}, only [N, L] or [N, C, L] are supported.")
    # ensure the shape is [N, 5, L]; convert if input is [N, L, 5]
    if pred.shape[1] == 5:
        pred = pred
        target = target
    elif pred.shape[2] == 5:
        pred = np.transpose(pred, (0, 2, 1))
        target = np.transpose(target, (0, 2, 1))
    else:
        raise ValueError(f"pred/target should have channel dimension size 5; got {pred.shape}")

    B, C, L = pred.shape
    p_pred = pred[:, channel_index['close']]   
    p_true = target[:, channel_index['close']] 

    # instance-wise normalize again to unify the magnitude
    p_pred_norm = _instance_normalize(p_pred)
    p_true_norm = _instance_normalize(p_true)
    
    # Price MAPE (mask near-zero denominator)
    mape_price = np.mean(np.abs((p_pred - p_true) / np.clip(p_true, eps, None)))
    

    # Price RMSE
    rmse_price = np.sqrt(np.mean((p_pred_norm - p_true_norm) ** 2))


    # ---------- Return metrics ----------
    c_true_hist = np.asarray(last_close).reshape(B)   # use provided last close as history
    c_true_fut  = target[:, channel_index['close']] 
    c_pred_fut  = p_pred                                     

    r_true = (c_true_fut / np.clip(c_true_hist[:, None], eps, None)) - 1  

    r_pred = (c_pred_fut / np.clip(c_true_hist[:, None], eps, None)) - 1  

    # Return RMSE across all horizons and samples
    rmse_return = np.sqrt(np.mean((r_pred - r_true) ** 2))

    def cum_return(close_series, horizon):
        base = c_true_hist[:, None]  
        idx = horizon - 1       
        idx = min(max(idx, 0), close_series.shape[1]-1)
        return (close_series[:, idx] / np.clip(base[:, 0], eps, None)) - 1.0  # [N]

    r_true_3  = cum_return(c_true_fut, 3)
    r_pred_3  = cum_return(c_pred_fut, 3)
    r_true_5  = cum_return(c_true_fut, 5)
    r_pred_5  = cum_return(c_pred_fut, 5)
    r_true_10 = cum_return(c_true_fut, 10)
    r_pred_10 = cum_return(c_pred_fut, 10)

    def DA(pred_cum, true_cum):
        return np.mean(np.sign(pred_cum) == np.sign(true_cum))

    da_3  = DA(r_pred_3,  r_true_3)
    da_5  = DA(r_pred_5,  r_true_5)
    da_10 = DA(r_pred_10, r_true_10)

    bins = np.array(sorted(trend_bins), dtype=float)

    def to_trend_class(x):
        # returns [0..9] class index
        return np.digitize(x, bins, right=False)  # [-inf,b0), [b0,b1), ...

    y_true_3  = to_trend_class(r_true_3)
    y_pred_3  = to_trend_class(r_pred_3)
    y_true_5  = to_trend_class(r_true_5)
    y_pred_5  = to_trend_class(r_pred_5)
    y_true_10 = to_trend_class(r_true_10)
    y_pred_10 = to_trend_class(r_pred_10)

    f1_3  = f1_score(y_true_3, y_pred_3, average='weighted', zero_division=0)
    f1_5  = f1_score(y_true_5, y_pred_5, average='weighted', zero_division=0)
    f1_10 = f1_score(y_true_10, y_pred_10, average='weighted', zero_division=0)

    # ---------- Realized Volatility (sum of squared log-returns) ----------
    # Horizon H = number of future days (here 10). We compute:
    #   RV = sum_{i=1}^{H-1} ( log(p_{i+1}) - log(p_i) )^2
    # for both predicted and true closing-price paths, then report MAE and R^2.

    # close prices over the future horizon
    # p_pred = pred[:, channel_index['close'], t_future]   # [N, H]
    # p_true = target[:, channel_index['close'], t_future] # [N, H]

    def calculate_rv(p_pred, p_true):
        log_p_pred = np.log(np.clip(p_pred, eps, None))
        log_p_true = np.log(np.clip(p_true, eps, None))

        lr_pred = log_p_pred[:, 1:] - log_p_pred[:, :-1]  # [N, H-1]
        lr_true = log_p_true[:, 1:] - log_p_true[:, :-1]  # [N, H-1]

        # realized volatility as sum of squared log-returns (variance-like)
        rv_pred = np.sum(lr_pred ** 2, axis=1)  # [N]
        rv_true = np.sum(lr_true ** 2, axis=1)  # [N]

        # MAE
        mae_rv = float(np.mean(np.abs(rv_pred - rv_true)))

        # R^2 = 1 - SSE/SST
        sse = np.sum((rv_pred - rv_true) ** 2)
        sst = np.sum((rv_true - np.mean(rv_true)) ** 2) + eps
        r2_rv = float(1.0 - sse / sst)
        return mae_rv, r2_rv
    
    mae_rv, r2_rv = calculate_rv(p_pred, p_true)
    
    
    # ---------- Volume metrics (MAPE / RMSE) ----------
    v_pred_norm = _instance_normalize(pred[:, channel_index['volume'],:])
    v_true_norm = _instance_normalize(target[:, channel_index['volume'],:])
    base_vol = np.clip(np.asarray(last_volume).reshape(B, 1), eps, None)
    v_pred_ret = pred[:, channel_index['volume'], :] / base_vol - 1
    v_true_ret = target[:, channel_index['volume'], :] / base_vol - 1
    mape_volume = np.mean(np.abs((v_pred_norm - v_true_norm) / np.maximum(np.abs(v_true_norm), eps)))
    rmse_volume = np.sqrt(np.mean((v_pred_norm - v_true_norm) ** 2))
    
    
    # collect metrics (rename keys for clarity)
    # ---------- Aggregate results ----------
    metrics = {
        # Price
        'price/mape':    float(mape_price),
        'price/rmse':    float(rmse_price),
        # 'price/ic':      float(ic_price),
        # 'price/rankic':  float(rankic_price),

        # Return
        'return/rmse_daily': float(rmse_return),
        'return/da_3':    float(da_3),
        'return/da_5':   float(da_5),
        'return/da_10':   float(da_10),
        'return/f1_3':    float(f1_3),
        'return/f1_5':    float(f1_5),
        'return/f1_10':   float(f1_10),

        # Volatility
        'vol/mae': float(mae_rv),
        'vol/r2':  float(r2_rv),


        # Volume
        'amount/mape': float(mape_volume),
        'amount/rmse': float(rmse_volume),
    }
    
    return metrics





if __name__ == "__main__":
    pred_norm = np.random.randn(1, 11, 5) * 0.01
    gt_norm = pred_norm
    tickers = ["AAPL"]
    price_stats = {"AAPL": {"mu": np.array([0.0, 0.0, 0.0, 0.0]), "sigma": np.array([1.0, 1.0, 1.0, 1.0])}}
    volume_stats = {"AAPL": {"mu": 0.0, "sigma": 1.0}}
    print(evaluate_metrics_with_denorm(pred_norm, gt_norm, tickers, price_stats, volume_stats))
