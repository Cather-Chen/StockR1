import numpy as np


def _f(x, nd=2):
    """Plain Python float (not numpy), rounded to nd decimals."""
    return float(round(float(x), nd))


def compute_future_stats(future_df):
    closes = future_df["Close"]
    highs = future_df["High"]
    lows = future_df["Low"]
    current_price = float(closes[0])
    forecast_window = len(future_df)
    stats = {
        "start_value": _f(closes[0]),
        "end_value": _f(closes[-1]),
        "max_value": _f(np.max(highs)),
        "min_value": _f(np.min(lows)),
        "mean_close": _f(np.mean(closes)),
        "end_change_pct": _f((float(closes[-1]) - current_price) / current_price * 100),
        "range_pct": _f((float(np.max(highs)) - float(np.min(lows))) / current_price * 100),
        "max_drawdown_from_start": _f((float(np.min(lows)) - current_price) / current_price * 100),
        "peak_day": int(np.argmax(highs)) + 1,
        "trough_day": int(np.argmin(lows)) + 1,
    }
    
    # Direction
    if stats["end_change_pct"] > 1.5:
        stats["direction"] = "up"
    elif stats["end_change_pct"] < -1.5:
        stats["direction"] = "down"
    else:
        stats["direction"] = "sideways"
    
    # Volatility (annualized from daily returns)
    daily_returns = np.diff(closes) / closes[:-1]
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    if ann_vol < 0.15:
        stats["volatility_level"] = "low"
    elif ann_vol < 0.30:
        stats["volatility_level"] = "medium"
    else:
        stats["volatility_level"] = "high"
    
    # Turning points (simplified: local extrema)
    turning_points = 0
    for i in range(1, len(closes) - 1):
        if (closes[i] > closes[i-1] and closes[i] > closes[i+1]) or \
           (closes[i] < closes[i-1] and closes[i] < closes[i+1]):
            turning_points += 1
    stats["turning_point_count"] = turning_points
    
    # Timing bins
    def to_bin(day, window):
        third = window // 3
        if day <= third:
            return f"t+1_t+{third}"
        elif day <= 2 * third:
            return f"t+{third+1}_t+{2*third}"
        else:
            return f"t+{2*third+1}_t+{window}"
        
    stats["peak_timing_bin"] = to_bin(stats["peak_day"], forecast_window)
    stats["trough_timing_bin"] = to_bin(stats["trough_day"], forecast_window)
    
    # Max drawdown bin
    dd = abs(stats["max_drawdown_from_start"])
    if dd < 1:
        stats["max_drawdown_bin"] = "negligible"
    elif dd < 3:
        stats["max_drawdown_bin"] = "small"
    elif dd < 5:
        stats["max_drawdown_bin"] = "moderate"
    else:
        stats["max_drawdown_bin"] = "large"
    
    # Range width bin
    rng = stats["range_pct"]
    if rng < 3:
        stats["range_width_bin"] = "narrow"
    elif rng < 6:
        stats["range_width_bin"] = "moderate"
    else:
        stats["range_width_bin"] = "wide"
    
    # Monotonicity
    increasing = all(closes[i] <= closes[i+1] for i in range(len(closes)-1))
    decreasing = all(closes[i] >= closes[i+1] for i in range(len(closes)-1))
    if increasing:
        stats["monotonicity"] = "monotonic_up"
    elif decreasing:
        stats["monotonicity"] = "monotonic_down"
    else:
        stats["monotonicity"] = "non_monotonic"
    
    # Trendline fit (R² of linear regression on closes)
    from scipy.stats import linregress
    x = np.arange(len(closes))
    slope, intercept, r_value, _, _ = linregress(x, closes)
    r_sq = r_value ** 2
    if r_sq > 0.85:
        stats["trendline_fit"] = "strong"
    elif r_sq > 0.5:
        stats["trendline_fit"] = "moderate"
    else:
        stats["trendline_fit"] = "weak"
    
    # Tail risk
    if abs(daily_returns).max() > 0.04:
        stats["tail_risk_level"] = "high"
    elif abs(daily_returns).max() > 0.02:
        stats["tail_risk_level"] = "medium"
    else:
        stats["tail_risk_level"] = "low"
    
    return stats