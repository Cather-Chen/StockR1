"""
Forecast Hint Feature Extractor

Converts <forecast_hint> JSON into a dense conditioning tensor
for injection into a time-series forecasting transformer.

Design:
  - Numerical fields → normalized scalars
  - Categorical fields → learned embeddings
  - All concatenated → single conditioning vector
  - Can be injected via cross-attention, adaptive LayerNorm, or prefix tokens
"""

import re
import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# 1. SCHEMA DEFINITION
# =============================================================================

# Categorical field vocabularies (order matters — index 0 = padding/unknown)
CATEGORICAL_VOCAB = {
    "direction":          ["<unk>", "up", "down", "sideways"],
    "volatility_level":   ["<unk>", "low", "medium", "high"],
    "range_width_bin":    ["<unk>", "narrow", "moderate", "wide"],
    "max_drawdown_bin":   ["<unk>", "negligible", "small", "moderate", "large"],
    "peak_timing_bin":    ["<unk>", "t+1_t+3", "t+4_t+7", "t+8_t+10",
                           # finer bins if you use them:
                           "t+1_t+1", "t+2_t+2", "t+3_t+3", "t+4_t+4", "t+5_t+5",
                           "t+6_t+6", "t+7_t+7", "t+8_t+8", "t+9_t+9", "t+10_t+10",
                           "t+1_t+2", "t+2_t+4", "t+3_t+5", "t+5_t+7", "t+6_t+8", "t+8_t+9"],
    "trough_timing_bin":  ["<unk>", "t+1_t+3", "t+4_t+7", "t+8_t+10",
                           "t+1_t+1", "t+2_t+2", "t+3_t+3", "t+4_t+4", "t+5_t+5",
                           "t+6_t+6", "t+7_t+7", "t+8_t+8", "t+9_t+9", "t+10_t+10",
                           "t+1_t+2", "t+2_t+4", "t+3_t+5", "t+5_t+7", "t+6_t+8", "t+8_t+9"],
    "monotonicity":       ["<unk>", "monotonic_up", "monotonic_down", "non_monotonic"],
    "trendline_fit":      ["<unk>", "strong", "moderate", "weak"],
    "tail_risk_level":    ["<unk>", "low", "medium", "high"],
}

# Numerical fields + normalization stats (mean, std) — update with your dataset stats
# These are reasonable defaults; replace with actual dataset statistics.
NUMERICAL_FIELDS = {
    #                      mean     std
    "start_value":       (200.0,  150.0),   # price-level — will be overridden by per-sample norm
    "end_value":         (200.0,  150.0),
    "max_value":         (200.0,  150.0),
    "min_value":         (200.0,  150.0),
    "mean_close":        (200.0,  150.0),
    "end_change_pct":    (0.0,    5.0),
    "range_pct":         (5.0,    4.0),
    "turning_point_count": (3.0,  2.0),
}


def build_vocab_lookup(vocab: dict) -> dict:
    """Build string → index lookup for each categorical field."""
    return {
        field: {v: i for i, v in enumerate(values)}
        for field, values in vocab.items()
    }

VOCAB_LOOKUP = build_vocab_lookup(CATEGORICAL_VOCAB)


# =============================================================================
# 2. PARSING
# =============================================================================

def parse_forecast_hint(raw_string: str) -> dict:
    """Extract forecast_hint JSON from a raw string that may contain XML tags."""
    # Try to extract content between <forecast_hint> tags
    match = re.search(
        r'<forecast_hint>\s*(.*?)\s*</forecast_hint>',
        raw_string,
        re.DOTALL
    )
    if match:
        json_str = match.group(1)
    else:
        # Assume the whole string is JSON or try to find { ... }
        match = re.search(r'\{.*\}', raw_string, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            raise ValueError("No JSON found in input string")

    return json.loads(json_str)


def prepare_batch(
    hint_strings: list[str],
    current_prices: list[float],
) -> dict[str, torch.Tensor]:
    """
    Collate function helper. Call OUTSIDE the forward pass
    (in DataLoader collate_fn or preprocessing).

    Args:
        hint_strings: list of raw forecast_hint strings (length B)
        current_prices: list of current prices (length B)

    Returns:
        dict of batched tensors ready for forward()
    """
    if len(hint_strings) == 0:
        return None
    all_num, all_cat, all_pr = [], [], []

    for hint_str, price in zip(hint_strings, current_prices):
        hint = parse_forecast_hint(hint_str)
        feats = extract_raw_features(hint, current_price=price)
        all_num.append(feats.numerical)
        all_cat.append(feats.categorical_ids)
        all_pr.append(feats.price_relative)

    return {
        "numerical": torch.stack(all_num),       # (B, num_numerical)
        "categorical_ids": torch.stack(all_cat),  # (B, num_categorical)
        "price_relative": torch.stack(all_pr),    # (B, num_price_relative)
    }


# =============================================================================
# 3. RAW FEATURE EXTRACTION (dict → tensors, no learnable params)
# =============================================================================

@dataclass
class ForecastHintFeatures:
    """Raw extracted features before embedding."""
    numerical: torch.Tensor      # (num_numerical,) float
    categorical_ids: torch.Tensor  # (num_categorical,) long

    # Price-relative numerical features (normalized by current price)
    price_relative: torch.Tensor  # (num_price_features,) float


def extract_raw_features(
    hint: dict,
    current_price: Optional[float] = None,
) -> ForecastHintFeatures:
    """
    Convert a forecast_hint dict into raw tensors.

    Args:
        hint: parsed forecast_hint dictionary
        current_price: if provided, price-level fields are normalized
                       relative to current price (recommended)
    """
    # --- Numerical features ---
    numerical_values = []
    for field, (mean, std) in NUMERICAL_FIELDS.items():
        val = hint.get(field, 0.0)
        if val is None:
            val = 0.0
        numerical_values.append((float(val) - mean) / (std + 1e-8))

    # --- Price-relative features (more robust than absolute prices) ---
    price_relative_values = []
    if current_price and current_price > 0:
        cp = current_price
        # Relative deviations from current price (scale-invariant)
        price_relative_values = [
            (hint.get("start_value", cp) - cp) / cp,
            (hint.get("end_value", cp) - cp) / cp,
            (hint.get("max_value", cp) - cp) / cp,
            (hint.get("min_value", cp) - cp) / cp,
            (hint.get("mean_close", cp) - cp) / cp,
            # Spread features
            (hint.get("max_value", cp) - hint.get("min_value", cp)) / cp,
            # Asymmetry: upside vs downside
            (hint.get("max_value", cp) - cp) / (cp - hint.get("min_value", cp) + 1e-8),
        ]
    else:
        price_relative_values = [0.0] * 7

    # --- Categorical features ---
    categorical_ids = []
    for field in CATEGORICAL_VOCAB:
        val = hint.get(field, "<unk>")
        lookup = VOCAB_LOOKUP[field]
        idx = lookup.get(val, 0)  # 0 = <unk>
        categorical_ids.append(idx)

    return ForecastHintFeatures(
        numerical=torch.tensor(numerical_values, dtype=torch.float32),
        categorical_ids=torch.tensor(categorical_ids, dtype=torch.long),
        price_relative=torch.tensor(price_relative_values, dtype=torch.float32),
    )


# =============================================================================
# 4. LEARNABLE ENCODER (features → conditioning vector)
# =============================================================================

class ForecastHintEncoder(nn.Module):
    """
    Encodes forecast_hint features into a dense conditioning vector.

    Output can be used to condition a time-series transformer via:
      (a) Cross-attention: use as key/value sequence
      (b) Adaptive LayerNorm: predict scale/shift from this vector
      (c) Prefix tokens: project to token-sized vectors, prepend to sequence
      (d) Additive bias: add to every timestep's representation

    Args:
        cat_embed_dim:  embedding dimension per categorical field
        num_proj_dim:   projection dimension for numerical features
        output_dim:     final conditioning vector dimension (match your transformer's d_model)
        n_prefix_tokens: if using prefix-token injection, how many tokens to produce
    """

    def __init__(
        self,
        output_dim: int = 256,
        cat_embed_dim: int = 16,
        num_proj_dim: int = 64,
        n_prefix_tokens: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.n_prefix_tokens = n_prefix_tokens

        # --- Categorical embeddings ---
        self.cat_embeddings = nn.ModuleDict({
            field: nn.Embedding(
                num_embeddings=len(vocab),
                embedding_dim=cat_embed_dim,
                padding_idx=0,  # <unk> gets zero vector
            )
            for field, vocab in CATEGORICAL_VOCAB.items()
        })
        self.cat_fields = list(CATEGORICAL_VOCAB.keys())  # preserve order
        total_cat_dim = cat_embed_dim * len(CATEGORICAL_VOCAB)

        # --- Numerical projection ---
        num_input_dim = len(NUMERICAL_FIELDS)
        price_rel_dim = 7  # from extract_raw_features
        total_num_input = num_input_dim + price_rel_dim

        self.num_proj = nn.Sequential(
            nn.Linear(total_num_input, num_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_proj_dim, num_proj_dim),
        )

        # --- Fusion → output ---
        fusion_dim = total_cat_dim + num_proj_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # --- Optional: prefix token projection ---
        # Produces n_prefix_tokens vectors of size output_dim
        if n_prefix_tokens > 0:
            self.prefix_proj = nn.Linear(output_dim, n_prefix_tokens * output_dim)

    def forward(
        self,
        numerical: torch.Tensor,       # (B, num_numerical)
        categorical_ids: torch.Tensor,  # (B, num_categorical)
        price_relative: torch.Tensor,   # (B, num_price_relative)
    ) -> dict:
        """
        Returns:
            "vector":  (B, output_dim)            — single conditioning vector
            "prefix":  (B, n_prefix_tokens, output_dim) — prefix tokens (if enabled)
        """
        B = numerical.shape[0]

        # Embed categoricals
        cat_embeds = []
        for i, field in enumerate(self.cat_fields):
            cat_embeds.append(self.cat_embeddings[field](categorical_ids[:, i]))
        cat_concat = torch.cat(cat_embeds, dim=-1)  # (B, total_cat_dim)

        # Project numericals
        num_input = torch.cat([numerical, price_relative], dim=-1)
        num_proj = self.num_proj(num_input)  # (B, num_proj_dim)

        # Fuse
        fused = torch.cat([cat_concat, num_proj], dim=-1)
        output_vec = self.fusion(fused)  # (B, output_dim)

        result = {"vector": output_vec}

        # Optional prefix tokens
        if self.n_prefix_tokens > 0:
            prefix = self.prefix_proj(output_vec)  # (B, n_tokens * d)
            prefix = prefix.view(B, self.n_prefix_tokens, self.output_dim)
            result["prefix"] = prefix

        return result


# =============================================================================
# 5. INTEGRATION HELPERS
# =============================================================================

class ForecastHintConditioner(nn.Module):
    """
    End-to-end: raw string → conditioning tensor.

    Wraps parsing + feature extraction + encoding in one module.
    The non-differentiable parsing step happens in `prepare_batch`,
    which should be called in your DataLoader collate_fn.
    """

    def __init__(self, output_dim: int = 256, **encoder_kwargs):
        super().__init__()
        self.encoder = ForecastHintEncoder(output_dim=output_dim, **encoder_kwargs)

    def forward(self, numerical, categorical_ids, price_relative):
        return self.encoder(numerical, categorical_ids, price_relative)


# =============================================================================
# 6. INJECTION STRATEGIES (for your transformer)
# =============================================================================

class PrefixInjection(nn.Module):
    """
    Prepend forecast_hint as prefix tokens to the time-series input sequence.

    transformer_input: (B, T, d_model)  → (B, n_prefix + T, d_model)
    """
    def __init__(self, conditioner: ForecastHintConditioner):
        super().__init__()
        self.conditioner = conditioner

    def forward(self, ts_input, hint_batch):
        """
        Args:
            ts_input: (B, T, d_model) — embedded time-series
            hint_batch: dict from prepare_batch
        Returns:
            (B, n_prefix + T, d_model)
        """
        cond = self.conditioner(**hint_batch)
        prefix = cond["prefix"]          # (B, n_prefix, d_model)
        return torch.cat([prefix, ts_input], dim=1)


class AdaptiveLayerNormInjection(nn.Module):
    """
    Condition each transformer layer via adaptive LayerNorm (like DiT).

    Given conditioning vector c, predict scale γ and shift β:
      output = γ * LayerNorm(x) + β
    """
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_model),  # predict γ and β
        )

    def forward(self, x: torch.Tensor, cond_vector: torch.Tensor):
        """
        Args:
            x: (B, T, d_model)
            cond_vector: (B, cond_dim)
        """
        gamma_beta = self.proj(cond_vector)  # (B, 2 * d_model)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (B, d_model)
        # Unsqueeze for broadcasting over T
        return gamma.unsqueeze(1) * self.norm(x) + beta.unsqueeze(1)


class CrossAttentionInjection(nn.Module):
    """
    Condition transformer via cross-attention to forecast_hint tokens.

    Time-series attends to hint prefix tokens.
    Insert this as an extra sub-layer in each transformer block.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hint_tokens: torch.Tensor):
        """
        Args:
            x: (B, T, d_model) — time-series hidden states
            hint_tokens: (B, n_prefix, d_model) — from encoder prefix output
        """
        residual = x
        x = self.norm(x)
        attn_out, _ = self.cross_attn(query=x, key=hint_tokens, value=hint_tokens)
        return residual + self.dropout(attn_out)


# =============================================================================
# 7. USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
#     raw_hint = """<forecast_hint>
# {
#   "future_window": "t+1_t+10",
#   "start_value": 133.31,
#   "end_value": 133.83,
#   "max_value": 137.49,
#   "min_value": 128.07,
#   "mean_close": 132.72,
#   "direction": "sideways",
#   "end_change_pct": 0.39,
#   "range_pct": 7.07,
#   "volatility_level": "high",
#   "range_width_bin": "wide",
#   "max_drawdown_bin": "moderate",
#   "turning_point_count": 6,
#   "peak_timing_bin": "t+2_t+2",
#   "trough_timing_bin": "t+3_t+5",
#   "monotonicity": "non_monotonic",
#   "trendline_fit": "weak",
#   "tail_risk_level": "high"
# }
# </forecast_hint>"""


#     # --- Step 1: Parse + extract ---
#     hint_dict = parse_forecast_hint(raw_hint)
#     feats = extract_raw_features(hint_dict)

#     print("=== Raw Features ===")
#     print(f"Numerical:      {feats.numerical.shape} → {feats.numerical}")
#     print(f"Categorical:    {feats.categorical_ids.shape} → {feats.categorical_ids}")
#     print(f"Price-relative: {feats.price_relative.shape} → {feats.price_relative}")

#     # --- Step 2: Batched encoding ---
#     d_model = 256
#     conditioner = ForecastHintConditioner(output_dim=d_model, n_prefix_tokens=4)

#     # Simulate a batch of 2
#     batch = prepare_batch(
#         hint_strings=[raw_hint, raw_hint],
#         current_prices=[133.31, 133.31],
#     )
#     output = conditioner(**batch)

#     print(f"\n=== Encoder Output ===")
#     print(f"Conditioning vector: {output['vector'].shape}")   # (2, 256)
#     print(f"Prefix tokens:       {output['prefix'].shape}")   # (2, 4, 256)

#     # --- Step 3: Injection demo ---
#     B, T = 2, 60
#     fake_ts_input = torch.randn(B, T, d_model)

#     # Option A: Prefix injection
#     prefix_injector = PrefixInjection(conditioner)
#     out_a = prefix_injector(fake_ts_input, batch)
#     print(f"\n=== Prefix Injection ===")
#     print(f"Input:  {fake_ts_input.shape}")   # (2, 60, 256)
#     print(f"Output: {out_a.shape}")            # (2, 64, 256) — 4 prefix + 60 ts

#     # Option B: Adaptive LayerNorm
#     adaln = AdaptiveLayerNormInjection(d_model=d_model, cond_dim=d_model)
#     cond_vec = output["vector"]
#     out_b = adaln(fake_ts_input, cond_vec)
#     print(f"\n=== AdaLN Injection ===")
#     print(f"Output: {out_b.shape}")            # (2, 60, 256)

#     # Option C: Cross-attention
#     xattn = CrossAttentionInjection(d_model=d_model)
#     out_c = xattn(fake_ts_input, output["prefix"])
#     print(f"\n=== Cross-Attention Injection ===")
#     print(f"Output: {out_c.shape}")            # (2, 60, 256)

    with open("data/nqsp_stock/processed/sft.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            hint_str = data.get("forecast_hint", "")
            if not hint_str:
                continue
            hint_dict = parse_forecast_hint(hint_str)
            feats = extract_raw_features(hint_dict)
            print(f"Numerical:      {feats.numerical.shape} → {feats.numerical}")
            print(f"Categorical:    {feats.categorical_ids.shape} → {feats.categorical_ids}")
            print(f"Price-relative: {feats.price_relative.shape} → {feats.price_relative}")
