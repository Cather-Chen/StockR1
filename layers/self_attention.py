import abc
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from einops import repeat, rearrange
from layers.attn_projection import QueryKeyProjection, RotaryProjection


class TimerMultivariateMask():
    def __init__(self, B, n_vars, n_tokens, device="cpu"):
        mask_shape = [B, 1, n_tokens, n_tokens]
        with torch.no_grad():
            self._mask1 = torch.ones((n_vars, n_vars), dtype=torch.bool).to(device)
            self._mask2 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            self._mask = torch.kron(self._mask1, self._mask2)
    @property
    def mask(self):
        return self._mask

class BinaryAttentionBias(nn.Module, abc.ABC):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=self.num_heads)

    def forward(self, query_id, kv_id):
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        weight = rearrange(
            self.emb.weight, "two num_heads -> two num_heads 1 1")
        bias = ~ind * weight[:1] + ind * weight[1:]
        return bias


class TimeAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, d_model=512, num_heads=8, max_len=100,flash_attention=False):
        super(TimeAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection, kwargs=dict(max_len=max_len),
                                          partial_factor=(0.0, 0.5),)
        self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=num_heads)

    def forward(self, queries, keys, values, n_vars, n_tokens):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # [B, H, L, E]
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        full_seq = torch.cat([torch.arange(0, n_tokens, device=queries.device).repeat(n_vars)])
        seq_id = repeat(full_seq, 'n -> b h n', b=B, h=H)   #[B, H, total_len]

        queries, keys = self.qk_proj(
            queries, keys, query_id=seq_id, kv_id=seq_id)

        scale = self.scale or 1. / sqrt(E)

        var_id = repeat(torch.arange(n_vars, device=queries.device),
                        'C -> (C n_tokens)', n_tokens=n_tokens)
        var_id = repeat(var_id, 'L -> b h L', b=B, h=1)

        attn_bias = self.attn_bias(var_id, var_id)


        attn_mask = TimerMultivariateMask(
            B, n_vars, n_tokens, device=queries.device)
        attn_mask = attn_bias.masked_fill(attn_mask.mask, float("-inf"))

        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            scores += attn_mask
            
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, n_vars, n_tokens):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            n_vars,
            n_tokens,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)