import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


from transformers import AutoModelForCausalLM, AutoTokenizer



class SinusoidalPositionalEmbedding(nn.Module):
   def __init__(self, dim: int, max_len: int = 4096):
       super().__init__()
       pe = torch.zeros(max_len, dim)
       position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
       div_term = torch.exp(
           torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
       )
       pe[:, 0::2] = torch.sin(position * div_term)
       pe[:, 1::2] = torch.cos(position * div_term)
       self.register_buffer("pe", pe, persistent=False)


   def forward(self, x: torch.Tensor) -> torch.Tensor:
       # x: [B, N, D]
       n = x.size(1)
       return x + self.pe[:n, :].unsqueeze(0).to(x.dtype)
    
    
class RMSNorm(nn.Module):
   def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
       super().__init__()
       self.eps = eps
       self.qwen3_compatible = qwen3_compatible
       self.scale = nn.Parameter(torch.ones(emb_dim))
       self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None


   def forward(self, x):
       input_dtype = x.dtype


       if self.qwen3_compatible:
           x = x.to(torch.float32)


       variance = x.pow(2).mean(dim=-1, keepdim=True)
       norm_x = x * torch.rsqrt(variance + self.eps)
       norm_x = norm_x * self.scale


       if self.shift is not None:
           norm_x = norm_x + self.shift


       return norm_x.to(input_dtype)


class PatchEmbed1D(nn.Module):
   def __init__(self, in_channels: int, d_model: int, patch_len: int):
       super().__init__()
       self.patch_len = patch_len
       self.proj = nn.Linear(in_channels, d_model) 


   def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
       # x: [B, L, C]
       B, L, C = x.shape
       x_proj = self.proj(x)  # [B, L, d_model]
       L_eff = (L // self.patch_len) * self.patch_len
       x_proj = x_proj[:, :L_eff, :]  # [B, L_eff, d_model]
       x_patch = rearrange(
           x_proj, "b (np p) d -> b np p d", p=self.patch_len
       ).mean(dim=2)  # [B, N_patch, d_model]
       return x_patch, L_eff // self.patch_len


class RotaryPositionalEmbedding(nn.Module):
   def __init__(self, dim: int, max_len: int = 8192):
       super().__init__()
       inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
       self.register_buffer("inv_freq", inv_freq, persistent=False)
       self.max_len = max_len


   def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
       # x: [B, N, D]
       seq_len = x.size(1)
       t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype) + offset
       freqs = torch.einsum("n,d->nd", t, self.inv_freq)
       emb = torch.cat((freqs, freqs), dim=-1)  # [N, D]
       return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
   # q, k: [B, N, H, D_head]
   # cos, sin: [1, N, D] where D = H * D_head
   # We need to extract the head dimension from the full d_model dimension
   B, N, H, D_head = q.shape
   D = cos.shape[-1]  # This should be d_model
   
   # Reshape cos/sin to [1, N, 1, D_head] by taking the first D_head dimensions
   cos = cos[:, :, None, :D_head].repeat(1, 1, H, 1)
   sin = sin[:, :, None, :D_head].repeat(1, 1, H, 1)


   def rotate_half(x):
       x1, x2 = x[..., ::2], x[..., 1::2]
       return torch.cat((-x2, x1), dim=-1)


   q_rot = (q * cos) + (rotate_half(q) * sin)
   k_rot = (k * cos) + (rotate_half(k) * sin)
   return q_rot, k_rot


class RoPEAttention(nn.Module):
   def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, batch_first: bool = True):
       super().__init__()
       assert batch_first, "RoPEAttention only supports batch_first=True"
       self.d_model = d_model
       self.nhead = nhead
       self.head_dim = d_model // nhead
       assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"


       self.q_proj = nn.Linear(d_model, d_model, bias=False)
       self.k_proj = nn.Linear(d_model, d_model, bias=False)
       self.v_proj = nn.Linear(d_model, d_model, bias=False)
       self.out_proj = nn.Linear(d_model, d_model, bias=False)


       self.dropout = nn.Dropout(dropout)
       self.scale = self.head_dim ** -0.5


   def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
       B, N, D = query.shape


       q = self.q_proj(query).view(B, N, self.nhead, self.head_dim)
       k = self.k_proj(key).view(B, N, self.nhead, self.head_dim)
       v = self.v_proj(value).view(B, N, self.nhead, self.head_dim)


       # Apply RoPE
       q, k = apply_rotary_pos_emb(q, k, cos, sin)


       # Rearrange for attention: [B, H, N, D_head]
       q = rearrange(q, "b n h d -> b h n d")
       k = rearrange(k, "b n h d -> b h n d")
       v = rearrange(v, "b n h d -> b h n d")


       # Scaled Dot-Product Attention
       attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
       if attn_mask is not None:
           # attn_mask: [B, N, N] -> [B, 1, N, N]
           attn_weights = attn_weights + attn_mask.unsqueeze(1)


       attn_weights = F.softmax(attn_weights, dim=-1)
       attn_weights = self.dropout(attn_weights)


       attn_output = torch.matmul(attn_weights, v)  # [B, H, N, D_head]
       attn_output = rearrange(attn_output, "b h n d -> b n (h d)") # [B, N, D]


       attn_output = self.out_proj(attn_output)
       return attn_output


class RoPETransformerEncoderLayer(nn.Module):
   def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                activation="gelu", batch_first: bool = True, norm_first: bool = True):
       super().__init__()
       self.norm_first = norm_first
       self.self_attn = RoPEAttention(d_model, nhead, dropout, batch_first)
       # Implementation of Feedforward layer
       self.linear1 = nn.Linear(d_model, dim_feedforward)
       self.dropout = nn.Dropout(dropout)
       self.linear2 = nn.Linear(dim_feedforward, d_model)


       self.norm1 = nn.LayerNorm(d_model)
       self.norm2 = nn.LayerNorm(d_model)
       self.dropout1 = nn.Dropout(dropout)
       self.dropout2 = nn.Dropout(dropout)
       self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()


   def forward(self, src: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
       # src: [B, N, D]
       if self.norm_first:
           # Attention block
           x = self.norm1(src)
           x = self.self_attn(x, x, x, cos, sin, attn_mask=src_mask)
           src = src + self.dropout1(x)


           # FFN block
           x = self.norm2(src)
           x = self.linear2(self.dropout(self.activation(self.linear1(x))))
           src = src + self.dropout2(x)
           return src
       else:
           # Attention block
           x = self.self_attn(src, src, src, cos, sin, attn_mask=src_mask)
           src = src + self.dropout1(x)
           src = self.norm1(src)


           # FFN block
           x = self.linear2(self.dropout(self.activation(self.linear1(src))))
           src = src + self.dropout2(x)
           src = self.norm2(src)
           return src

class TS2Encoder(nn.Module):
   def __init__(
       self,
       in_channels: int,
       d_model: int = 512,
       nhead: int = 8,
       num_layers: int = 8,
       patch_len: int = 8,
       dropout: float = 0.1,
       use_global_token: bool = True,
       out_days: int = 10,
   ):
       super().__init__()
       self.use_global_token = use_global_token
       self.patch = PatchEmbed1D(in_channels, d_model, patch_len)
       # Changed to RoPETransformerEncoderLayer
       encoder_layer = RoPETransformerEncoderLayer(
           d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
           dropout=dropout, activation="gelu", batch_first=True, norm_first=True
       )
       self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
       # Removed SinusoidalPositionalEmbedding, added RotaryPositionalEmbedding
       # Use head dimension for RoPE since we apply it per head
       self.rope = RotaryPositionalEmbedding(d_model // nhead, max_len=8192)
       if use_global_token:
           self.global_token = nn.Parameter(torch.randn(1, in_channels, d_model))

       self.norm = nn.LayerNorm(d_model)
       self.forecasting_head = nn.Linear(d_model, out_days)

   def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
       """
       x: [B, L, C]
       returns:
         ts_tokens: [B, N_tokens, D]
         ts_global: [B, D]
       """
       B = x.size(0)
       raw_tokens, n_patch = self.patch(x)  # [B, Np, D]
       if self.use_global_token:
           g = self.global_token.expand(B, -1, -1)
           tokens = torch.cat([g, raw_tokens], dim=1)  # [B, 5+Np, D]
       else:
           tokens = raw_tokens  # [B, Np, D]


       # Generate RoPE embeddings for the current sequence length
       cos, sin = self.rope(tokens)


       # Pass cos and sin to each layer of the encoder
       for layer in self.enc.layers:
           tokens = layer(tokens, cos, sin)


       tokens = self.norm(tokens)


       if self.use_global_token:
           ts_global = tokens[:, :5]      # [B, 5, D]
           ts_tokens = tokens[:, 5:]     # [B, Np, D]
       else:
           ts_global = tokens.mean(1) 
           ts_tokens = tokens


       return ts_tokens, ts_global
   
   def forecasting(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,L,C]
        returns forecasting: [B, out_days, C]
        """
        B,L,C = x.size()
        ts_tokens, ts_global = self.forward(x) #[B, C, D]
        forecasting = self.forecasting_head(ts_global) # [B,C, out_days]
        forecasting = forecasting.reshape(B, -1, C)
        return forecasting



class SoftTokenProjector(nn.Module):
   def __init__(self, ts_dim: int, llm_dim: int, num_soft_tokens: int = 8):
       super().__init__()
       self.num_soft_tokens = num_soft_tokens
       self.llm_dim = llm_dim
       if self.num_soft_tokens != 5:
            self.token_proj = nn.Linear(5, num_soft_tokens)
       self.proj = nn.Sequential(
            nn.Linear(ts_dim, ts_dim * 2),
            nn.GELU(),
            nn.Linear(ts_dim * 2, llm_dim),
        )



   def forward(self, ts_global: torch.Tensor) -> torch.Tensor:
       """
       ts_global: [B, 5, ts_dim]
       returns soft_tokens: [B, num_soft_tokens, llm_dim]
       """
       B = ts_global.size(0)
       if self.num_soft_tokens != 5:
           ts_global = ts_global.reshape(B, -1, 5) # [B, ts_dim, 5]
           ts_global = self.token_proj(ts_global)  # [B, ts_dim, num_soft_tokens]
           ts_global = ts_global.reshape(B, self.num_soft_tokens, -1) # [B, num_soft_tokens, ts_dim]
       out = self.proj(ts_global)  # [B, num_soft_tokens, llm_dim]
       return out
