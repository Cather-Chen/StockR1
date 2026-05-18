import torch
from torch import nn
import torch.nn.functional as F
from layers.self_attention import AttentionLayer, TimeAttention



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, ffn_dropout_p=0.0):
        super().__init__()

        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self,  d_model, n_heads, ff_dim=1024, ffn_dropout_p=0.0, attn_dropout_p=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = AttentionLayer(TimeAttention(attention_dropout=attn_dropout_p, d_model=d_model, num_heads=n_heads), d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, ffn_dropout_p)

    def forward(self, x, in_channels, n_tokens):
        residual = x
        x = self.norm1(x)
        attn_out = self.self_attn(x,x,x,in_channels, n_tokens)  # [B, C * N, D]
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        return x

class MultichannelTSEncoder(nn.Module):
    def __init__(
       self,
       input_len: int = 100,
       in_channels: int = 5,
       d_model: int = 512,
       nhead: int = 8,
       num_layers: int = 8,
       patch_len: int = 8,
       dropout: float = 0.1,
       out_days: int = 10,
       num_soft_tokens: int = 5,
       llm_dim: int = 4096,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_heads = nhead
        self.enc_layers = num_layers
        self.patch_len = patch_len

        self.dropout = dropout
        self.out_days = out_days
        self.num_soft_tokens = num_soft_tokens
        self.llm_dim = llm_dim
        
        
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        self.encoder = nn.ModuleList([
            TransformerBlock(d_model=self.d_model, 
                             n_heads=self.n_heads, 
                             ffn_dropout_p=self.dropout, 
                             attn_dropout_p=self.dropout) for _ in range(self.enc_layers)])

        if self.num_soft_tokens != self.in_channels:
            self.token_proj = nn.Linear(self.in_channels, self.num_soft_tokens)
        self.soft_token_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.llm_dim),
        )
        self.num_patches = input_len // patch_len
        self.flatten_dim = self.num_patches * self.d_model
        self.price_head  = nn.Linear(self.flatten_dim, self.out_days * 4)
        self.vol_head    = nn.Linear(self.flatten_dim, self.out_days)
        self.eps = 1e-6
        
    def embedding(self, x):
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)  # [B, C, N, P]
        x = self.patch_embedding(x)  # [B, C, N, D]
        N = x.shape[2]
        
        assert N == self.num_patches, "Number of patches does not match"
        x = x.reshape(B, C * N, -1)  # [B, C * N, D]
        for layer in self.encoder:
            x = layer(x, self.in_channels, N) # [B, C * N, D]
            
        x = x.reshape(B, C, N, -1)
        return x  # [B, C, N, D]


    def forecasting(self, x):
        x = self.embedding(x)                 # [B, C, N, D]
        B, C, N, D = x.shape
        x = x.reshape(B, C, -1)               # [B, C, N*D]
        x = F.gelu(x)

        x_agg = x.mean(dim=1)                 # [B, N*D]   Aggregate to sample level

        raw_price = self.price_head(x_agg)    # [B, out_days*4]
        raw_vol   = self.vol_head(x_agg)      # [B, out_days]

        raw_price = raw_price.view(B, self.out_days, 4)  # [:, :, (L, dO, dC, dH)]

        L_hat  = F.softplus(raw_price[..., 0]) + self.eps    # [B, out_days]
        dO_hat = F.softplus(raw_price[..., 1])               # [B, out_days]
        dC_hat = F.softplus(raw_price[..., 2])               # [B, out_days]
        dH_hat = F.softplus(raw_price[..., 3])               # [B, out_days]
        V_hat  = F.softplus(raw_vol)                         # [B, out_days]

        out = torch.stack([L_hat, dO_hat, dC_hat, dH_hat, V_hat], dim=-1)   # [B, out_days, 5]
        return out
    
    def get_soft_tokens(self, x):
        x = self.embedding(x)  # [B, C, N, D]
        x = x.mean(2)  # [B, C, D]
        B, C, D = x.shape
        if self.num_soft_tokens != self.in_channels:
            x = x.reshape(B, -1, self.in_channels)  # [B, D, C]
            x = self.token_proj(x)  # [B, D, num_soft_tokens]
            x = x.reshape(B, self.num_soft_tokens, -1)  # [B, num_soft_tokens, D]
        soft_tokens = self.soft_token_proj(x)  # [B, num_soft_tokens, llm_dim]
        return soft_tokens  # [B, num_soft_tokens, llm_dim]
    
    
class MultichannelTSEncoder_rm(nn.Module):
    def __init__(
        self,
        input_len: int = 100,
        in_channels: int = 5,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        patch_len: int = 8,
        dropout: float = 0.1,
        out_days: int = 10,
        num_soft_tokens: int = 5,
        llm_dim: int = 4096,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_heads = nhead
        self.enc_layers = num_layers
        self.patch_len = patch_len
        self.dropout = dropout
        self.out_days = out_days
        self.num_soft_tokens = num_soft_tokens
        self.llm_dim = llm_dim
        self.eps = 1e-6

        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    ffn_dropout_p=self.dropout,
                    attn_dropout_p=self.dropout,
                )
                for _ in range(self.enc_layers)
            ]
        )

        self.num_patches = input_len // patch_len
        self.flatten_dim = self.num_patches * self.d_model
        if self.num_soft_tokens != self.in_channels:
            self.token_proj = nn.Linear(self.in_channels, self.num_soft_tokens)
        # self.soft_token_proj = nn.Sequential(
        #     nn.Linear(self.d_model * self.num_patches, self.d_model),
        #     nn.GELU(),
        #     nn.Linear(self.d_model, self.llm_dim),
        # )
        self.param_head = nn.Linear(self.flatten_dim, self.out_days * 8)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)  # [B, C, N, P]
        x = self.patch_embedding(x)  # [B, C, N, D]
        N = x.shape[2]
        assert N == self.num_patches, "Number of patches does not match"
        x = x.reshape(B, C * N, -1)  # [B, C * N, D]
        for layer in self.encoder:
            x = layer(x, self.in_channels, N)  # [B, C * N, D]
        x = x.reshape(B, C, N, -1)
        return x

    def forecasting(self, x: torch.Tensor) -> dict:
        x = self.embedding(x)  # [B, C, N, D]
        B, C, N, D = x.shape
        x = x.reshape(B, C, -1)  # [B, C, N*D]
        x = F.gelu(x)
        x_agg = x.mean(dim=1)  # [B, N*D]

        raw_params = self.param_head(x_agg).view(B, self.out_days, 8)
        mu_on = raw_params[..., 0]  # [B, out_days]
        log_sigma_on = raw_params[..., 1]
        mu_c = raw_params[..., 2]
        log_sigma_c = raw_params[..., 3]
        mu_v = raw_params[..., 4]
        log_sigma_v = raw_params[..., 5]
        q_high = raw_params[..., 6]
        q_low = raw_params[..., 7]

        sigma_on = torch.exp(log_sigma_on)
        sigma_c = torch.exp(log_sigma_c)
        sigma_v = torch.exp(log_sigma_v)

        mean_returns = torch.stack([mu_on, mu_c, mu_v, q_high, q_low], dim=-1)
        return {
            "mu_on": mu_on,
            "sigma_on": sigma_on,
            "mu_c": mu_c,
            "sigma_c": sigma_c,
            "mu_v": mu_v,
            "sigma_v": sigma_v,
            "q_high": q_high,
            "q_low": q_low,
            "mean_returns": mean_returns,  # [B, out_days, 5]
        }

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Draw samples of future return series."""
        params = self.forecasting(x)
        mu_on = params["mu_on"].unsqueeze(1).expand(-1, n_samples, -1)
        sigma_on = params["sigma_on"].unsqueeze(1).expand_as(mu_on)
        mu_c = params["mu_c"].unsqueeze(1).expand(-1, n_samples, -1)
        sigma_c = params["sigma_c"].unsqueeze(1).expand_as(mu_c)
        mu_v = params["mu_v"].unsqueeze(1).expand(-1, n_samples, -1)
        sigma_v = params["sigma_v"].unsqueeze(1).expand_as(mu_v)
        q_high = params["q_high"].unsqueeze(1).expand(-1, n_samples, -1)
        q_low = params["q_low"].unsqueeze(1).expand(-1, n_samples, -1)

        r_on = mu_on + sigma_on * torch.randn_like(mu_on)
        r_c = mu_c + sigma_c * torch.randn_like(mu_c)
        r_v = mu_v + sigma_v * torch.randn_like(mu_v)

        sampled = torch.stack([r_on, r_c, r_v, q_high, q_low], dim=-1)  # [B, n_samples, H, 5]
        return sampled

    # def get_soft_tokens(self, x):
    #     """
    #     [B,C,N,D] => [B,C,N*D] => [B, num_soft_tokens, llm_dim]
    #     """
    #     B, C, N, D = x.shape
    #     x = self.embedding(x)  # [B, C, N, D]
    #     x = x.reshape(B, C, -1)  # [B, C, N*D]
    #     if self.num_soft_tokens != self.in_channels:
    #         x = x.permute(0, 2, 1)  # [B, N*D, C]
    #         x = self.token_proj(x)  # [B, N*D, num_soft_tokens]
    #         x = x.permute(0, 2, 1)  # [B, num_soft_tokens, N*D]
    #     soft_tokens = self.soft_token_proj(x)  # [B, num_soft_tokens, llm_dim]
    #     return soft_tokens  # [B, num_soft_tokens, llm_dim]