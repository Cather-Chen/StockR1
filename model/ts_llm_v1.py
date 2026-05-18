from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model.multichannel_ts_enc import MultichannelTSEncoder_rm
import torch.nn.functional as F
from model.loss import compute_return_loss
from utils.hint_extract import ForecastHintConditioner

from peft import LoraConfig, get_peft_model, TaskType, PeftModel

@dataclass
class TS2LLM_v2Config:
    qwen_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer: AutoTokenizer = None
    llm_dim: int = 4096     
    out_days: int = 10
    ts_encoder_checkpoint: str = "" 
    freeze_qwen: bool = False
    # LoRA settings
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # e.g., ["q_proj", "k_proj", "v_proj", "o_proj"]
    is_eval: bool = False,
    ts_soft_tokens: bool = False

class TS2QwenModel_v2(nn.Module):
    def __init__(self, cfg: TS2LLM_v2Config):
        super().__init__()
        self.cfg = cfg

        # 1) load tokenizer & Qwen
        self.tokenizer = cfg.tokenizer
        self.analysis_id = self.tokenizer.convert_tokens_to_ids("<analysis>")
        self.answer_open_id = self.tokenizer.convert_tokens_to_ids("<answer>")
        self.answer_close_id = self.tokenizer.convert_tokens_to_ids("</answer>")
        self.thinking_open_id = self.tokenizer.convert_tokens_to_ids("<think>")
        self.thinking_close_id = self.tokenizer.convert_tokens_to_ids("</think>")

        self.qwen = AutoModelForCausalLM.from_pretrained(
            cfg.qwen_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True
        )
        self.qwen.config.recompute_loss = True
        # Ensure JSON-serializable values in HF config (avoid torch.dtype objects)
        self.qwen.config.torch_dtype = "bfloat16"
        self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if cfg.lora_enable:
            if cfg.is_eval:
                self.qwen = PeftModel.from_pretrained(self.qwen, cfg.qwen_model_name)
            else:
                target_modules = cfg.lora_target_modules
                # reasonable defaults for Qwen style attention/MLP names
                if target_modules is None:
                    # Try common names used in HF Qwen architectures
                    target_modules = [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ]
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=cfg.lora_r,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=target_modules,
                    bias="none",
                )
                self.qwen = get_peft_model(self.qwen, lora_cfg)
                self.qwen.print_trainable_parameters()
        else:
            if cfg.freeze_qwen:
                for p in self.qwen.parameters():
                    p.requires_grad = False
            else:
                for name, p in self.qwen.named_parameters():
                    p.requires_grad = True

        self.qwen.get_input_embeddings().weight.requires_grad = True
        self.qwen.get_output_embeddings().weight.requires_grad = True
        
        self.qwen.to(dtype=torch.bfloat16) 
        self.llm_dim = getattr(self.qwen.config, "hidden_size", cfg.llm_dim)
        self.out_days = cfg.out_days
        self.qwen.eval()

        self._load_ts_encoder(cfg.ts_encoder_checkpoint + "/best_val.pt")
        self.llm_residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.llm_residual_head = nn.Linear(self.llm_dim, self.out_days * 8)
        nn.init.zeros_(self.llm_residual_head.weight)
        nn.init.zeros_(self.llm_residual_head.bias)
        self.ts_soft_tokens = cfg.ts_soft_tokens
        if self.ts_soft_tokens:
            self.ts_soft_token_layer = nn.Linear(self.ts_encoder.flatten_dim, self.llm_dim)
            self.ts_soft_token_layer.to(dtype=torch.bfloat16)
            self.ts_soft_token_layer.requires_grad = True
        self.to(dtype=torch.bfloat16) 

    
    def _load_ts_encoder(self, checkpoint_path):
        # Load checkpoint without specifying device - let it load on the current device
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        saved_args = checkpoint["args"]  
        model = MultichannelTSEncoder_rm(
            input_len=saved_args["input_len"],
            in_channels=saved_args["ts_in_channels"],
            d_model=saved_args["ts_d_model"],
            nhead=saved_args["ts_heads"],
            num_layers=saved_args["ts_num_layers"],
            patch_len=saved_args["ts_patch_len"],
            dropout=saved_args["ts_dropout"],
            out_days=saved_args["out_days"],
        )
    
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        if unexpected_keys:
            print(f"TS encoder load: skipped unexpected keys: {unexpected_keys}")
        if missing_keys:
            print(f"TS encoder load: missing keys: {missing_keys}")
        self.ts_encoder = model
        for p in self.ts_encoder.parameters():
            p.requires_grad = True
            
    def llm_forecasting(self, x, llm_hidden=None):
        """
        x: [B, L_ts, C]
        llm_hidden: [B, D_llm]
        """
        x = self.ts_encoder.embedding(x)
        B, C, N, D = x.shape
        x = x.reshape(B, C, -1)               # [B, C, N*D]
        x = F.gelu(x)
        x_agg = x.mean(dim=1)                 # [B, N*D]   Aggregate to sample level

        base_params = self.ts_encoder.param_head(x_agg).view(B, self.out_days, 8)
        raw_params = base_params
        if llm_hidden is not None:
            llm_hidden = llm_hidden.to(device=x_agg.device, dtype=x_agg.dtype)
            llm_residual = self.llm_residual_head(llm_hidden.detach()).view(B, self.out_days, 8)
            raw_params = raw_params + self.llm_residual_scale.to(dtype=raw_params.dtype) * torch.tanh(llm_residual)

        mu_on = raw_params[..., 0]
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
        uncertainty = torch.sqrt(sigma_on**2 + sigma_c**2 + sigma_v**2).mean(dim=-1) / (3.0 ** 0.5) # [B]
        uncertainty = torch.clamp(uncertainty, 0.0, 1.0)
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
            "mean_returns": mean_returns,
            "uncertainty": uncertainty
        }


    def forward(self, batch):
        ts_inputs = batch["input_features"]  # [B, L_ts, C]
        ts_inputs = ts_inputs.to(dtype=torch.bfloat16)
        B, _, _ = ts_inputs.shape
        labels = batch["text_ids"].clone()
        labels[:] = -100  

        for i in range(B):
            start_idx = None
            for token_id in (self.thinking_open_id, self.analysis_id, self.answer_open_id):
                if token_id is None or token_id < 0:
                    continue
                pos = (batch["text_ids"][i] == token_id).nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    start_idx = pos[0].item()
                    break
            if start_idx is None:
                valid_pos = (batch["text_attention_mask"][i] > 0).nonzero(as_tuple=True)[0]
                start_idx = valid_pos[0].item() if len(valid_pos) > 0 else 0
            labels[i, start_idx:] = batch["text_ids"][i, start_idx:]
            
        if not self.ts_soft_tokens:
            outputs = self.qwen(
                input_ids=batch["text_ids"],
                attention_mask=batch["text_attention_mask"],
                labels=labels,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            inputs_embeddings = self.qwen.get_input_embeddings()(batch["text_ids"])  # [B, S, D_llm]

            ts_embeddings = self.ts_encoder.embedding(ts_inputs)   # [B, C, N, D]
            B, C, N, D = ts_embeddings.shape
            ts_embeddings = ts_embeddings.reshape(B, C, -1)        # [B, C, N*D]
            ts_embeddings = F.gelu(ts_embeddings)
            ts_embeddings = self.ts_soft_token_layer(ts_embeddings)  # [B, C, D_llm]

            llm_dtype = inputs_embeddings.dtype
            ts_embeddings = ts_embeddings.to(llm_dtype)

            concat_embeddings = torch.cat([ts_embeddings, inputs_embeddings], dim=1)  # [B, C+S, D_llm]

            prefix_mask = torch.ones(B, C, dtype=torch.long, device=ts_embeddings.device)
            text_mask   = batch["text_attention_mask"].to(torch.long)
            attention_mask = torch.cat([prefix_mask, text_mask], dim=1)               # [B, C+S]

            prefix_labels = torch.full((B, C), -100, dtype=torch.long, device=labels.device)
            labels = torch.cat([prefix_labels, labels], dim=1)                    # [B, C+S]

            outputs = self.qwen(
                inputs_embeds=concat_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            
        last_hidden = outputs.hidden_states[-1]  # [B, S, D]

        analysis_hidden = []
        for i in range(B):
            analysis_pos = (batch["text_ids"][i] == self.analysis_id).nonzero(as_tuple=True)[0]
            if len(analysis_pos) > 0:
                analysis_hidden.append(last_hidden[i, analysis_pos[-1], :])  # [D_llm]
            else:
                analysis_hidden.append(last_hidden[i, -1, :])  # [D_llm]
        analysis_hidden = torch.stack(analysis_hidden, dim=0)  # [B, D_llm]

        pred_params = self.llm_forecasting(ts_inputs, analysis_hidden)  # [B, out_days, 5]
        ts_loss_dict = compute_return_loss(pred_params, batch["output_features"])
        total_loss = ts_loss_dict["loss"] + outputs.loss
        sft_loss = outputs.loss
        return {
            "loss": total_loss,
            "sft_loss": sft_loss,
            "ts_loss": ts_loss_dict["loss"],
            # "ts_loss_on": ts_loss_dict["loss_on"],
            # "ts_loss_c": ts_loss_dict["loss_c"],
            # "ts_loss_v": ts_loss_dict["loss_v"],
            # "ts_loss_h": ts_loss_dict["loss_h"],
            # "ts_loss_l": ts_loss_dict["loss_l"],
            "forecasting": pred_params["mean_returns"],
        }

        
        
