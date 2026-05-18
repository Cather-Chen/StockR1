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
        self.answer_open_id = self.tokenizer.convert_tokens_to_ids("<answer>")
        self.answer_close_id = self.tokenizer.convert_tokens_to_ids("</answer>")
        self.forecast_ts_open_id = self.tokenizer.convert_tokens_to_ids("<forecast_ts>")
        self.forecast_ts_close_id = self.tokenizer.convert_tokens_to_ids("</forecast_ts>")
        self.forecast_router_id = self.tokenizer.convert_tokens_to_ids("<forecast_router>")
        self.forecast_hint_open_id = self.tokenizer.convert_tokens_to_ids("<forecast_hint>")
        self.forecast_hint_close_id = self.tokenizer.convert_tokens_to_ids("</forecast_hint>")
        self.assistant_prefix_ids = self.tokenizer(
            "<|im_start|>assistant",
            add_special_tokens=False,
        )["input_ids"]

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
                # self.qwen.print_trainable_parameters()
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
        self.hint_conditioner = ForecastHintConditioner(output_dim=self.out_days * 8)
        # Keep hint residual bounded at init to avoid destabilizing pretrained TS param_head.
        self.hint_residual_scale = 0.05
        for p in self.ts_encoder.parameters():
            p.requires_grad = True
        for p in self.hint_conditioner.parameters():
            p.requires_grad = True
            
    def llm_forecasting(self, x, hint_batch=None, sample=False):
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
        if hint_batch is not None:
            cond_out = self.hint_conditioner(
                numerical=hint_batch["numerical"],
                categorical_ids=hint_batch["categorical_ids"],
                price_relative=hint_batch["price_relative"],
            )
            hint_params = cond_out["vector"].view(B, self.out_days, 8)
            hint_params = self.hint_residual_scale * torch.tanh(hint_params)
            raw_params = raw_params + hint_params

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

    @staticmethod
    def _find_subseq(seq: torch.Tensor, subseq: List[int]) -> int:
        if len(subseq) == 0 or seq.numel() < len(subseq):
            return -1
        for i in range(seq.numel() - len(subseq) + 1):
            if seq[i:i + len(subseq)].tolist() == subseq:
                return i
        return -1

    def _qwen_forward_trimmed_loss(self, **kwargs):
        labels = kwargs.pop("labels")
        # Qwen3 can restrict LM-head logits with logits_to_keep.  Compute the
        # same causal-LM loss, but only over the suffix containing supervised
        # targets, avoiding a full [B, S, vocab] logits tensor for long prompts.
        shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
        active = shift_labels.ne(-100)
        if active.any():
            first_active = int(active.nonzero(as_tuple=False)[:, 1].min().item())
            logits_to_keep = labels.shape[1] - first_active
            shift_labels = shift_labels[:, first_active:].contiguous()
        else:
            logits_to_keep = 1
            shift_labels = shift_labels[:, -1:].contiguous()

        return self.qwen(
            labels=labels,
            logits_to_keep=logits_to_keep,
            shift_labels=shift_labels,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )

    def pure_forecast(self, batch):
        ts_inputs = batch["input_features"].to(dtype=torch.bfloat16)  # [B, L_ts, C]
        hint_batch = batch.get("hint_feature_batch", None)
        if hint_batch is not None:
            hint_batch = {
                "numerical": hint_batch["numerical"].to(ts_inputs.device, dtype=torch.float32),
                "categorical_ids": hint_batch["categorical_ids"].to(ts_inputs.device, dtype=torch.long),
                "price_relative": hint_batch["price_relative"].to(ts_inputs.device, dtype=torch.float32),
            }

        pred_params = self.llm_forecasting(ts_inputs, hint_batch=hint_batch)
        ts_loss_dict = compute_return_loss(pred_params, batch["output_features"])
        return pred_params, ts_loss_dict


    def forward(self, batch):
        ts_only = bool(batch.get("ts_only", False))
        pred_params, ts_loss_dict = self.pure_forecast(batch)

        if ts_only:
            device = batch["input_features"].device
            sft_loss = torch.zeros((), device=device, dtype=ts_loss_dict["loss"].dtype)
            total_loss = ts_loss_dict["loss"]
            return {
                "loss": total_loss,
                "sft_loss": sft_loss,
                "ts_loss": ts_loss_dict["loss"],
                "forecasting": pred_params["mean_returns"],
            }

        ts_inputs = batch["input_features"].to(dtype=torch.bfloat16)  # [B, L_ts, C]
        B, _, _ = ts_inputs.shape
        if "text_loss_mask" in batch:
            labels = batch["text_ids"].clone()
            labels[batch["text_loss_mask"] == 0] = -100
        else:
            labels = batch["text_ids"].clone()
            labels[:] = -100  

            for i in range(B):
                pos = self._find_subseq(batch["text_ids"][i], self.assistant_prefix_ids)
                if pos != -1:
                    start_idx = pos + len(self.assistant_prefix_ids)
                    labels[i, start_idx:] = batch["text_ids"][i, start_idx:]
                else:
                    labels[i, :] = -100
                    print(f"Assistant prefix not found in text_ids for sample {i}")
            
        if not self.ts_soft_tokens:
            outputs = self._qwen_forward_trimmed_loss(
                input_ids=batch["text_ids"],
                attention_mask=batch["text_attention_mask"],
                labels=labels,
                use_cache=False,
            )
        else:
            inputs_embeddings = self.qwen.get_input_embeddings()(batch["text_ids"])  # [B, S, D_llm]
            ## get soft tokens from ts encoder ###################
            ts_embeddings = self.ts_encoder.embedding(ts_inputs)   # [B, C, N, D]
            B, C, N, D = ts_embeddings.shape
            ts_embeddings = ts_embeddings.reshape(B, C, -1)        # [B, C, N*D]
            ts_embeddings = F.gelu(ts_embeddings)
            ts_embeddings = self.ts_soft_token_layer(ts_embeddings)  # [B, C, D_llm]

            llm_dtype = inputs_embeddings.dtype
            ts_embeddings = ts_embeddings.to(llm_dtype)
            #######################################################
            concat_embeddings = torch.cat([ts_embeddings, inputs_embeddings], dim=1)  # [B, C+S, D_llm]

            prefix_mask = torch.ones(B, C, dtype=torch.long, device=ts_embeddings.device)
            text_mask   = batch["text_attention_mask"].to(torch.long)
            attention_mask = torch.cat([prefix_mask, text_mask], dim=1)               # [B, C+S]

            prefix_labels = torch.full((B, C), -100, dtype=torch.long, device=labels.device)
            labels = torch.cat([prefix_labels, labels], dim=1)                    # [B, C+S]

            outputs = self._qwen_forward_trimmed_loss(
                inputs_embeds=concat_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
        sft_loss = outputs.loss
        total_loss = ts_loss_dict["loss"] +  sft_loss

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

        
