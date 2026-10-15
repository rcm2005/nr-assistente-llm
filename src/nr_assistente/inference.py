from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .records import QARecord, messages_for


class Generator:
    def __init__(self, model_or_adapter: str, load_in_4bit: bool = False):
        candidate = Path(model_or_adapter)
        adapter_path = candidate if (candidate / "adapter_config.json").exists() else None
        base_model = PeftConfig.from_pretrained(str(adapter_path)).base_model_name_or_path if adapter_path else model_or_adapter
        kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        self.model = PeftModel.from_pretrained(model, str(adapter_path)) if adapter_path else model
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def generate(
        self,
        record: QARecord,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens: int = 192,
    ) -> tuple[str, dict[str, float | int]]:
        prompt = self.tokenizer.apply_chat_template(
            messages_for(record, include_answer=False), tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        generated = self.model.generate(
            **encoded,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        peak = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
        return text, {
            "generated_tokens": int(new_tokens.numel()),
            "latency_seconds": elapsed,
            "tokens_per_second": float(new_tokens.numel() / elapsed) if elapsed else 0.0,
            "peak_allocated_vram_mib": peak,
        }
