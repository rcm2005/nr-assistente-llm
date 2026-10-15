from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from src.nr_assistente.data import read_jsonl
from src.nr_assistente.records import messages_for, normalize_record
from src.nr_assistente.telemetry import NvidiaMonitor, summarize_gpu_csv


class RuntimeLimitCallback(TrainerCallback):
    def __init__(self, minutes: float):
        self.limit_seconds = minutes * 60
        self.started = time.monotonic()
        self.limit_reached = False

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if time.monotonic() - self.started >= self.limit_seconds:
            self.limit_reached = True
            control.should_training_stop = True
            control.should_save = True
        return control


def load_config(path: Path, run_id: str) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    selected = next((run for run in document["runs"] if run["id"] == run_id), None)
    if selected is None:
        available = ", ".join(run["id"] for run in document["runs"])
        raise ValueError(f"Run desconhecido: {run_id}. Disponiveis: {available}")
    return {**document.get("defaults", {}), **selected}


def quantization_config(mode: str) -> tuple[Any | None, bool]:
    if mode not in {"auto", "4bit"}:
        return None, False
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ), True
    except (ImportError, RuntimeError):
        if mode == "4bit":
            raise
        return None, False


def run_training(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Treinamento fast-track exige CUDA; nenhum dispositivo CUDA foi detectado")
    set_seed(int(config["seed"]))
    output_dir = output_root / config["id"]
    if (output_dir / "adapter_config.json").exists():
        raise FileExistsError(f"Adapter ja existe, recusando sobrescrever: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(Path(config["train_file"])))
    records = [normalize_record(row, index) for index, row in enumerate(rows, 1)]
    if not records:
        raise ValueError("Dataset de treino vazio")

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [
        tokenizer.apply_chat_template(messages_for(record), tokenize=False, add_generation_prompt=False)
        for record in records
    ]
    maximum = int(config["max_seq_length"])

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=maximum)

    dataset = Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])
    quantization_mode = str(config.get("quantization", "none"))
    qconfig, used_4bit = quantization_config(quantization_mode)
    quantization_fallback: str | None = None
    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    if qconfig is not None:
        model_kwargs.update(quantization_config=qconfig, device_map="auto")
    try:
        model = AutoModelForCausalLM.from_pretrained(config["model"], **model_kwargs)
    except Exception as exc:
        if quantization_mode != "auto" or qconfig is None:
            raise
        quantization_fallback = f"{type(exc).__name__}: {exc}"
        used_4bit = False
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            config["model"],
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    model.config.use_cache = False
    if used_4bit:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=int(config["rank"]),
        lora_alpha=int(config["rank"]) * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    bf16 = torch.cuda.is_bf16_supported()
    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=float(config["epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config["warmup_ratio"]),
        weight_decay=float(config["weight_decay"]),
        logging_steps=int(config["logging_steps"]),
        save_strategy=str(config["save_strategy"]),
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        report_to="none",
        seed=int(config["seed"]),
    )
    runtime_callback = RuntimeLimitCallback(float(config["max_runtime_minutes"]))
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[runtime_callback],
    )
    telemetry_path = output_dir / "nvidia-smi.csv"
    started = time.time()
    with NvidiaMonitor(telemetry_path):
        result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "status": "completed",
        "run_id": config["id"],
        "config": config,
        "examples": len(records),
        "used_4bit": used_4bit,
        "quantization_fallback": quantization_fallback,
        "started_unix": started,
        "finished_unix": time.time(),
        "train_metrics": result.metrics,
        "global_steps": trainer.state.global_step,
        "epochs_completed": trainer.state.epoch,
        "runtime_limit_reached": runtime_callback.limit_reached,
        "gpu": summarize_gpu_csv(telemetry_path),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
    }
    (output_dir / "training_result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina um adapter LoRA do NR-Assistente")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--run", required=True, help="ID de um run da matriz")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    metadata = run_training(load_config(args.config, args.run), args.output_root)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
