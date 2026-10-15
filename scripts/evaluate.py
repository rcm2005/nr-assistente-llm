from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from src.nr_assistente.data import read_jsonl, write_jsonl
from src.nr_assistente.inference import Generator
from src.nr_assistente.metrics import citation_scores, refusal_scores, rouge_l
from src.nr_assistente.records import normalize_record
from src.nr_assistente.telemetry import NvidiaMonitor, summarize_gpu_csv


def parse_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NOME=MODELO_OU_DIRETORIO")
    name, location = value.split("=", 1)
    if not name.strip() or not location.strip():
        raise argparse.ArgumentTypeError("Nome e modelo/diretorio nao podem estar vazios")
    return name.strip(), location.strip()


def evaluate_one(
    name: str,
    location: str,
    records: list,
    temperatures: list[float],
    max_new_tokens: int,
    output_dir: Path,
) -> list[dict]:
    generator = Generator(location)
    rows: list[dict] = []
    temperature_label = "-".join(str(value).replace(".", "p") for value in temperatures)
    telemetry_path = output_dir / f"nvidia-smi-{name}-t{temperature_label}.csv"
    with NvidiaMonitor(telemetry_path):
        for temperature in temperatures:
            for record in records:
                prediction, performance = generator.generate(
                    record, temperature=temperature, max_new_tokens=max_new_tokens
                )
                rows.append(
                    {
                        "model": name,
                        "model_location": location,
                        "temperature": temperature,
                        "id": record.identifier,
                        "question": record.question,
                        "reference_answer": record.answer,
                        "reference_citations": list(record.citations),
                        "is_refusal": record.is_refusal,
                        "prediction": prediction,
                        **performance,
                    }
                )
    gpu = summarize_gpu_csv(telemetry_path)
    for row in rows:
        row["telemetry_peak_vram_mib"] = gpu.get("peak_vram_mib")
    del generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["temperature"]), []).append(row)
    summaries = []
    for (model, temperature), group in sorted(groups.items()):
        predictions = [row["prediction"] for row in group]
        summary = {
            "model": model,
            "temperature": temperature,
            "examples": len(group),
            **citation_scores(predictions, [row["reference_citations"] for row in group]),
            **refusal_scores(predictions, [row["is_refusal"] for row in group]),
            "rouge_l": rouge_l(predictions, [row["reference_answer"] for row in group]),
            "tokens_per_second": sum(row["tokens_per_second"] for row in group) / len(group),
            "peak_allocated_vram_mib": max(row["peak_allocated_vram_mib"] for row in group),
            "telemetry_peak_vram_mib": max(
                (row["telemetry_peak_vram_mib"] or 0.0) for row in group
            ),
        }
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia modelos no teste retido e grava resultados auditaveis")
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--temperature", action="append", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output-dir", type=Path, default=Path("results/eval"))
    parser.add_argument(
        "--append", action="store_true",
        help="preserva predicoes existentes e acrescenta apenas casos modelo/temperatura novos",
    )
    args = parser.parse_args()

    records = [normalize_record(row, index) for index, row in enumerate(read_jsonl(args.test_file), 1)]
    if not records:
        raise ValueError("Arquivo de teste vazio")
    temperatures = args.temperature or [0.2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    rows: list[dict] = list(read_jsonl(predictions_path)) if args.append and predictions_path.exists() else []
    existing = {(row["model"], float(row["temperature"]), str(row["id"])) for row in rows}
    for name, location in args.model:
        requested = {(name, float(temperature), str(record.identifier))
                     for temperature in temperatures for record in records}
        overlap = existing & requested
        if overlap:
            raise ValueError(f"--append recusou {len(overlap)} predicoes duplicadas")
        new_rows = evaluate_one(
            name, location, records, temperatures, args.max_new_tokens, args.output_dir
        )
        rows.extend(new_rows)
        existing.update(requested)
    write_jsonl(predictions_path, rows)
    summaries = aggregate(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
