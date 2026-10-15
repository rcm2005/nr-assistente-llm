from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.nr_assistente.data import read_jsonl, write_jsonl

DEFAULT_ACCOUNTS = ("g2", "g4", "g6", "g7", "g8")
TRANSIENT_RE = re.compile(r"quota|resource.?exhausted|rate.?limit|429|temporar|overload", re.I)
SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "judge_id": {"type": "string"},
                    "correcao": {"type": "integer", "minimum": 0, "maximum": 2},
                    "fundamentacao": {"type": "integer", "minimum": 0, "maximum": 2},
                    "utilidade": {"type": "integer", "minimum": 0, "maximum": 2},
                    "justificativa": {"type": "string"},
                },
                "required": [
                    "judge_id", "correcao", "fundamentacao", "utilidade", "justificativa"
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def prompt_for(batch: list[dict]) -> str:
    compact = [
        {
            "judge_id": row["judge_id"],
            "pergunta": row["question"],
            "resposta_referencia": row["reference_answer"],
            "citacoes_referencia": row["reference_citations"],
            "resposta_avaliada": row["prediction"],
        }
        for row in batch
    ]
    return (
        "Avalie cada resposta usando somente a respectiva referencia. Retorne exatamente um score "
        "por judge_id e na mesma ordem. Use inteiros de 0 a 2 para correcao, fundamentacao e "
        "utilidade. Penalize afirmacoes nao sustentadas e citacoes erradas. Justifique brevemente "
        "em portugues. Retorne somente o objeto JSON exigido.\nCASOS:\n"
        + json.dumps(compact, ensure_ascii=False)
    )


def decode(stdout: str) -> list[dict]:
    payload = json.loads(stdout)
    candidates = [payload]
    if isinstance(payload, dict):
        candidates.extend(payload.get(key) for key in ("result", "response", "output", "text"))
    for candidate in candidates:
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if candidate.startswith("```"):
                candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
            candidate = json.loads(candidate)
        if isinstance(candidate, dict) and isinstance(candidate.get("scores"), list):
            return candidate["scores"]
    raise ValueError("saida do agy nao contem scores")


def judge_batch(
    batch_id: int,
    batch: list[dict],
    accounts: tuple[str, ...],
    model: str,
    timeout: int,
) -> list[dict]:
    errors: list[str] = []
    for attempt in range(len(accounts)):
        account = accounts[(batch_id + attempt) % len(accounts)]
        command = [
            "agy-acct", "run", account, "-p", prompt_for(batch), "--model", model,
            "--output-format", "json", "--json-schema", json.dumps(SCORE_SCHEMA),
            "--print-timeout", f"{timeout}s", "--mode", "plan", "--disable-slash-commands",
        ]
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=timeout + 20)
        except subprocess.TimeoutExpired:
            errors.append(f"{account}: timeout")
            continue
        if result.returncode == 0:
            scores = decode(result.stdout)
            expected = [row["judge_id"] for row in batch]
            if [row.get("judge_id") for row in scores] != expected:
                errors.append(f"{account}: judge_ids divergentes")
                continue
            return [
                {**source, "judge_model": model, "judge_scores": score}
                for source, score in zip(batch, scores)
            ]
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")[-400:]
        errors.append(f"{account}: exit {result.returncode} ({detail})")
        if not TRANSIENT_RE.search(detail):
            break
    raise RuntimeError(f"lote de julgamento {batch_id} falhou: {'; '.join(errors)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Julga 40 perguntas com Gemini Flash via agy")
    parser.add_argument("--predictions", type=Path, default=Path("results/eval/predictions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/eval/gemini_judge.jsonl"))
    parser.add_argument("--limit", type=int, default=40, help="quantidade de perguntas distintas")
    parser.add_argument("--batch-size", type=int, default=15, choices=range(5, 21))
    parser.add_argument("--parallel", type=int, default=2, choices=(1, 2))
    parser.add_argument("--model", default="gemini-3.8-flash-medium")
    parser.add_argument("--accounts", nargs="+", default=list(DEFAULT_ACCOUNTS))
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    all_rows = list(read_jsonl(args.predictions))
    selected_ids: list[str] = []
    for row in all_rows:
        identifier = str(row["id"])
        if identifier not in selected_ids:
            selected_ids.append(identifier)
        if len(selected_ids) == args.limit:
            break
    selected = set(selected_ids)
    rows = [
        dict(row, judge_id=f"j{index:05d}")
        for index, row in enumerate(all_rows)
        if str(row["id"]) in selected
    ]
    if len(selected) != args.limit:
        raise ValueError(f"somente {len(selected)} perguntas distintas disponiveis")
    batches = [rows[start : start + args.batch_size] for start in range(0, len(rows), args.batch_size)]
    judged: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(
                judge_batch, index, batch, tuple(args.accounts), args.model, args.timeout
            ): index
            for index, batch in enumerate(batches)
        }
        results: dict[int, list[dict]] = {}
        for future in as_completed(futures):
            batch_id = futures[future]
            results[batch_id] = future.result()
            print(f"lote de julgamento {batch_id + 1}/{len(batches)} concluido", flush=True)
    for batch_id in range(len(batches)):
        judged.extend(results[batch_id])
    write_jsonl(args.output, judged)

    dimensions = ("correcao", "fundamentacao", "utilidade")
    groups: dict[tuple[str, float], list[dict]] = {}
    for row in judged:
        groups.setdefault((row["model"], row["temperature"]), []).append(row)
    summary = [
        {
            "judge_model": args.model,
            "evaluated_model": model,
            "temperature": temperature,
            "examples": len(group),
            **{
                dimension: sum(row["judge_scores"][dimension] for row in group) / len(group)
                for dimension in dimensions
            },
        }
        for (model, temperature), group in sorted(groups.items())
    ]
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
