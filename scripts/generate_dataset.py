#!/usr/bin/env python3
"""Generate grounded NR Q&A batches through authenticated agy accounts.

Every accepted row is subsequently checked by validate_dataset.py. This script
never synthesizes fallback rows: a failed API batch remains failed and can be
resumed in a later invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nr_assistente.data import read_jsonl

DEFAULT_ACCOUNTS = ("g2", "g4", "g6", "g7", "g8")
TRANSIENT_RE = re.compile(r"quota|resource.?exhausted|rate.?limit|429|temporar|overload", re.I)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        temporary = Path(tmp.name)
    os.replace(temporary, path)

SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "citation": {"type": ["string", "null"]},
                    "expected_refusal": {"type": "boolean"},
                },
                "required": ["source_id", "question", "answer", "citation", "expected_refusal"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["pairs"],
    "additionalProperties": False,
}


def select_items(rows: list[dict], target: int) -> list[dict]:
    """Deterministically interleave NRs so small documents remain represented."""
    by_nr: dict[int, list[dict]] = {}
    for row in rows:
        by_nr.setdefault(int(row["nr"]), []).append(row)
    for values in by_nr.values():
        values.sort(key=lambda row: row["id"])
    selected: list[dict] = []
    while len(selected) < target and any(by_nr.values()):
        for nr in sorted(by_nr):
            if by_nr[nr] and len(selected) < target:
                selected.append(by_nr[nr].pop(0))
    return selected


def make_prompt(batch: list[dict], first_index: int) -> str:
    sources = []
    for offset, row in enumerate(batch):
        is_trap = (first_index + offset) % 5 == 4
        sources.append(
            {
                "source_id": row["id"],
                "citation": row["citation"],
                "make_trap": is_trap,
                "official_excerpt": row["text"][:3000],
            }
        )
    return (
        "Você cria dados acadêmicos em pt-BR a partir de trechos oficiais de NRs. "
        "Retorne SOMENTE o objeto JSON exigido, com a chave pairs contendo exatamente um objeto por fonte "
        "e na mesma ordem. "
        "Para make_trap=false, formule pergunta útil respondível apenas pelo trecho; dê resposta curta e "
        "factual terminando com a citação fornecida exatamente como escrita. Não invente requisitos. "
        "Para make_trap=true, formule pergunta plausível cuja resposta NÃO esteja no trecho e responda "
        "explicitamente que não há base suficiente no item fornecido; citation deve ser null. "
        "Copie source_id sem alterações. Não use conhecimento externo.\nFONTES:\n"
        + json.dumps(sources, ensure_ascii=False)
    )


def decode_agy_output(stdout: str) -> list[dict]:
    payload = json.loads(stdout)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("pairs"), list):
            return payload["pairs"]
        for key in ("result", "response", "output", "text"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                value = value.strip()
                if value.startswith("```"):
                    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I)
                decoded = json.loads(value)
                if isinstance(decoded, dict) and isinstance(decoded.get("pairs"), list):
                    return decoded["pairs"]
                if isinstance(decoded, list):
                    return decoded
    raise ValueError("saída do agy não contém um array JSON")


def run_batch(
    batch_id: int,
    batch: list[dict],
    first_index: int,
    output_dir: Path,
    accounts: tuple[str, ...],
    model: str,
    timeout: int,
) -> tuple[int, str]:
    destination = output_dir / f"batch-{batch_id:04d}.json"
    audit_path = output_dir / f"batch-{batch_id:04d}.meta.json"
    if destination.exists() and audit_path.exists():
        return batch_id, "já existente"
    if destination.exists() != audit_path.exists():
        raise RuntimeError(f"lote {batch_id} tem dados/metadados incompletos; mova ambos e retome")
    prompt = make_prompt(batch, first_index)
    errors: list[str] = []
    audit = {
        "batch_id": batch_id,
        "model": model,
        "started_at": now_utc(),
        "source_ids": [row["id"] for row in batch],
        "attempts": [],
        "status": "running",
    }
    start_account = batch_id % len(accounts)
    for attempt in range(len(accounts)):
        account = accounts[(start_account + attempt) % len(accounts)]
        attempt_record = {"account": account, "started_at": now_utc()}
        command = [
            "agy-acct", "run", account, "-p", prompt,
            "--model", model, "--output-format", "json",
            "--json-schema", json.dumps(SCHEMA, ensure_ascii=False),
            "--print-timeout", f"{timeout}s", "--mode", "plan", "--disable-slash-commands",
        ]
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=timeout + 20)
        except subprocess.TimeoutExpired:
            errors.append(f"{account}: timeout")
            attempt_record.update({"ended_at": now_utc(), "status": "timeout"})
            audit["attempts"].append(attempt_record)
            continue
        attempt_record.update(
            {"ended_at": now_utc(), "returncode": result.returncode, "status": "returned"}
        )
        if result.returncode == 0:
            try:
                decoded = decode_agy_output(result.stdout)
                if len(decoded) != len(batch):
                    raise ValueError(f"esperados {len(batch)}, recebidos {len(decoded)}")
                attempt_record["status"] = "success"
                audit["attempts"].append(attempt_record)
                atomic_json(destination, decoded)
                audit.update(
                    {
                        "ended_at": now_utc(),
                        "status": "success",
                        "successful_account": account,
                        "row_count": len(decoded),
                        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    }
                )
                atomic_json(audit_path, audit)
                return batch_id, f"ok via {account}"
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{account}: JSON inválido ({exc})")
                attempt_record.update({"status": "invalid_json", "detail": str(exc)[:400]})
                audit["attempts"].append(attempt_record)
        else:
            detail = (result.stderr or result.stdout).strip().replace("\n", " ")[-400:]
            errors.append(f"{account}: exit {result.returncode} ({detail})")
            attempt_record.update(
                {"status": "failed", "transient": bool(TRANSIENT_RE.search(detail)), "detail": detail}
            )
            audit["attempts"].append(attempt_record)
            if not TRANSIENT_RE.search(detail):
                break
        time.sleep(min(2**attempt, 8))
    audit.update({"ended_at": now_utc(), "status": "failed", "errors": errors})
    atomic_json(audit_path, audit)
    raise RuntimeError(f"lote {batch_id} falhou: {'; '.join(errors)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=Path("data/interim/items-subset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/generated-batches"))
    parser.add_argument("--target", type=int, default=720)
    parser.add_argument("--batch-size", type=int, default=18, choices=range(15, 21))
    parser.add_argument("--parallel", type=int, default=2, choices=(1, 2))
    parser.add_argument("--accounts", nargs="+", default=list(DEFAULT_ACCOUNTS))
    parser.add_argument("--model", default="gemini-3.8-flash-medium")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--audit-output", type=Path, default=Path("data/processed/generation-metadata.json")
    )
    args = parser.parse_args()
    if args.target <= 0:
        parser.error("--target deve ser positivo")
    rows = select_items(list(read_jsonl(args.items)), args.target)
    if len(rows) < args.target:
        parser.error(f"somente {len(rows)} itens disponíveis para target={args.target}")
    jobs = []
    for batch_id, start in enumerate(range(0, len(rows), args.batch_size)):
        jobs.append((batch_id, rows[start : start + args.batch_size], start))
    run_started = now_utc()
    failures = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(
                run_batch, batch_id, batch, start, args.output_dir,
                tuple(args.accounts), args.model, args.timeout,
            ): batch_id
            for batch_id, batch, start in jobs
        }
        for future in as_completed(futures):
            try:
                batch_id, status = future.result()
                print(f"lote {batch_id:04d}: {status}", flush=True)
            except Exception as exc:  # preserve successful batches for resume
                failures += 1
                print(f"ERRO: {exc}", flush=True)
    if failures:
        batch_audits = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(args.output_dir.glob("batch-*.meta.json"))
        ]
        atomic_json(
            args.audit_output,
            {
                "started_at": run_started, "ended_at": now_utc(), "model": args.model,
                "requested_rows": args.target, "batch_size": args.batch_size,
                "parallel_calls": args.parallel, "status": "incomplete", "batches": batch_audits,
            },
        )
        raise SystemExit(f"{failures} lote(s) falharam; execute novamente para retomar")
    batch_audits = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.output_dir.glob("batch-*.meta.json"))
    ]
    atomic_json(
        args.audit_output,
        {
            "started_at": run_started, "ended_at": now_utc(), "model": args.model,
            "requested_rows": args.target, "batch_size": args.batch_size,
            "parallel_calls": args.parallel, "status": "complete", "batches": batch_audits,
        },
    )


if __name__ == "__main__":
    main()
