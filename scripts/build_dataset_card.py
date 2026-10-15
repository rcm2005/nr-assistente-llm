#!/usr/bin/env python3
"""Build a dataset card using counts computed from files on disk."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nr_assistente.data import read_jsonl


def stats(path: Path) -> tuple[list[dict], Counter, int]:
    rows = list(read_jsonl(path))
    return rows, Counter(int(row["nr"]) for row in rows), sum(bool(r["expected_refusal"]) for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/manifest.json"))
    parser.add_argument(
        "--generation-metadata",
        type=Path,
        default=Path("data/processed/generation-metadata.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("docs/dataset-card.md"))
    args = parser.parse_args()
    train, train_nrs, train_refusals = stats(args.train)
    test, test_nrs, test_refusals = stats(args.test)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    generation = json.loads(args.generation_metadata.read_text(encoding="utf-8"))
    if generation.get("status") != "complete":
        raise ValueError("metadados indicam geracao incompleta")
    requested = int(generation["requested_rows"])
    accepted = len(train) + len(test)
    attempts = [attempt for batch in generation["batches"] for attempt in batch["attempts"]]
    retries = len(attempts) - len(generation["batches"])
    all_nrs = sorted(train_nrs | test_nrs)
    lines = [
        "# Cartão do dataset", "",
        f"Gerado automaticamente em {datetime.now(timezone.utc).isoformat()} a partir dos arquivos locais.", "",
        "## Origem e escopo", "",
        "Trechos dos PDFs públicos oficiais das Normas Regulamentadoras do MTE. "
        f"SHA-256 do manifesto de fontes: `{manifest_hash}`.", "",
        "## Composição medida", "",
        f"O Gemini `{generation['model']}` produziu {requested} saídas em "
        f"{len(generation['batches'])} lotes, com até {generation['parallel_calls']} chamadas "
        f"simultâneas. A validação aceitou {accepted} e rejeitou {requested - accepted}.", "",
        "| Split | Pares | Recusas esperadas | Itens normativos distintos |", "|---|---:|---:|---:|",
        f"| Treino | {len(train)} | {train_refusals} | {len({r['source_id'] for r in train})} |",
        f"| Teste | {len(test)} | {test_refusals} | {len({r['source_id'] for r in test})} |", "",
        "| NR | Treino | Teste |", "|---:|---:|---:|",
    ]
    lines += [f"| {nr:02d} | {train_nrs[nr]} | {test_nrs[nr]} |" for nr in all_nrs]
    lines += [
        "", "## Metodologia", "",
        f"Os pares foram gerados pelo `{generation['model']}` via `agy`, entre "
        f"`{generation['started_at']}` e `{generation['ended_at']}`. Foram "
        f"{len(attempts)} tentativas no total ({retries} repetições após saída inválida ou falha). "
        "Depois, os dados foram validados localmente. "
        "Exemplos comuns exigem a citação exata do item-fonte; armadilhas exigem recusa e não podem "
        "conter citação. O split é feito por `source_id`, portanto um mesmo item normativo não aparece "
        "simultaneamente em treino e teste.", "",
        "## Limitações", "",
        "Dados sintéticos podem conter formulações pouco naturais ou inferências imperfeitas. A validação "
        "local confirma esquema, identidade da fonte, duplicatas, recusas e formato de citação, mas não "
        "substitui revisão jurídica humana. O assistente não oferece parecer legal.", "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"cartão escrito em {args.output}")


if __name__ == "__main__":
    main()
