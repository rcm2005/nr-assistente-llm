#!/usr/bin/env python3
"""Validate generated batches against source items and emit accepted/rejected JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nr_assistente.data import canonical_text, extract_citations, read_jsonl, stable_id, write_jsonl

REFUSAL_MARKERS = ("não há", "não consta", "não informa", "base suficiente", "não é possível")


def validate(row: dict, sources: dict[str, dict]) -> tuple[dict | None, str | None]:
    required = {"source_id", "question", "answer", "citation", "expected_refusal"}
    if not required.issubset(row):
        return None, f"campos ausentes: {sorted(required - set(row))}"
    source = sources.get(str(row["source_id"]))
    if source is None:
        return None, "source_id desconhecido"
    question, answer = str(row["question"]).strip(), str(row["answer"]).strip()
    if len(question) < 15 or len(answer) < 15:
        return None, "pergunta ou resposta curta demais"
    refusal = row["expected_refusal"]
    if not isinstance(refusal, bool):
        return None, "expected_refusal não é booleano"
    citations = extract_citations(answer)
    if refusal:
        if row["citation"] is not None or citations:
            return None, "recusa contém citação normativa"
        if not any(marker in canonical_text(answer) for marker in REFUSAL_MARKERS):
            return None, "resposta não explicita recusa"
    else:
        expected = source["citation"]
        if row["citation"] != expected or citations != [expected]:
            return None, f"citação deve ser exatamente {expected}"
    accepted = {
        "id": stable_id(source["id"], question),
        "source_id": source["id"],
        "nr": source["nr"],
        "item": source["item"],
        "question": question,
        "answer": answer,
        "citation": row["citation"],
        "expected_refusal": refusal,
    }
    return accepted, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=Path("data/interim/items-subset.jsonl"))
    parser.add_argument("--batches", type=Path, default=Path("data/interim/generated-batches"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/dataset.jsonl"))
    parser.add_argument("--rejected", type=Path, default=Path("data/interim/rejected.jsonl"))
    args = parser.parse_args()
    sources = {row["id"]: row for row in read_jsonl(args.items)}
    accepted, rejected = [], []
    seen_questions: set[str] = set()
    batch_paths = [path for path in args.batches.glob("batch-*.json") if ".meta." not in path.name]
    for path in sorted(batch_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for index, row in enumerate(payload):
            valid, reason = validate(row, sources)
            if valid is not None:
                normalized = canonical_text(valid["question"])
                if normalized in seen_questions:
                    reason = "pergunta duplicada"
                    valid = None
                else:
                    seen_questions.add(normalized)
            if valid is not None:
                accepted.append(valid)
            else:
                rejected.append({"batch": path.name, "index": index, "reason": reason, "row": row})
    write_jsonl(args.output, accepted)
    write_jsonl(args.rejected, rejected)
    print(f"aceitos={len(accepted)} rejeitados={len(rejected)}")
    if not accepted:
        raise SystemExit("nenhum exemplo válido")


if __name__ == "__main__":
    main()
