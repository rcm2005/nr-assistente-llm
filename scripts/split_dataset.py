#!/usr/bin/env python3
"""Create deterministic train/test splits grouped by normative item."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nr_assistente.data import read_jsonl, write_jsonl


def group_split(rows: list[dict], test_size: int, seed: str) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_id"]].append(row)
    if not 0 < test_size < len(rows):
        raise ValueError("test_size deve estar entre 1 e len(dataset)-1")
    ranked = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}\x1f{key}".encode()).hexdigest(),
    )
    chosen: set[str] = set()
    count = 0
    for key in ranked:
        size = len(groups[key])
        if count + size <= test_size or not chosen:
            chosen.add(key)
            count += size
        if count == test_size:
            break
    if count != test_size:
        raise ValueError(
            f"não foi possível obter exatamente {test_size} linhas sem quebrar grupos; obtido {count}"
        )
    train = [row for row in rows if row["source_id"] not in chosen]
    test = [row for row in rows if row["source_id"] in chosen]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/dataset.jsonl"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--test-size", type=int, default=120)
    parser.add_argument("--seed", default="nr-assistente-v1")
    args = parser.parse_args()
    rows = list(read_jsonl(args.input))
    train, test = group_split(rows, args.test_size, args.seed)
    train_groups = {row["source_id"] for row in train}
    test_groups = {row["source_id"] for row in test}
    assert train_groups.isdisjoint(test_groups)
    write_jsonl(args.train, train)
    write_jsonl(args.test, test)
    print(f"treino={len(train)} teste={len(test)} itens_teste={len(test_groups)}")


if __name__ == "__main__":
    main()
