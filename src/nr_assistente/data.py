from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

CITATION_RE = re.compile(
    r"\bNR\s*[-–—]?\s*0?(?P<nr>\d{1,2})\s+(?:item|itens?)\s+"
    r"(?P<item>\d{1,2}(?:\.\d+){1,7})\b",
    re.IGNORECASE,
)


def canonical_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def citation_key(nr: int | str, item: str) -> str:
    return f"NR-{int(nr):02d} item {item}"


def extract_citations(text: str) -> list[str]:
    return [citation_key(m.group("nr"), m.group("item")) for m in CITATION_RE.finditer(text)]


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON inválido em {path}:{line_no}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
