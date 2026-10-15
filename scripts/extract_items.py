#!/usr/bin/env python3
"""Extract hierarchical NR items from official PDFs into auditable JSONL chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nr_assistente.data import citation_key, stable_id, write_jsonl

HEADER_RE = re.compile(r"^(?P<item>\d{1,2}(?:\.\d+){1,7})\.?\s*(?P<body>.*)$")
FOOTER_RE = re.compile(
    r"^(?:Este texto n[aã]o substitui o publicado no DOU|Minist[eé]rio do Trabalho.*)$",
    re.IGNORECASE,
)


def clean_lines(text: str) -> list[str]:
    output: list[str] = []
    for raw in text.replace("\u00ad", "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if (
            not line
            or re.fullmatch(r"\d+", line)
            or line.casefold().startswith("nr ")
            or FOOTER_RE.match(line)
        ):
            continue
        output.append(line)
    return output


def extract_pdf(path: Path, nr: int) -> list[dict]:
    lines: list[tuple[int, str]] = []
    with fitz.open(path) as document:
        for page_no, page in enumerate(document, 1):
            lines.extend((page_no, line) for line in clean_lines(page.get_text("text")))
    # Most PDFs repeat the table of contents on page 1. When the first section
    # heading occurs again, the second occurrence marks the normative text.
    root_item = f"{nr}.1"
    root_pages = [page for page, line in lines if (m := HEADER_RE.match(line)) and m["item"] == root_item]
    normative_start = root_pages[1] if len(root_pages) > 1 else 1
    rows: list[dict] = []
    current: dict | None = None
    for page_no, line in lines:
        match = HEADER_RE.match(line)
        if page_no >= normative_start and match and int(match.group("item").split(".")[0]) == nr:
            if current:
                current["text"] = " ".join(current.pop("parts"))
                current["page_end"] = page_no
                rows.append(current)
            item = match.group("item")
            current = {
                "id": stable_id(f"NR-{nr:02d}", item),
                "nr": nr,
                "item": item,
                "citation": citation_key(nr, item),
                "source_file": path.name,
                "page_start": page_no,
                "parts": [match.group("body")],
            }
        elif current:
            current["parts"].append(line)
    if current:
        current["text"] = " ".join(current.pop("parts"))
        current["page_end"] = lines[-1][0]
        rows.append(current)
    # Two-component numbers (e.g. 12.2) are normally section titles, not
    # self-contained requirements. Keeping clauses from x.y.z onward also
    # prevents annex section numbers from masquerading as main-body items.
    eligible = [
        row
        for row in rows
        if row["item"].count(".") >= 2 and 25 <= len(row["text"]) <= 8_000
    ]
    # Anexos can restart numbering (notably NR-12). Keep the first substantive
    # occurrence in the normative body rather than replacing it with an annex.
    deduplicated: dict[str, dict] = {}
    for row in eligible:
        deduplicated.setdefault(row["item"], row)
    return list(deduplicated.values())


def verify_manifest(input_dir: Path, only: set[int] | None) -> None:
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {int(row["nr"]): row for row in manifest["documents"]}
    selected = only or set(records)
    missing = selected - set(records)
    if missing:
        raise ValueError(f"NRs ausentes no manifesto: {sorted(missing)}")
    for nr in selected:
        row = records[nr]
        path = input_dir / row["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"SHA-256 divergente para {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/items.jsonl"))
    parser.add_argument("--only", type=int, nargs="*", help="Extrair apenas estas NRs")
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args()
    only = set(args.only) if args.only else None
    if not args.skip_hash_check:
        verify_manifest(args.input, only)
    rows: list[dict] = []
    for path in sorted(args.input.glob("NR-??.pdf")):
        nr = int(path.stem.split("-")[1])
        if only is not None and nr not in only:
            continue
        extracted = extract_pdf(path, nr)
        print(f"{path.stem}: {len(extracted)} itens")
        rows.extend(extracted)
    count = write_jsonl(args.output, rows)
    print(f"Total: {count} itens em {args.output}")


if __name__ == "__main__":
    main()
