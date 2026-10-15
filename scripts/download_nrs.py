#!/usr/bin/env python3
"""Download current NR PDFs from the official MTE index and write a SHA manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

INDEX_URL = (
    "https://www.gov.br/trabalho-e-emprego/pt-br/acesso-a-informacao/participacao-social/"
    "conselhos-e-orgaos-colegiados/comissao-tripartite-partitaria-permanente/"
    "normas-regulamentadora/normas-regulamentadoras-vigentes"
)
REVOKED = {2, 27}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str, str]] = []
        self.href: str | None = None
        self.text: list[str] = []
        self.context: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            self.href = values["href"]
            self.text = []
        if tag == "p":
            self.context.append(values.get("class") or "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.href:
            self.links.append((self.href, " ".join(self.text), " ".join(self.context)))
            self.href = None
        if tag == "p" and self.context:
            self.context.pop()

    def handle_data(self, data: str) -> None:
        if self.href:
            self.text.append(data.strip())


def fetch(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response


def nr_pages(html: str) -> dict[int, str]:
    parser = LinkParser()
    parser.feed(html)
    pages: dict[int, str] = {}
    for href, text, _ in parser.links:
        match = re.fullmatch(r"NR\s*-?\s*(\d{1,2}).*", " ".join(text.split()), re.I)
        is_nr_page = "norma-regulamentadora" in href or "/normas-regulamentadoras-vigentes/nr-1" in href
        if match and is_nr_page and ".pdf" not in href.casefold():
            pages[int(match.group(1))] = href
    return pages


def choose_pdf(nr: int, page_url: str, html: str) -> str:
    parser = LinkParser()
    parser.feed(html)
    candidates: list[tuple[tuple[int, int, int, int], str]] = []
    for href, text, context in parser.links:
        href = urljoin(page_url, href)
        normalized = " ".join(text.split())
        if ".pdf" not in href.casefold() or "callout" not in context:
            continue
        if not re.match(rf"NR\s*-?\s*0?{nr}\b", normalized, re.I):
            continue
        years = [int(year) for year in re.findall(r"20\d{2}", href + " " + normalized)]
        is_annex = "anexo" in (href + " " + normalized).casefold()
        score = (
            int(not is_annex),
            max(years, default=0),
            int("atual" in href.casefold()),
            -len(normalized),
        )
        candidates.append((score, href))
    if not candidates:
        raise RuntimeError(f"Nenhum PDF normativo atual encontrado para NR-{nr:02d}: {page_url}")
    return max(candidates)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--only", type=int, nargs="*")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "NR-Assistente-academic-research/1.0"
    index = fetch(session, INDEX_URL)
    pages = nr_pages(index.text)
    selected = sorted(set(args.only or pages) - REVOKED)
    manifest = {
        "index_url": INDEX_URL,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "documents": [],
    }
    for nr in selected:
        page_url = pages[nr]
        page = fetch(session, page_url)
        pdf_url = choose_pdf(nr, page_url, page.text)
        payload = fetch(session, pdf_url).content
        if not payload.startswith(b"%PDF"):
            raise RuntimeError(f"Resposta não é PDF para NR-{nr:02d}: {pdf_url}")
        target = args.output / f"NR-{nr:02d}.pdf"
        target.write_bytes(payload)
        manifest["documents"].append(
            {
                "nr": nr,
                "page_url": page_url,
                "pdf_url": pdf_url,
                "file": target.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        print(f"NR-{nr:02d}: {len(payload) / 1_048_576:.1f} MiB")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
