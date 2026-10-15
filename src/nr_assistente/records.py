from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data import extract_citations

SYSTEM_PROMPT = (
    "Voce e um assistente de Normas Regulamentadoras brasileiras. Responda apenas com "
    "base no contexto fornecido, cite NR e item e recuse quando o contexto nao sustentar a resposta."
)


@dataclass(frozen=True)
class QARecord:
    question: str
    answer: str
    context: str
    citations: tuple[str, ...]
    is_refusal: bool
    identifier: str


def _first(row: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def normalize_record(row: dict[str, Any], row_number: int = 0) -> QARecord:
    question = str(_first(row, "question", "pergunta", "prompt")).strip()
    answer = str(_first(row, "answer", "resposta", "response")).strip()
    context = str(_first(row, "context", "contexto", "source_text", "trecho")).strip()
    if not question or not answer:
        raise ValueError(f"Registro {row_number}: pergunta ou resposta ausente")

    raw_citations = _first(row, "citations", "citation", "citacoes", "references", default=[])
    if isinstance(raw_citations, str):
        citations = extract_citations(raw_citations) or [raw_citations.strip()]
    else:
        citations = []
        for value in raw_citations:
            text = str(value).strip()
            citations.extend(extract_citations(text) or ([text] if text else []))
    if not citations:
        citations = extract_citations(answer)

    raw_refusal = _first(
        row, "is_refusal", "expected_refusal", "trap", "is_trap", "recusa", default=False
    )
    if isinstance(raw_refusal, str):
        is_refusal = raw_refusal.strip().lower() in {"1", "true", "sim", "yes"}
    else:
        is_refusal = bool(raw_refusal)
    identifier = str(_first(row, "id", "example_id", default=row_number))
    return QARecord(question, answer, context, tuple(citations), is_refusal, identifier)


def messages_for(record: QARecord, include_answer: bool = True) -> list[dict[str, str]]:
    user = record.question
    if record.context:
        user = f"Contexto normativo:\n{record.context}\n\nPergunta:\n{record.question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": record.answer})
    return messages
