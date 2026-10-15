from __future__ import annotations

import re
from collections.abc import Iterable

from .data import extract_citations

REFUSAL_RE = re.compile(
    r"\b(n[aã]o (?:h[aá]|posso|[eé] poss[ií]vel|encontrei)|fora do contexto|"
    r"informa[cç][aã]o insuficiente|consulte (?:um|uma) profissional)\b",
    re.IGNORECASE,
)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text))


def citation_scores(predictions: Iterable[str], references: Iterable[Iterable[str]]) -> dict[str, float]:
    exact: list[float] = []
    precision: list[float] = []
    recall: list[float] = []
    for prediction, reference in zip(predictions, references):
        predicted = set(extract_citations(prediction))
        expected = set(reference)
        exact.append(float(predicted == expected))
        precision.append(len(predicted & expected) / len(predicted) if predicted else float(not expected))
        recall.append(len(predicted & expected) / len(expected) if expected else float(not predicted))
    return {
        "citation_exact_accuracy": _mean(exact),
        "citation_precision": _mean(precision),
        "citation_recall": _mean(recall),
    }


def refusal_scores(predictions: Iterable[str], labels: Iterable[bool]) -> dict[str, float]:
    pairs = [(is_refusal(text), bool(label)) for text, label in zip(predictions, labels)]
    tp = sum(predicted and expected for predicted, expected in pairs)
    fp = sum(predicted and not expected for predicted, expected in pairs)
    fn = sum(not predicted and expected for predicted, expected in pairs)
    return {
        "refusal_precision": tp / (tp + fp) if tp + fp else 0.0,
        "refusal_recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def rouge_l(predictions: Iterable[str], references: Iterable[str]) -> float:
    values = [_rouge_l_f1(prediction, reference) for prediction, reference in zip(predictions, references)]
    return _mean(values)


def _rouge_l_f1(prediction: str, reference: str) -> float:
    predicted = prediction.casefold().split()
    expected = reference.casefold().split()
    if not predicted or not expected:
        return float(predicted == expected)
    previous = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for column, expected_token in enumerate(expected, 1):
            if token == expected_token:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(predicted)
    recall = lcs / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
