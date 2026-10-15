from src.nr_assistente.data import canonical_text, citation_key, extract_citations, stable_id
from scripts.generate_dataset import decode_agy_output, select_items
from scripts.split_dataset import group_split
from scripts.validate_dataset import validate


def test_citation_normalization() -> None:
    text = "Conforme NR 1 item 1.5.3 e NR-35 itens 35.4.2.1."
    assert extract_citations(text) == ["NR-01 item 1.5.3", "NR-35 item 35.4.2.1"]


def test_stable_id_and_text() -> None:
    assert stable_id("a", "b") == stable_id("a", "b")
    assert canonical_text("  AÇÃO\nsegura ") == "ação segura"
    assert citation_key("7", "7.5.1") == "NR-07 item 7.5.1"


def test_decode_agy_wrapped_output() -> None:
    assert decode_agy_output('{"response":"```json\\n[{\\"x\\": 1}]\\n```"}') == [{"x": 1}]
    assert decode_agy_output('{"pairs":[{"x": 1}]}') == [{"x": 1}]


def test_select_items_interleaves_nrs() -> None:
    rows = [{"id": str(i), "nr": nr} for i, nr in enumerate([6, 6, 6, 10, 10, 12])]
    assert [row["nr"] for row in select_items(rows, 5)] == [6, 10, 12, 6, 10]


def test_validate_grounded_and_refusal_rows() -> None:
    source = {"abc": {"id": "abc", "nr": 6, "item": "6.5.1", "citation": "NR-06 item 6.5.1"}}
    good = {
        "source_id": "abc", "question": "Qual é a obrigação indicada no item?",
        "answer": "A organização deve cumprir a obrigação. NR-06 item 6.5.1",
        "citation": "NR-06 item 6.5.1", "expected_refusal": False,
    }
    accepted, reason = validate(good, source)
    assert accepted and reason is None
    trap = {
        "source_id": "abc", "question": "Qual é o prazo exato em dias corridos?",
        "answer": "Não há base suficiente no item fornecido para informar esse prazo.",
        "citation": None, "expected_refusal": True,
    }
    accepted, reason = validate(trap, source)
    assert accepted and reason is None


def test_split_has_no_item_leakage() -> None:
    rows = [
        {"source_id": f"s{i}", "id": f"q{i}", "nr": 6, "expected_refusal": False}
        for i in range(10)
    ]
    train, test = group_split(rows, 3, "seed")
    assert len(test) == 3
    assert {x["source_id"] for x in train}.isdisjoint({x["source_id"] for x in test})
