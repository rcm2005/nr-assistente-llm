from src.nr_assistente.metrics import citation_scores, is_refusal, refusal_scores, rouge_l
from src.nr_assistente.records import messages_for, normalize_record


def test_normalize_bilingual_record_and_messages() -> None:
    record = normalize_record(
        {
            "pergunta": "Qual requisito?",
            "resposta": "Segundo a NR-35 item 35.4.2.1, este e o requisito.",
            "contexto": "Texto oficial.",
            "citacoes": ["NR 35 item 35.4.2.1"],
            "trap": "false",
        }
    )
    assert record.citations == ("NR-35 item 35.4.2.1",)
    assert "Contexto normativo" in messages_for(record)[1]["content"]

    generated_schema = normalize_record(
        {"question": "Pergunta valida?", "answer": "Não há base suficiente.", "citation": None,
         "expected_refusal": True}
    )
    assert generated_schema.is_refusal
    assert generated_schema.citations == ()


def test_citation_and_refusal_metrics() -> None:
    predictions = ["Conforme NR-35 item 35.4.2.1.", "Nao ha informacao suficiente."]
    citations = [["NR-35 item 35.4.2.1"], []]
    assert citation_scores(predictions, citations)["citation_exact_accuracy"] == 1.0
    scores = refusal_scores(predictions, [False, True])
    assert scores == {"refusal_precision": 1.0, "refusal_recall": 1.0}
    assert is_refusal("Não há informação suficiente no contexto.")
    assert rouge_l(["a b c"], ["a x c"]) == 2 / 3
