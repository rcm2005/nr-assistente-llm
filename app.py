from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import gradio as gr

from src.nr_assistente.data import extract_citations
from src.nr_assistente.inference import Generator
from src.nr_assistente.records import QARecord
from src.nr_assistente.telemetry import query_gpu

BASE_MODEL = os.environ.get("NR_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def available_models() -> dict[str, str]:
    models = {"Base Qwen2.5-0.5B": BASE_MODEL}
    for adapter in sorted(Path("outputs").glob("*/adapter_config.json")):
        models[f"Fine-tune: {adapter.parent.name}"] = str(adapter.parent)
    return models


@lru_cache(maxsize=2)
def load_generator(location: str) -> Generator:
    return Generator(location)


def gpu_status() -> str:
    try:
        gpu = query_gpu()
        return (
            f'{gpu["name"]} | VRAM {gpu["memory.used"]}/{gpu["memory.total"]} MiB | '
            f'utilizacao {gpu["utilization.gpu"]}%'
        )
    except Exception as exc:  # UI must show absence instead of crashing.
        return f"GPU indisponivel: {type(exc).__name__}: {exc}"


def respond(
    question: str,
    model_label: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[str, str, str, str]:
    if not question.strip():
        return "Informe uma pergunta.", "", "", gpu_status()
    models = available_models()
    if model_label not in models:
        raise gr.Error("Modelo nao encontrado; recarregue a pagina")
    record = QARecord(question.strip(), "", "", (), False, "ui")
    answer, metrics = load_generator(models[model_label]).generate(
        record, temperature=temperature, top_p=top_p, max_new_tokens=int(max_tokens)
    )
    citations = extract_citations(answer)
    citation_text = "\n".join(citations) if citations else "Nenhuma citacao reconhecida na resposta."
    speed = f'{metrics["generated_tokens"]} tokens em {metrics["latency_seconds"]:.2f}s | {metrics["tokens_per_second"]:.2f} tokens/s'
    return answer, citation_text, speed, gpu_status()


def build_app() -> gr.Blocks:
    choices = list(available_models())
    with gr.Blocks(title="NR-Assistente") as demo:
        gr.Markdown("# NR-Assistente\nRespostas experimentais: confira sempre a NR oficial.")
        with gr.Row():
            question = gr.Textbox(label="Pergunta", lines=5, placeholder="Digite uma pergunta sobre as NRs")
            with gr.Column():
                model = gr.Dropdown(choices=choices, value=choices[0], label="Modelo")
                temperature = gr.Slider(0, 1, value=0.2, step=0.05, label="Temperatura")
                top_p = gr.Slider(0.1, 1, value=0.9, step=0.05, label="top_p")
                max_tokens = gr.Slider(32, 512, value=192, step=16, label="max_tokens")
        submit = gr.Button("Responder", variant="primary")
        answer = gr.Textbox(label="Resposta", lines=9)
        citations = gr.Textbox(label="Citacoes extraidas")
        speed = gr.Textbox(label="Desempenho")
        gpu = gr.Textbox(value=gpu_status, label="GPU", interactive=False)
        submit.click(
            respond,
            inputs=[question, model, temperature, top_p, max_tokens],
            outputs=[answer, citations, speed, gpu],
        )
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0")
