from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera graficos somente a partir de avaliacao existente")
    parser.add_argument("--summary", type=Path, default=Path("results/eval/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/img"))
    args = parser.parse_args()
    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("Resumo vazio: nenhum grafico foi gerado")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = [f'{row["model"]}\nT={row["temperature"]}' for row in rows]

    figures = {
        "metricas-qualidade.png": (
            ["citation_exact_accuracy", "refusal_precision", "refusal_recall", "rouge_l"],
            "Metricas no teste retido",
            "proporcao",
        ),
        "desempenho-inferencia.png": (
            ["tokens_per_second", "peak_allocated_vram_mib"],
            "Desempenho de inferencia",
            "valor medido",
        ),
    }
    for filename, (metrics, title, ylabel) in figures.items():
        figure, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), squeeze=False)
        for axis, metric in zip(axes[0], metrics):
            values = [row[metric] for row in rows]
            axis.bar(labels, values)
            axis.set_title(metric.replace("_", " "))
            axis.tick_params(axis="x", labelrotation=30)
        figure.suptitle(title)
        axes[0][0].set_ylabel(ylabel)
        figure.tight_layout()
        figure.savefig(args.output_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
