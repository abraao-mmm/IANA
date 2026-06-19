#!/usr/bin/env python3
"""
Projeto IANA — Tabela comparativa dos modelos do benchmark.

Le os arquivos {model}_metrics.json gerados por compute_metrics.py com
--matching=both e produz comparison.md com tabelas exact vs fuzzy.

Uso:
    python compare_models.py
    python compare_models.py --results-dir ../results
"""

import argparse
import json
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent.parent

MODELS = ["biobertpt", "medgemma", "gemma4_e4b", "qwen35_4b"]

NER_CATEGORIES = [
    "disease_or_syndrome", "sign_or_symptom", "pharmacologic_substance",
    "laboratory_or_test_result", "diagnostic_procedure", "organism_or_virus",
]


def _agg_table(all_metrics: dict, mode: str) -> list:
    """mode: 'exact' ou 'fuzzy' — extrai do bloco correspondente, com
    fallback para o top-level (compat com runs antigos sem matching=both)."""
    lines = [f"## Métricas Agregadas ({mode})\n",
             "| Modelo | Micro-P | Micro-R | Micro-F1 | Macro-F1 |",
             "|---|---|---|---|---|"]
    for model, m in all_metrics.items():
        block = m.get(mode) or m  # fallback top-level
        mi = block.get("micro", {})
        lines.append(f"| {model} | {mi.get('precision', '-')} | {mi.get('recall', '-')} | "
                     f"{mi.get('f1', '-')} | {block.get('macro_f1', '-')} |")
    lines.append("")
    return lines


def _cat_table(all_metrics: dict, mode: str) -> list:
    lines = [f"## F1 por Categoria ({mode})\n"]
    header = "| Categoria | " + " | ".join(all_metrics.keys()) + " |"
    sep = "|---|" + "|".join(["---"] * len(all_metrics)) + "|"
    lines.append(header)
    lines.append(sep)
    for cat in NER_CATEGORIES:
        row = f"| {cat} |"
        for model, m in all_metrics.items():
            block = m.get(mode) or m
            f1 = block.get("per_category", {}).get(cat, {}).get("f1", "-")
            row += f" {f1} |"
        lines.append(row)
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description="IANA — Comparação de modelos")
    parser.add_argument("--results-dir", default=str(_TRAINING_DIR / "results"))
    parser.add_argument("--output", default=str(_TRAINING_DIR / "results" / "comparison.md"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    all_metrics = {}
    for model in MODELS:
        path = results_dir / f"{model}_metrics.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                all_metrics[model] = json.load(f)

    if not all_metrics:
        print("Nenhum resultado encontrado. Rode compute_metrics.py primeiro.")
        return

    has_dual = any("exact" in m and "fuzzy" in m for m in all_metrics.values())

    lines = ["# Comparação de Modelos — Benchmark IANA\n"]

    if has_dual:
        lines += _agg_table(all_metrics, "exact")
        lines += _agg_table(all_metrics, "fuzzy")
        lines += _cat_table(all_metrics, "exact")
        lines += _cat_table(all_metrics, "fuzzy")
    else:
        lines += _agg_table(all_metrics, "exact")
        lines += _cat_table(all_metrics, "exact")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # CSV: linha por modelo×modo
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,mode,micro_precision,micro_recall,micro_f1,macro_f1\n")
        for model, m in all_metrics.items():
            for mode in (("exact", "fuzzy") if has_dual else ("exact",)):
                block = m.get(mode) or m
                mi = block.get("micro", {})
                f.write(f"{model},{mode},{mi.get('precision', '')},{mi.get('recall', '')},"
                        f"{mi.get('f1', '')},{block.get('macro_f1', '')}\n")

    print(f"Comparação salva em {output_path} e {csv_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
