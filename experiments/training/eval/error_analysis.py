#!/usr/bin/env python3
"""
Projeto IANA — Análise qualitativa de erros por modelo.

Lista os top-10 erros mais comuns por categoria para cada modelo.

Uso:
    python error_analysis.py --model biobertpt
    python error_analysis.py --model qwen35_4b
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _TRAINING_DIR.parent

NER_CATEGORIES = [
    "disease_or_syndrome", "sign_or_symptom", "pharmacologic_substance",
    "laboratory_or_test_result", "diagnostic_procedure", "organism_or_virus",
]


def _norm(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    no_acc = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_acc.strip().lower())


def main():
    parser = argparse.ArgumentParser(description="IANA — Análise de erros")
    parser.add_argument("--model", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--gold", default=str(_EXPERIMENTS_DIR / "resultados" / "gold_test_set_30.json"))
    parser.add_argument("--output-dir", default=str(_TRAINING_DIR / "results"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    pred_path = args.predictions or str(_TRAINING_DIR / "predictions" / f"{args.model}_predictions.json")

    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)
    with open(args.gold, encoding="utf-8") as f:
        gold_data = json.load(f)

    gold_notes = gold_data.get("notes", gold_data) if isinstance(gold_data, dict) else gold_data
    gold_by_id = {n["paciente_id"]: n for n in gold_notes}

    # Collect errors
    false_positives: dict[str, Counter] = defaultdict(Counter)
    false_negatives: dict[str, Counter] = defaultdict(Counter)

    for pred in predictions:
        pid = pred["paciente_id"]
        gold = gold_by_id.get(pid, {})
        gold_ner = gold.get("ner", {})
        pred_ner = pred.get("predictions", {})

        for cat in NER_CATEGORIES:
            gold_items = {_norm(x) for x in gold_ner.get(cat, []) if isinstance(x, str)}
            pred_raw = pred_ner.get(cat, []) if isinstance(pred_ner, dict) else []
            pred_items = {_norm(x) for x in pred_raw if isinstance(x, str)}

            for fp in pred_items - gold_items:
                false_positives[cat][fp] += 1
            for fn in gold_items - pred_items:
                false_negatives[cat][fn] += 1

    # Report
    report = {"model": args.model, "categories": {}}
    for cat in NER_CATEGORIES:
        top_fp = false_positives[cat].most_common(args.top_k)
        top_fn = false_negatives[cat].most_common(args.top_k)
        report["categories"][cat] = {
            "top_false_positives": [{"entity": e, "count": c} for e, c in top_fp],
            "top_false_negatives": [{"entity": e, "count": c} for e, c in top_fn],
            "total_fp": sum(false_positives[cat].values()),
            "total_fn": sum(false_negatives[cat].values()),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.model}_error_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
