#!/usr/bin/env python3
"""
Projeto IANA — Calcula precision/recall/F1 por categoria e doença.

Suporta dois modos de matching entre gold e predicao:
  --matching exact (default): comparacao exata por string normalizada
  --matching fuzzy:           usa overlap de tokens / substring / Levenshtein
                              (mais informativo para entidades clinicas com
                              variantes lexicais como "HIV" vs "HIV/AIDS").
  --matching both:            roda ambos e salva nos campos exact/fuzzy

Uso:
    python compute_metrics.py --model qwen35_4b
    python compute_metrics.py --model qwen35_4b --matching fuzzy
    python compute_metrics.py --model qwen35_4b --matching both
"""

import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _TRAINING_DIR.parent

NER_CATEGORIES = [
    "disease_or_syndrome", "sign_or_symptom", "pharmacologic_substance",
    "laboratory_or_test_result", "diagnostic_procedure", "organism_or_virus",
]

# Threshold de similaridade para matching fuzzy (Levenshtein normalizado).
FUZZY_SIM_THRESHOLD = 0.85


def _norm(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    no_acc = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_acc.strip().lower())


def _is_fuzzy_match(a: str, b: str) -> bool:
    """Match com 2 estrategias em cascata:
    1. Word-boundary match: a string menor aparece como palavra completa na
       maior. Cobre "HIV" vs "HIV/AIDS" (HIV e palavra), "febre" vs "febre alta".
       Usa \\b do regex para evitar falsos positivos como "ar" em "tratar"
       (\\b nao casa entre letras).
    2. Levenshtein normalizado >= FUZZY_SIM_THRESHOLD (cobre typos e
       variantes pequenas como "diabetes" vs "diabete").
    """
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if shorter and re.search(r"\b" + re.escape(shorter) + r"\b", longer):
        return True
    return SequenceMatcher(None, a, b).ratio() >= FUZZY_SIM_THRESHOLD


def _set_f1_exact(gold_set: set, pred_set: set) -> dict:
    if not gold_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def _set_f1_fuzzy(gold_set: set, pred_set: set) -> dict:
    """Matching fuzzy: cada gold pode casar com 1 pred via _is_fuzzy_match.
    Greedy: itera gold, marca o primeiro pred nao-usado que da match."""
    if not gold_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    gold_list = list(gold_set)
    pred_list = list(pred_set)
    matched_pred = [False] * len(pred_list)
    tp = 0
    for g in gold_list:
        for i, p in enumerate(pred_list):
            if matched_pred[i]:
                continue
            if _is_fuzzy_match(g, p):
                matched_pred[i] = True
                tp += 1
                break
    fp = len(pred_list) - tp
    fn = len(gold_list) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def _compute_metrics(gold_by_id: dict, predictions: list, matcher) -> dict:
    """matcher: _set_f1_exact ou _set_f1_fuzzy."""
    category_metrics = {}
    all_tp, all_fp, all_fn = 0, 0, 0

    for cat in NER_CATEGORIES:
        cat_gold_all: set = set()
        cat_pred_all: set = set()

        for pred in predictions:
            pid = pred["paciente_id"]
            gold = gold_by_id.get(pid, {})
            gold_ner = gold.get("ner", {})
            pred_ner = pred.get("predictions", {})

            gold_items = {_norm(x) for x in gold_ner.get(cat, []) if isinstance(x, str)}
            pred_raw = pred_ner.get(cat, []) if isinstance(pred_ner, dict) else []
            pred_items = {_norm(x) for x in pred_raw if isinstance(x, str)}

            cat_gold_all.update(gold_items)
            cat_pred_all.update(pred_items)

        m = matcher(cat_gold_all, cat_pred_all)
        category_metrics[cat] = m
        all_tp += m["tp"]
        all_fp += m["fp"]
        all_fn += m["fn"]

    micro_p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    micro_r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    macro_f1 = sum(m["f1"] for m in category_metrics.values()) / len(category_metrics)

    return {
        "per_category": category_metrics,
        "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4)},
        "macro_f1": round(macro_f1, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="IANA — Métricas de avaliação")
    parser.add_argument("--model", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--gold", default=str(_EXPERIMENTS_DIR / "resultados" / "gold_test_set_30.json"))
    parser.add_argument("--output-dir", default=str(_TRAINING_DIR / "results"))
    parser.add_argument("--matching", choices=["exact", "fuzzy", "both"], default="both")
    args = parser.parse_args()

    pred_path = args.predictions or str(_TRAINING_DIR / "predictions" / f"{args.model}_predictions.json")

    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)
    with open(args.gold, encoding="utf-8") as f:
        gold_data = json.load(f)

    gold_notes = gold_data.get("notes", gold_data) if isinstance(gold_data, dict) else gold_data
    gold_by_id = {n["paciente_id"]: n for n in gold_notes}

    results = {"model": args.model}

    if args.matching in ("exact", "both"):
        results["exact"] = _compute_metrics(gold_by_id, predictions, _set_f1_exact)
    if args.matching in ("fuzzy", "both"):
        results["fuzzy"] = _compute_metrics(gold_by_id, predictions, _set_f1_fuzzy)

    # Mantem chaves no top-level pra retro-compat (compare_models le 'micro'/'macro_f1'/'per_category')
    primary = "fuzzy" if args.matching == "fuzzy" else "exact"
    if primary in results:
        results.update({
            "per_category": results[primary]["per_category"],
            "micro": results[primary]["micro"],
            "macro_f1": results[primary]["macro_f1"],
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.model}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
