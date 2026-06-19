#!/usr/bin/env python3
"""
Projeto IANA — Converte silver standard para formato ChatML (Qwen3.5-4B).

Para cada nota, gera 2 exemplos: NER e SOAP, em formato messages ChatML.

Uso:
    python format_chatml.py
    python format_chatml.py --smoke-test
    python format_chatml.py --split train --config ../config/splits.yaml
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _TRAINING_DIR.parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from config.prompts import PROMPT_NER_TRAIN, PROMPT_SOAP_TRAIN


class _JSONFormatter(logging.Formatter):
    def format(self, record):
        entry = {"ts": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
        if hasattr(record, "data"):
            entry["data"] = record.data
        return json.dumps(entry, ensure_ascii=False)

_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(_JSONFormatter())
log = logging.getLogger("format_chatml")
if not log.handlers:
    log.addHandler(_h)
log.setLevel(logging.INFO)


NER_CATEGORIES = [
    "disease_or_syndrome", "sign_or_symptom", "pharmacologic_substance",
    "laboratory_or_test_result", "diagnostic_procedure", "organism_or_virus",
]

# Orçamento de caracteres do texto clínico (ver format_gemma.py para detalhes).
TEXT_BUDGET_NER = 9000
TEXT_BUDGET_SOAP = 2200


def _truncate_head_tail(text: str, max_chars: int) -> str:
    """Trunca texto preservando começo e fim (HPI + Assessment/Plan)."""
    if len(text) <= max_chars:
        return text
    half = (max_chars - 20) // 2
    return text[:half] + "\n[...]\n" + text[-half:]


def _load_records(silver_path: Path, ids_path: Path) -> list[dict]:
    """Carrega notas do silver filtradas pelos IDs do split."""
    with open(silver_path, encoding="utf-8") as f:
        silver = json.load(f)
    with open(ids_path, encoding="utf-8") as f:
        ids = set(json.load(f))
    return [r for r in silver if r["paciente_id"] in ids]


def _get_original_texts(parquet_path: Path, patient_ids: set[str]) -> dict[str, str]:
    """Carrega textos originais do parquet."""
    import polars as pl
    df = pl.read_parquet(str(parquet_path))
    df_f = df.filter(pl.col("hadm_id").cast(pl.Utf8).is_in(patient_ids))
    return {str(row["hadm_id"]): row.get("text", "") for row in df_f.to_dicts()}


def record_to_chatml_ner(record: dict, original_text: str) -> dict:
    """Converte um registro para formato ChatML para treino de NER."""
    ner = record.get("ner", {})
    ner_output = {cat: ner.get(cat, []) for cat in NER_CATEGORIES}
    text_trunc = _truncate_head_tail(original_text, TEXT_BUDGET_NER)
    return {
        "messages": [
            {"role": "system", "content": PROMPT_NER_TRAIN},
            {"role": "user", "content": text_trunc},
            {"role": "assistant", "content": json.dumps(ner_output, ensure_ascii=False)},
        ],
        "hadm_id": record["paciente_id"],
        "task": "ner",
    }


def record_to_chatml_soap(record: dict, original_text: str) -> dict:
    """Converte um registro para formato ChatML para treino de SOAP."""
    soap = record.get("soap", {})
    text_trunc = _truncate_head_tail(original_text, TEXT_BUDGET_SOAP)
    return {
        "messages": [
            {"role": "system", "content": PROMPT_SOAP_TRAIN},
            {"role": "user", "content": text_trunc},
            {"role": "assistant", "content": json.dumps(soap, ensure_ascii=False)},
        ],
        "hadm_id": record["paciente_id"],
        "task": "soap",
    }


def main():
    parser = argparse.ArgumentParser(description="IANA — Formato ChatML para Qwen")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--silver", default=str(_EXPERIMENTS_DIR / "resultados" / "banco_dados_iana_v3_clean.json"))
    parser.add_argument("--splits-dir", default=str(_TRAINING_DIR / "data" / "splits"))
    parser.add_argument("--parquet", default=str(_EXPERIMENTS_DIR / "dados" / "mimic_filtrado_tb_hiv_sifilis.parquet"))
    parser.add_argument("--output-dir", default=str(_TRAINING_DIR / "data" / "chatml"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    silver_path = Path(args.silver)
    splits_dir = Path(args.splits_dir)
    parquet_path = Path(args.parquet)
    output_dir = Path(args.output_dir)

    if args.smoke_test:
        # Smoke test: 3 notas do silver com seed=42
        with open(silver_path, encoding="utf-8") as f:
            all_records = json.load(f)
        random.seed(42)
        records = random.sample(all_records, min(3, len(all_records)))
        log.info("Smoke test: 3 notas selecionadas")
    else:
        ids_path = splits_dir / f"{args.split}_ids.json"
        records = _load_records(silver_path, ids_path)

    # Carrega textos originais
    pids = {r["paciente_id"] for r in records}
    texts = _get_original_texts(parquet_path, pids)

    # Converte
    examples = []
    for r in records:
        pid = r["paciente_id"]
        text = texts.get(pid, "")
        if not text:
            log.warning(f"Texto não encontrado para {pid}")
            continue
        examples.append(record_to_chatml_ner(r, text))
        examples.append(record_to_chatml_soap(r, text))

    log.info("Conversão concluída", extra={"data": {
        "records": len(records), "examples": len(examples),
        "ner": sum(1 for e in examples if e["task"] == "ner"),
        "soap": sum(1 for e in examples if e["task"] == "soap"),
    }})

    if args.smoke_test:
        # Valida formato
        errors = 0
        for ex in examples:
            if "messages" not in ex:
                errors += 1
                continue
            msgs = ex["messages"]
            if len(msgs) != 3:
                errors += 1
                continue
            if msgs[0]["role"] != "system" or msgs[1]["role"] != "user" or msgs[2]["role"] != "assistant":
                errors += 1
                continue
            try:
                json.loads(msgs[2]["content"])
            except json.JSONDecodeError:
                errors += 1
        log.info("Smoke test resultado", extra={"data": {
            "total": len(examples), "errors": errors,
            "status": "PASS" if errors == 0 else "FAIL",
        }})
        return

    # Salva
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.split}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info("Salvo", extra={"data": {"path": str(out_path), "lines": len(examples)}})


if __name__ == "__main__":
    main()
