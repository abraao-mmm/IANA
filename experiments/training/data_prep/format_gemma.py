#!/usr/bin/env python3
"""
Projeto IANA — Converte silver standard para formato Gemma instruction-following.

Reutilizado por MedGemma 4B e Gemma 4 E4B (mesma família).
Para cada nota, gera 2 exemplos: NER e SOAP.

Uso:
    python format_gemma.py
    python format_gemma.py --smoke-test
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
log = logging.getLogger("format_gemma")
if not log.handlers:
    log.addHandler(_h)
log.setLevel(logging.INFO)

NER_CATEGORIES = [
    "disease_or_syndrome", "sign_or_symptom", "pharmacologic_substance",
    "laboratory_or_test_result", "diagnostic_procedure", "organism_or_virus",
]

# Orçamento de caracteres do texto clínico (estimativa ~4 chars/token).
# Alvo: max_seq_length=4096 para os 3 decoders.
#   NER:  prompt 203 + resposta_p95 ~1400 + overhead 50 = 1653 tok → sobra 2443 tok ≈ 9700 chars
#   SOAP: prompt 139 + resposta_p95 ~3300 + overhead 50 = 3489 tok → sobra 607 tok ≈ 2400 chars
# Margem de segurança: 9000 e 2200.
TEXT_BUDGET_NER = 9000
TEXT_BUDGET_SOAP = 2200


def _truncate_head_tail(text: str, max_chars: int) -> str:
    """Trunca texto preservando começo e fim (HPI + Assessment/Plan)."""
    if len(text) <= max_chars:
        return text
    half = (max_chars - 20) // 2
    return text[:half] + "\n[...]\n" + text[-half:]


def _get_original_texts(parquet_path: Path, pids: set[str]) -> dict[str, str]:
    import polars as pl
    df = pl.read_parquet(str(parquet_path))
    df_f = df.filter(pl.col("hadm_id").cast(pl.Utf8).is_in(pids))
    return {str(row["hadm_id"]): row.get("text", "") for row in df_f.to_dicts()}


def record_to_gemma_ner(record: dict, original_text: str) -> dict:
    ner = record.get("ner", {})
    ner_output = {cat: ner.get(cat, []) for cat in NER_CATEGORIES}
    text_trunc = _truncate_head_tail(original_text, TEXT_BUDGET_NER)
    return {
        "text": (
            f"<start_of_turn>user\n{PROMPT_NER_TRAIN}\n\n{text_trunc}<end_of_turn>\n"
            f"<start_of_turn>model\n{json.dumps(ner_output, ensure_ascii=False)}<end_of_turn>"
        ),
        "hadm_id": record["paciente_id"],
        "task": "ner",
    }


def record_to_gemma_soap(record: dict, original_text: str) -> dict:
    soap = record.get("soap", {})
    text_trunc = _truncate_head_tail(original_text, TEXT_BUDGET_SOAP)
    return {
        "text": (
            f"<start_of_turn>user\n{PROMPT_SOAP_TRAIN}\n\n{text_trunc}<end_of_turn>\n"
            f"<start_of_turn>model\n{json.dumps(soap, ensure_ascii=False)}<end_of_turn>"
        ),
        "hadm_id": record["paciente_id"],
        "task": "soap",
    }


def main():
    parser = argparse.ArgumentParser(description="IANA — Formato Gemma")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--silver", default=str(_EXPERIMENTS_DIR / "resultados" / "banco_dados_iana_v3_clean.json"))
    parser.add_argument("--splits-dir", default=str(_TRAINING_DIR / "data" / "splits"))
    parser.add_argument("--parquet", default=str(_EXPERIMENTS_DIR / "dados" / "mimic_filtrado_tb_hiv_sifilis.parquet"))
    parser.add_argument("--output-dir", default=str(_TRAINING_DIR / "data" / "gemma_format"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    silver_path = Path(args.silver)

    if args.smoke_test:
        with open(silver_path, encoding="utf-8") as f:
            all_records = json.load(f)
        random.seed(42)
        records = random.sample(all_records, min(3, len(all_records)))
    else:
        splits_dir = Path(args.splits_dir)
        ids_path = splits_dir / f"{args.split}_ids.json"
        with open(silver_path, encoding="utf-8") as f:
            silver = json.load(f)
        with open(ids_path, encoding="utf-8") as f:
            ids = set(json.load(f))
        records = [r for r in silver if r["paciente_id"] in ids]

    pids = {r["paciente_id"] for r in records}
    texts = _get_original_texts(Path(args.parquet), pids)

    examples = []
    for r in records:
        pid = r["paciente_id"]
        text = texts.get(pid, "")
        if not text:
            continue
        examples.append(record_to_gemma_ner(r, text))
        examples.append(record_to_gemma_soap(r, text))

    log.info("Conversão concluída", extra={"data": {
        "records": len(records), "examples": len(examples)}})

    if args.smoke_test:
        errors = 0
        for ex in examples:
            if "text" not in ex:
                errors += 1
                continue
            if "<start_of_turn>user" not in ex["text"]:
                errors += 1
            if "<start_of_turn>model" not in ex["text"]:
                errors += 1
        log.info("Smoke test", extra={"data": {
            "total": len(examples), "errors": errors,
            "status": "PASS" if errors == 0 else "FAIL"}})
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.split}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info("Salvo", extra={"data": {"path": str(out_path), "lines": len(examples)}})


if __name__ == "__main__":
    main()
