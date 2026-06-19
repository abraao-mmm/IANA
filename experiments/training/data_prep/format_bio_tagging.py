#!/usr/bin/env python3
"""
Projeto IANA — Converte silver standard para formato BIO tagging (BioBERTpt).

Tokeniza com o tokenizer do BioBERTpt, alinha entidades NER com tokens
usando BIO scheme, e aplica chunking de 512 tokens com overlap de 50.

Uso:
    python format_bio_tagging.py
    python format_bio_tagging.py --smoke-test
"""

import argparse
import json
import logging
import random
import re
import sys
import unicodedata
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _TRAINING_DIR.parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))


class _JSONFormatter(logging.Formatter):
    def format(self, record):
        entry = {"ts": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
        if hasattr(record, "data"):
            entry["data"] = record.data
        return json.dumps(entry, ensure_ascii=False)

_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(_JSONFormatter())
log = logging.getLogger("format_bio")
if not log.handlers:
    log.addHandler(_h)
log.setLevel(logging.INFO)


NER_CATEGORIES = [
    "disease_or_syndrome", "sign_or_symptom", "pharmacologic_substance",
    "laboratory_or_test_result", "diagnostic_procedure", "organism_or_virus",
]

# Short labels for BIO tags
CATEGORY_SHORT = {
    "disease_or_syndrome": "DISEASE",
    "sign_or_symptom": "SYMPTOM",
    "pharmacologic_substance": "MEDICATION",
    "laboratory_or_test_result": "LAB",
    "diagnostic_procedure": "PROCEDURE",
    "organism_or_virus": "ORGANISM",
}


def _norm_for_match(text: str) -> str:
    """Normaliza para matching fuzzy."""
    nfkd = unicodedata.normalize("NFKD", text)
    no_acc = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_acc.strip().lower())


def _find_entity_spans(text: str, entities_by_cat: dict[str, list[str]]) -> list[tuple[int, int, str]]:
    """Encontra spans (start, end, label) no texto para cada entidade."""
    text_lower = text.lower()
    spans = []
    for cat, entities in entities_by_cat.items():
        label = CATEGORY_SHORT[cat]
        for entity in entities:
            entity_lower = entity.lower()
            start = text_lower.find(entity_lower)
            if start >= 0:
                spans.append((start, start + len(entity_lower), label))
    # Ordena e remove overlaps (mantém o mais longo)
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered = []
    last_end = -1
    for s, e, label in spans:
        if s >= last_end:
            filtered.append((s, e, label))
            last_end = e
    return filtered


def _align_tokens_to_bio(tokens: list[str], offsets: list[tuple[int, int]],
                          spans: list[tuple[int, int, str]]) -> list[str]:
    """Alinha tokens com spans de entidades usando BIO scheme."""
    labels = ["O"] * len(tokens)
    for span_start, span_end, label in spans:
        in_entity = False
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= span_start:
                continue
            if tok_start >= span_end:
                break
            if not in_entity:
                labels[i] = f"B-{label}"
                in_entity = True
            else:
                labels[i] = f"I-{label}"
    return labels


def _chunk_with_overlap(tokens: list[str], labels: list[str],
                         max_len: int = 512, overlap: int = 50) -> list[dict]:
    """Divide sequência em chunks com overlap."""
    chunks = []
    step = max_len - overlap
    for start in range(0, len(tokens), step):
        end = min(start + max_len, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_labels = labels[start:end]
        if len(chunk_tokens) > 0:
            chunks.append({"tokens": chunk_tokens, "labels": chunk_labels})
        if end >= len(tokens):
            break
    return chunks


def _get_original_texts(parquet_path: Path, pids: set[str]) -> dict[str, str]:
    import polars as pl
    df = pl.read_parquet(str(parquet_path))
    df_f = df.filter(pl.col("hadm_id").cast(pl.Utf8).is_in(pids))
    return {str(row["hadm_id"]): row.get("text", "") for row in df_f.to_dicts()}


def process_record(record: dict, text: str, tokenizer, max_len: int = 512,
                    overlap: int = 50) -> list[dict]:
    """Processa um registro completo para BIO tagging."""
    ner = record.get("ner", {})
    entities_by_cat = {cat: ner.get(cat, []) for cat in NER_CATEGORIES}

    # Tokeniza
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False,
                         truncation=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]

    # Encontra spans
    spans = _find_entity_spans(text, entities_by_cat)

    # Alinha com BIO
    labels = _align_tokens_to_bio(tokens, offsets, spans)

    # Chunk
    chunks = _chunk_with_overlap(tokens, labels, max_len, overlap)

    # Adiciona metadados
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"{record['paciente_id']}_{i}"
        chunk["hadm_id"] = record["paciente_id"]

    return chunks


def main():
    parser = argparse.ArgumentParser(description="IANA — Formato BIO para BioBERTpt")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--silver", default=str(_EXPERIMENTS_DIR / "resultados" / "banco_dados_iana_v3_clean.json"))
    parser.add_argument("--splits-dir", default=str(_TRAINING_DIR / "data" / "splits"))
    parser.add_argument("--parquet", default=str(_EXPERIMENTS_DIR / "dados" / "mimic_filtrado_tb_hiv_sifilis.parquet"))
    parser.add_argument("--output-dir", default=str(_TRAINING_DIR / "data" / "bio_tagging"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
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

    # Tenta carregar tokenizer (smoke test pode falhar sem transformers)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("pucpr/biobertpt-clin")
    except Exception as e:
        if args.smoke_test:
            # Fallback: tokenizer simples por whitespace para smoke test
            log.warning("Tokenizer não disponível, usando whitespace split para smoke test")

            class _FakeTokenizer:
                def __call__(self, text, **kwargs):
                    words = text.split()
                    offsets = []
                    pos = 0
                    for w in words:
                        start = text.find(w, pos)
                        offsets.append((start, start + len(w)))
                        pos = start + len(w)
                    return {"input_ids": list(range(len(words))), "offset_mapping": offsets}
                def convert_ids_to_tokens(self, ids):
                    return [f"tok_{i}" for i in ids]

            tokenizer = _FakeTokenizer()
        else:
            raise

    all_chunks = []
    for r in records:
        pid = r["paciente_id"]
        text = texts.get(pid, "")
        if not text:
            continue
        chunks = process_record(r, text, tokenizer, args.max_length, args.overlap)
        all_chunks.extend(chunks)

    log.info("Conversão concluída", extra={"data": {
        "records": len(records), "chunks": len(all_chunks)}})

    if args.smoke_test:
        # Valida BIO tags
        valid_prefixes = {"B-", "I-", "O"}
        errors = 0
        for chunk in all_chunks:
            for label in chunk["labels"]:
                if label != "O" and label[:2] not in {"B-", "I-"}:
                    errors += 1
        log.info("Smoke test", extra={"data": {
            "chunks": len(all_chunks), "label_errors": errors,
            "status": "PASS" if errors == 0 else "FAIL"}})
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.split}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info("Salvo", extra={"data": {"path": str(out_path), "lines": len(all_chunks)}})


if __name__ == "__main__":
    main()
