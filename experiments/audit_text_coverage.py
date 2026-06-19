#!/usr/bin/env python3
"""
Projeto IANA - Auditoria de cobertura textual do dataset.

Verifica se cada nota clinica selecionada realmente menciona a doenca alvo
no corpo do texto (nao apenas no codigo ICD). Notas com zero mencoes indicam
que o filtro seq_num==1 nao garantiu relevancia narrativa.

Uso:
    python audit_text_coverage.py
    python audit_text_coverage.py --parquet dados/mimic_filtrado_tb_hiv_sifilis.parquet
    python audit_text_coverage.py --output coverage_audit.csv
"""

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
# pyrefly: ignore [missing-import]
import polars as pl


_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

# ---------------------------------------------------------------------------
# Logging estruturado em JSON
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if hasattr(record, "data"):
            entry["data"] = record.data  # type: ignore[attr-defined]
        return json.dumps(entry, ensure_ascii=False)


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_JSONFormatter())
log = logging.getLogger("audit_coverage")
log.addHandler(_handler)
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Regex por doenca alvo
# ---------------------------------------------------------------------------

DISEASE_PATTERNS: dict[str, re.Pattern] = {
    "HIV": re.compile(
        r"\b("
        r"hiv|aids|human immunodeficiency"
        r"|cd4|antiretroviral|haart|\bart\b"
        r"|truvada|ritonavir|darunavir|raltegravir"
        r"|dolutegravir|tenofovir|emtricitabine"
        r"|abacavir|lamivudine|efavirenz|nevirapine"
        r"|atazanavir|lopinavir|elvitegravir|biktarvy"
        r")\b",
        re.IGNORECASE,
    ),
    "Tuberculose": re.compile(
        r"\b("
        r"tuberculosis|tuberculous|\btb\b"
        r"|mycobacterium|ppd|igra|quantiferon"
        r"|afb|isoniazid|rifampin|rifampicin"
        r"|ethambutol|pyrazinamide|\binh\b|\brip\b|\bmtb\b"
        r"|mantoux|ghon"
        r")\b",
        re.IGNORECASE,
    ),
    "Sifilis": re.compile(
        r"\b("
        r"syphilis|treponema|rpr|vdrl"
        r"|fta-abs|penicillin g benzathine"
        r"|neurosyphilis|chancre|gumma"
        r"|treponemal|nontreponemal"
        r")\b",
        re.IGNORECASE,
    ),
}

# Mapeamento doenca_alvo do parquet para chave do DISEASE_PATTERNS
_ALIAS = {
    "hiv": "HIV",
    "tuberculose": "Tuberculose",
    "tuberculosis": "Tuberculose",
    "sifilis": "Sifilis",
    "syphilis": "Sifilis",
}


def _resolve_disease(raw: str) -> str:
    """Normaliza o nome da doenca para a chave do dicionario de regex."""
    key = raw.strip().lower()
    if key in _ALIAS:
        return _ALIAS[key]
    for canon in DISEASE_PATTERNS:
        if canon.lower() == key:
            return canon
    return raw


# ---------------------------------------------------------------------------
# Header adicionado pelo pipeline — deve ser ignorado na contagem
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(
    r"^={3,}.*?^={3,}",
    re.MULTILINE | re.DOTALL,
)


def _strip_header(text: str) -> str:
    """Remove cabecalho 'DIAGNOSTICO PRINCIPAL CONFIRMADO' se presente."""
    return _HEADER_RE.sub("", text, count=1)


# ---------------------------------------------------------------------------
# Contagem de mencoes
# ---------------------------------------------------------------------------

def count_mentions(text: str, disease: str) -> int:
    """Retorna numero de matches do regex da doenca no texto."""
    pattern = DISEASE_PATTERNS.get(disease)
    if pattern is None:
        return 0
    clean = _strip_header(text)
    return len(pattern.findall(clean))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def classify_coverage(n: int) -> str:
    if n == 0:
        return "zero"
    if n <= 2:
        return "minimal"
    return "adequate"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IANA - Auditoria de cobertura textual do dataset",
    )
    parser.add_argument(
        "--parquet",
        default="dados/mimic_filtrado_tb_hiv_sifilis.parquet",
        help="Caminho do parquet com as notas filtradas.",
    )
    parser.add_argument(
        "--output",
        default="resultados/coverage_audit.csv",
        help="Caminho do CSV de saida.",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = _EXPERIMENTS_DIR / parquet_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _EXPERIMENTS_DIR / output_path

    # Carrega parquet

    df = pl.read_parquet(str(parquet_path))
    log.info("Parquet carregado", extra={"data": {"rows": df.height, "cols": df.columns}})

    # Processa cada nota
    rows: list[dict] = []
    status_counter: Counter[str] = Counter()
    disease_counter: Counter[str] = Counter()
    mention_values: list[int] = []

    for record in df.to_dicts():
        hadm_id = str(record.get("hadm_id", "___"))
        icd_code = str(record.get("icd_code", record.get("codigo_cid", "")))
        raw_disease = str(record.get("doenca_alvo", ""))
        text = record.get("text", "")

        disease = _resolve_disease(raw_disease)
        mentions = count_mentions(text, disease)
        status = classify_coverage(mentions)

        rows.append({
            "paciente_id": hadm_id,
            "codigo_cid": icd_code,
            "doenca_alvo": disease,
            "mention_count": mentions,
            "coverage_status": status,
        })

        status_counter[status] += 1
        disease_counter[disease] += 1
        mention_values.append(mentions)

    # Salva CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "paciente_id", "codigo_cid", "doenca_alvo", "mention_count", "coverage_status",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Relatorio
    total = len(rows)
    zero = status_counter.get("zero", 0)
    minimal = status_counter.get("minimal", 0)
    adequate = status_counter.get("adequate", 0)

    log.info("Auditoria concluida", extra={"data": {
        "total_notas": total,
        "zero_mencoes": zero,
        "minimal_1_2_mencoes": minimal,
        "adequate_3plus_mencoes": adequate,
        "pct_zero": round(zero / total * 100, 1) if total else 0,
        "pct_minimal": round(minimal / total * 100, 1) if total else 0,
        "pct_adequate": round(adequate / total * 100, 1) if total else 0,
    }})

    log.info("Distribuicao por doenca", extra={"data": dict(disease_counter)})

    if mention_values:
        import statistics
        log.info("Estatisticas de mencoes", extra={"data": {
            "media": round(statistics.mean(mention_values), 1),
            "mediana": statistics.median(mention_values),
            "min": min(mention_values),
            "max": max(mention_values),
            "desvio_padrao": round(statistics.stdev(mention_values), 1) if len(mention_values) > 1 else 0,
        }})

    # Histograma textual simplificado
    bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, 500]
    hist: Counter[str] = Counter()
    for v in mention_values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                hist[f"{bins[i]}-{bins[i+1]-1}"] = hist.get(f"{bins[i]}-{bins[i+1]-1}", 0) + 1
                break
        else:
            hist[f"{bins[-1]}+"] = hist.get(f"{bins[-1]}+", 0) + 1

    log.info("Histograma de mencoes", extra={"data": dict(sorted(hist.items()))})

    log.info("CSV salvo", extra={"data": {"path": str(output_path)}})

    # Zero-coverage notes detalhadas (para investigacao)
    zero_notes = [r for r in rows if r["coverage_status"] == "zero"]
    if zero_notes:
        log.warning("Notas com ZERO mencoes da doenca alvo", extra={"data": {
            "count": len(zero_notes),
            "ids": [n["paciente_id"] for n in zero_notes[:20]],
        }})


if __name__ == "__main__":
    main()
