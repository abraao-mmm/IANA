#!/usr/bin/env python3
"""
Projeto IANA — Seleção estratificada de candidatas para teste do pipeline v3.

Gera um inventário com 3-5 candidatas para cada uma das 6 categorias de teste,
classificadas por complexidade clínica. O orientador escolhe as 10 finais.

Categorias alvo:
  - 2 notas HIV "complexas" (múltiplos diagnósticos, testes negativos, HAART)
  - 2 notas HIV "simples" (diagnóstico claro, poucas comorbidades)
  - 2 notas Sífilis com cobertura adequada (≥3 menções)
  - 1 nota Sífilis com cobertura mínima/zero (estilo Amostra 2)
  - 2 notas TB "complexas" (testes pendentes Quantiferon/PPD/AFB)
  - 1 nota TB "simples" (diagnóstico confirmado)

Pré-requisito: rodar audit_text_coverage.py primeiro para gerar coverage_audit.csv.

Uso:
    python select_test_samples.py
    python select_test_samples.py --coverage resultados/coverage_audit.csv
    python select_test_samples.py --parquet dados/mimic_filtrado_tb_hiv_sifilis.parquet
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))


# ---------------------------------------------------------------------------
# Logging estruturado JSON
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if hasattr(record, "data"):
            entry["data"] = record.data
        return json.dumps(entry, ensure_ascii=False)


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_JSONFormatter())
log = logging.getLogger("select_samples")
log.addHandler(_handler)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Regex de complexidade
# ---------------------------------------------------------------------------

_NEGATION_RE = re.compile(
    r"\b(negative|negativ[oe]|ruled out|no evidence|no history|denies|"
    r"not detected|non-reactive|no growth|sem crescimento)\b",
    re.IGNORECASE,
)

_PENDING_RE = re.compile(
    r"\b(pending|pnd|indeterminate|to be determined|awaiting)\b",
    re.IGNORECASE,
)

_DIFFERENTIAL_RE = re.compile(
    r"\b(differential diagnosis|ddx|rule out|r/o|consider|"
    r"suspected|possible|probable|cannot exclude)\b",
    re.IGNORECASE,
)

_HAART_RE = re.compile(
    r"\b(haart|antiretroviral|art regimen|truvada|biktarvy|"
    r"ritonavir|darunavir|dolutegravir|raltegravir|tenofovir|"
    r"emtricitabine|abacavir|lamivudine|efavirenz|atazanavir|"
    r"lopinavir|elvitegravir|nevirapine)\b",
    re.IGNORECASE,
)

_MEDICATION_RE = re.compile(
    r"\b(mg|mcg|units|tablet|capsule|oral|iv|inhaler|"
    r"daily|bid|tid|qid|prn|q\d+h)\b",
    re.IGNORECASE,
)

_TB_TESTS_RE = re.compile(
    r"\b(quantiferon|ppd|igra|afb|acid.fast|mantoux|"
    r"sputum.*culture|bam.*culture|mycobacterium.*culture)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Análise de complexidade
# ---------------------------------------------------------------------------

def analyze_complexity(text: str, disease: str) -> dict:
    """Computa métricas de complexidade para uma nota clínica."""
    num_negations = len(_NEGATION_RE.findall(text))
    num_pending = len(_PENDING_RE.findall(text))
    num_differentials = len(_DIFFERENTIAL_RE.findall(text))
    num_medications = len(_MEDICATION_RE.findall(text))
    has_haart = bool(_HAART_RE.search(text))
    has_tb_tests = bool(_TB_TESTS_RE.search(text))
    char_count = len(text)

    # Score composto de complexidade (maior = mais complexo)
    complexity_score = (
        num_negations * 2
        + num_pending * 3
        + num_differentials * 2
        + (num_medications // 5)  # cada 5 medicações = +1
        + (10 if has_haart else 0)
        + (10 if has_tb_tests else 0)
        + (5 if char_count > 15000 else 0)
        + (3 if char_count > 10000 else 0)
    )

    return {
        "char_count": char_count,
        "num_negations": num_negations,
        "num_pending_tests": num_pending,
        "num_differentials": num_differentials,
        "num_medications": num_medications,
        "has_haart": has_haart,
        "has_tb_tests": has_tb_tests,
        "complexity_score": complexity_score,
    }


# ---------------------------------------------------------------------------
# Seleção de candidatas
# ---------------------------------------------------------------------------

def select_candidates(
    notes: list[dict],
    coverage: dict[str, dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Gera candidatas estratificadas para as 6 categorias de teste.

    Args:
        notes: lista de dicts do parquet (hadm_id, doenca_alvo, text)
        coverage: dict {hadm_id: {mention_count, coverage_status}} do CSV de auditoria
        top_n: número de candidatas por categoria

    Returns:
        Lista de dicts com metadados e categoria sugerida.
    """
    # Enriquecer notas com métricas
    enriched: list[dict] = []
    for note in notes:
        hadm_id = str(note.get("hadm_id", "___"))
        text = note.get("text", "")
        disease = str(note.get("doenca_alvo", ""))
        cov = coverage.get(hadm_id, {})

        metrics = analyze_complexity(text, disease)
        metrics["paciente_id"] = hadm_id
        metrics["doenca_alvo"] = disease
        metrics["mention_count"] = cov.get("mention_count", 0)
        metrics["coverage_status"] = cov.get("coverage_status", "unknown")
        enriched.append(metrics)

    candidates: list[dict] = []

    # --- HIV complexas ---
    hiv_notes = [n for n in enriched if "hiv" in n["doenca_alvo"].lower()]
    hiv_complex = sorted(
        [n for n in hiv_notes if n["coverage_status"] == "adequate"],
        key=lambda x: x["complexity_score"],
        reverse=True,
    )
    for n in hiv_complex[:top_n]:
        candidates.append({**n, "suggested_category": "HIV_complexa"})

    # --- HIV simples ---
    hiv_simple = sorted(
        [n for n in hiv_notes if n["coverage_status"] == "adequate"],
        key=lambda x: x["complexity_score"],
    )
    for n in hiv_simple[:top_n]:
        candidates.append({**n, "suggested_category": "HIV_simples"})

    # --- Sífilis com cobertura adequada ---
    sif_notes = [n for n in enriched if "sifilis" in n["doenca_alvo"].lower()
                 or "syphilis" in n["doenca_alvo"].lower()]
    sif_adequate = sorted(
        [n for n in sif_notes if n["mention_count"] >= 3],
        key=lambda x: x["complexity_score"],
        reverse=True,
    )
    for n in sif_adequate[:top_n]:
        candidates.append({**n, "suggested_category": "Sifilis_adequada"})

    # --- Sífilis com cobertura mínima/zero ---
    sif_poor = sorted(
        [n for n in sif_notes if n["mention_count"] <= 2],
        key=lambda x: x["mention_count"],
    )
    for n in sif_poor[:top_n]:
        candidates.append({**n, "suggested_category": "Sifilis_minima"})

    # --- TB complexas ---
    tb_notes = [n for n in enriched if "tuberculose" in n["doenca_alvo"].lower()
                or "tuberculosis" in n["doenca_alvo"].lower()
                or n["doenca_alvo"].lower() == "tb"]
    tb_complex = sorted(
        [n for n in tb_notes if n["coverage_status"] == "adequate" and n["has_tb_tests"]],
        key=lambda x: x["complexity_score"],
        reverse=True,
    )
    # Fallback: se poucas com has_tb_tests, pega as mais complexas gerais
    if len(tb_complex) < top_n:
        tb_complex_fallback = sorted(
            [n for n in tb_notes if n["coverage_status"] == "adequate"],
            key=lambda x: x["complexity_score"],
            reverse=True,
        )
        for n in tb_complex_fallback:
            if n not in tb_complex:
                tb_complex.append(n)
            if len(tb_complex) >= top_n:
                break

    for n in tb_complex[:top_n]:
        candidates.append({**n, "suggested_category": "TB_complexa"})

    # --- TB simples ---
    tb_simple = sorted(
        [n for n in tb_notes if n["coverage_status"] == "adequate"],
        key=lambda x: x["complexity_score"],
    )
    for n in tb_simple[:top_n]:
        candidates.append({**n, "suggested_category": "TB_simples"})

    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IANA — Seleção estratificada de candidatas para teste v3",
    )
    parser.add_argument(
        "--coverage",
        default="resultados/coverage_audit.csv",
        help="CSV de auditoria de cobertura textual (gerado por audit_text_coverage.py).",
    )
    parser.add_argument(
        "--parquet",
        default="dados/mimic_filtrado_tb_hiv_sifilis.parquet",
        help="Parquet com as notas clínicas.",
    )
    parser.add_argument(
        "--output",
        default="resultados/test_sample_candidates.csv",
        help="CSV de saída com candidatas.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Número de candidatas por categoria (default: 5).",
    )
    args = parser.parse_args()

    # Resolve caminhos
    coverage_path = Path(args.coverage)
    if not coverage_path.is_absolute():
        coverage_path = _EXPERIMENTS_DIR / coverage_path

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = _EXPERIMENTS_DIR / parquet_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _EXPERIMENTS_DIR / output_path

    # Carrega CSV de cobertura
    coverage: dict[str, dict] = {}
    with open(coverage_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coverage[row["paciente_id"]] = {
                "mention_count": int(row["mention_count"]),
                "coverage_status": row["coverage_status"],
            }

    log.info("Cobertura carregada", extra={"data": {"notas": len(coverage)}})

    # Carrega parquet
    import polars as pl
    df = pl.read_parquet(str(parquet_path))
    notes = df.to_dicts()
    log.info("Parquet carregado", extra={"data": {"notas": len(notes)}})

    # Seleciona candidatas
    candidates = select_candidates(notes, coverage, top_n=args.top_n)
    log.info("Candidatas selecionadas", extra={"data": {"total": len(candidates)}})

    # Conta por categoria
    from collections import Counter
    cat_counts = Counter(c["suggested_category"] for c in candidates)
    log.info("Distribuição por categoria", extra={"data": dict(cat_counts)})

    # Salva CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paciente_id", "doenca_alvo", "suggested_category",
        "complexity_score", "mention_count", "coverage_status",
        "char_count", "num_negations", "num_pending_tests",
        "num_differentials", "num_medications",
        "has_haart", "has_tb_tests",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)

    log.info("CSV salvo", extra={"data": {"path": str(output_path)}})


if __name__ == "__main__":
    main()
