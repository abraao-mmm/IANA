#!/usr/bin/env python3
"""
Projeto IANA - Validacao automatizada de qualidade da extracao NER.

Implementa 8 checks de qualidade que verificam os erros sistematicos
identificados na analise qualitativa das 3 amostras representativas.

Uso:
    python validate_extraction_quality.py
    python validate_extraction_quality.py --json resultados/banco_dados_iana_oficial.json
    python validate_extraction_quality.py --json resultados/teste_10_notas.json --verbose

Checks:
    1. Cross-category duplication (mesma entidade em 2+ categorias)
    2. Negative test leakage (teste negativo gerando entidade positiva)
    3. Pending test leakage (teste pendente gerando entidade positiva)
    4. Symptom in disease_or_syndrome (sintoma na categoria errada)
    5. Imaging finding in disease_or_syndrome (achado de imagem como diagnostico)
    6. Intra-category synonym duplication (sinonimos duplicados)
    7. Language violation (termos em ingles que deveriam estar em portugues)
    8. Vital sign in laboratory (SpO2 etc. em lab em vez de exame fisico)
"""

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

# ---------------------------------------------------------------------------
# Logging
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
log = logging.getLogger("validate_quality")
log.addHandler(_handler)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    no_acc = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_acc.strip().lower())


# ---------------------------------------------------------------------------
# Check definitions
# ---------------------------------------------------------------------------

NER_CATEGORIES = [
    "disease_or_syndrome",
    "sign_or_symptom",
    "pharmacologic_substance",
    "laboratory_or_test_result",
    "diagnostic_procedure",
    "organism_or_virus",
]

# Common symptoms that should NOT be in disease_or_syndrome
_SYMPTOM_TERMS = {
    "febre", "tosse", "dispneia", "dor toracica", "dor abdominal",
    "nausea", "vomito", "diarreia", "cefaleia", "fadiga", "mal estar",
    "calafrios", "sudorese noturna", "perda de peso", "perda de apetite",
    "mialgia", "artralgia", "dor", "edema", "prurido", "rash",
    "tosse seca", "tosse produtiva", "hemoptise", "ortopneia",
    "dor toracica pleuritica", "aperto toracico", "dormencia",
    "parestesia", "fraqueza", "tontura", "sincope",
}

# Imaging findings that should NOT be in disease_or_syndrome
# NÃO incluir termos ambíguos como "massa" e "lesão" (podem ser diagnósticos clínicos)
_IMAGING_TERMS = {
    "atelectasia", "edema pulmonar", "derrame pleural", "derrame pericardico",
    "consolidacao", "opacidade", "infiltrado", "nodulo pulmonar",
    "linfonodo aumentado", "linfonodo hilar", "cicatrizacao biapical",
    "espessamento pleural", "pneumotorax", "cardiomegalia",
    "calcificacao", "efusao",
}

# Vital signs that should NOT be in laboratory_or_test_result
_VITAL_SIGN_PATTERNS = re.compile(
    r"(?i)\b(spo2|sao2|saturacao de o2|oximetria|temperatura|"
    r"pressao arterial|pa |fc |fr |frequencia cardiaca|"
    r"frequencia respiratoria)\b"
)

# Negation markers in lab results
_NEGATIVE_MARKERS = re.compile(
    r"(?i)(negativ[oe]|negative|sem crescimento|no growth|"
    r"not detected|nao detectado|nao reativ[oa]|non-reactive)"
)

# Pending markers
_PENDING_MARKERS = re.compile(
    r"(?i)(pendente|pending|pnd|indeterminad[oa]|indeterminate)"
)

# English terms that should be in Portuguese
# NÃO incluir termos válidos em PT-BR: edema, eritema, dispneia, taquicardia
_ENGLISH_PATTERNS = re.compile(
    r"\b(fever|cough|chest pain|shortness of breath|nausea|vomiting|"
    r"diarrhea|headache|fatigue|weakness|numbness|tightness|"
    r"weight loss|night sweats|chills|rash|swelling|"
    r"abdominal pain|back pain|joint pain|muscle pain|"
    r"runny nose|sore throat|wheezing)\b",
    re.IGNORECASE,
)

# Synonym pairs for intra-category dedup check
_SYNONYM_GROUPS: list[set[str]] = [
    {"hipertensao", "hipertensao arterial", "hipertensao arterial sistemica", "has"},
    {"hiperlipidemia", "dislipidemia"},
    {"diabetes", "diabetes mellitus"},
    {"drge", "doenca do refluxo gastroesofagico", "doenca do refluxo gastroesofagico (drge)"},
    {"aids", "sindrome da imunodeficiencia adquirida", "aids (sindrome da imunodeficiencia adquirida)"},
    {"sindrome de raynaud", "fenomeno de raynaud", "doenca de raynaud"},
    {"insuficiencia renal cronica", "doenca renal cronica"},
    {"insuficiencia renal aguda", "lesao renal aguda"},
    {"embolia pulmonar", "tromboembolismo pulmonar"},
    {"insuficiencia cardiaca", "insuficiencia cardiaca congestiva"},
    {"dpoc", "doenca pulmonar obstrutiva cronica", "doenca pulmonar obstrutiva cronica (dpoc)"},
]


def _check_cross_category_duplication(ner: dict) -> list[dict]:
    """Check 1: mesma entidade em 2+ categorias."""
    issues = []
    entity_map: dict[str, list[str]] = defaultdict(list)

    for cat in NER_CATEGORIES:
        for item in ner.get(cat, []):
            entity_map[_norm(item)].append(cat)

    for key, cats in entity_map.items():
        if len(cats) > 1:
            issues.append({
                "check": "cross_category_duplication",
                "entity": key,
                "categories": cats,
            })
    return issues


def _check_negative_test_leakage(ner: dict) -> list[dict]:
    """Check 2: teste negativo gerando entidade em disease/organism."""
    issues = []
    lab_items = ner.get("laboratory_or_test_result", [])
    negative_tests = [item for item in lab_items if _NEGATIVE_MARKERS.search(item)]

    disease_norms = {_norm(d) for d in ner.get("disease_or_syndrome", [])}
    organism_norms = {_norm(o) for o in ner.get("organism_or_virus", [])}

    for test in negative_tests:
        # Extract test name (before the negative marker)
        name = _NEGATIVE_MARKERS.split(test)[0].strip(" -–—:")
        name_norm = _norm(name)
        if name_norm in disease_norms:
            issues.append({
                "check": "negative_test_in_disease",
                "test": test,
                "leaked_to": "disease_or_syndrome",
                "leaked_entity": name,
            })
        if name_norm in organism_norms:
            issues.append({
                "check": "negative_test_in_organism",
                "test": test,
                "leaked_to": "organism_or_virus",
                "leaked_entity": name,
            })
    return issues


def _check_pending_test_leakage(ner: dict) -> list[dict]:
    """Check 3: teste pendente gerando entidade em disease/organism."""
    issues = []
    lab_items = ner.get("laboratory_or_test_result", [])
    pending_tests = [item for item in lab_items if _PENDING_MARKERS.search(item)]

    disease_norms = {_norm(d) for d in ner.get("disease_or_syndrome", [])}
    organism_norms = {_norm(o) for o in ner.get("organism_or_virus", [])}

    for test in pending_tests:
        name = _PENDING_MARKERS.split(test)[0].strip(" -–—:")
        name_norm = _norm(name)
        if name_norm in disease_norms:
            issues.append({
                "check": "pending_test_in_disease",
                "test": test,
                "leaked_entity": name,
            })
        if name_norm in organism_norms:
            issues.append({
                "check": "pending_test_in_organism",
                "test": test,
                "leaked_entity": name,
            })
    return issues


def _check_symptom_in_disease(ner: dict) -> list[dict]:
    """Check 4: sintomas comuns na categoria disease_or_syndrome."""
    issues = []
    for item in ner.get("disease_or_syndrome", []):
        if _norm(item) in _SYMPTOM_TERMS:
            issues.append({
                "check": "symptom_in_disease",
                "entity": item,
                "should_be": "sign_or_symptom",
            })
    return issues


def _check_imaging_finding_in_disease(ner: dict, soap: dict | None = None) -> list[dict]:
    """Check 5: achados de imagem na categoria disease_or_syndrome.

    Se o termo também aparece na avaliação ou exame físico do SOAP,
    é um diagnóstico clínico confirmado e não é flagado.
    """
    issues = []
    # Texto do SOAP para verificar se é diagnóstico real
    soap_clinical = ""
    if soap:
        soap_clinical = " ".join([
            soap.get("avaliacao", ""),
            soap.get("objetivo_exame_fisico", ""),
            soap.get("subjetivo", ""),
        ]).lower()

    for item in ner.get("disease_or_syndrome", []):
        norm = _norm(item)
        for imaging_term in _IMAGING_TERMS:
            if imaging_term in norm:
                # Se o termo aparece na avaliação/exame do SOAP, é diagnóstico real
                if soap_clinical and norm in soap_clinical:
                    break  # não é violação
                issues.append({
                    "check": "imaging_finding_in_disease",
                    "entity": item,
                    "should_be": "soap.objetivo_imagem",
                })
                break
    return issues


def _check_synonym_duplication(ner: dict) -> list[dict]:
    """Check 6: sinonimos duplicados dentro da mesma categoria."""
    issues = []
    for cat in NER_CATEGORIES:
        items_norm = [_norm(item) for item in ner.get(cat, [])]
        for group in _SYNONYM_GROUPS:
            found = [item for item in items_norm if item in group]
            if len(found) > 1:
                issues.append({
                    "check": "synonym_duplication",
                    "category": cat,
                    "synonyms": found,
                })
    return issues


def _check_language_violation(ner: dict) -> list[dict]:
    """Check 7: termos em ingles que deveriam estar em portugues."""
    issues = []
    for cat in NER_CATEGORIES:
        for item in ner.get(cat, []):
            matches = _ENGLISH_PATTERNS.findall(item)
            if matches:
                issues.append({
                    "check": "language_violation",
                    "category": cat,
                    "entity": item,
                    "english_terms": matches,
                })
    return issues


def _check_vital_sign_in_lab(ner: dict) -> list[dict]:
    """Check 8: sinais vitais em laboratory_or_test_result."""
    issues = []
    for item in ner.get("laboratory_or_test_result", []):
        if _VITAL_SIGN_PATTERNS.search(item):
            issues.append({
                "check": "vital_sign_in_lab",
                "entity": item,
                "should_be": "soap.objetivo_exame_fisico",
            })
    return issues


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    _check_cross_category_duplication,
    _check_negative_test_leakage,
    _check_pending_test_leakage,
    _check_symptom_in_disease,
    _check_imaging_finding_in_disease,
    _check_synonym_duplication,
    _check_language_violation,
    _check_vital_sign_in_lab,
]


def _run_check(check_fn, ner: dict, soap: dict | None = None) -> list[dict]:
    """Executa um check, passando soap se a assinatura aceitar."""
    import inspect
    sig = inspect.signature(check_fn)
    if "soap" in sig.parameters:
        return check_fn(ner, soap=soap)
    return check_fn(ner)


def validate_record(record: dict) -> dict:
    """Roda os 8 checks em um único registro e retorna relatório."""
    ner = record.get("ner", {})
    soap = record.get("soap", {})
    issues: list[dict] = []
    for check_fn in ALL_CHECKS:
        issues.extend(_run_check(check_fn, ner, soap))

    return {
        "paciente_id": record.get("paciente_id", "___"),
        "doenca_alvo": record.get("doenca_alvo_identificada", ""),
        "total_issues": len(issues),
        "issues": issues,
        "check_summary": {
            check_fn.__name__.replace("_check_", ""): len(_run_check(check_fn, ner, soap))
            for check_fn in ALL_CHECKS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IANA - Validacao de qualidade da extracao NER",
    )
    parser.add_argument(
        "--json",
        default="resultados/banco_dados_iana_oficial.json",
        help="Caminho do JSON de resultados.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Caminho do JSON de relatorio (default: stdout).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostra detalhes de cada issue por nota.",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.is_absolute():
        json_path = _EXPERIMENTS_DIR / json_path

    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    log.info("Validando", extra={"data": {"total_records": len(records)}})

    reports: list[dict] = []
    total_issues = 0
    check_totals: dict[str, int] = defaultdict(int)

    for record in records:
        report = validate_record(record)
        reports.append(report)
        total_issues += report["total_issues"]
        for check_name, count in report["check_summary"].items():
            check_totals[check_name] += count

    # Resumo global
    clean_records = sum(1 for r in reports if r["total_issues"] == 0)

    summary = {
        "total_records": len(records),
        "clean_records": clean_records,
        "records_with_issues": len(records) - clean_records,
        "total_issues": total_issues,
        "issues_per_check": dict(check_totals),
    }

    log.info("Validacao concluida", extra={"data": summary})

    if args.verbose:
        for report in reports:
            if report["total_issues"] > 0:
                log.info(
                    f"Issues for {report['paciente_id']}",
                    extra={"data": report},
                )

    # Salvar relatorio
    output = {
        "summary": summary,
        "reports": reports,
    }

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = _EXPERIMENTS_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        log.info("Relatorio salvo", extra={"data": {"path": str(out_path)}})
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
