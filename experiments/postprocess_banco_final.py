#!/usr/bin/env python3
"""
Projeto IANA — Pós-processamento determinístico do silver standard.

Aplica 5 camadas de correção taxonômica sobre o banco de dados bruto
gerado pelo pipeline LangGraph, sem invocar LLM. As regras são baseadas
em ontologia clínica e são transparentes, reprodutíveis e documentadas.

Uso:
    python postprocess_banco_final.py
    python postprocess_banco_final.py --input resultados/banco_dados_iana_v3.json
    python postprocess_banco_final.py --input X.json --output Y.json --no-validate

Saídas:
    resultados/banco_dados_iana_v3_clean.json   (banco processado)
    logs/postprocess_stats.json                  (estatísticas agregadas)
"""

import argparse
import copy
import json
import logging
import re
import sys
import time
import unicodedata
from collections import defaultdict
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
log = logging.getLogger("postprocess_final")
if not log.handlers:
    log.addHandler(_handler)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Normalização (mesma do validate_extraction_quality.py)
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """NFKD, remove acentos, lowercase, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", text)
    no_acc = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_acc.strip().lower())


# ---------------------------------------------------------------------------
# Camada 1 — Achados de imagem em disease_or_syndrome → sign_or_symptom
# ---------------------------------------------------------------------------

IMAGING_FINDINGS: set[str] = {
    # Pulmonares
    "derrame pleural", "derrame pericardico",
    "consolidacao", "consolidacoes",
    "opacidade", "opacidades", "vidro fosco", "vidro-fosco",
    "atelectasia", "atelectasias",
    "nodulo pulmonar", "nodulos pulmonares",
    "cavitacao", "cavitacoes",
    "massa pulmonar", "massa cerebelar", "massa cerebral",
    "espessamento pleural", "espessamento da parede", "espessamento parietal",
    "infiltrado", "infiltrados", "infiltrado intersticial",
    "cardiomegalia", "hipertransparencia",
    "pneumotorax", "efusao",
    # Cerebrais/neurológicos
    "edema cerebral", "edema vasogenico",
    "desvio de linha media", "efeito de massa",
    "hidrocefalia", "ventriculomegalia",
    "hernia uncal", "herniacao",
    # Abdominais
    "dilatacao biliar", "dilatacao do ducto biliar comum",
    "esteatose hepatica",
    "esplenomegalia", "hepatomegalia", "hepatoesplenomegalia",
    "espessamento da parede retal", "espessamento parietal intestinal",
    # Linfonodais
    "linfadenopatia", "linfadenomegalia",
    "linfonodos aumentados", "linfonodos calcificados",
    "linfonodo hilar", "linfonodos mediastinais",
    "nodulos tireoidianos",
    # Genéricos
    "lesao", "lesoes",
    "nodulo", "nodulos",
}


def _matches_imaging(norm_item: str) -> bool:
    """Verifica se o item normalizado é um achado de imagem."""
    if norm_item in IMAGING_FINDINGS:
        return True
    for term in IMAGING_FINDINGS:
        if norm_item.startswith(term + " ") or norm_item.startswith(term + ","):
            return True
    return False


def layer1_imaging_findings(ner: dict, stats: dict) -> dict:
    """Move achados de imagem de disease_or_syndrome para sign_or_symptom."""
    diseases = ner.get("disease_or_syndrome", [])
    symptoms = list(ner.get("sign_or_symptom", []))
    symptom_norms = {_norm(s) for s in symptoms}

    new_diseases = []
    moved = []
    for item in diseases:
        if _matches_imaging(_norm(item)):
            if _norm(item) not in symptom_norms:
                symptoms.append(item)
                symptom_norms.add(_norm(item))
            moved.append(item)
        else:
            new_diseases.append(item)

    stats["layer1_moved"] = len(moved)
    if moved:
        stats["layer1_examples"] = moved[:5]

    ner["disease_or_syndrome"] = new_diseases
    ner["sign_or_symptom"] = symptoms
    return ner


# ---------------------------------------------------------------------------
# Camada 2 — Sinais vitais em laboratory_or_test_result → remover
# ---------------------------------------------------------------------------

_VITAL_PATTERNS = [
    re.compile(r"^(spo2|sao2|sat(?:uracao)?\s+de\s+o2|saturacao)", re.IGNORECASE),
    re.compile(r"^(pa\b|pressao arterial|pressao sistolica|pressao diastolica)", re.IGNORECASE),
    re.compile(r"^(fc\b|frequencia cardiaca|pulso\b)", re.IGNORECASE),
    re.compile(r"^(fr\b|frequencia respiratoria)", re.IGNORECASE),
    re.compile(r"^(temperatura|t\s|tax\b|t°)", re.IGNORECASE),
    re.compile(r"^(pam\b|pressao arterial media)", re.IGNORECASE),
    re.compile(r"^(oximetria)", re.IGNORECASE),
]


def _is_vital_sign(item: str) -> bool:
    norm = _norm(item)
    for pat in _VITAL_PATTERNS:
        if pat.search(norm):
            return True
    return False


def layer2_vital_signs(ner: dict, stats: dict) -> dict:
    """Remove sinais vitais de laboratory_or_test_result."""
    labs = ner.get("laboratory_or_test_result", [])
    new_labs = []
    removed = []
    for item in labs:
        if _is_vital_sign(item):
            removed.append(item)
        else:
            new_labs.append(item)

    stats["layer2_removed"] = len(removed)
    if removed:
        stats["layer2_examples"] = removed[:5]

    ner["laboratory_or_test_result"] = new_labs
    return ner


# ---------------------------------------------------------------------------
# Camada 3 — Normalização EN→PT
# ---------------------------------------------------------------------------

EN_TO_PT_MAPPING: dict[str, str] = {
    "rash": "exantema",
    "shortness of breath": "dispneia",
    "wheezing": "sibilos",
    "numbness": "dormência",
    "tightness": "opressão",
    "lightheadedness": "tontura",
    "swelling": "edema",
    "tenderness": "sensibilidade à palpação",
    "bruising": "equimose",
    "itching": "prurido",
    "clubbing": "baqueteamento digital",
    "cyanosis": "cianose",
    "altered mental status": "alteração do estado mental",
    "unresponsive": "não responsivo",
    "lethargic": "letárgico",
    "tender": "doloroso",
    "non-tender": "indolor",
    "soft": "macio",
    "fever": "febre",
    "cough": "tosse",
    "chest pain": "dor torácica",
    "nausea": "náusea",
    "vomiting": "vômito",
    "diarrhea": "diarreia",
    "headache": "cefaleia",
    "fatigue": "fadiga",
    "weakness": "fraqueza",
    "weight loss": "perda de peso",
    "night sweats": "sudorese noturna",
    "chills": "calafrios",
    "abdominal pain": "dor abdominal",
    "back pain": "dor lombar",
    "joint pain": "dor articular",
    "muscle pain": "mialgia",
    "sore throat": "dor de garganta",
    "runny nose": "rinorreia",
}

# Compile regex patterns para cada termo EN (word boundary)
_EN_REPLACEMENTS: list[tuple[re.Pattern, str]] = []
for en_term, pt_term in sorted(EN_TO_PT_MAPPING.items(), key=lambda x: -len(x[0])):
    _EN_REPLACEMENTS.append((
        re.compile(r"\b" + re.escape(en_term) + r"\b", re.IGNORECASE),
        pt_term,
    ))


def _translate_item(item: str) -> tuple[str, bool]:
    """Substitui termos EN por PT. Retorna (item_traduzido, foi_modificado)."""
    result = item
    changed = False
    for pattern, replacement in _EN_REPLACEMENTS:
        new_result = pattern.sub(replacement, result)
        if new_result != result:
            result = new_result
            changed = True
    return result, changed


NER_CATEGORIES = [
    "disease_or_syndrome", "sign_or_symptom", "pharmacologic_substance",
    "laboratory_or_test_result", "diagnostic_procedure", "organism_or_virus",
]


def layer3_language_normalization(ner: dict, stats: dict) -> dict:
    """Substitui termos em inglês por equivalentes em português."""
    total_subs = 0
    term_counts: dict[str, int] = defaultdict(int)
    examples: list[str] = []

    for cat in NER_CATEGORIES:
        items = ner.get(cat, [])
        new_items = []
        for item in items:
            translated, changed = _translate_item(item)
            if changed:
                total_subs += 1
                if len(examples) < 5:
                    examples.append(f"{item} → {translated}")
                # Contar quais termos foram substituídos
                for pattern, replacement in _EN_REPLACEMENTS:
                    if pattern.search(item):
                        term_counts[replacement] += 1
            new_items.append(translated)
        ner[cat] = new_items

    stats["layer3_substitutions"] = total_subs
    if examples:
        stats["layer3_examples"] = examples

    return ner


# ---------------------------------------------------------------------------
# Camada 4 — Deduplicação cross-category
# ---------------------------------------------------------------------------

# Prioridade: menor índice = maior prioridade
_CATEGORY_PRIORITY = {
    "organism_or_virus": 0,
    "pharmacologic_substance": 1,
    "laboratory_or_test_result": 2,
    "diagnostic_procedure": 3,
    "disease_or_syndrome": 4,
    "sign_or_symptom": 5,
}

# Stopwords clínicos genéricos demais para deduplicar
_CLINICAL_STOPWORDS = {
    "paciente", "exame", "teste", "resultado", "tratamento",
    "terapia", "consulta", "acompanhamento",
}


def layer4_cross_category_dedup(ner: dict, stats: dict) -> dict:
    """Remove duplicatas entre categorias, mantendo na mais específica."""
    # Mapeia entidade_norm → [(categoria, indice_original)]
    entity_locations: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for cat in NER_CATEGORIES:
        for idx, item in enumerate(ner.get(cat, [])):
            n = _norm(item)
            if n not in _CLINICAL_STOPWORDS:
                entity_locations[n].append((cat, idx))

    # Identificar entidades duplicadas
    to_remove: dict[str, set[int]] = defaultdict(set)  # cat → indices a remover
    removed_count = 0
    examples: list[str] = []

    for entity_norm, locations in entity_locations.items():
        if len(locations) <= 1:
            continue
        # Ordena por prioridade (menor = manter)
        locations_sorted = sorted(locations, key=lambda x: _CATEGORY_PRIORITY.get(x[0], 99))
        keep_cat = locations_sorted[0][0]
        for cat, idx in locations_sorted[1:]:
            to_remove[cat].add(idx)
            removed_count += 1
            if len(examples) < 5:
                examples.append(f"'{entity_norm}' removido de {cat} (mantido em {keep_cat})")

    # Aplicar remoções
    for cat in NER_CATEGORIES:
        items = ner.get(cat, [])
        indices_to_remove = to_remove.get(cat, set())
        if indices_to_remove:
            ner[cat] = [item for idx, item in enumerate(items) if idx not in indices_to_remove]

    stats["layer4_removed"] = removed_count
    if examples:
        stats["layer4_examples"] = examples

    return ner


# ---------------------------------------------------------------------------
# Camada 5 — Deduplicação intra-category
# ---------------------------------------------------------------------------

def layer5_intra_category_dedup(ner: dict, stats: dict) -> dict:
    """Remove duplicatas exatas dentro de cada categoria."""
    total_removed = 0
    examples: list[str] = []

    for cat in NER_CATEGORIES:
        items = ner.get(cat, [])
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            n = _norm(item)
            if n not in seen:
                seen.add(n)
                deduped.append(item)
            else:
                total_removed += 1
                if len(examples) < 5:
                    examples.append(f"'{item}' duplicado em {cat}")
        ner[cat] = deduped

    stats["layer5_removed"] = total_removed
    if examples:
        stats["layer5_examples"] = examples

    return ner


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def postprocess_record(record: dict) -> tuple[dict, dict]:
    """Aplica as 5 camadas em um registro. Retorna (record_modificado, stats)."""
    record = copy.deepcopy(record)
    ner = record.get("ner", {})
    stats: dict = {}

    ner = layer1_imaging_findings(ner, stats)
    ner = layer2_vital_signs(ner, stats)
    ner = layer3_language_normalization(ner, stats)
    ner = layer4_cross_category_dedup(ner, stats)
    ner = layer5_intra_category_dedup(ner, stats)

    record["ner"] = ner
    return record, stats


# ---------------------------------------------------------------------------
# Validação comparativa
# ---------------------------------------------------------------------------

def run_validation(json_path: str) -> dict:
    """Roda validate_extraction_quality sobre um arquivo JSON."""
    from validate_extraction_quality import validate_record, ALL_CHECKS

    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    check_totals: dict[str, int] = defaultdict(int)
    total_issues = 0
    clean = 0

    for record in records:
        report = validate_record(record)
        total_issues += report["total_issues"]
        if report["total_issues"] == 0:
            clean += 1
        for name, count in report["check_summary"].items():
            check_totals[name] += count

    return {
        "total_records": len(records),
        "clean_records": clean,
        "total_issues": total_issues,
        "checks": dict(check_totals),
    }


def print_comparison(before: dict, after: dict) -> None:
    """Imprime tabela comparativa antes/depois."""
    print(f"\n{'='*66}")
    print(f"  COMPARATIVO ANTES / DEPOIS DO PÓS-PROCESSAMENTO")
    print(f"{'='*66}")
    print(f"  {'Check':<35} {'Antes':>7}  {'Depois':>7}  {'Delta':>7}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*7}  {'-'*7}")

    all_checks = sorted(set(list(before["checks"].keys()) + list(after["checks"].keys())))
    for check in all_checks:
        b = before["checks"].get(check, 0)
        a = after["checks"].get(check, 0)
        delta = a - b
        delta_str = f"{delta:+d}" if delta != 0 else "0"
        print(f"  {check:<35} {b:>7}  {a:>7}  {delta_str:>7}")

    print(f"  {'-'*35}  {'-'*7}  {'-'*7}  {'-'*7}")
    b_total = before["total_issues"]
    a_total = after["total_issues"]
    print(f"  {'TOTAL ISSUES':<35} {b_total:>7}  {a_total:>7}  {a_total - b_total:>+7d}")
    print(f"  {'NOTAS LIMPAS':<35} {before['clean_records']:>7}  {after['clean_records']:>7}  {after['clean_records'] - before['clean_records']:>+7d}")
    print(f"{'='*66}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IANA — Pós-processamento determinístico do silver standard",
    )
    parser.add_argument(
        "--input",
        default="resultados/banco_dados_iana_v3.json",
        help="JSON de entrada (silver bruto).",
    )
    parser.add_argument(
        "--output",
        default="resultados/banco_dados_iana_v3_clean.json",
        help="JSON de saída (silver pós-processado).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Pular validação comparativa.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = _EXPERIMENTS_DIR / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _EXPERIMENTS_DIR / output_path

    # Carregar
    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)
    log.info("Carregado", extra={"data": {"notas": len(records), "input": str(input_path)}})

    # Processar
    t0 = time.perf_counter()
    processed_records = []
    aggregate_stats: dict[str, int] = defaultdict(int)
    all_examples: dict[str, list] = defaultdict(list)

    for record in records:
        new_record, stats = postprocess_record(record)
        processed_records.append(new_record)
        for key, val in stats.items():
            if key.endswith("_examples"):
                layer = key.split("_examples")[0]
                if len(all_examples[layer]) < 5:
                    all_examples[layer].extend(val[:5 - len(all_examples[layer])])
            else:
                aggregate_stats[key] += val

    duration = time.perf_counter() - t0

    # Salvar output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, indent=2, ensure_ascii=False)

    # Salvar stats
    logs_dir = _EXPERIMENTS_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    stats_path = logs_dir / "postprocess_stats.json"
    stats_output = {
        "duration_seconds": round(duration, 2),
        "total_records": len(records),
        "aggregate": dict(aggregate_stats),
        "examples": {k: v for k, v in all_examples.items()},
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)

    log.info("Processamento concluído", extra={"data": {
        "duration_seconds": round(duration, 2),
        "total_records": len(records),
        "output": str(output_path),
        **dict(aggregate_stats),
    }})

    # Validação comparativa
    if not args.no_validate:
        log.info("Rodando validação comparativa...")
        before = run_validation(str(input_path))
        after = run_validation(str(output_path))
        print_comparison(before, after)

        # Salvar comparação nos stats
        stats_output["validation_before"] = before
        stats_output["validation_after"] = after
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
