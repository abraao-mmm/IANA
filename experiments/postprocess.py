"""
Projeto IANA - Pos-processamento deterministico de entidades NER (v3).

Tres funcoes aplicadas em sequencia apos a saida do Agent 1 (NER Extractor)
e novamente apos o Agent auditor:

1. deduplicate_within_category  — remove sinonimos/duplicatas intra-categoria
2. enforce_mutual_exclusivity   — garante que cada entidade aparece em 1 categoria
3. normalize_canonical_terms    — aplica dicionario de termos canonicos

Todas as funcoes recebem e retornam EntidadeClinica.
"""

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from models.schemas import EntidadeClinica
from config.canonical_terms import ALL_CANONICAL

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
log = logging.getLogger("postprocess")
log.addHandler(_handler)
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_key(text: str) -> str:
    """Produz chave de comparacao: lowercase, sem acentos, espacos normalizados."""
    # Remove acentos
    nfkd = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase, normaliza espacos
    return re.sub(r"\s+", " ", without_accents.strip().lower())


# ---------------------------------------------------------------------------
# Regex para detectar negacoes/pendencias (usado em enforce_mutual_exclusivity)
# ---------------------------------------------------------------------------

_NEGATION_MARKERS = re.compile(
    r"(?i)\b("
    r"negativ[oe]|negative|sem crescimento|no growth|not detected"
    r"|pendente|pending|pnd|indeterminate"
    r"|ruled out|descartad[oa]|excluíd[oa]"
    r")\b"
)


# ---------------------------------------------------------------------------
# 1. Deduplicacao intra-categoria
# ---------------------------------------------------------------------------

def deduplicate_within_category(ner: EntidadeClinica) -> EntidadeClinica:
    """
    Remove duplicatas dentro de cada categoria usando comparacao
    case-insensitive e sem acentos.

    Mantem a primeira ocorrencia (preserva a forma original).
    """
    changes = 0
    data = ner.model_dump()

    for field_name, items in data.items():
        if not isinstance(items, list):
            continue
        seen: dict[str, str] = {}
        deduped: list[str] = []
        for item in items:
            key = _normalize_key(item)
            if key not in seen:
                seen[key] = item
                deduped.append(item)
            else:
                changes += 1
        data[field_name] = deduped

    if changes > 0:
        log.info(
            "dedup_within_category",
            extra={"data": {"removed": changes}},
        )

    return EntidadeClinica(**data)


# ---------------------------------------------------------------------------
# 2. Exclusividade mutua entre categorias
# ---------------------------------------------------------------------------

# Ordem de precedencia (indice menor = maior precedencia)
_CATEGORY_PRECEDENCE = [
    "laboratory_or_test_result",
    "organism_or_virus",
    "diagnostic_procedure",
    "disease_or_syndrome",
    "sign_or_symptom",
    "pharmacologic_substance",
]


def enforce_mutual_exclusivity(ner: EntidadeClinica) -> EntidadeClinica:
    """
    Se uma entidade aparece em mais de uma categoria, mantem na de
    maior precedencia e remove das demais.

    Precedencia: laboratory > organism > procedure > disease > symptom > medication
    """
    data = ner.model_dump()
    changes = 0

    # Mapa: normalized_key -> (categoria de maior precedencia, forma original)
    seen: dict[str, tuple[str, str]] = {}

    # Primeira passada: registra a categoria de maior precedencia para cada entidade
    for cat in _CATEGORY_PRECEDENCE:
        items = data.get(cat, [])
        if not isinstance(items, list):
            continue
        for item in items:
            key = _normalize_key(item)
            if key not in seen:
                seen[key] = (cat, item)

    # Segunda passada: remove entidades das categorias de menor precedencia
    for cat in _CATEGORY_PRECEDENCE:
        items = data.get(cat, [])
        if not isinstance(items, list):
            continue
        filtered: list[str] = []
        for item in items:
            key = _normalize_key(item)
            winner_cat, _ = seen.get(key, (cat, item))
            if winner_cat == cat:
                filtered.append(item)
            else:
                changes += 1
        data[cat] = filtered

    if changes > 0:
        log.info(
            "enforce_mutual_exclusivity",
            extra={"data": {"moved": changes}},
        )

    return EntidadeClinica(**data)


# ---------------------------------------------------------------------------
# 3. Normalizacao canonica
# ---------------------------------------------------------------------------

def normalize_canonical_terms(ner: EntidadeClinica) -> EntidadeClinica:
    """
    Aplica o dicionario de termos canonicos para normalizar variacoes.
    Ex: "Hipertensao" -> "Hipertensao arterial sistemica"
    """
    data = ner.model_dump()
    changes = 0

    for field_name, items in data.items():
        if not isinstance(items, list):
            continue
        normalized: list[str] = []
        seen_keys: set[str] = set()
        for item in items:
            key = _normalize_key(item)
            canonical = ALL_CANONICAL.get(key, item)
            canon_key = _normalize_key(canonical)
            if canon_key not in seen_keys:
                seen_keys.add(canon_key)
                if canonical != item:
                    changes += 1
                normalized.append(canonical)
        data[field_name] = normalized

    if changes > 0:
        log.info(
            "normalize_canonical_terms",
            extra={"data": {"normalized": changes}},
        )

    return EntidadeClinica(**data)


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def run_postprocess(ner: EntidadeClinica) -> EntidadeClinica:
    """Aplica as 3 etapas de pos-processamento em sequencia."""
    ner = deduplicate_within_category(ner)
    ner = enforce_mutual_exclusivity(ner)
    ner = normalize_canonical_terms(ner)
    return ner
