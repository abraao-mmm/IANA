r"""
Projeto IANA — Grafo de extração clínica v3.1 (LangGraph).

Arquitetura híbrida: LLM + pós-processamento determinístico + auditoria LLM

  START
    |
    +---> extrair_ner ──────────┐
    |                           |
    +---> extrair_soap ─────────┤
                                ↓
                  postprocess_deterministico  (Python puro, ~10ms)
                                ↓
                  audit_quality_llm           (1 chamada LLM, ~25s)
                                ↓
                  postprocess_final           (Python puro, ~10ms)
                                ↓
                  montar_resultado → END

Changelog v3.1:
- max_tokens dinâmico por tamanho de input (P1)
- Fallback do auditor em estouro de tokens (P1B)
- Status explícito por agente: ok/token_overflow/error/skipped (P2)
- failed_notes.jsonl para reprocessamento (P2D)
"""

import json
import logging
import sys
import time
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from models.schemas import (
    AgentStatus,
    EntidadeClinica,
    SOAP,
    EstadoExtracao,
)
from config.prompts import PROMPT_NER, PROMPT_SOAP, PROMPT_AUDIT_QUALITY
from postprocess import run_postprocess


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
log = logging.getLogger("extracao_graph")
if not log.handlers:
    log.addHandler(_handler)
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Audit metrics file logger (JSONL)
# ---------------------------------------------------------------------------

_LOGS_DIR = _EXPERIMENTS_DIR / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_AUDIT_METRICS_PATH = _LOGS_DIR / "audit_metrics.jsonl"

_audit_file_handler = logging.FileHandler(_AUDIT_METRICS_PATH, encoding="utf-8")
_audit_file_handler.setFormatter(_JSONFormatter())
_audit_log = logging.getLogger("audit_metrics")
if not _audit_log.handlers:
    _audit_log.addHandler(_audit_file_handler)
_audit_log.setLevel(logging.INFO)
_audit_log.propagate = False

# ---------------------------------------------------------------------------
# Failed notes logger (JSONL)
# ---------------------------------------------------------------------------

_FAILED_NOTES_PATH = _LOGS_DIR / "failed_notes.jsonl"
_failed_handler = logging.FileHandler(_FAILED_NOTES_PATH, encoding="utf-8")
_failed_handler.setFormatter(_JSONFormatter())
_failed_log = logging.getLogger("failed_notes")
if not _failed_log.handlers:
    _failed_log.addHandler(_failed_handler)
_failed_log.setLevel(logging.INFO)
_failed_log.propagate = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_token_overflow(error_msg: str) -> bool:
    """Detecta se o erro é estouro de max_tokens."""
    lower = error_msg.lower()
    return any(marker in lower for marker in [
        "length", "max_tokens", "maximum", "token limit",
        "finish_reason", "lengthfinishreason",
    ])


def _calcular_max_tokens(text: str) -> int:
    """Escala max_tokens conforme tamanho do input."""
    char_count = len(text)
    if char_count < 15000:
        return 16384
    elif char_count < 30000:
        return 24576
    elif char_count < 45000:
        return 32768
    else:
        return 49152


def _count_all_entities(ner_obj: EntidadeClinica) -> int:
    """Conta total de entidades em todas as categorias."""
    return sum(len(getattr(ner_obj, f)) for f in EntidadeClinica.model_fields)


def _compute_changes(before: EntidadeClinica, after: EntidadeClinica) -> int:
    """Conta diferenças totais entre dois NER (adições + remoções)."""
    changes = 0
    for field in EntidadeClinica.model_fields:
        set_before = {item.lower().strip() for item in getattr(before, field)}
        set_after = {item.lower().strip() for item in getattr(after, field)}
        changes += len(set_after - set_before)
        changes += len(set_before - set_after)
    return changes


# ---------------------------------------------------------------------------
# Grafo
# ---------------------------------------------------------------------------

def criar_grafo_extracao(llm):
    """
    Cria e compila o grafo de extração v3.1 para uma única nota clínica.

    Args:
        llm: Instância ChatOpenAI conectada ao vLLM.

    Returns:
        Grafo compilado pronto para .invoke() ou .stream()
    """

    # ---------------------------------------------------------
    # Nó 1: Extrator NER (paralelo com SOAP)
    # ---------------------------------------------------------
    def extrair_ner(state: EstadoExtracao):
        hadm_id = state.get("hadm_id", "___")
        max_tok = _calcular_max_tokens(state.get("texto_prontuario", ""))

        try:
            llm_ner = llm.bind(max_tokens=max_tok).with_structured_output(EntidadeClinica)
            ner = llm_ner.invoke([
                SystemMessage(content=PROMPT_NER),
                HumanMessage(content=state["texto_prontuario"]),
            ])
            return {"ner": ner, "ner_status": "ok", "ner_error": None}

        except Exception as e:
            error_msg = str(e)
            status = "token_overflow" if _is_token_overflow(error_msg) else "error"
            log.warning("extrair_ner_failed", extra={"data": {
                "hadm_id": hadm_id, "status": status, "error": error_msg[:200],
            }})
            _failed_log.info("ner_failed", extra={"data": {
                "hadm_id": hadm_id, "agent": "ner", "status": status,
                "char_count": len(state.get("texto_prontuario", "")),
                "max_tokens": max_tok, "error": error_msg[:500],
            }})
            return {
                "ner": EntidadeClinica(),
                "ner_status": status,
                "ner_error": error_msg[:500],
            }

    # ---------------------------------------------------------
    # Nó 2: Estruturador SOAP (paralelo com NER)
    # ---------------------------------------------------------
    def extrair_soap(state: EstadoExtracao):
        hadm_id = state.get("hadm_id", "___")
        max_tok = _calcular_max_tokens(state.get("texto_prontuario", ""))

        try:
            llm_soap = llm.bind(max_tokens=max_tok).with_structured_output(SOAP)
            soap = llm_soap.invoke([
                SystemMessage(content=PROMPT_SOAP),
                HumanMessage(content=state["texto_prontuario"]),
            ])
            return {"soap": soap, "soap_status": "ok", "soap_error": None}

        except Exception as e:
            error_msg = str(e)
            status = "token_overflow" if _is_token_overflow(error_msg) else "error"
            log.warning("extrair_soap_failed", extra={"data": {
                "hadm_id": hadm_id, "status": status, "error": error_msg[:200],
            }})
            _failed_log.info("soap_failed", extra={"data": {
                "hadm_id": hadm_id, "agent": "soap", "status": status,
                "char_count": len(state.get("texto_prontuario", "")),
                "max_tokens": max_tok, "error": error_msg[:500],
            }})
            return {
                "soap": SOAP(subjetivo="ERRO: extração falhou"),
                "soap_status": status,
                "soap_error": error_msg[:500],
            }

    # ---------------------------------------------------------
    # Nó 3: Pós-processamento determinístico (Python puro)
    # ---------------------------------------------------------
    def postprocess_deterministico(state: EstadoExtracao):
        ner = state.get("ner")
        if ner is None or not isinstance(ner, EntidadeClinica):
            return {"ner": EntidadeClinica()}
        return {"ner": run_postprocess(ner)}

    # ---------------------------------------------------------
    # Nó 4: Auditor LLM unificado (com métricas e fallback)
    # ---------------------------------------------------------
    def audit_quality_llm(state: EstadoExtracao):
        hadm_id = state.get("hadm_id", "___")
        ner = state.get("ner")

        if ner is None or not isinstance(ner, EntidadeClinica):
            return {"ner": EntidadeClinica(), "audit_status": "skipped", "audit_error": None}

        total_ent = _count_all_entities(ner)
        if total_ent == 0:
            _audit_log.info("audit_skipped_empty", extra={"data": {
                "hadm_id": hadm_id, "reason": "NER vazio",
            }})
            return {"ner": ner, "audit_status": "not_needed", "audit_error": None}

        t0 = time.perf_counter()
        audit_max_tokens = 16384  # dobrado para acomodar NER grandes

        try:
            llm_audit = llm.bind(max_tokens=audit_max_tokens).with_structured_output(EntidadeClinica)
            ner_json = ner.model_dump_json(indent=2)

            result = llm_audit.invoke([
                SystemMessage(content=PROMPT_AUDIT_QUALITY),
                HumanMessage(content=(
                    f"TEXTO ORIGINAL DO PRONTUÁRIO:\n{state['texto_prontuario']}\n\n"
                    f"---\n\n"
                    f"EXTRAÇÃO NER (pós-processada):\n{ner_json}\n\n"
                    f"---\n\n"
                    f"Retorne as 6 listas CORRIGIDAS e FINAIS."
                )),
            ])

            duration = time.perf_counter() - t0
            changes = _compute_changes(ner, result)
            output_json = result.model_dump_json()
            estimated_output_tokens = len(output_json) // 4

            _audit_log.info("audit_completed", extra={"data": {
                "hadm_id": hadm_id,
                "audit_duration_seconds": round(duration, 2),
                "audit_output_tokens": estimated_output_tokens,
                "audit_max_tokens_used_pct": round(estimated_output_tokens / audit_max_tokens * 100, 1),
                "audit_made_changes": changes > 0,
                "audit_changes_count": changes,
                "entities_before": total_ent,
                "entities_after": _count_all_entities(result),
            }})

            return {"ner": result, "audit_status": "ok", "audit_error": None}

        except Exception as e:
            duration = time.perf_counter() - t0
            error_msg = str(e)
            status = "token_overflow" if _is_token_overflow(error_msg) else "error"

            _audit_log.warning("audit_failed", extra={"data": {
                "hadm_id": hadm_id,
                "audit_duration_seconds": round(duration, 2),
                "status": status,
                "error": error_msg[:200],
                "fallback": "using_pre_audit_ner",
            }})
            _failed_log.info("audit_failed", extra={"data": {
                "hadm_id": hadm_id, "agent": "audit", "status": status,
                "char_count": len(state.get("texto_prontuario", "")),
                "entities_in_ner": total_ent,
                "error": error_msg[:500],
            }})

            # Fallback: retorna NER pós-processado sem auditoria LLM
            return {"ner": ner, "audit_status": status, "audit_error": error_msg[:500]}

    # ---------------------------------------------------------
    # Nó 5: Pós-processamento final
    # ---------------------------------------------------------
    def postprocess_final(state: EstadoExtracao):
        ner = state.get("ner")
        if ner is None or not isinstance(ner, EntidadeClinica):
            return {"ner": EntidadeClinica()}
        return {"ner": run_postprocess(ner)}

    # ---------------------------------------------------------
    # Nó 6: Monta o JSON final (com agent_status)
    # ---------------------------------------------------------
    def montar_resultado(state: EstadoExtracao):
        ner = state.get("ner")
        soap = state.get("soap")

        agent_status = AgentStatus(
            ner_status=state.get("ner_status", "ok"),
            soap_status=state.get("soap_status", "ok"),
            audit_status=state.get("audit_status", "ok"),
            ner_error_message=state.get("ner_error"),
            soap_error_message=state.get("soap_error"),
            audit_error_message=state.get("audit_error"),
        )

        resultado = {
            "paciente_id": state.get("hadm_id", "___"),
            "codigo_cid": state.get("codigo_cid", ""),
            "doenca_alvo_identificada": state.get("doenca_alvo", ""),
            "ner": ner.model_dump() if isinstance(ner, EntidadeClinica) else EntidadeClinica().model_dump(),
            "soap": soap.model_dump() if isinstance(soap, SOAP) else SOAP().model_dump(),
            "agent_status": agent_status.model_dump(),
        }
        return {"resultado_json": resultado}

    # ---------------------------------------------------------
    # Montagem do grafo
    # ---------------------------------------------------------
    builder = StateGraph(EstadoExtracao)

    builder.add_node("extrair_ner", extrair_ner)
    builder.add_node("extrair_soap", extrair_soap)
    builder.add_node("postprocess_det", postprocess_deterministico)
    builder.add_node("audit_llm", audit_quality_llm)
    builder.add_node("postprocess_final", postprocess_final)
    builder.add_node("montar", montar_resultado)

    # Fan-out: NER e SOAP em paralelo
    builder.add_edge(START, "extrair_ner")
    builder.add_edge(START, "extrair_soap")

    # NER → pós-processamento → auditoria → pós-processamento final
    builder.add_edge("extrair_ner", "postprocess_det")
    builder.add_edge("postprocess_det", "audit_llm")
    builder.add_edge("audit_llm", "postprocess_final")

    # SOAP e postprocess_final convergem no montar
    builder.add_edge("extrair_soap", "montar")
    builder.add_edge("postprocess_final", "montar")

    builder.add_edge("montar", END)

    return builder.compile()
