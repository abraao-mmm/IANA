#!/usr/bin/env python3
"""
Projeto IANA — Execução do test batch de 10 notas selecionadas (pipeline v3).

Processa 10 notas estratificadas pelo orientador, roda validação de qualidade
automática e gera relatório consolidado.

Uso:
    python run_test_batch.py
    python run_test_batch.py --url http://localhost:8000/v1
    python run_test_batch.py --parquet dados/mimic_filtrado_tb_hiv_sifilis.parquet
"""

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from langchain_openai import ChatOpenAI

from graphs.extracao import criar_grafo_extracao
from validate_extraction_quality import validate_record, ALL_CHECKS


# ---------------------------------------------------------------------------
# 10 notas selecionadas pelo orientador
# ---------------------------------------------------------------------------

TEST_PATIENT_IDS = [
    "25557330",  # HIV complexa - B20, 51 menções, 18 diferenciais, HAART+TB
    "22924630",  # HIV complexa - 50K chars, contexto longo
    "24918106",  # HIV simples - 6 menções, sem negações
    "22413631",  # HIV simples - validação não-regressão
    "23080963",  # Sífilis adequada - 46 testes pendentes
    "22978216",  # Sífilis adequada - 21 negações, coinfecção HIV
    "27306123",  # Sífilis zero - caso patológico Amostra 2
    "20250010",  # TB complexa - 52K chars, 17 diferenciais, TB miliar A199
    "27321074",  # TB complexa - TB do SNC A1781, HAART+TB
    "20248623",  # TB simples - TB pleural A182
]

TEST_LABELS = {
    "25557330": "HIV complexa",
    "22924630": "HIV complexa",
    "24918106": "HIV simples",
    "22413631": "HIV simples",
    "23080963": "Sífilis adequada",
    "22978216": "Sífilis adequada",
    "27306123": "Sífilis zero (caso patológico)",
    "20250010": "TB complexa",
    "27321074": "TB complexa",
    "20248623": "TB simples",
}


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
            entry["data"] = record.data
        return json.dumps(entry, ensure_ascii=False)


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_JSONFormatter())
log = logging.getLogger("test_batch")
log.addHandler(_handler)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Carregamento das 10 notas
# ---------------------------------------------------------------------------

def carregar_notas_teste(caminho_parquet: str) -> list[dict]:
    """Carrega apenas as 10 notas de teste do parquet."""
    import polars as pl

    df = pl.read_parquet(caminho_parquet)
    df_filtrado = df.filter(
        pl.col("hadm_id").cast(pl.Utf8).is_in(set(TEST_PATIENT_IDS))
    )

    log.info("Notas de teste carregadas", extra={"data": {
        "solicitadas": len(TEST_PATIENT_IDS),
        "encontradas": df_filtrado.height,
    }})

    if df_filtrado.height < len(TEST_PATIENT_IDS):
        encontradas = {str(r["hadm_id"]) for r in df_filtrado.to_dicts()}
        faltantes = set(TEST_PATIENT_IDS) - encontradas
        log.warning("Notas não encontradas no parquet", extra={"data": {
            "faltantes": list(faltantes),
        }})

    return df_filtrado.to_dicts()


# ---------------------------------------------------------------------------
# Processamento
# ---------------------------------------------------------------------------

def processar_test_batch(
    llm,
    notas: list[dict],
    saida_json: str,
    saida_metrics: str,
) -> tuple[list[dict], list[dict]]:
    """Processa as 10 notas e retorna (resultados, timings)."""
    import graphs.extracao as _gmod
    # Redireciona audit_metrics para arquivo de teste
    metrics_path = Path(saida_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _audit_log = logging.getLogger("audit_metrics")
    # Adiciona handler de arquivo para o test
    test_handler = logging.FileHandler(metrics_path, mode="w", encoding="utf-8")
    test_handler.setFormatter(_JSONFormatter())
    _audit_log.addHandler(test_handler)

    grafo = criar_grafo_extracao(llm)

    resultados = []
    timings = []

    total = len(notas)
    print(f"\n{'='*60}")
    print(f"  TEST BATCH v3 — {total} NOTAS SELECIONADAS")
    print(f"{'='*60}\n")

    for i, nota in enumerate(notas):
        hadm = str(nota.get("hadm_id", "___"))
        tamanho = len(nota.get("text", ""))
        label = TEST_LABELS.get(hadm, "?")

        print(f"[{i+1}/{total}] hadm={hadm} ({label}, {tamanho} chars)...", end=" ", flush=True)

        t0 = time.perf_counter()

        try:
            estado_final = grafo.invoke({
                "hadm_id": hadm,
                "codigo_cid": str(nota.get("icd_code", nota.get("codigo_cid", ""))),
                "doenca_alvo": str(nota.get("doenca_alvo", "")),
                "texto_prontuario": nota.get("text", ""),
            })

            resultado = estado_final.get("resultado_json")
            duracao = time.perf_counter() - t0

            if resultado:
                resultados.append(resultado)
                ner = resultado.get("ner", {})
                total_ent = sum(len(v) for v in ner.values() if isinstance(v, list))
                agent_st = resultado.get("agent_status", {})

                # Reporta status parcial se algum agente falhou
                warnings = []
                if agent_st.get("ner_status", "ok") != "ok":
                    warnings.append(f"NER:{agent_st['ner_status']}")
                if agent_st.get("soap_status", "ok") != "ok":
                    warnings.append(f"SOAP:{agent_st['soap_status']}")
                if agent_st.get("audit_status", "ok") not in ("ok", "not_needed"):
                    warnings.append(f"AUDIT:{agent_st['audit_status']}")

                if warnings:
                    print(f"PARCIAL ({total_ent} ent, {duracao:.1f}s) — {', '.join(warnings)}")
                else:
                    print(f"OK ({total_ent} entidades, {duracao:.1f}s)")

                timings.append({
                    "hadm_id": hadm, "label": label, "duration": duracao,
                    "entities": total_ent,
                    "status": "partial" if warnings else "ok",
                    "agent_status": agent_st,
                })
            else:
                print(f"WARN: sem resultado ({duracao:.1f}s)")
                timings.append({"hadm_id": hadm, "label": label, "duration": duracao, "entities": 0, "status": "empty"})

        except Exception as e:
            duracao = time.perf_counter() - t0
            print(f"ERRO ({duracao:.1f}s): {e}")
            timings.append({"hadm_id": hadm, "label": label, "duration": duracao, "entities": 0, "status": "error", "error": str(e)})

    # Salva resultados
    Path(saida_json).parent.mkdir(parents=True, exist_ok=True)
    with open(saida_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print(f"\n[SALVO] {len(resultados)} resultados -> {saida_json}")

    # Remove handler de teste para não poluir execuções futuras
    _audit_log.removeHandler(test_handler)
    test_handler.close()

    return resultados, timings


# ---------------------------------------------------------------------------
# Validação automática
# ---------------------------------------------------------------------------

def rodar_validacao(resultados: list[dict]) -> dict:
    """Roda os 8 checks em todos os resultados e retorna resumo."""
    from collections import defaultdict

    reports = []
    check_totals: dict[str, int] = defaultdict(int)

    for record in resultados:
        report = validate_record(record)
        reports.append(report)
        for check_name, count in report["check_summary"].items():
            check_totals[check_name] += count

    clean = sum(1 for r in reports if r["total_issues"] == 0)
    total_issues = sum(r["total_issues"] for r in reports)

    return {
        "total_records": len(resultados),
        "clean_records": clean,
        "records_with_issues": len(resultados) - clean,
        "total_issues": total_issues,
        "issues_per_check": dict(check_totals),
        "reports": reports,
    }


# ---------------------------------------------------------------------------
# Geração do relatório markdown
# ---------------------------------------------------------------------------

def gerar_relatorio(
    timings: list[dict],
    validation: dict,
    audit_metrics_path: str,
    saida_report: str,
) -> None:
    """Gera relatório consolidado em markdown."""
    lines: list[str] = []
    lines.append("# Relatório — Test Batch v3 (10 notas)\n")
    lines.append(f"**Data**: {time.strftime('%Y-%m-%d %H:%M')}\n")

    # --- Seção 1: Tempos ---
    lines.append("## 1. Tempos de processamento\n")
    durations = [t["duration"] for t in timings]
    lines.append(f"| Métrica | Valor |")
    lines.append(f"|---|---|")
    lines.append(f"| Tempo total | {sum(durations):.1f}s ({sum(durations)/60:.1f} min) |")
    lines.append(f"| Tempo médio | {statistics.mean(durations):.1f}s |")
    lines.append(f"| Mediana | {statistics.median(durations):.1f}s |")
    lines.append(f"| Mais rápida | {min(durations):.1f}s |")
    lines.append(f"| Mais lenta | {max(durations):.1f}s |")
    lines.append("")

    lines.append("### Detalhamento por nota\n")
    lines.append("| hadm_id | Categoria | Entidades | Tempo (s) | Status | NER | SOAP | Audit |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in timings:
        ast = t.get("agent_status", {})
        lines.append(
            f"| {t['hadm_id']} | {t['label']} | {t['entities']} | {t['duration']:.1f} "
            f"| {t['status']} | {ast.get('ner_status', '?')} "
            f"| {ast.get('soap_status', '?')} | {ast.get('audit_status', '?')} |"
        )
    lines.append("")

    # --- Seção 1b: Resumo de status por agente ---
    from collections import Counter
    ner_counts: Counter = Counter()
    soap_counts: Counter = Counter()
    audit_counts: Counter = Counter()
    for t in timings:
        ast = t.get("agent_status", {})
        ner_counts[ast.get("ner_status", "?")] += 1
        soap_counts[ast.get("soap_status", "?")] += 1
        audit_counts[ast.get("audit_status", "?")] += 1

    lines.append("### Status agregado por agente\n")
    lines.append("| Agente | ok | token_overflow | error | skipped | not_needed |")
    lines.append("|---|---|---|---|---|---|")
    for name, counts in [("NER", ner_counts), ("SOAP", soap_counts), ("Audit", audit_counts)]:
        lines.append(
            f"| {name} | {counts.get('ok', 0)} | {counts.get('token_overflow', 0)} "
            f"| {counts.get('error', 0)} | {counts.get('skipped', 0)} "
            f"| {counts.get('not_needed', 0)} |"
        )
    lines.append("")

    # --- Seção 2: Métricas do Auditor ---
    lines.append("## 2. Métricas do auditor LLM\n")
    audit_path = Path(audit_metrics_path)
    if audit_path.exists():
        audit_entries = []
        with open(audit_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if entry.get("msg") == "audit_completed":
                            audit_entries.append(entry.get("data", {}))
                    except json.JSONDecodeError:
                        pass

        if audit_entries:
            durations_audit = [e["audit_duration_seconds"] for e in audit_entries]
            pcts = [e["audit_max_tokens_used_pct"] for e in audit_entries]
            changes_count = [e["audit_changes_count"] for e in audit_entries]
            made_changes = sum(1 for e in audit_entries if e["audit_made_changes"])

            lines.append(f"| Métrica | Valor |")
            lines.append(f"|---|---|")
            lines.append(f"| Notas auditadas | {len(audit_entries)} |")
            lines.append(f"| Duração média | {statistics.mean(durations_audit):.1f}s |")
            lines.append(f"| % max_tokens médio | {statistics.mean(pcts):.1f}% |")
            lines.append(f"| % max_tokens máximo | {max(pcts):.1f}% |")
            lines.append(f"| Notas com mudanças | {made_changes}/{len(audit_entries)} |")
            lines.append(f"| Mudanças médias por nota | {statistics.mean(changes_count):.1f} |")
            lines.append("")

            lines.append("### Detalhamento por nota\n")
            lines.append("| hadm_id | Duração (s) | % max_tokens | Mudanças | Ent. antes | Ent. depois |")
            lines.append("|---|---|---|---|---|---|")
            for e in audit_entries:
                lines.append(
                    f"| {e['hadm_id']} | {e['audit_duration_seconds']:.1f} "
                    f"| {e['audit_max_tokens_used_pct']:.1f}% "
                    f"| {e['audit_changes_count']} "
                    f"| {e['entities_before']} | {e['entities_after']} |"
                )
            lines.append("")
        else:
            lines.append("*Nenhuma entrada de auditoria encontrada no log.*\n")
    else:
        lines.append(f"*Arquivo de métricas não encontrado: {audit_metrics_path}*\n")

    # --- Seção 3: Validação de qualidade ---
    lines.append("## 3. Validação de qualidade (8 checks)\n")
    summary = {k: v for k, v in validation.items() if k != "reports"}
    lines.append(f"| Métrica | Valor |")
    lines.append(f"|---|---|")
    lines.append(f"| Notas limpas (0 issues) | {summary['clean_records']}/{summary['total_records']} |")
    lines.append(f"| Notas com issues | {summary['records_with_issues']} |")
    lines.append(f"| Total de issues | {summary['total_issues']} |")
    lines.append("")

    lines.append("### Issues por check\n")
    lines.append("| Check | Ocorrências |")
    lines.append("|---|---|")
    for check_name, count in sorted(summary.get("issues_per_check", {}).items()):
        status = "✅" if count == 0 else f"⚠️ {count}"
        lines.append(f"| {check_name} | {status} |")
    lines.append("")

    # Detalhamento por nota com issues
    reports_with_issues = [r for r in validation.get("reports", []) if r["total_issues"] > 0]
    if reports_with_issues:
        lines.append("### Notas com issues\n")
        for r in reports_with_issues:
            lines.append(f"**{r['paciente_id']}** ({r['doenca_alvo']}) — {r['total_issues']} issues:\n")
            for issue in r["issues"]:
                lines.append(f"- `{issue['check']}`: {json.dumps({k: v for k, v in issue.items() if k != 'check'}, ensure_ascii=False)}")
            lines.append("")

    # Salva
    Path(saida_report).parent.mkdir(parents=True, exist_ok=True)
    with open(saida_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[RELATÓRIO] Salvo em {saida_report}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IANA — Test batch v3 com 10 notas selecionadas",
    )
    parser.add_argument("--url", default="http://localhost:8000/v1", help="URL do servidor vLLM.")
    parser.add_argument("--api-key", default="iana-local-key", help="API key do vLLM.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B", help="Modelo.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperatura.")
    parser.add_argument("--parquet", default="dados/mimic_filtrado_tb_hiv_sifilis.parquet", help="Parquet.")
    parser.add_argument("--saida", default="resultados/test_batch_v3.json", help="JSON de saída.")
    parser.add_argument("--report", default="resultados/test_batch_v3_report.md", help="Relatório markdown.")
    args = parser.parse_args()

    # Resolve caminhos
    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = _EXPERIMENTS_DIR / parquet_path

    saida_path = Path(args.saida)
    if not saida_path.is_absolute():
        saida_path = _EXPERIMENTS_DIR / saida_path

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = _EXPERIMENTS_DIR / report_path

    metrics_path = _EXPERIMENTS_DIR / "logs" / "audit_metrics_test.jsonl"

    # Cria LLM
    print(f"[CONFIG] Servidor: {args.url}")
    print(f"[CONFIG] Modelo:   {args.model}")
    print(f"[CONFIG] Temp:     {args.temperature}")

    llm = ChatOpenAI(
        base_url=args.url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=16384,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        timeout=120,
    )

    # Carrega notas de teste
    notas = carregar_notas_teste(str(parquet_path))

    # Processa
    resultados, timings = processar_test_batch(
        llm, notas,
        saida_json=str(saida_path),
        saida_metrics=str(metrics_path),
    )

    # Validação automática
    print(f"\n{'='*60}")
    print(f"  VALIDAÇÃO DE QUALIDADE")
    print(f"{'='*60}\n")

    validation = rodar_validacao(resultados)

    print(f"  Notas limpas:     {validation['clean_records']}/{validation['total_records']}")
    print(f"  Total de issues:  {validation['total_issues']}")
    for check, count in sorted(validation["issues_per_check"].items()):
        marker = "✅" if count == 0 else f"⚠️  {count}"
        print(f"    {check}: {marker}")

    # Gera relatório
    gerar_relatorio(
        timings=timings,
        validation=validation,
        audit_metrics_path=str(metrics_path),
        saida_report=str(report_path),
    )

    print(f"\n{'='*60}")
    print(f"  CONCLUÍDO")
    print(f"{'='*60}")
    print(f"  Resultados:  {saida_path}")
    print(f"  Métricas:    {metrics_path}")
    print(f"  Relatório:   {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
