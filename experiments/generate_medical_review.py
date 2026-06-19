#!/usr/bin/env python3
"""
Projeto IANA — Gera material HTML de revisão para especialista médico.

Produz 10 arquivos HTML (um por nota do test batch) + index.html,
para que o especialista compare a nota original (inglês) com a
extração estruturada (NER + SOAP em português).

Uso:
    python generate_medical_review.py
    python generate_medical_review.py --json resultados/test_batch_v3.json
"""

import argparse
import html
import json
import logging
import os
import sys
import time
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {"ts": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
        if hasattr(record, "data"):
            entry["data"] = record.data
        return json.dumps(entry, ensure_ascii=False)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_JSONFormatter())
log = logging.getLogger("generate_review")
log.addHandler(_handler)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Metadados das notas (labels conhecidos + fallback por doença)
# ---------------------------------------------------------------------------

TEST_LABELS: dict[str, str] = {
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

NER_CATEGORY_LABELS = {
    "disease_or_syndrome": "Doenças e Síndromes",
    "sign_or_symptom": "Sinais e Sintomas",
    "pharmacologic_substance": "Substâncias Farmacológicas",
    "laboratory_or_test_result": "Resultados Laboratoriais",
    "diagnostic_procedure": "Procedimentos Diagnósticos",
    "organism_or_virus": "Organismos e Vírus",
}

SOAP_FIELD_LABELS = {
    "subjetivo": "Subjetivo",
    "objetivo_exame_fisico": "Objetivo — Exame Físico",
    "objetivo_laboratorio": "Objetivo — Laboratório",
    "objetivo_imagem": "Objetivo — Imagem",
    "avaliacao": "Avaliação",
    "plano": "Plano",
}

SOAP_COLORS = {
    "subjetivo": "#2563eb",
    "objetivo_exame_fisico": "#059669",
    "objetivo_laboratorio": "#7c3aed",
    "objetivo_imagem": "#d97706",
    "avaliacao": "#dc2626",
    "plano": "#0891b2",
}


# ---------------------------------------------------------------------------
# CSS embutido
# ---------------------------------------------------------------------------

CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6; color: #1a1a2e; background: #f8f9fa;
    max-width: 920px; margin: 0 auto; padding: 24px 16px;
}
h1 { font-size: 1.5rem; color: #1e3a5f; margin-bottom: 8px; }
h2 {
    font-size: 1.25rem; color: #1e3a5f; margin: 32px 0 12px;
    padding-bottom: 6px; border-bottom: 2px solid #dbe4f0;
}
h3 { font-size: 1.05rem; margin: 16px 0 8px; }
.header-meta {
    background: #e8eef6; border-radius: 8px; padding: 16px 20px;
    margin-bottom: 24px; font-size: 0.9rem; line-height: 1.8;
}
.header-meta strong { color: #1e3a5f; }
.badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.8rem; font-weight: 600;
}
.badge-ok { background: #d1fae5; color: #065f46; }
.badge-warn { background: #fef3c7; color: #92400e; }
.badge-error { background: #fecaca; color: #991b1b; }
.alert-pathological {
    background: #fef3c7; border-left: 4px solid #f59e0b;
    padding: 12px 16px; margin: 16px 0; border-radius: 4px;
    font-size: 0.95rem;
}
.original-text {
    background: #f1f3f5; border: 1px solid #dee2e6; border-radius: 6px;
    padding: 16px; font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 0.82rem; line-height: 1.5; white-space: pre-wrap;
    word-wrap: break-word; max-height: 600px; overflow-y: auto;
}
.ner-category {
    margin-bottom: 16px; padding: 12px 16px;
    background: #fff; border: 1px solid #e5e7eb; border-radius: 6px;
}
.ner-category h3 {
    color: #1e3a5f; font-size: 0.95rem; margin: 0 0 8px;
}
.ner-category .count {
    color: #6b7280; font-size: 0.8rem; font-weight: normal;
}
.ner-category ul { padding-left: 20px; margin: 0; }
.ner-category li { font-size: 0.9rem; margin: 2px 0; }
.ner-empty { color: #9ca3af; font-style: italic; font-size: 0.9rem; }
.soap-field {
    margin-bottom: 16px; background: #fff;
    border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden;
}
.soap-field-header {
    padding: 8px 16px; color: #fff; font-weight: 600; font-size: 0.95rem;
}
.soap-field-body {
    padding: 12px 16px; font-size: 0.9rem; white-space: pre-wrap;
    line-height: 1.6;
}
.soap-empty { color: #9ca3af; font-style: italic; }
table.annotation {
    width: 100%; border-collapse: collapse; margin-top: 12px;
    font-size: 0.85rem;
}
table.annotation th, table.annotation td {
    border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left;
    vertical-align: top;
}
table.annotation th {
    background: #e8eef6; color: #1e3a5f; font-weight: 600;
}
table.annotation td { min-height: 28px; }
.annotation-instructions {
    background: #eff6ff; border-radius: 6px; padding: 12px 16px;
    margin-bottom: 12px; font-size: 0.88rem; color: #1e40af;
}
.footer {
    margin-top: 40px; padding-top: 16px; border-top: 1px solid #dee2e6;
    font-size: 0.8rem; color: #6b7280; text-align: center;
}
@media print {
    body { max-width: 100%; padding: 12px; }
    .original-text { max-height: none; overflow: visible; }
    .alert-pathological { break-inside: avoid; }
    .footer { display: none; }
}
"""


# ---------------------------------------------------------------------------
# Geração HTML
# ---------------------------------------------------------------------------

def _badge(status: str) -> str:
    if status in ("ok", "not_needed"):
        cls = "badge-ok"
    elif status in ("token_overflow", "skipped"):
        cls = "badge-warn"
    else:
        cls = "badge-error"
    return f'<span class="badge {cls}">{html.escape(status)}</span>'


def _generate_note_html(record: dict, original_text: str) -> str:
    pid = record.get("paciente_id", "???")
    cid = record.get("codigo_cid", "")
    doenca = record.get("doenca_alvo_identificada", "")
    label = TEST_LABELS.get(pid, doenca)
    ner = record.get("ner", {})
    soap = record.get("soap", {})
    agent_st = record.get("agent_status", {})

    total_ent = sum(len(v) for v in ner.values() if isinstance(v, list))

    parts: list[str] = []
    parts.append(f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Revisão — {html.escape(pid)} ({html.escape(doenca)})</title>
<style>{CSS}</style>
</head>
<body>
<h1>Revisão Clínica — Nota {html.escape(pid)}</h1>
<div class="header-meta">
    <strong>Paciente ID:</strong> {html.escape(pid)} &nbsp;|&nbsp;
    <strong>Doença alvo:</strong> {html.escape(doenca)} &nbsp;|&nbsp;
    <strong>CID:</strong> {html.escape(cid)} &nbsp;|&nbsp;
    <strong>Categoria:</strong> {html.escape(label)}<br>
    <strong>Entidades extraídas:</strong> {total_ent} &nbsp;|&nbsp;
    <strong>NER:</strong> {_badge(agent_st.get('ner_status', '?'))}
    <strong>SOAP:</strong> {_badge(agent_st.get('soap_status', '?'))}
    <strong>Audit:</strong> {_badge(agent_st.get('audit_status', '?'))}
</div>""")

    # Alerta para caso patológico 27306123
    if pid == "27306123":
        parts.append("""
<div class="alert-pathological">
    <strong>⚠️ ATENÇÃO:</strong> Esta nota foi incluída intencionalmente como
    <strong>caso patológico</strong> — o resumo de alta original NÃO menciona
    sífilis em momento algum, apesar do código CID indicar neurossífilis.
    O pipeline deve retornar listas vazias ou apenas as condições reais do
    paciente (pé de Charcot, neuropatia). Por favor, valide se o pipeline
    lidou corretamente com esse caso.
</div>""")

    # Seção 1 — Nota Original
    parts.append(f"""
<h2>1. Nota Original (inglês)</h2>
<div class="original-text">{html.escape(original_text)}</div>""")

    # Seção 2 — NER
    parts.append("\n<h2>2. Entidades Clínicas Extraídas (NER)</h2>")
    for field_key, field_label in NER_CATEGORY_LABELS.items():
        items = ner.get(field_key, [])
        count = len(items)
        parts.append(f'<div class="ner-category">')
        parts.append(f'  <h3>{html.escape(field_label)} <span class="count">({count})</span></h3>')
        if items:
            parts.append("  <ul>")
            for item in items:
                parts.append(f"    <li>{html.escape(item)}</li>")
            parts.append("  </ul>")
        else:
            parts.append('  <p class="ner-empty">Nenhuma entidade extraída nesta categoria.</p>')
        parts.append("</div>")

    # Seção 3 — SOAP
    parts.append("\n<h2>3. Estrutura SOAP (português)</h2>")
    for field_key, field_label in SOAP_FIELD_LABELS.items():
        content = soap.get(field_key, "")
        color = SOAP_COLORS.get(field_key, "#374151")
        parts.append(f'<div class="soap-field">')
        parts.append(f'  <div class="soap-field-header" style="background:{color};">{html.escape(field_label)}</div>')
        if content and content.strip():
            parts.append(f'  <div class="soap-field-body">{html.escape(content)}</div>')
        else:
            parts.append('  <div class="soap-field-body soap-empty">Campo não preenchido.</div>')
        parts.append("</div>")

    # Seção 4 — Anotação do especialista
    parts.append("""
<h2>4. Anotação do Especialista</h2>
<div class="annotation-instructions">
    <strong>Instruções:</strong> Por favor, anote abaixo qualquer erro encontrado
    durante a revisão. Compare o texto original (Seção 1) com as entidades
    extraídas (Seção 2) e a estrutura SOAP (Seção 3). Esta tabela será usada
    para calibrar o pipeline antes do processamento das 738 notas restantes.<br><br>
    <strong>Tipos de erro:</strong>
    Omissão (entidade presente no texto mas não extraída) |
    Categoria errada (entidade extraída na categoria incorreta) |
    Negação vazada (teste negativo/pendente gerando entidade positiva) |
    Tradução (erro de tradução EN→PT ou termo em inglês residual) |
    Invenção (entidade que não existe no texto original) |
    Outro
</div>
<table class="annotation">
    <thead>
        <tr>
            <th style="width:18%">Categoria</th>
            <th style="width:22%">Item</th>
            <th style="width:18%">Tipo de erro</th>
            <th style="width:12%">Severidade</th>
            <th style="width:30%">Comentário</th>
        </tr>
    </thead>
    <tbody>""")
    for _ in range(12):
        parts.append("        <tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>")
    parts.append("""    </tbody>
</table>""")

    # Footer
    parts.append(f"""
<div class="footer">
    Projeto IANA — Pipeline v3.1 | Gerado em {time.strftime('%Y-%m-%d %H:%M')} |
    Material para revisão por especialista médico
</div>
</body>
</html>""")

    return "\n".join(parts)


def _generate_index_html(records: list[dict]) -> str:
    rows: list[str] = []
    for r in records:
        pid = r.get("paciente_id", "???")
        doenca = r.get("doenca_alvo_identificada", "")
        label = TEST_LABELS.get(pid, doenca)
        ner = r.get("ner", {})
        total_ent = sum(len(v) for v in ner.values() if isinstance(v, list))
        agent_st = r.get("agent_status", {})

        fname = f"{pid}_{doenca}_review.html"
        rows.append(
            f"<tr>"
            f'<td><a href="{html.escape(fname)}">{html.escape(pid)}</a></td>'
            f"<td>{html.escape(doenca)}</td>"
            f"<td>{html.escape(label)}</td>"
            f"<td style='text-align:center'>{total_ent}</td>"
            f"<td>{_badge(agent_st.get('ner_status', '?'))} "
            f"{_badge(agent_st.get('soap_status', '?'))} "
            f"{_badge(agent_st.get('audit_status', '?'))}</td>"
            f"</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IANA — Índice de Revisão Médica</title>
<style>{CSS}
table.index {{ width: 100%; border-collapse: collapse; }}
table.index th, table.index td {{ border: 1px solid #cbd5e1; padding: 10px 12px; text-align: left; }}
table.index th {{ background: #1e3a5f; color: #fff; font-weight: 600; }}
table.index tr:nth-child(even) {{ background: #f8f9fa; }}
table.index a {{ color: #2563eb; text-decoration: none; font-weight: 600; }}
table.index a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>Projeto IANA — Revisão por Especialista Médico</h1>
<p style="margin:12px 0 24px; color:#4b5563; font-size:0.95rem;">
    Pipeline v3.1 — Test batch com 10 notas estratificadas (HIV, Sífilis, Tuberculose).<br>
    Clique no ID do paciente para abrir a revisão detalhada de cada nota.
</p>

<table class="index">
    <thead>
        <tr>
            <th>Paciente ID</th>
            <th>Doença</th>
            <th>Categoria</th>
            <th>Entidades</th>
            <th>Status (NER / SOAP / Audit)</th>
        </tr>
    </thead>
    <tbody>
        {"".join(rows)}
    </tbody>
</table>

<div class="annotation-instructions" style="margin-top:24px;">
    <strong>Instruções gerais:</strong> Cada arquivo contém o texto original
    da nota clínica (em inglês) e a extração estruturada produzida pelo
    pipeline (em português). Ao final de cada arquivo há uma tabela para
    anotar erros encontrados. Foque em: entidades omitidas, entidades na
    categoria errada, testes negativos extraídos como positivos, e erros
    de tradução.
</div>

<div class="footer">
    Projeto IANA — Pipeline v3.1 | Gerado em {time.strftime('%Y-%m-%d %H:%M')}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IANA — Gera material HTML de revisão para especialista médico",
    )
    parser.add_argument("--json", default="resultados/test_batch_v3.json", help="JSON do test batch.")
    parser.add_argument("--parquet", default="dados/mimic_filtrado_tb_hiv_sifilis.parquet", help="Parquet original.")
    parser.add_argument("--output-dir", default="resultados/medical_review", help="Diretório de saída.")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.is_absolute():
        json_path = _EXPERIMENTS_DIR / json_path

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = _EXPERIMENTS_DIR / parquet_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _EXPERIMENTS_DIR / output_dir

    # Carrega resultados
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)
    log.info("Resultados carregados", extra={"data": {"notas": len(records)}})

    # Carrega textos originais do parquet
    import polars as pl
    df = pl.read_parquet(str(parquet_path))
    patient_ids = {r.get("paciente_id", "") for r in records}
    df_filtered = df.filter(pl.col("hadm_id").cast(pl.Utf8).is_in(patient_ids))
    texts: dict[str, str] = {}
    for row in df_filtered.to_dicts():
        texts[str(row.get("hadm_id", ""))] = row.get("text", "")
    log.info("Textos originais carregados", extra={"data": {"encontrados": len(texts)}})

    # Gera HTMLs
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes: list[int] = []

    for record in records:
        pid = record.get("paciente_id", "???")
        doenca = record.get("doenca_alvo_identificada", "")
        original_text = texts.get(pid, "[TEXTO ORIGINAL NÃO ENCONTRADO NO PARQUET]")

        html_content = _generate_note_html(record, original_text)
        fname = f"{pid}_{doenca}_review.html"
        fpath = output_dir / fname

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(html_content)
        sizes.append(os.path.getsize(fpath))

    # Gera index
    index_html = _generate_index_html(records)
    index_path = output_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)

    avg_size = sum(sizes) / len(sizes) if sizes else 0

    log.info("Revisão médica gerada", extra={"data": {
        "total_arquivos": len(sizes) + 1,
        "diretorio": str(output_dir),
        "tamanho_medio_bytes": round(avg_size),
        "tamanho_medio_kb": round(avg_size / 1024, 1),
    }})

    print(f"\n{'='*60}")
    print(f"  MATERIAL DE REVISÃO GERADO")
    print(f"{'='*60}")
    print(f"  Diretório: {output_dir}")
    print(f"  Arquivos:  {len(sizes)} notas + index.html")
    print(f"  Tamanho médio: {avg_size/1024:.1f} KB por nota")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
