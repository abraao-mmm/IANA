#!/usr/bin/env python3
"""
Projeto IANA — Gera HTML interativo de revisão clínica (versão final, editável).

Diferencas vs generate_medical_review.py (v1):
  * Avaliacao por CATEGORIA (12 selects por nota) ao inves de por entidade
    (~1500 inputs). Reduz drasticamente a carga do medico revisor.
  * Likert simples: Correto / Parcial / Incorreto + textarea opcional.
  * Auto-save em localStorage (nao perde progresso ao fechar o browser).
  * Botao "Exportar JSON" por nota e botao "Exportar TUDO" no index.
  * Indice mostra progresso visual (X/30 revisadas) lido do localStorage.
  * Identificacao do revisor no topo (salva globalmente em localStorage).

Uso:
    python generate_medical_review_v2.py
    python generate_medical_review_v2.py --gold resultados/gold_test_set_30.json \\
        --output-dir resultados/medical_review_final
"""

import argparse
import html
import json
import os
import sys
import time
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent

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

CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6; color: #1a1a2e; background: #f8f9fa;
    max-width: 980px; margin: 0 auto; padding: 24px 16px;
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
.reviewer-bar {
    background: #fff; border: 1px solid #cbd5e1; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 16px;
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
}
.reviewer-bar label { font-weight: 600; color: #1e3a5f; }
.reviewer-bar input[type="text"] {
    padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 4px;
    font-size: 0.95rem; min-width: 220px;
}
.reviewer-bar .nav { margin-left: auto; }
.reviewer-bar a {
    color: #2563eb; text-decoration: none; padding: 6px 12px;
    border: 1px solid #2563eb; border-radius: 4px; font-size: 0.9rem;
}
.reviewer-bar a:hover { background: #eff6ff; }
.original-text {
    background: #f1f3f5; border: 1px solid #dee2e6; border-radius: 6px;
    padding: 16px; font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 0.82rem; line-height: 1.5; white-space: pre-wrap;
    word-wrap: break-word; max-height: 500px; overflow-y: auto;
}
.ner-category, .soap-field {
    margin-bottom: 16px; padding: 12px 16px;
    background: #fff; border: 1px solid #e5e7eb; border-radius: 6px;
}
.ner-category h3 { color: #1e3a5f; font-size: 0.95rem; margin: 0 0 8px; }
.ner-category .count { color: #6b7280; font-size: 0.8rem; font-weight: normal; }
.ner-category ul { padding-left: 20px; margin: 0; }
.ner-category li { font-size: 0.9rem; margin: 2px 0; }
.ner-empty { color: #9ca3af; font-style: italic; font-size: 0.9rem; }
.soap-field-header {
    padding: 8px 16px; color: #fff; font-weight: 600; font-size: 0.95rem;
    margin: -12px -16px 12px; border-radius: 6px 6px 0 0;
}
.soap-field-body {
    padding: 0; font-size: 0.9rem; white-space: pre-wrap; line-height: 1.6;
}
.soap-empty { color: #9ca3af; font-style: italic; }
.eval-block {
    margin-top: 12px; padding-top: 12px; border-top: 1px dashed #cbd5e1;
    display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap;
}
.eval-block label { font-weight: 600; color: #1e3a5f; font-size: 0.88rem; }
.eval-block select {
    padding: 4px 8px; border: 1px solid #cbd5e1; border-radius: 4px;
    font-size: 0.9rem; background: #fff; cursor: pointer;
}
.eval-block select.status-correto { background: #d1fae5; color: #065f46; }
.eval-block select.status-parcial { background: #fef3c7; color: #92400e; }
.eval-block select.status-incorreto { background: #fecaca; color: #991b1b; }
.eval-block textarea {
    flex: 1; min-width: 300px; min-height: 38px; padding: 6px 10px;
    border: 1px solid #cbd5e1; border-radius: 4px; font-size: 0.88rem;
    font-family: inherit; resize: vertical;
}
.export-bar {
    position: sticky; bottom: 0; background: #1e3a5f; color: #fff;
    padding: 12px 16px; margin: 32px -16px -24px;
    display: flex; gap: 12px; align-items: center; box-shadow: 0 -2px 8px rgba(0,0,0,0.1);
}
.export-bar .progress { flex: 1; font-size: 0.9rem; }
.export-bar button {
    padding: 8px 16px; background: #fff; color: #1e3a5f; border: none;
    border-radius: 4px; font-weight: 600; cursor: pointer; font-size: 0.9rem;
}
.export-bar button:hover { background: #e8eef6; }
.export-bar button.secondary {
    background: transparent; color: #fff; border: 1px solid #fff;
}
.export-bar button.secondary:hover { background: rgba(255,255,255,0.1); }
.alert-pathological {
    background: #fef3c7; border-left: 4px solid #f59e0b;
    padding: 12px 16px; margin: 16px 0; border-radius: 4px;
    font-size: 0.95rem;
}
.footer {
    margin-top: 24px; padding: 16px 0; font-size: 0.8rem;
    color: #6b7280; text-align: center;
}
table.index { width: 100%; border-collapse: collapse; }
table.index th, table.index td {
    border: 1px solid #cbd5e1; padding: 10px 12px; text-align: left;
}
table.index th { background: #1e3a5f; color: #fff; font-weight: 600; }
table.index tr:nth-child(even) { background: #f8f9fa; }
table.index a { color: #2563eb; text-decoration: none; font-weight: 600; }
table.index a:hover { text-decoration: underline; }
.badge-status {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75rem; font-weight: 600;
}
.status-pendente { background: #f3f4f6; color: #6b7280; }
.status-iniciado { background: #fef3c7; color: #92400e; }
.status-completo { background: #d1fae5; color: #065f46; }
"""

# JavaScript embutido — comum a todas as paginas
JS_PER_NOTE = """\
const PID = '__PID__';
const STORAGE_KEY = 'iana_review_' + PID;
const REVIEWER_KEY = 'iana_reviewer';

function loadState() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); }
    catch (e) { return {}; }
}
function saveState(state) {
    state._updated = new Date().toISOString();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    updateProgressBar();
}
function updateProgressBar() {
    const state = loadState();
    const selects = document.querySelectorAll('select.eval-status');
    let filled = 0;
    selects.forEach(s => { if (s.value) filled++; });
    document.getElementById('progress-text').textContent =
        `${filled} de ${selects.length} categorias avaliadas`;
}
function applyStatusClass(select) {
    select.classList.remove('status-correto','status-parcial','status-incorreto');
    if (select.value) select.classList.add('status-' + select.value);
}
function attachHandlers() {
    document.querySelectorAll('select.eval-status').forEach(sel => {
        const key = sel.dataset.key;
        const state = loadState();
        if (state[key]?.status) { sel.value = state[key].status; applyStatusClass(sel); }
        sel.addEventListener('change', () => {
            applyStatusClass(sel);
            const state = loadState();
            state[key] = state[key] || {};
            state[key].status = sel.value;
            saveState(state);
        });
    });
    document.querySelectorAll('textarea.eval-comment').forEach(ta => {
        const key = ta.dataset.key;
        const state = loadState();
        if (state[key]?.comment) ta.value = state[key].comment;
        ta.addEventListener('input', () => {
            const state = loadState();
            state[key] = state[key] || {};
            state[key].comment = ta.value;
            saveState(state);
        });
    });
    const ta = document.getElementById('general-comments');
    if (ta) {
        const state = loadState();
        if (state._general) ta.value = state._general;
        ta.addEventListener('input', () => {
            const state = loadState();
            state._general = ta.value;
            saveState(state);
        });
    }
    const reviewer = document.getElementById('reviewer-name');
    if (reviewer) {
        reviewer.value = localStorage.getItem(REVIEWER_KEY) || '';
        reviewer.addEventListener('input', () => {
            localStorage.setItem(REVIEWER_KEY, reviewer.value);
        });
    }
}
function exportJSON() {
    const state = loadState();
    const reviewer = localStorage.getItem(REVIEWER_KEY) || 'anonimo';
    const out = {
        paciente_id: PID,
        reviewer: reviewer,
        timestamp: new Date().toISOString(),
        evaluations: state,
    };
    const blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `review_${reviewer.replace(/\\W+/g,'_')}_${PID}.json`;
    a.click();
    URL.revokeObjectURL(url);
}
function clearLocal() {
    if (confirm('Apagar avaliacao desta nota? (sem desfazer)')) {
        localStorage.removeItem(STORAGE_KEY);
        location.reload();
    }
}
window.addEventListener('DOMContentLoaded', () => {
    attachHandlers();
    updateProgressBar();
});
"""

JS_INDEX = """\
const REVIEWER_KEY = 'iana_reviewer';

function refreshTable() {
    const reviewer = localStorage.getItem(REVIEWER_KEY) || '';
    document.getElementById('reviewer-display').textContent = reviewer || '(nao identificado)';

    let totalCompleted = 0;
    let totalStarted = 0;
    document.querySelectorAll('tr[data-pid]').forEach(row => {
        const pid = row.dataset.pid;
        const cell = row.querySelector('.status-cell');
        const raw = localStorage.getItem('iana_review_' + pid);
        if (!raw) {
            cell.innerHTML = '<span class="badge-status status-pendente">pendente</span>';
            return;
        }
        try {
            const state = JSON.parse(raw);
            const cats = Object.keys(state).filter(k => !k.startsWith('_'));
            const filled = cats.filter(k => state[k]?.status).length;
            // 12 categorias por nota: 6 NER + 6 SOAP
            const total = 12;
            if (filled >= total) {
                cell.innerHTML = `<span class="badge-status status-completo">completo (${filled}/${total})</span>`;
                totalCompleted++;
            } else {
                cell.innerHTML = `<span class="badge-status status-iniciado">iniciado (${filled}/${total})</span>`;
                totalStarted++;
            }
        } catch (e) {
            cell.innerHTML = '<span class="badge-status status-pendente">erro</span>';
        }
    });
    document.getElementById('progress-summary').textContent =
        `${totalCompleted} completas, ${totalStarted} iniciadas, ${30 - totalCompleted - totalStarted} pendentes`;
}
function exportAll() {
    const reviewer = localStorage.getItem(REVIEWER_KEY) || 'anonimo';
    const allData = { reviewer: reviewer, timestamp: new Date().toISOString(), notes: [] };
    document.querySelectorAll('tr[data-pid]').forEach(row => {
        const pid = row.dataset.pid;
        const raw = localStorage.getItem('iana_review_' + pid);
        if (raw) {
            try {
                allData.notes.push({ paciente_id: pid, evaluations: JSON.parse(raw) });
            } catch (e) {}
        }
    });
    const blob = new Blob([JSON.stringify(allData, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `iana_review_${reviewer.replace(/\\W+/g,'_')}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}
function clearAll() {
    if (confirm('Apagar TODAS as avaliacoes deste browser? (sem desfazer)')) {
        const keys = [];
        for (let i = 0; i < localStorage.length; i++) {
            const k = localStorage.key(i);
            if (k.startsWith('iana_review_')) keys.push(k);
        }
        keys.forEach(k => localStorage.removeItem(k));
        refreshTable();
    }
}
function setReviewer() {
    const r = document.getElementById('reviewer-name').value.trim();
    if (r) {
        localStorage.setItem(REVIEWER_KEY, r);
        refreshTable();
    }
}
window.addEventListener('DOMContentLoaded', () => {
    const r = localStorage.getItem(REVIEWER_KEY);
    if (r) document.getElementById('reviewer-name').value = r;
    refreshTable();
});
"""


def _generate_note_html(record: dict, original_text: str) -> str:
    pid = record.get("paciente_id", "???")
    cid = record.get("codigo_cid", "")
    doenca = record.get("doenca_alvo_identificada", "")
    ner = record.get("ner", {})
    soap = record.get("soap", {})
    total_ent = sum(len(v) for v in ner.values() if isinstance(v, list))

    def _eval_block(key: str) -> str:
        return f"""
<div class="eval-block">
    <label for="eval-{key}-status">Avaliação:</label>
    <select id="eval-{key}-status" class="eval-status" data-key="{key}">
        <option value="">— selecionar —</option>
        <option value="correto">Correto</option>
        <option value="parcial">Parcial</option>
        <option value="incorreto">Incorreto</option>
    </select>
    <textarea class="eval-comment" data-key="{key}"
              placeholder="Comentario (opcional): correcoes, omissoes, categoria errada..."></textarea>
</div>"""

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
<div class="reviewer-bar">
    <label for="reviewer-name">Revisor:</label>
    <input type="text" id="reviewer-name" placeholder="Seu nome"/>
    <span class="nav"><a href="index.html">← Índice</a></span>
</div>

<h1>Revisão Clínica — Nota {html.escape(pid)}</h1>
<div class="header-meta">
    <strong>Paciente ID:</strong> {html.escape(pid)} &nbsp;|&nbsp;
    <strong>Doença alvo:</strong> {html.escape(doenca)} &nbsp;|&nbsp;
    <strong>CID:</strong> {html.escape(cid)} &nbsp;|&nbsp;
    <strong>Entidades extraídas:</strong> {total_ent}
</div>""")

    if pid == "27306123":
        parts.append("""
<div class="alert-pathological">
    <strong>⚠️ Caso patológico:</strong> esta nota foi incluída intencionalmente
    como caso de erro de codificação CID — o resumo de alta original NÃO menciona
    sífilis, apesar do CID indicar neurossífilis. O pipeline deve retornar listas
    vazias ou apenas as condições reais do paciente. Por favor, valide se houve
    extração correta ou alucinação.
</div>""")

    # Seção 1 — Texto Original
    parts.append(f"""
<h2>1. Nota Original (inglês)</h2>
<div class="original-text">{html.escape(original_text)}</div>""")

    # Seção 2 — NER (visualização + 1 avaliação por categoria)
    parts.append("\n<h2>2. Entidades Clínicas (NER) — avaliar cada categoria</h2>")
    for field_key, field_label in NER_CATEGORY_LABELS.items():
        items = ner.get(field_key, [])
        count = len(items)
        parts.append(f'<div class="ner-category">')
        parts.append(f'  <h3>{html.escape(field_label)} <span class="count">({count})</span></h3>')
        if items:
            parts.append("  <ul>")
            for item in items:
                if isinstance(item, str):
                    parts.append(f"    <li>{html.escape(item)}</li>")
            parts.append("  </ul>")
        else:
            parts.append('  <p class="ner-empty">Nenhuma entidade nesta categoria.</p>')
        parts.append(_eval_block(f"ner_{field_key}"))
        parts.append("</div>")

    # Seção 3 — SOAP (1 avaliação por campo)
    parts.append("\n<h2>3. Estrutura SOAP — avaliar cada campo</h2>")
    for field_key, field_label in SOAP_FIELD_LABELS.items():
        content = soap.get(field_key, "")
        color = SOAP_COLORS.get(field_key, "#374151")
        parts.append(f'<div class="soap-field">')
        parts.append(f'  <div class="soap-field-header" style="background:{color};">{html.escape(field_label)}</div>')
        if content and isinstance(content, str) and content.strip():
            parts.append(f'  <div class="soap-field-body">{html.escape(content)}</div>')
        else:
            parts.append('  <div class="soap-field-body soap-empty">Campo não preenchido.</div>')
        parts.append(_eval_block(f"soap_{field_key}"))
        parts.append("</div>")

    # Seção 4 — Comentários gerais
    parts.append("""
<h2>4. Comentários Gerais (opcional)</h2>
<div class="ner-category">
    <textarea id="general-comments" class="eval-comment" style="width:100%; min-height:80px;"
              placeholder="Observacoes sobre a nota como um todo: padroes de erro, qualidade geral, recomendacoes..."></textarea>
</div>""")

    # Barra de export
    parts.append("""
<div class="export-bar">
    <span class="progress" id="progress-text">0 de 12 categorias avaliadas</span>
    <button class="secondary" onclick="clearLocal()">Limpar</button>
    <button onclick="exportJSON()">Exportar JSON</button>
</div>""")

    parts.append(f"""
<div class="footer">
    Projeto IANA — Validação clínica do gold test set | Gerado em {time.strftime('%Y-%m-%d')}
</div>
<script>{JS_PER_NOTE.replace("__PID__", pid)}</script>
</body>
</html>""")

    return "\n".join(parts)


def _generate_index_html(records: list[dict]) -> str:
    rows: list[str] = []
    for r in records:
        pid = r.get("paciente_id", "???")
        doenca = r.get("doenca_alvo_identificada", "")
        ner = r.get("ner", {})
        total_ent = sum(len(v) for v in ner.values() if isinstance(v, list))
        fname = f"{pid}_{html.escape(doenca)}_review.html"
        rows.append(
            f'<tr data-pid="{html.escape(pid)}">'
            f'<td><a href="{fname}">{html.escape(pid)}</a></td>'
            f"<td>{html.escape(doenca)}</td>"
            f"<td style='text-align:center'>{total_ent}</td>"
            f"<td class='status-cell'><span class='badge-status status-pendente'>pendente</span></td>"
            f"</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IANA — Validação Clínica do Gold Test Set</title>
<style>{CSS}</style>
</head>
<body>
<h1>Projeto IANA — Validação Clínica</h1>
<p style="margin:12px 0 24px; color:#4b5563; font-size:0.95rem;">
    Gold test set com 30 notas estratificadas (HIV, Sífilis, Tuberculose) extraído
    de prontuários MIMIC-IV. Sua avaliação calibrará o gold standard usado para
    benchmark dos modelos compactos. Avaliação por <strong>categoria</strong>
    (12 selects por nota), não por entidade individual — minimiza carga.
</p>

<div class="reviewer-bar">
    <label for="reviewer-name">Identifique-se:</label>
    <input type="text" id="reviewer-name" placeholder="Seu nome"
           oninput="setReviewer()"/>
    <span style="color:#6b7280; font-size:0.85rem;">
        Revisor atual: <strong id="reviewer-display">(não identificado)</strong>
    </span>
</div>

<div class="annotation-instructions" style="background:#eff6ff; border-radius:6px;
     padding:12px 16px; margin-bottom:16px; font-size:0.88rem; color:#1e40af;">
    <strong>Como avaliar cada categoria:</strong><br>
    <strong>Correto</strong> — extração captura adequadamente o que está no texto.<br>
    <strong>Parcial</strong> — captura parte, mas omite ou inclui demais (use comentário pra detalhar).<br>
    <strong>Incorreto</strong> — categoria errada, alucinação, ou erro grave de tradução.<br><br>
    <strong>Auto-save:</strong> tudo é salvo automaticamente no seu navegador. Pode fechar
    e voltar depois sem perder. Ao terminar todas as 30, clique em <strong>Exportar TUDO</strong>
    abaixo para baixar o JSON consolidado e enviar de volta à equipe.
</div>

<table class="index">
    <thead>
        <tr>
            <th>Paciente ID</th>
            <th>Doença</th>
            <th>Entidades</th>
            <th>Status</th>
        </tr>
    </thead>
    <tbody>
        {"".join(rows)}
    </tbody>
</table>

<div class="export-bar">
    <span class="progress" id="progress-summary">carregando...</span>
    <button class="secondary" onclick="clearAll()">Limpar tudo</button>
    <button onclick="exportAll()">Exportar TUDO (JSON)</button>
</div>

<div class="footer">
    Projeto IANA — Gerado em {time.strftime('%Y-%m-%d')} |
    <a href="instrucoes.html">Instruções detalhadas</a>
</div>
<script>{JS_INDEX}</script>
</body>
</html>"""


def _generate_instructions_html() -> str:
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>IANA — Instruções para revisão</title>
<style>{CSS}</style>
</head>
<body>
<div class="reviewer-bar"><span class="nav"><a href="index.html">← Índice</a></span></div>
<h1>Instruções para revisão clínica</h1>

<h2>Contexto</h2>
<p>Este conjunto de 30 notas será o <strong>gold test set</strong> usado para
avaliar 4 modelos compactos de extração de entidades nomeadas (NER) e
sumarização SOAP. As notas foram extraídas do MIMIC-IV (prontuários reais
de UTI em inglês) e já tiveram a extração feita automaticamente por um modelo
maior (Qwen 122B). <strong>Sua tarefa é validar essa extração.</strong></p>

<h2>O que avaliar</h2>
<p>Para <strong>cada categoria</strong> de NER (6) e <strong>cada campo</strong>
de SOAP (6), atribua um dos 3 status:</p>
<ul style="padding-left:24px; margin:12px 0;">
    <li><strong>Correto</strong> — extração faz sentido clínico e cobre o que está no texto.</li>
    <li><strong>Parcial</strong> — captura parte, mas há omissões importantes ou inclusões inadequadas.</li>
    <li><strong>Incorreto</strong> — erro grave: categoria errada, alucinação, contradição com o texto.</li>
</ul>
<p>Comentário é <strong>opcional</strong> — só preencha se quiser detalhar uma
correção específica. Médicos com tempo limitado podem só clicar nos status.</p>

<h2>Funcionamento</h2>
<ul style="padding-left:24px; margin:12px 0;">
    <li>Todas as suas avaliações são salvas <strong>automaticamente no navegador</strong>
    (localStorage). Pode fechar e voltar depois sem perder.</li>
    <li>Cada nota tem botão <strong>Exportar JSON</strong> ao final (opcional).</li>
    <li>Quando terminar todas as 30, volte ao índice e clique
    <strong>Exportar TUDO (JSON)</strong> — baixa um arquivo consolidado com
    todas as suas avaliações.</li>
    <li>Envie esse JSON de volta para a equipe.</li>
</ul>

<h2>Aviso sobre caso patológico</h2>
<p>Uma das notas (paciente 27306123) é um <strong>caso de erro de codificação</strong>:
o CID indica neurossífilis, mas o texto da nota não menciona sífilis em momento
algum. O pipeline deve ter retornado listas vazias ou apenas as condições reais
(pé de Charcot, neuropatia). Se viu sífilis sendo extraída, é alucinação.</p>

<div class="footer">Projeto IANA — Validação clínica</div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="IANA — Gera HTML editavel de revisao clinica")
    parser.add_argument("--gold", default="resultados/gold_test_set_30.json",
                        help="JSON do gold test set.")
    parser.add_argument("--parquet", default="dados/mimic_filtrado_tb_hiv_sifilis.parquet",
                        help="Parquet com textos originais.")
    parser.add_argument("--output-dir", default="resultados/medical_review_final",
                        help="Diretorio de saida.")
    args = parser.parse_args()

    json_path = Path(args.gold)
    if not json_path.is_absolute():
        json_path = _EXPERIMENTS_DIR / json_path
    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = _EXPERIMENTS_DIR / parquet_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _EXPERIMENTS_DIR / output_dir

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("notes", data) if isinstance(data, dict) else data
    print(f"[info] {len(records)} notas carregadas de {json_path.name}")

    import polars as pl
    df = pl.read_parquet(str(parquet_path))
    pids = {r.get("paciente_id", "") for r in records}
    df_f = df.filter(pl.col("hadm_id").cast(pl.Utf8).is_in(pids))
    texts = {str(row["hadm_id"]): row.get("text", "") for row in df_f.to_dicts()}
    print(f"[info] {len(texts)} textos originais carregados")

    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = []
    for record in records:
        pid = record.get("paciente_id", "???")
        doenca = record.get("doenca_alvo_identificada", "")
        text = texts.get(pid, "[texto original nao encontrado no parquet]")
        html_content = _generate_note_html(record, text)
        fname = f"{pid}_{doenca}_review.html"
        fpath = output_dir / fname
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(html_content)
        sizes.append(os.path.getsize(fpath))

    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(_generate_index_html(records))
    with open(output_dir / "instrucoes.html", "w", encoding="utf-8") as f:
        f.write(_generate_instructions_html())

    avg_kb = (sum(sizes) / len(sizes) / 1024) if sizes else 0
    print(f"\n{'=' * 60}")
    print(f"  VALIDACAO CLINICA GERADA")
    print(f"{'=' * 60}")
    print(f"  Diretorio: {output_dir}")
    print(f"  Arquivos:  {len(sizes)} notas + index.html + instrucoes.html")
    print(f"  Tamanho medio: {avg_kb:.1f} KB por nota")
    print(f"\n  Para o medico: abrir index.html em qualquer navegador.")
    print(f"  Tudo funciona offline (auto-save em localStorage).")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
