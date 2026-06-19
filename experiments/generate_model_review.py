#!/usr/bin/env python3
"""
Projeto IANA — Gera HTML side-by-side comparando extracoes dos modelos.

Layout: para cada nota, mostra 4 colunas lado a lado (Silver/Qwen122B teacher
+ os 3 decoders compactos), cada uma com NER + SOAP + avaliacao Likert
independente. Auto-save em localStorage com namespace por modelo.

Diferencas vs generate_medical_review_v2.py:
  * 4 modelos lado a lado (vs 1 modelo isolado)
  * Avaliacao independente por modelo: 12 categorias x 4 modelos = 48 selects
  * Layout responsivo: 4 colunas em tela larga, 2x2 em tela media,
    1xN com scroll em tela pequena
  * Index mostra progresso por modelo

Uso:
    python generate_model_review.py
    python generate_model_review.py --output-dir resultados/medical_review_models
"""

import argparse
import html
import json
import os
import sys
import time
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_TRAINING_DIR = _EXPERIMENTS_DIR / "training"

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

# Os modelos das predictions usam chaves SOAP em ingles (subjective/objective/etc)
# enquanto o silver usa em portugues. Mapeamento:
SOAP_KEY_VARIANTS = {
    "subjetivo": ["subjetivo", "subjective"],
    "objetivo_exame_fisico": ["objetivo_exame_fisico", "objective", "objective_physical_exam"],
    "objetivo_laboratorio": ["objetivo_laboratorio", "objective_laboratory"],
    "objetivo_imagem": ["objetivo_imagem", "objective_imaging"],
    "avaliacao": ["avaliacao", "assessment"],
    "plano": ["plano", "plan"],
}

# Modelos no comparativo (ordem de exibicao). Silver primeiro como referencia.
MODELS = [
    {"key": "silver", "label": "Silver (Qwen 122B teacher)", "color": "#1e3a5f"},
    {"key": "qwen35_4b", "label": "Qwen 3.5-4B", "color": "#7c3aed"},
    {"key": "gemma4_e4b", "label": "Gemma 4 E4B", "color": "#059669"},
    {"key": "medgemma", "label": "MedGemma 4B", "color": "#dc2626"},
]


def _get_soap_value(soap: dict, key: str) -> str:
    if not isinstance(soap, dict):
        return ""
    for variant in SOAP_KEY_VARIANTS.get(key, [key]):
        if variant in soap:
            v = soap[variant]
            if isinstance(v, str):
                return v
            if isinstance(v, list):
                return "\n".join(str(x) for x in v if x)
    return ""


def _get_ner_items(ner: dict, key: str) -> list[str]:
    if not isinstance(ner, dict):
        return []
    items = ner.get(key, [])
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, str)]


CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.55; color: #1a1a2e; background: #f8f9fa;
    max-width: 100%; margin: 0 auto; padding: 16px;
}
h1 { font-size: 1.5rem; color: #1e3a5f; margin-bottom: 8px; }
h2 {
    font-size: 1.2rem; color: #1e3a5f; margin: 28px 0 12px;
    padding-bottom: 6px; border-bottom: 2px solid #dbe4f0;
}
h3 { font-size: 0.95rem; margin: 12px 0 6px; }
.header-meta {
    background: #e8eef6; border-radius: 8px; padding: 14px 18px;
    margin-bottom: 20px; font-size: 0.88rem; line-height: 1.7;
}
.reviewer-bar {
    background: #fff; border: 1px solid #cbd5e1; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 14px;
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
}
.reviewer-bar label { font-weight: 600; color: #1e3a5f; }
.reviewer-bar input[type="text"] {
    padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 4px;
    font-size: 0.92rem; min-width: 200px;
}
.reviewer-bar .nav { margin-left: auto; }
.reviewer-bar a {
    color: #2563eb; text-decoration: none; padding: 6px 12px;
    border: 1px solid #2563eb; border-radius: 4px; font-size: 0.88rem;
}
.reviewer-bar a:hover { background: #eff6ff; }
.original-text {
    background: #f1f3f5; border: 1px solid #dee2e6; border-radius: 6px;
    padding: 14px; font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 0.78rem; line-height: 1.5; white-space: pre-wrap;
    word-wrap: break-word; max-height: 360px; overflow-y: auto;
}
.models-grid {
    display: grid; gap: 12px;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
}
.model-col {
    background: #fff; border: 2px solid #e5e7eb; border-radius: 8px;
    padding: 12px; min-width: 0;
}
.model-col-header {
    padding: 8px 12px; color: #fff; font-weight: 600; font-size: 0.95rem;
    margin: -12px -12px 12px; border-radius: 6px 6px 0 0;
    display: flex; justify-content: space-between; align-items: center;
}
.model-col-header button {
    background: rgba(255,255,255,0.2); color: #fff; border: 1px solid rgba(255,255,255,0.4);
    padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; cursor: pointer;
}
.model-col-header button:hover { background: rgba(255,255,255,0.35); }
.cat-block {
    margin-bottom: 10px; padding: 8px 10px;
    background: #fafafa; border: 1px solid #e5e7eb; border-radius: 4px;
}
.cat-block h3 {
    font-size: 0.82rem; color: #1e3a5f; margin: 0 0 4px;
    display: flex; justify-content: space-between;
}
.cat-block .count { color: #6b7280; font-size: 0.7rem; font-weight: normal; }
.cat-block ul { padding-left: 16px; margin: 4px 0; }
.cat-block li { font-size: 0.82rem; margin: 1px 0; line-height: 1.4; }
.cat-block .empty { color: #9ca3af; font-style: italic; font-size: 0.8rem; }
.cat-block .soap-body {
    font-size: 0.82rem; padding: 4px 0; white-space: pre-wrap;
    line-height: 1.4; max-height: 140px; overflow-y: auto;
}
.eval-row {
    margin-top: 6px; display: flex; gap: 6px; align-items: stretch;
    flex-wrap: wrap;
}
.eval-row select {
    padding: 3px 6px; border: 1px solid #cbd5e1; border-radius: 4px;
    font-size: 0.78rem; background: #fff; cursor: pointer; flex: 1; min-width: 110px;
}
.eval-row select.status-correto { background: #d1fae5; color: #065f46; }
.eval-row select.status-parcial { background: #fef3c7; color: #92400e; }
.eval-row select.status-incorreto { background: #fecaca; color: #991b1b; }
.eval-row textarea {
    flex: 2; min-width: 150px; min-height: 28px; padding: 4px 6px;
    border: 1px solid #cbd5e1; border-radius: 4px; font-size: 0.78rem;
    font-family: inherit; resize: vertical;
}
.export-bar {
    position: sticky; bottom: 0; background: #1e3a5f; color: #fff;
    padding: 10px 14px; margin: 24px -16px -16px;
    display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
    box-shadow: 0 -2px 8px rgba(0,0,0,0.1); z-index: 10;
}
.export-bar .progress { flex: 1; font-size: 0.88rem; min-width: 200px; }
.export-bar button {
    padding: 7px 14px; background: #fff; color: #1e3a5f; border: none;
    border-radius: 4px; font-weight: 600; cursor: pointer; font-size: 0.88rem;
}
.export-bar button:hover { background: #e8eef6; }
.export-bar button.secondary {
    background: transparent; color: #fff; border: 1px solid #fff;
}
.alert-pathological {
    background: #fef3c7; border-left: 4px solid #f59e0b;
    padding: 12px 16px; margin: 12px 0; border-radius: 4px;
    font-size: 0.9rem;
}
.footer {
    margin-top: 20px; padding: 14px 0; font-size: 0.78rem;
    color: #6b7280; text-align: center;
}
table.index { width: 100%; border-collapse: collapse; }
table.index th, table.index td {
    border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left;
}
table.index th { background: #1e3a5f; color: #fff; font-weight: 600; font-size: 0.88rem; }
table.index tr:nth-child(even) { background: #f8f9fa; }
table.index a { color: #2563eb; text-decoration: none; font-weight: 600; }
table.index a:hover { text-decoration: underline; }
.badge-status {
    display: inline-block; padding: 2px 6px; border-radius: 4px;
    font-size: 0.72rem; font-weight: 600; margin-right: 4px;
}
.bs-pendente { background: #f3f4f6; color: #6b7280; }
.bs-iniciado { background: #fef3c7; color: #92400e; }
.bs-completo { background: #d1fae5; color: #065f46; }
"""

JS_PER_NOTE = r"""
const PID = '__PID__';
const STORAGE_KEY = 'iana_models_' + PID;
const REVIEWER_KEY = 'iana_reviewer';
const MODELS = __MODELS__;
const TOTAL_CATS = 12;  // 6 NER + 6 SOAP

function loadState() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); }
    catch (e) { return {}; }
}
function saveState(state) {
    state._updated = new Date().toISOString();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    updateProgress();
}
function updateProgress() {
    const state = loadState();
    let totalFilled = 0;
    let totalSlots = MODELS.length * TOTAL_CATS;
    MODELS.forEach(m => {
        const ms = state[m] || {};
        Object.keys(ms).filter(k => !k.startsWith('_')).forEach(k => {
            if (ms[k]?.status) totalFilled++;
        });
    });
    document.getElementById('progress-text').textContent =
        `${totalFilled} de ${totalSlots} avaliações preenchidas (` +
        `${MODELS.length} modelos × ${TOTAL_CATS} categorias)`;
}
function applyStatusClass(select) {
    select.classList.remove('status-correto','status-parcial','status-incorreto');
    if (select.value) select.classList.add('status-' + select.value);
}
function attachHandlers() {
    document.querySelectorAll('select.eval-status').forEach(sel => {
        const model = sel.dataset.model;
        const cat = sel.dataset.cat;
        const state = loadState();
        if (state[model]?.[cat]?.status) {
            sel.value = state[model][cat].status;
            applyStatusClass(sel);
        }
        sel.addEventListener('change', () => {
            applyStatusClass(sel);
            const state = loadState();
            state[model] = state[model] || {};
            state[model][cat] = state[model][cat] || {};
            state[model][cat].status = sel.value;
            saveState(state);
        });
    });
    document.querySelectorAll('textarea.eval-comment').forEach(ta => {
        const model = ta.dataset.model;
        const cat = ta.dataset.cat;
        const state = loadState();
        if (state[model]?.[cat]?.comment) ta.value = state[model][cat].comment;
        ta.addEventListener('input', () => {
            const state = loadState();
            state[model] = state[model] || {};
            state[model][cat] = state[model][cat] || {};
            state[model][cat].comment = ta.value;
            saveState(state);
        });
    });
    const general = document.getElementById('general-comments');
    if (general) {
        const state = loadState();
        if (state._general) general.value = state._general;
        general.addEventListener('input', () => {
            const state = loadState();
            state._general = general.value;
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
function approveAllModel(model) {
    if (!confirm(`Marcar todas as 12 categorias de "${model}" como Correto?`)) return;
    document.querySelectorAll(`select.eval-status[data-model="${model}"]`).forEach(sel => {
        sel.value = 'correto';
        sel.dispatchEvent(new Event('change'));
    });
}
function exportJSON() {
    const state = loadState();
    const reviewer = localStorage.getItem(REVIEWER_KEY) || 'anonimo';
    const out = {
        paciente_id: PID,
        reviewer: reviewer,
        timestamp: new Date().toISOString(),
        models_evaluations: state,
    };
    const blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `model_review_${reviewer.replace(/\W+/g,'_')}_${PID}.json`;
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
    updateProgress();
});
"""

JS_INDEX = r"""
const REVIEWER_KEY = 'iana_reviewer';
const TOTAL_CATS = 12;

function refreshTable() {
    const reviewer = localStorage.getItem(REVIEWER_KEY) || '';
    document.getElementById('reviewer-display').textContent = reviewer || '(nao identificado)';

    let totalSlots = 0;
    let totalFilled = 0;
    document.querySelectorAll('tr[data-pid]').forEach(row => {
        const pid = row.dataset.pid;
        const cell = row.querySelector('.status-cell');
        const raw = localStorage.getItem('iana_models_' + pid);
        if (!raw) {
            cell.innerHTML = '<span class="badge-status bs-pendente">pendente</span>';
            totalSlots += 4 * TOTAL_CATS;
            return;
        }
        try {
            const state = JSON.parse(raw);
            let badges = '';
            ['silver','qwen35_4b','gemma4_e4b','medgemma'].forEach(m => {
                const ms = state[m] || {};
                const filled = Object.keys(ms).filter(k => !k.startsWith('_') && ms[k]?.status).length;
                totalFilled += filled;
                totalSlots += TOTAL_CATS;
                let cls = 'bs-pendente';
                if (filled >= TOTAL_CATS) cls = 'bs-completo';
                else if (filled > 0) cls = 'bs-iniciado';
                badges += `<span class="badge-status ${cls}" title="${m}">${m.split('_')[0]}: ${filled}/${TOTAL_CATS}</span>`;
            });
            cell.innerHTML = badges;
        } catch (e) {
            cell.innerHTML = '<span class="badge-status bs-pendente">erro</span>';
        }
    });
    document.getElementById('progress-summary').textContent =
        `${totalFilled} de ${totalSlots} avaliações preenchidas no total`;
}
function exportAll() {
    const reviewer = localStorage.getItem(REVIEWER_KEY) || 'anonimo';
    const allData = { reviewer: reviewer, timestamp: new Date().toISOString(), notes: [] };
    document.querySelectorAll('tr[data-pid]').forEach(row => {
        const pid = row.dataset.pid;
        const raw = localStorage.getItem('iana_models_' + pid);
        if (raw) {
            try {
                allData.notes.push({ paciente_id: pid, models_evaluations: JSON.parse(raw) });
            } catch (e) {}
        }
    });
    const blob = new Blob([JSON.stringify(allData, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `iana_models_review_${reviewer.replace(/\W+/g,'_')}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}
function clearAll() {
    if (confirm('Apagar TODAS as avaliacoes deste browser? (sem desfazer)')) {
        const keys = [];
        for (let i = 0; i < localStorage.length; i++) {
            const k = localStorage.key(i);
            if (k.startsWith('iana_models_')) keys.push(k);
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


def _eval_block_html(model_key: str, cat_key: str) -> str:
    return f"""
<div class="eval-row">
    <select class="eval-status" data-model="{model_key}" data-cat="{cat_key}">
        <option value="">— avaliar —</option>
        <option value="correto">Correto</option>
        <option value="parcial">Parcial</option>
        <option value="incorreto">Incorreto</option>
    </select>
    <textarea class="eval-comment" data-model="{model_key}" data-cat="{cat_key}"
              placeholder="comentario (opcional)"></textarea>
</div>"""


def _model_column_html(model_key: str, model_label: str, color: str,
                       ner: dict, soap: dict) -> str:
    parts = [f'<div class="model-col">',
             f'  <div class="model-col-header" style="background:{color};">',
             f'    <span>{html.escape(model_label)}</span>',
             f'    <button onclick="approveAllModel(\'{model_key}\')">✓ tudo correto</button>',
             f'  </div>']

    # NER
    for cat_key, cat_label in NER_CATEGORY_LABELS.items():
        items = _get_ner_items(ner, cat_key)
        parts.append(f'<div class="cat-block">')
        parts.append(f'  <h3>{html.escape(cat_label)} <span class="count">({len(items)})</span></h3>')
        if items:
            parts.append("  <ul>")
            for item in items:
                parts.append(f"    <li>{html.escape(item)}</li>")
            parts.append("  </ul>")
        else:
            parts.append('  <p class="empty">vazio</p>')
        parts.append(_eval_block_html(model_key, f"ner_{cat_key}"))
        parts.append("</div>")

    # SOAP
    for cat_key, cat_label in SOAP_FIELD_LABELS.items():
        content = _get_soap_value(soap, cat_key)
        parts.append(f'<div class="cat-block">')
        parts.append(f'  <h3>{html.escape(cat_label)}</h3>')
        if content and content.strip():
            parts.append(f'  <div class="soap-body">{html.escape(content)}</div>')
        else:
            parts.append('  <p class="empty">vazio</p>')
        parts.append(_eval_block_html(model_key, f"soap_{cat_key}"))
        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


def _generate_note_html(pid: str, doenca: str, cid: str, original_text: str,
                        records_by_model: dict) -> str:
    parts = [f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Comparação Modelos — {html.escape(pid)} ({html.escape(doenca)})</title>
<style>{CSS}</style>
</head>
<body>
<div class="reviewer-bar">
    <label for="reviewer-name">Revisor:</label>
    <input type="text" id="reviewer-name" placeholder="Seu nome"/>
    <span class="nav"><a href="index.html">← Índice</a></span>
</div>

<h1>Comparação de Modelos — Nota {html.escape(pid)}</h1>
<div class="header-meta">
    <strong>Paciente ID:</strong> {html.escape(pid)} &nbsp;|&nbsp;
    <strong>Doença alvo:</strong> {html.escape(doenca)} &nbsp;|&nbsp;
    <strong>CID:</strong> {html.escape(cid)}
</div>"""]

    if pid == "27306123":
        parts.append("""
<div class="alert-pathological">
    <strong>⚠️ Caso patológico de erro de codificação:</strong> esta nota tem CID
    de neurossífilis mas o texto NÃO menciona sífilis. Modelos corretos devem
    extrair listas vazias ou apenas as condições reais (pé de Charcot, etc).
    Avalie se houve alucinação.
</div>""")

    parts.append(f"""
<h2>1. Nota Original (inglês)</h2>
<div class="original-text">{html.escape(original_text)}</div>

<h2>2. Extrações dos modelos — comparação lado a lado</h2>
<div class="models-grid">""")

    for model in MODELS:
        rec = records_by_model.get(model["key"], {})
        ner = rec.get("ner", {})
        soap = rec.get("soap", {})
        parts.append(_model_column_html(model["key"], model["label"], model["color"], ner, soap))

    parts.append("</div>")  # /models-grid

    parts.append("""
<h2>3. Comentários Gerais (opcional)</h2>
<textarea id="general-comments" style="width:100%; min-height:60px; padding:8px;
          border:1px solid #cbd5e1; border-radius:4px; font-family:inherit; font-size:0.9rem;"
          placeholder="Observacoes sobre a nota e os modelos: padroes de erro, qual modelo melhor, etc."></textarea>""")

    parts.append("""
<div class="export-bar">
    <span class="progress" id="progress-text">0 de 48 avaliações preenchidas</span>
    <button class="secondary" onclick="clearLocal()">Limpar nota</button>
    <button onclick="exportJSON()">Exportar JSON desta nota</button>
</div>""")

    models_js = json.dumps([m["key"] for m in MODELS])
    js = JS_PER_NOTE.replace("__PID__", pid).replace("__MODELS__", models_js)

    parts.append(f"""
<div class="footer">
    Projeto IANA — Comparação de modelos | Gerado em {time.strftime('%Y-%m-%d')}
</div>
<script>{js}</script>
</body>
</html>""")
    return "\n".join(parts)


def _generate_index_html(records: list[dict]) -> str:
    rows = []
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
            f"<td class='status-cell'><span class='badge-status bs-pendente'>pendente</span></td>"
            f"</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IANA — Comparação de Modelos (validação clínica)</title>
<style>{CSS}</style>
</head>
<body>
<h1>Projeto IANA — Comparação de Modelos</h1>
<p style="margin:12px 0 24px; color:#4b5563; font-size:0.92rem;">
    Para cada uma das 30 notas estratificadas, compare lado a lado as extrações
    de 4 modelos: <strong>Silver/Qwen 122B (teacher)</strong> + 3 modelos compactos
    finetuned (<strong>Qwen 3.5-4B</strong>, <strong>Gemma 4 E4B</strong>,
    <strong>MedGemma 4B</strong>). Avalie cada categoria de cada modelo
    independentemente.
</p>

<div class="reviewer-bar">
    <label for="reviewer-name">Identifique-se:</label>
    <input type="text" id="reviewer-name" placeholder="Seu nome" oninput="setReviewer()"/>
    <span style="color:#6b7280; font-size:0.85rem;">
        Revisor atual: <strong id="reviewer-display">(não identificado)</strong>
    </span>
</div>

<div style="background:#eff6ff; border-radius:6px; padding:12px 16px;
     margin-bottom:16px; font-size:0.88rem; color:#1e40af;">
    <strong>Como avaliar:</strong><br>
    Cada nota tem 4 colunas (uma por modelo). Para cada categoria (6 NER + 6 SOAP)
    de cada modelo, atribua: <strong>Correto</strong> / <strong>Parcial</strong> /
    <strong>Incorreto</strong>. Comentário é opcional. Botão "✓ tudo correto"
    no topo de cada coluna marca tudo daquele modelo de uma vez.<br><br>
    <strong>Carga estimada:</strong> ~10-15 min por nota se avaliar todos os 4
    modelos. Para reduzir, foque nos modelos que mais te interessam.<br><br>
    <strong>Auto-save:</strong> tudo é salvo no navegador automaticamente.
    Quando terminar, clique em <strong>Exportar TUDO</strong> abaixo.
</div>

<table class="index">
    <thead>
        <tr>
            <th>Paciente ID</th>
            <th>Doença</th>
            <th>Entidades (Silver)</th>
            <th>Status por modelo</th>
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
<title>IANA — Instruções (comparação modelos)</title>
<style>{CSS}</style>
</head>
<body>
<div class="reviewer-bar"><span class="nav"><a href="index.html">← Índice</a></span></div>
<h1>Instruções para validação clínica dos modelos</h1>

<h2>Contexto</h2>
<p>Você vai avaliar 30 notas clínicas processadas por <strong>4 modelos de IA</strong>:</p>
<ul style="padding-left:24px; margin:12px 0;">
    <li><strong>Silver / Qwen 122B</strong>: modelo grande "professor" que gerou
    o conjunto de treino. Serve como referência (não é necessariamente perfeito).</li>
    <li><strong>Qwen 3.5-4B</strong>: modelo compacto multilingual (4B parâmetros), fine-tuned no silver.</li>
    <li><strong>Gemma 4 E4B</strong>: modelo compacto Google (7.9B params, ~3.4B ativos), multilingual, fine-tuned.</li>
    <li><strong>MedGemma 4B</strong>: modelo Google com pretraining médico em inglês, fine-tuned.</li>
</ul>
<p>O objetivo é validar a qualidade clínica das extrações: NER (6 categorias) e SOAP (6 campos).</p>

<h2>Como avaliar</h2>
<ul style="padding-left:24px; margin:12px 0;">
    <li>Para cada nota, há <strong>4 colunas lado a lado</strong> (uma por modelo).</li>
    <li>Em cada coluna, há <strong>12 blocos</strong>: 6 categorias NER + 6 campos SOAP.</li>
    <li>Em cada bloco há <strong>1 select</strong>: Correto / Parcial / Incorreto, e
    1 textarea de comentário <strong>opcional</strong>.</li>
    <li>Atalho: o botão "✓ tudo correto" no topo de cada coluna marca todos os 12
    blocos daquele modelo como Correto de uma vez (útil quando o modelo acertou em geral).</li>
</ul>

<h2>Critério</h2>
<ul style="padding-left:24px; margin:12px 0;">
    <li><strong>Correto</strong>: extração captura adequadamente o que está no texto.</li>
    <li><strong>Parcial</strong>: captura parte mas omite ou inclui demais (use comentário).</li>
    <li><strong>Incorreto</strong>: categoria errada, alucinação, ou erro grave.</li>
</ul>

<h2>Funcionamento técnico</h2>
<ul style="padding-left:24px; margin:12px 0;">
    <li>Tudo salvo automaticamente no navegador (localStorage).</li>
    <li>Pode fechar e voltar quando quiser — os dados ficam.</li>
    <li>Use sempre o mesmo navegador na mesma máquina.</li>
    <li>Quando terminar, no índice clique <strong>Exportar TUDO (JSON)</strong> e me envie o arquivo.</li>
</ul>

<h2>Carga de trabalho estimada</h2>
<p>~10-15 min por nota se avaliar todos os 4 modelos = ~5-7h total para 30 notas.
Para reduzir: foque em 1 ou 2 modelos de interesse (ex.: só "Qwen 3.5-4B"),
deixando outros pendentes. Mesmo avaliação parcial é valiosa.</p>

<h2>Aviso sobre caso patológico</h2>
<p>A nota do paciente <strong>27306123</strong> tem CID indicando neurossífilis,
mas o texto da nota não menciona sífilis em momento algum (erro de codificação).
Modelos corretos devem retornar listas vazias para sífilis ou apenas extrair
as condições reais (pé de Charcot, neuropatia). Se algum modelo extraiu sífilis,
isso é alucinação.</p>

<div class="footer">Projeto IANA — Validação clínica de modelos compactos</div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="IANA — Gera HTMLs side-by-side de comparacao de modelos")
    parser.add_argument("--gold", default=str(_EXPERIMENTS_DIR / "resultados" / "gold_test_set_30.json"),
                        help="JSON do gold (silver) — usado como coluna 'Silver'.")
    parser.add_argument("--predictions-dir", default=str(_TRAINING_DIR / "predictions"),
                        help="Diretorio com {model}_predictions.json para os 3 decoders.")
    parser.add_argument("--parquet", default=str(_EXPERIMENTS_DIR / "dados" / "mimic_filtrado_tb_hiv_sifilis.parquet"),
                        help="Parquet com textos originais.")
    parser.add_argument("--output-dir", default=str(_EXPERIMENTS_DIR / "resultados" / "medical_review_models"),
                        help="Diretorio de saida.")
    args = parser.parse_args()

    json_path = Path(args.gold)
    pred_dir = Path(args.predictions_dir)
    parquet_path = Path(args.parquet)
    output_dir = Path(args.output_dir)

    # Carrega silver (gold)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("notes", data) if isinstance(data, dict) else data
    print(f"[info] {len(records)} notas carregadas de {json_path.name}")

    silver_by_pid = {r["paciente_id"]: r for r in records}

    # Carrega predictions dos 3 decoders
    preds_by_model = {}
    for model_key in ("qwen35_4b", "gemma4_e4b", "medgemma"):
        pred_path = pred_dir / f"{model_key}_predictions.json"
        if not pred_path.exists():
            print(f"[warn] predictions nao encontradas: {pred_path}")
            continue
        with open(pred_path, encoding="utf-8") as f:
            pred_list = json.load(f)
        preds_by_model[model_key] = {p["paciente_id"]: p.get("predictions", {}) for p in pred_list}
        print(f"[info] {len(preds_by_model[model_key])} predictions carregadas para {model_key}")

    # Carrega textos originais
    import polars as pl
    df = pl.read_parquet(str(parquet_path))
    pids = set(silver_by_pid.keys())
    df_f = df.filter(pl.col("hadm_id").cast(pl.Utf8).is_in(pids))
    texts = {str(row["hadm_id"]): row.get("text", "") for row in df_f.to_dicts()}
    print(f"[info] {len(texts)} textos originais carregados")

    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = []
    for record in records:
        pid = record["paciente_id"]
        doenca = record.get("doenca_alvo_identificada", "")
        cid = record.get("codigo_cid", "")
        text = texts.get(pid, "[texto original nao encontrado]")

        records_by_model = {
            "silver": {"ner": record.get("ner", {}), "soap": record.get("soap", {})},
        }
        for model_key in ("qwen35_4b", "gemma4_e4b", "medgemma"):
            pred = preds_by_model.get(model_key, {}).get(pid, {})
            if isinstance(pred, dict):
                records_by_model[model_key] = {
                    "ner": {k: v for k, v in pred.items() if k in NER_CATEGORY_LABELS},
                    "soap": {k: v for k, v in pred.items() if k in SOAP_FIELD_LABELS or k in
                             {v2 for variants in SOAP_KEY_VARIANTS.values() for v2 in variants}},
                }
            else:
                records_by_model[model_key] = {"ner": {}, "soap": {}}

        html_content = _generate_note_html(pid, doenca, cid, text, records_by_model)
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
    print(f"  COMPARACAO DE MODELOS GERADA")
    print(f"{'=' * 60}")
    print(f"  Diretorio: {output_dir}")
    print(f"  Arquivos:  {len(sizes)} notas + index.html + instrucoes.html")
    print(f"  Tamanho medio: {avg_kb:.1f} KB por nota")
    print(f"\n  Para o medico: abrir index.html em qualquer navegador.")
    print(f"  Tudo offline (auto-save em localStorage).")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
