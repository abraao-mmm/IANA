#!/usr/bin/env python3
"""
Projeto IANA — Seleção do test set gold de 30 notas para validação clínica.

Seleciona 10 HIV + 10 TB + 10 Sífilis, ancorando as 10 notas já revisadas
pelo especialista e completando com sorteio estratificado por complexidade.

Uso:
    python select_gold_test_set.py
    python select_gold_test_set.py --seed 42
    python select_gold_test_set.py --input resultados/banco_dados_iana_v3_clean.json
"""

import argparse
import copy
import datetime
import html as html_mod
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
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
log = logging.getLogger("gold_selection")
if not log.handlers:
    log.addHandler(_handler)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# 10 notas já revisadas pelo especialista (fonte de verdade: run_test_batch.py)
ANCHORED_IDS: dict[str, str] = {
    "25557330": "HIV",
    "22924630": "HIV",
    "24918106": "HIV",
    "22413631": "HIV",
    "23080963": "Sifilis",
    "22978216": "Sifilis",
    "27306123": "Sifilis",
    "20250010": "Tuberculose",
    "27321074": "Tuberculose",
    "20248623": "Tuberculose",
}

# Distribuição alvo por doença
TARGET_PER_DISEASE = 10

# Distribuição de novas notas por complexidade (curtas, médias, longas)
NEW_DISTRIBUTION: dict[str, dict[str, int]] = {
    "HIV":          {"curtas": 2, "medias": 3, "longas": 1},  # 6 novas
    "Tuberculose":  {"curtas": 3, "medias": 3, "longas": 1},  # 7 novas
    "Sifilis":      {"curtas": 3, "medias": 4, "longas": 0},  # 7 novas (sem longas — corpus sífilis < 20K chars)
}

# Limites de complexidade por tamanho de texto
BIN_THRESHOLDS = {"curtas": 8000, "medias": 20000}  # longas: > 20000


# ---------------------------------------------------------------------------
# Normalização de nome de doença
# ---------------------------------------------------------------------------

_DISEASE_ALIASES: dict[str, str] = {
    "hiv": "HIV", "hiv/aids": "HIV",
    "tuberculose": "Tuberculose", "tuberculosis": "Tuberculose", "tb": "Tuberculose",
    "sifilis": "Sifilis", "syphilis": "Sifilis",
}

def _normalize_disease(raw: str) -> str:
    key = raw.strip().lower()
    return _DISEASE_ALIASES.get(key, raw)


def _classify_bin(char_count: int) -> str:
    if char_count < BIN_THRESHOLDS["curtas"]:
        return "curtas"
    elif char_count < BIN_THRESHOLDS["medias"]:
        return "medias"
    return "longas"


def _total_entities(ner: dict) -> int:
    return sum(len(v) for v in ner.values() if isinstance(v, list))


# ---------------------------------------------------------------------------
# Seleção principal
# ---------------------------------------------------------------------------

def select_gold(records: list[dict], seed: int) -> tuple[list[dict], dict]:
    """Seleciona 30 notas com ancoragem e estratificação.

    Returns:
        (selected_records, selection_log)
    """
    random.seed(seed)
    sel_log: dict = {
        "seed": seed,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # Indexar por paciente_id
    by_id: dict[str, dict] = {}
    for r in records:
        by_id[r["paciente_id"]] = r

    # --- Ancoragem ---
    # Nota 27306123 está excluída do banco clean (zero cobertura textual),
    # mas precisa ser ancorada. Carrega do banco bruto se não encontrada.
    if "27306123" not in by_id:
        bruto_path = Path(records[0].get("_source", "")) if records else None
        # Tenta carregar do test_batch_v3.json
        test_batch_path = _EXPERIMENTS_DIR / "resultados" / "test_batch_v3.json"
        if test_batch_path.exists():
            with open(test_batch_path, encoding="utf-8") as f:
                for r in json.load(f):
                    if r.get("paciente_id") == "27306123":
                        by_id["27306123"] = r
                        log.info("Nota 27306123 carregada do test_batch_v3.json (caso patológico)")
                        break

    anchored: list[dict] = []
    anchored_by_disease: dict[str, list[str]] = defaultdict(list)
    missing_anchors: list[str] = []

    for pid, disease in ANCHORED_IDS.items():
        if pid in by_id:
            anchored.append(by_id[pid])
            anchored_by_disease[disease].append(pid)
        else:
            missing_anchors.append(pid)

    if missing_anchors:
        log.warning("Notas ancoradas não encontradas no banco", extra={"data": {"missing": missing_anchors}})

    sel_log["anchored"] = {
        "total": len(anchored),
        "by_disease": {k: len(v) for k, v in anchored_by_disease.items()},
        "missing": missing_anchors,
    }

    anchored_ids_set = set(ANCHORED_IDS.keys())

    # --- Filtros ---
    eligible: list[dict] = []
    filter_stats = {"total": len(records), "excluded_ner": 0, "excluded_soap": 0,
                    "excluded_sparse": 0, "excluded_outlier": 0, "excluded_anchored": 0}

    for r in records:
        pid = r["paciente_id"]
        if pid in anchored_ids_set:
            filter_stats["excluded_anchored"] += 1
            continue
        agent_st = r.get("agent_status", {})
        if agent_st.get("ner_status", "ok") != "ok":
            filter_stats["excluded_ner"] += 1
            continue
        if agent_st.get("soap_status", "ok") != "ok":
            filter_stats["excluded_soap"] += 1
            continue
        ner = r.get("ner", {})
        ent_count = _total_entities(ner)
        if ent_count < 20:
            filter_stats["excluded_sparse"] += 1
            continue
        if ent_count > 300:
            filter_stats["excluded_outlier"] += 1
            continue
        eligible.append(r)

    filter_stats["eligible"] = len(eligible)
    sel_log["filters"] = filter_stats
    log.info("Filtros aplicados", extra={"data": filter_stats})

    # --- Agrupar por doença e bin ---
    pools: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in eligible:
        disease = _normalize_disease(r.get("doenca_alvo_identificada", ""))
        text = r.get("soap", {}).get("subjetivo", "") + r.get("soap", {}).get("avaliacao", "")
        # Estima char_count pelo tamanho total dos campos NER + SOAP
        ner_text = json.dumps(r.get("ner", {}), ensure_ascii=False)
        soap_text = json.dumps(r.get("soap", {}), ensure_ascii=False)
        char_count = len(ner_text) + len(soap_text)
        r["_char_count"] = char_count
        complexity_bin = _classify_bin(char_count)
        r["_bin"] = complexity_bin
        pools[disease][complexity_bin].append(r)

    # Reportar disponibilidade
    pool_counts = {d: {b: len(items) for b, items in bins.items()} for d, bins in pools.items()}
    sel_log["pool_availability"] = pool_counts
    log.info("Pools disponíveis", extra={"data": pool_counts})

    # Verificar suficiência
    for disease in ["HIV", "Tuberculose", "Sifilis"]:
        n_anchored = len(anchored_by_disease.get(disease, []))
        n_needed = TARGET_PER_DISEASE - n_anchored
        n_available = sum(len(pools[disease][b]) for b in ["curtas", "medias", "longas"])
        if n_available < n_needed:
            msg = f"Notas insuficientes para {disease}: precisa {n_needed}, tem {n_available}"
            log.error(msg)
            raise ValueError(msg)

    # --- Sorteio estratificado ---
    new_selected: list[dict] = []
    new_ids: list[str] = []
    selection_details: dict[str, dict] = {}
    adjustments: list[str] = []

    for disease in ["HIV", "Tuberculose", "Sifilis"]:
        dist = NEW_DISTRIBUTION[disease]
        disease_selected: list[dict] = []
        disease_detail: dict[str, list[str]] = {}

        for bin_name in ["curtas", "medias", "longas"]:
            needed = dist[bin_name]
            pool = pools[disease][bin_name]

            if len(pool) >= needed:
                chosen = random.sample(pool, needed)
            else:
                # Bin insuficiente — toma tudo e busca no adjacente
                chosen = list(pool)
                deficit = needed - len(chosen)
                adj_msg = f"{disease}/{bin_name}: pediu {needed}, tem {len(pool)}, deficit {deficit}"
                adjustments.append(adj_msg)
                log.warning(f"Bin insuficiente: {adj_msg}")

                # Tenta bins adjacentes
                if bin_name == "curtas":
                    fallback_bins = ["medias", "longas"]
                elif bin_name == "medias":
                    fallback_bins = ["longas", "curtas"]
                else:
                    fallback_bins = ["medias", "curtas"]

                already_chosen = {r["paciente_id"] for r in chosen}
                for fb_bin in fallback_bins:
                    if deficit <= 0:
                        break
                    fb_pool = [r for r in pools[disease][fb_bin]
                               if r["paciente_id"] not in already_chosen]
                    take = min(deficit, len(fb_pool))
                    if take > 0:
                        extra = random.sample(fb_pool, take)
                        chosen.extend(extra)
                        already_chosen.update(r["paciente_id"] for r in extra)
                        deficit -= take
                        adjustments.append(f"  → compensado com {take} de {disease}/{fb_bin}")

            disease_selected.extend(chosen)
            disease_detail[bin_name] = [r["paciente_id"] for r in chosen]

        new_selected.extend(disease_selected)
        new_ids.extend(r["paciente_id"] for r in disease_selected)
        selection_details[disease] = disease_detail

    sel_log["selection_details"] = selection_details
    sel_log["adjustments"] = adjustments
    sel_log["new_ids"] = new_ids

    # --- Montar resultado final ---
    all_selected: list[dict] = []

    # Ancoradas primeiro
    for r in anchored:
        record = copy.deepcopy(r)
        ner_text = json.dumps(r.get("ner", {}), ensure_ascii=False)
        soap_text = json.dumps(r.get("soap", {}), ensure_ascii=False)
        char_count = len(ner_text) + len(soap_text)
        record["gold_metadata"] = {
            "anchored": True,
            "complexity_bin": _classify_bin(char_count),
            "char_count": char_count,
        }
        # Limpa campos temporários
        record.pop("_char_count", None)
        record.pop("_bin", None)
        all_selected.append(record)

    # Novas
    for r in new_selected:
        record = copy.deepcopy(r)
        record["gold_metadata"] = {
            "anchored": False,
            "complexity_bin": r.get("_bin", "?"),
            "char_count": r.get("_char_count", 0),
        }
        record.pop("_char_count", None)
        record.pop("_bin", None)
        all_selected.append(record)

    # Estatísticas finais
    final_dist: dict[str, dict] = defaultdict(lambda: {"total": 0, "anchored": 0, "new": 0})
    complexity_dist: dict[str, Counter] = defaultdict(Counter)
    for r in all_selected:
        d = _normalize_disease(r.get("doenca_alvo_identificada", ""))
        gm = r.get("gold_metadata", {})
        final_dist[d]["total"] += 1
        if gm.get("anchored"):
            final_dist[d]["anchored"] += 1
        else:
            final_dist[d]["new"] += 1
        complexity_dist[d][gm.get("complexity_bin", "?")] += 1

    sel_log["final_distribution"] = {k: dict(v) for k, v in final_dist.items()}
    sel_log["complexity_distribution"] = {k: dict(v) for k, v in complexity_dist.items()}
    sel_log["total_selected"] = len(all_selected)

    log.info("Seleção concluída", extra={"data": {
        "total": len(all_selected),
        "distribution": {k: dict(v) for k, v in final_dist.items()},
    }})

    return all_selected, sel_log


# ---------------------------------------------------------------------------
# Geração de HTML para as novas notas
# ---------------------------------------------------------------------------

def generate_gold_htmls(
    new_records: list[dict],
    parquet_path: Path,
    output_dir: Path,
) -> int:
    """Gera HTMLs de revisão para as notas NOVAS (não ancoradas)."""
    from generate_medical_review import _generate_note_html, _generate_index_html, CSS

    # Carrega textos originais
    import polars as pl
    df = pl.read_parquet(str(parquet_path))
    pids = {r["paciente_id"] for r in new_records}
    df_f = df.filter(pl.col("hadm_id").cast(pl.Utf8).is_in(pids))
    texts: dict[str, str] = {}
    for row in df_f.to_dicts():
        texts[str(row.get("hadm_id", ""))] = row.get("text", "")

    output_dir.mkdir(parents=True, exist_ok=True)

    for record in new_records:
        pid = record["paciente_id"]
        doenca = record.get("doenca_alvo_identificada", "")
        original = texts.get(pid, "[TEXTO NÃO ENCONTRADO]")
        html_content = _generate_note_html(record, original)
        fname = f"{pid}_{doenca}_review.html"
        with open(output_dir / fname, "w", encoding="utf-8") as f:
            f.write(html_content)

    # Index
    index_html = _generate_index_html(new_records)
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(index_html)

    # Instruções do especialista
    instructions = _generate_instructions_html(new_records)
    with open(output_dir / "instrucoes_especialista.html", "w", encoding="utf-8") as f:
        f.write(instructions)

    return len(new_records) + 2  # HTMLs de notas + index + instruções


def _generate_instructions_html(records: list[dict]) -> str:
    from generate_medical_review import CSS

    disease_counts: Counter = Counter()
    for r in records:
        disease_counts[r.get("doenca_alvo_identificada", "?")] += 1

    dist_rows = "\n".join(
        f"<tr><td>{d}</td><td>{c}</td></tr>" for d, c in sorted(disease_counts.items())
    )

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IANA — Instruções para Revisão do Gold Test Set</title>
<style>{CSS}</style>
</head>
<body>
<h1>Instruções para Revisão — Gold Test Set</h1>

<div class="annotation-instructions" style="margin:24px 0;">
<strong>Propósito:</strong> Esta é a <strong>validação final</strong> do test set gold
do Projeto IANA. Os resultados desta revisão serão usados para:
<ul style="margin:8px 0 0 20px;">
<li>Avaliar os 5 modelos compactos treinados no silver standard</li>
<li>Calcular métricas finais (precision, recall, F1) para o paper na MDPI Diagnostics</li>
<li>Documentar a qualidade do silver standard gerado pelo Qwen 3.5-122B</li>
</ul>
</div>

<h2>Contexto</h2>
<p>Estas <strong>20 notas novas</strong> complementam as <strong>10 notas já revisadas
anteriormente</strong>, totalizando 30 notas no gold test set (10 HIV + 10 TB + 10 Sífilis).
As 10 notas anteriores não precisam de nova revisão — estão ancoradas no gold set
com as anotações já realizadas.</p>

<h2>Distribuição das 20 notas novas</h2>
<table class="annotation" style="width:40%;">
<thead><tr><th>Doença</th><th>Notas</th></tr></thead>
<tbody>{dist_rows}</tbody>
</table>

<h2>O que revisar em cada nota</h2>
<p>Cada arquivo HTML contém 4 seções:</p>
<ol style="margin-left:20px;">
<li><strong>Nota Original (inglês)</strong> — texto bruto do MIMIC-IV</li>
<li><strong>NER Extraído (português)</strong> — 6 categorias de entidades</li>
<li><strong>SOAP Estruturado (português)</strong> — 6 campos</li>
<li><strong>Tabela de Anotação</strong> — para registrar erros encontrados</li>
</ol>

<h2>Tipos de erro a anotar</h2>
<table class="annotation">
<thead><tr><th>Tipo</th><th>Descrição</th><th>Exemplo</th></tr></thead>
<tbody>
<tr><td><strong>Omissão</strong></td><td>Entidade presente no texto original mas não extraída</td>
<td>Hipertensão no PMH não aparece em disease_or_syndrome</td></tr>
<tr><td><strong>Categoria errada</strong></td><td>Entidade extraída na categoria incorreta</td>
<td>"Leucocitose" em disease em vez de lab</td></tr>
<tr><td><strong>Negação vazada</strong></td><td>Teste negativo/pendente gerando entidade positiva</td>
<td>"RPR-Negativo" gerando "Sífilis" em disease</td></tr>
<tr><td><strong>Tradução</strong></td><td>Erro de tradução EN→PT ou termo em inglês residual</td>
<td>"Rash" em vez de "Exantema"</td></tr>
<tr><td><strong>Invenção</strong></td><td>Entidade que não existe no texto original</td>
<td>Sintoma inventado a partir do diagnóstico</td></tr>
<tr><td><strong>Outro</strong></td><td>Qualquer outro tipo de erro</td><td>—</td></tr>
</tbody>
</table>

<h2>Prazo sugerido</h2>
<p>2-3 semanas a partir do recebimento.</p>

<div class="footer">
    Projeto IANA — Pipeline v3.2 | Gerado em {time.strftime('%Y-%m-%d %H:%M')} |
    Material para revisão final do gold test set
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IANA — Seleção do gold test set de 30 notas",
    )
    parser.add_argument("--input", default="resultados/banco_dados_iana_v3_clean.json")
    parser.add_argument("--output-json", default="resultados/gold_test_set_30.json")
    parser.add_argument("--output-html-dir", default="resultados/medical_review_gold")
    parser.add_argument("--parquet", default="dados/mimic_filtrado_tb_hiv_sifilis.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-html", action="store_true", help="Pular geração de HTMLs.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = _EXPERIMENTS_DIR / input_path

    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = _EXPERIMENTS_DIR / output_json

    output_html = Path(args.output_html_dir)
    if not output_html.is_absolute():
        output_html = _EXPERIMENTS_DIR / output_html

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = _EXPERIMENTS_DIR / parquet_path

    # Carregar banco
    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)
    log.info("Banco carregado", extra={"data": {"notas": len(records)}})

    # Selecionar
    selected, sel_log = select_gold(records, seed=args.seed)

    # Montar output JSON
    output_data = {
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "source_file": str(input_path.name),
            "source_total_notes": len(records),
            "selection_seed": args.seed,
            "total_selected": len(selected),
            "distribution": sel_log.get("final_distribution", {}),
            "complexity_distribution": sel_log.get("complexity_distribution", {}),
            "anchored_ids": list(ANCHORED_IDS.keys()),
            "new_ids": sel_log.get("new_ids", []),
        },
        "notes": selected,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    log.info("JSON salvo", extra={"data": {"path": str(output_json)}})

    # Salvar log
    logs_dir = _EXPERIMENTS_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    with open(logs_dir / "gold_selection.json", "w", encoding="utf-8") as f:
        json.dump(sel_log, f, indent=2, ensure_ascii=False)

    # Gerar HTMLs das novas notas
    if not args.no_html:
        new_records = [r for r in selected if not r.get("gold_metadata", {}).get("anchored", False)]
        n_files = generate_gold_htmls(new_records, parquet_path, output_html)
        log.info("HTMLs gerados", extra={"data": {"files": n_files, "dir": str(output_html)}})

    # Resumo
    print(f"\n{'='*60}")
    print(f"  GOLD TEST SET SELECIONADO")
    print(f"{'='*60}")
    print(f"  Total: {len(selected)} notas (seed={args.seed})")
    for disease, dist in sorted(sel_log.get("final_distribution", {}).items()):
        print(f"  {disease}: {dist['total']} (ancoradas: {dist['anchored']}, novas: {dist['new']})")
    print(f"  JSON: {output_json}")
    if not args.no_html:
        print(f"  HTMLs: {output_html}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
