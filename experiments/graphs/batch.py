"""
Projeto IANA - Grafo de processamento em lote (LangGraph Map-Reduce).

Usa a Send() API do LangGraph para distribuir notas clinicas
em paralelo pelo grafo de extracao, acumulando resultados
via reducer (operator.add).
"""

import json
import sys
from pathlib import Path

# Garante que experiments/ esta no sys.path para imports entre modulos
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import polars as pl
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from models.schemas import EstadoBatch
from graphs.extracao import criar_grafo_extracao


def criar_grafo_batch(llm, caminho_saida: str = "resultados/banco_dados_iana_oficial.json"):
    """
    Cria o grafo de processamento em lote.

    Args:
        llm: Instancia ChatOpenAI conectada ao vLLM.
        caminho_saida: Caminho do arquivo JSON de saida.

    Returns:
        Grafo compilado pronto para .invoke()
    """

    grafo_extracao = criar_grafo_extracao(llm)

    def distribuir_notas(state: EstadoBatch):
        """Map: cria uma instancia do grafo para cada nota."""
        return [
            Send("processar_nota", {
                "hadm_id": str(nota.get("hadm_id", "___")),
                "codigo_cid": str(nota.get("codigo_cid", "")),
                "doenca_alvo": str(nota.get("doenca_alvo", "")),
                "texto_prontuario": nota.get("text", nota.get("texto_prontuario", "")),
            })
            for nota in state["notas"]
        ]

    def processar_nota(state):
        """Executa o grafo completo de extracao para uma nota."""
        resultado = grafo_extracao.invoke(state)
        return {"resultados": [resultado.get("resultado_json", {})]}

    def salvar_resultados(state: EstadoBatch):
        """Reduce: salva todos os resultados em JSON."""
        Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
        with open(caminho_saida, "w", encoding="utf-8") as f:
            json.dump(state["resultados"], f, indent=2, ensure_ascii=False)
        print(f"\n[SALVO] {len(state['resultados'])} prontuarios -> {caminho_saida}")
        return state

    builder = StateGraph(EstadoBatch)
    builder.add_node("processar_nota", processar_nota)
    builder.add_node("salvar", salvar_resultados)

    builder.add_conditional_edges(START, distribuir_notas, ["processar_nota"])
    builder.add_edge("processar_nota", "salvar")
    builder.add_edge("salvar", END)

    return builder.compile()


def carregar_notas_parquet(
    caminho_parquet: str,
    excluir_ids: set[str] | None = None,
) -> list[dict]:
    """
    Carrega notas clínicas de um arquivo Parquet.

    Args:
        caminho_parquet: Caminho do arquivo Parquet.
        excluir_ids: Conjunto de hadm_ids a excluir (ex: notas sem cobertura textual).
                     Se None, carrega a lista padrão de config/excluded_notes.py.

    Returns:
        Lista de dicts com chaves: hadm_id, text (e opcionalmente codigo_cid, doenca_alvo).
    """
    if excluir_ids is None:
        from config.excluded_notes import EXCLUDED_PATIENT_IDS
        excluir_ids = EXCLUDED_PATIENT_IDS

    df = pl.read_parquet(caminho_parquet)
    total_antes = df.height
    print(f"[PARQUET] {total_antes} notas carregadas de {caminho_parquet}")
    print(f"[PARQUET] Colunas: {df.columns}")

    if excluir_ids:
        df = df.filter(
            ~pl.col("hadm_id").cast(pl.Utf8).is_in(excluir_ids)
        )
        excluidas = total_antes - df.height
        if excluidas > 0:
            print(f"[PARQUET] {excluidas} notas excluídas por falha de cobertura textual")
        print(f"[PARQUET] {df.height} notas no dataset final")

    return df.to_dicts()


def carregar_notas_amostras(diretorio_amostras: str = "amostras") -> list[dict]:
    """
    Carrega as 3 notas de amostra (.txt) do diretorio de amostras.

    Returns:
        Lista de dicts com chaves: hadm_id, codigo_cid, doenca_alvo, text.
    """
    import re

    notas = []
    diretorio = Path(diretorio_amostras)

    for arquivo in sorted(diretorio.glob("amostra_*.txt")):
        texto = arquivo.read_text(encoding="utf-8")

        # Extrai metadados do nome do arquivo
        # Ex: amostra_HIV_CID_042_subj_18170517_hadm_21473169.txt
        nome = arquivo.stem
        match = re.search(r"amostra_(\w+)_CID_(\w+)_subj_\d+_hadm_(\d+)", nome)

        if match:
            doenca = match.group(1)
            cid = match.group(2)
            hadm_id = match.group(3)
        else:
            doenca = "Desconhecida"
            cid = ""
            hadm_id = "___"

        notas.append({
            "hadm_id": hadm_id,
            "codigo_cid": cid,
            "doenca_alvo": doenca,
            "text": texto,
        })
        print(f"[AMOSTRA] {arquivo.name} -> {doenca} (CID {cid}, hadm {hadm_id})")

    return notas
