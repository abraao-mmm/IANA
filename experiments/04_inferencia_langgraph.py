#!/usr/bin/env python3
"""
Projeto IANA - Pipeline de Inferencia LangGraph + vLLM.

Uso:
    # Teste com 3 amostras (recomendado primeiro):
    python 04_inferencia_langgraph.py --modo amostras

    # Producao com todo o parquet:
    python 04_inferencia_langgraph.py --modo parquet

    # Opcoes adicionais:
    python 04_inferencia_langgraph.py --modo amostras --url http://localhost:8000/v1
    python 04_inferencia_langgraph.py --modo parquet --parquet dados/mimic_filtrado_tb_hiv_sifilis.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Garante que imports relativos funcionem independente do CWD
# (ex: VS Code pode rodar com CWD no root do repo, nao em experiments/)
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from langchain_openai import ChatOpenAI

from graphs.extracao import criar_grafo_extracao
from graphs.batch import carregar_notas_amostras, carregar_notas_parquet


def criar_llm(base_url: str, api_key: str, model: str, temperature: float = 0.1):
    """Cria a instancia do ChatOpenAI conectada ao servidor vLLM."""
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=16384,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )





def executar_amostras(llm, diretorio: str = "amostras", saida: str = "resultados/banco_dados_iana_langgraph_amostras.json"):
    """
    Executa o pipeline nas 3 amostras de teste.
    Processa uma a uma para mostrar progresso e metricas detalhadas.
    """
    notas = carregar_notas_amostras(diretorio)
    grafo = criar_grafo_extracao(llm)

    resultados = []
    tempos = []

    print(f"\n{'='*60}")
    print(f"  PIPELINE LANGGRAPH - TESTE COM {len(notas)} AMOSTRAS")
    print(f"{'='*60}\n")

    for i, nota in enumerate(notas):
        doenca = nota["doenca_alvo"]
        hadm = nota["hadm_id"]
        tamanho = len(nota["text"])

        print(f"[{i+1}/{len(notas)}] Processando: {doenca} (hadm={hadm}, {tamanho} chars)")

        t0 = time.perf_counter()

        # Executa o grafo com streaming para ver os passos
        estado_final = None
        for step in grafo.stream({
            "hadm_id": hadm,
            "codigo_cid": nota["codigo_cid"],
            "doenca_alvo": doenca,
            "texto_prontuario": nota["text"],
        }):
            # Cada step e um dict {nome_no: estado_parcial}
            for nome_no, _ in step.items():
                t_parcial = time.perf_counter() - t0
                print(f"  -> {nome_no} concluido ({t_parcial:.1f}s)")
            estado_final = step

        t1 = time.perf_counter()
        duracao = t1 - t0
        tempos.append(duracao)

        # Extrai resultado do ultimo step
        resultado = None
        if estado_final:
            for valor in estado_final.values():
                if isinstance(valor, dict) and "resultado_json" in valor:
                    resultado = valor["resultado_json"]

        if resultado:
            resultados.append(resultado)

            # Metricas do resultado
            ner = resultado.get("ner", {})
            total_entidades = sum(len(v) for v in ner.values() if isinstance(v, list))
            print(f"  -> TOTAL: {total_entidades} entidades extraidas em {duracao:.1f}s")

            # Detalhamento por categoria
            for cat, items in ner.items():
                if items:
                    print(f"     {cat}: {len(items)}")
        else:
            print(f"  -> ERRO: Resultado nao obtido")

        print()

    # Salvar resultados
    Path(saida).parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # Resumo final
    print(f"{'='*60}")
    print(f"  RESUMO FINAL")
    print(f"{'='*60}")
    print(f"  Notas processadas: {len(resultados)}/{len(notas)}")
    print(f"  Tempo total:       {sum(tempos):.1f}s")
    print(f"  Tempo medio:       {sum(tempos)/len(tempos):.1f}s por nota")
    if tempos:
        print(f"  Mais rapido:       {min(tempos):.1f}s")
        print(f"  Mais lento:        {max(tempos):.1f}s")
    print(f"  Resultado salvo:   {saida}")
    print(f"{'='*60}\n")

    return resultados


def executar_parquet(llm, caminho_parquet: str, saida: str = "resultados/banco_dados_iana_oficial.json"):
    """
    Executa o pipeline em todas as notas do parquet.
    Processa sequencialmente com salvamento incremental.
    """
    notas = carregar_notas_parquet(caminho_parquet)
    grafo = criar_grafo_extracao(llm)

    resultados = []
    erros = []
    t_global = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"  PIPELINE LANGGRAPH - PRODUCAO ({len(notas)} NOTAS)")
    print(f"{'='*60}\n")

    for i, nota in enumerate(notas):
        hadm = str(nota.get("hadm_id", "___"))
        tamanho = len(nota.get("text", ""))

        print(f"[{i+1}/{len(notas)}] hadm={hadm} ({tamanho} chars)...", end=" ", flush=True)

        t0 = time.perf_counter()

        try:
            estado_final = grafo.invoke({
                "hadm_id": hadm,
                "codigo_cid": str(nota.get("codigo_cid", "")),
                "doenca_alvo": str(nota.get("doenca_alvo", "")),
                "texto_prontuario": nota.get("text", ""),
            })

            resultado = estado_final.get("resultado_json")
            if resultado:
                resultados.append(resultado)
                duracao = time.perf_counter() - t0
                ner = resultado.get("ner", {})
                total_ent = sum(len(v) for v in ner.values() if isinstance(v, list))
                print(f"OK ({total_ent} entidades, {duracao:.1f}s)")
            else:
                print("WARN: sem resultado")
                erros.append(hadm)

        except Exception as e:
            duracao = time.perf_counter() - t0
            print(f"ERRO ({duracao:.1f}s): {e}")
            erros.append(hadm)

        # Salvamento incremental a cada 10 notas
        if (i + 1) % 10 == 0:
            Path(saida).parent.mkdir(parents=True, exist_ok=True)
            with open(saida, "w", encoding="utf-8") as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)
            print(f"  [CHECKPOINT] {len(resultados)} salvos em {saida}")

    # Salvamento final
    Path(saida).parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    t_total = time.perf_counter() - t_global

    print(f"\n{'='*60}")
    print(f"  RESUMO FINAL - PRODUCAO")
    print(f"{'='*60}")
    print(f"  Notas processadas: {len(resultados)}/{len(notas)}")
    print(f"  Erros:             {len(erros)}")
    if erros:
        print(f"  IDs com erro:      {erros[:20]}{'...' if len(erros) > 20 else ''}")
    print(f"  Tempo total:       {t_total:.0f}s ({t_total/60:.1f} min)")
    if resultados:
        print(f"  Tempo medio:       {t_total/len(notas):.1f}s por nota")
    print(f"  Resultado salvo:   {saida}")
    print(f"{'='*60}\n")

    return resultados


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline IANA - Extracao clinica com LangGraph + vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python 04_inferencia_langgraph.py --modo amostras
  python 04_inferencia_langgraph.py --modo parquet
  python 04_inferencia_langgraph.py --modo amostras --url http://10.0.0.1:8000/v1
        """,
    )
    parser.add_argument(
        "--modo",
        choices=["amostras", "parquet"],
        required=True,
        help="'amostras' para testar com 3 notas, 'parquet' para producao completa.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/v1",
        help="URL do servidor vLLM (default: http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--api-key",
        default="iana-local-key",
        help="API key do servidor vLLM (default: iana-local-key).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-122B-A10B",
        help="Nome do modelo no servidor vLLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperatura da LLM (default: 0.1).",
    )
    parser.add_argument(
        "--parquet",
        default="dados/mimic_filtrado_tb_hiv_sifilis.parquet",
        help="Caminho do parquet (modo parquet).",
    )
    parser.add_argument(
        "--saida",
        default=None,
        help="Caminho do arquivo JSON de saida.",
    )

    args = parser.parse_args()

    # Conecta ao servidor vLLM
    print(f"[CONFIG] Servidor: {args.url}")
    print(f"[CONFIG] Modelo:   {args.model}")
    print(f"[CONFIG] Temp:     {args.temperature}")

    llm = criar_llm(
        base_url=args.url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
    )

    if args.modo == "amostras":
        saida = args.saida or "resultados/banco_dados_iana_langgraph_amostras.json"
        executar_amostras(llm, saida=saida)

    elif args.modo == "parquet":
        saida = args.saida or "resultados/banco_dados_iana_oficial.json"
        executar_parquet(llm, caminho_parquet=args.parquet, saida=saida)


if __name__ == "__main__":
    main()
