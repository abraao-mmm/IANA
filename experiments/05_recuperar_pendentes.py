#!/usr/bin/env python3
"""
Projeto IANA - Recuperacao de notas com extracao falha.

Identifica notas no banco_dados_iana_oficial.json que produziram
0 entidades (falha silenciosa do Qwen thinking mode) e re-processa
com parametros ajustados para maximizar a taxa de recuperacao.

Uso:
    python 05_recuperar_pendentes.py
    python 05_recuperar_pendentes.py --max-tokens 32768
    python 05_recuperar_pendentes.py --json resultados/banco_dados_iana_oficial.json
    python 05_recuperar_pendentes.py --url http://localhost:8000/v1

Criterio de falha:
    - Todos os campos NER sao listas vazias, OU
    - soap.subjetivo comeca com "ERRO"
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Garante que imports relativos funcionem independente do CWD
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from langchain_openai import ChatOpenAI

from graphs.extracao import criar_grafo_extracao


# ============================================================
# 1. Identificacao de notas pendentes
# ============================================================

def _ner_vazio(ner: dict) -> bool:
    """Retorna True se todas as listas NER estao vazias."""
    return all(
        isinstance(v, list) and len(v) == 0
        for v in ner.values()
    )


def _soap_falhou(soap: dict) -> bool:
    """Retorna True se o SOAP indica falha de extracao."""
    subj = soap.get("subjetivo", "")
    return isinstance(subj, str) and subj.startswith("ERRO")


def identificar_pendentes(caminho_json: str) -> list[dict]:
    """
    Le o JSON de resultados e retorna as entradas com extracao falha.

    Returns:
        Lista de dicts com chaves: paciente_id, codigo_cid, doenca_alvo_identificada.
    """
    with open(caminho_json, encoding="utf-8") as f:
        resultados = json.load(f)

    pendentes = []
    for r in resultados:
        ner = r.get("ner", {})
        soap = r.get("soap", {})
        if _ner_vazio(ner) or _soap_falhou(soap):
            pendentes.append({
                "paciente_id": r.get("paciente_id", "___"),
                "codigo_cid": r.get("codigo_cid", ""),
                "doenca_alvo_identificada": r.get("doenca_alvo_identificada", ""),
            })

    print(f"[PENDENTES] {len(pendentes)} notas com extracao falha de {len(resultados)} total")
    return pendentes


# ============================================================
# 2. Carregamento do texto original pelo parquet
# ============================================================

def carregar_textos_parquet(
    caminho_parquet: str,
    hadm_ids: set[str],
) -> dict[str, dict]:
    """
    Carrega do parquet apenas as notas cujos hadm_id estao em hadm_ids.

    Returns:
        Dict {hadm_id: {hadm_id, codigo_cid, doenca_alvo, text}}
    """
    import polars as pl

    df = pl.read_parquet(caminho_parquet)
    print(f"[PARQUET] {df.height} notas no parquet, colunas: {df.columns}")

    # Filtra pelas notas pendentes
    col_id = "hadm_id"
    df_filtrado = df.filter(
        pl.col(col_id).cast(pl.Utf8).is_in(hadm_ids)
    )
    print(f"[PARQUET] {df_filtrado.height} notas pendentes encontradas no parquet")

    notas = {}
    for row in df_filtrado.to_dicts():
        hid = str(row.get("hadm_id", "___"))
        notas[hid] = {
            "hadm_id": hid,
            "codigo_cid": str(row.get("codigo_cid", "")),
            "doenca_alvo": str(row.get("doenca_alvo", "")),
            "text": row.get("text", row.get("texto_prontuario", "")),
        }

    return notas


# ============================================================
# 3. Reprocessamento
# ============================================================

def reprocessar_pendentes(
    llm,
    notas_texto: dict[str, dict],
    pendentes: list[dict],
    max_retries: int = 2,
) -> tuple[list[dict], list[str]]:
    grafo = criar_grafo_extracao(llm)

    recuperados = []
    ainda_falhos = []

    total = len(pendentes)
    print(f"\n{'='*60}")
    print(f"  RECUPERACAO - {total} NOTAS PENDENTES (max {max_retries} tentativas)")
    print(f"{'='*60}\n")

    for i, pendente in enumerate(pendentes):
        hadm = pendente["paciente_id"]
        nota = notas_texto.get(hadm)

        if not nota:
            print(f"[{i+1}/{total}] hadm={hadm}: NAO encontrado no parquet")
            ainda_falhos.append(hadm)
            continue

        tamanho = len(nota.get("text", ""))
        resultado = None
        sucesso = False

        for tentativa in range(1, max_retries + 1):
            print(f"[{i+1}/{total}] hadm={hadm} ({tamanho} chars) tentativa {tentativa}/{max_retries}...", end=" ", flush=True)
            t0 = time.perf_counter()

            try:
                estado_final = grafo.invoke({
                    "hadm_id": hadm,
                    "codigo_cid": nota.get("codigo_cid", ""),
                    "doenca_alvo": nota.get("doenca_alvo", pendente.get("doenca_alvo_identificada", "")),
                    "texto_prontuario": nota["text"],
                })

                resultado = estado_final.get("resultado_json")
                duracao = time.perf_counter() - t0

                if resultado:
                    ner = resultado.get("ner", {})
                    soap = resultado.get("soap", {})
                    total_ent = sum(len(v) for v in ner.values() if isinstance(v, list))
                    soap_ok = not _soap_falhou(soap)

                    if total_ent > 0 and soap_ok:
                        recuperados.append(resultado)
                        print(f"RECUPERADO ({total_ent} entidades, {duracao:.1f}s)")
                        sucesso = True
                        break
                    else:
                        print(f"VAZIO ({duracao:.1f}s)")
                else:
                    print(f"SEM RESULTADO ({time.perf_counter() - t0:.1f}s)")

            except Exception as e:
                print(f"EXCECAO ({time.perf_counter() - t0:.1f}s): {e}")

        if not sucesso:
            ainda_falhos.append(hadm)
            if resultado:
                recuperados.append(resultado)

    return recuperados, ainda_falhos



# ============================================================
# 4. Merge de resultados
# ============================================================

def mesclar_resultados(
    caminho_json_original: str,
    novos_resultados: list[dict],
    caminho_saida: str,
):
    """
    Substitui entradas falhas no JSON original pelos novos resultados.
    Notas recuperadas sobrescrevem as originais pelo paciente_id.
    """
    with open(caminho_json_original, encoding="utf-8") as f:
        resultados_originais = json.load(f)

    # Indexa novos resultados por paciente_id
    novos_por_id = {r["paciente_id"]: r for r in novos_resultados}

    substituidos = 0
    resultados_finais = []
    for r in resultados_originais:
        pid = r.get("paciente_id", "___")
        if pid in novos_por_id:
            resultados_finais.append(novos_por_id[pid])
            substituidos += 1
        else:
            resultados_finais.append(r)

    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(resultados_finais, f, indent=2, ensure_ascii=False)

    print(f"\n[MERGE] {substituidos} notas substituidas no banco de dados")
    print(f"[SALVO] {len(resultados_finais)} prontuarios -> {caminho_saida}")


# ============================================================
# 5. Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="IANA - Recuperacao de notas com extracao falha",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python 05_recuperar_pendentes.py
  python 05_recuperar_pendentes.py --max-tokens 32768
  python 05_recuperar_pendentes.py --url http://10.0.0.1:8000/v1
  python 05_recuperar_pendentes.py --json resultados/banco_dados_iana_oficial.json \\
                                   --parquet dados/mimic_filtrado_tb_hiv_sifilis.parquet
        """,
    )
    parser.add_argument(
        "--json",
        default="resultados/banco_dados_iana_oficial.json",
        help="Caminho do JSON de resultados gerado pelo pipeline principal.",
    )
    parser.add_argument(
        "--parquet",
        default="dados/mimic_filtrado_tb_hiv_sifilis.parquet",
        help="Caminho do parquet com os textos originais das notas.",
    )
    parser.add_argument(
        "--saida",
        default=None,
        help="Caminho do JSON de saida apos merge (default: sobrescreve --json).",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/v1",
        help="URL do servidor vLLM.",
    )
    parser.add_argument(
        "--api-key",
        default="iana-local-key",
        help="API key do servidor vLLM.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-122B-A10B",
        help="Nome do modelo no servidor vLLM.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="max_tokens para a LLM na recuperacao (default: 32768, maior que no pipeline principal).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperatura (default: 0.0 - deterministico para maximizar recuperacao).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout em segundos por nota (default: 300 - mais generoso para notas longas).",
    )

    args = parser.parse_args()
    saida = args.saida or args.json

    # Resolve caminhos relativos ao diretorio experiments/
    json_path = Path(args.json)
    if not json_path.is_absolute():
        json_path = _EXPERIMENTS_DIR / json_path

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = _EXPERIMENTS_DIR / parquet_path

    saida_path = Path(saida)
    if not saida_path.is_absolute():
        saida_path = _EXPERIMENTS_DIR / saida_path

    print(f"[CONFIG] JSON:       {json_path}")
    print(f"[CONFIG] Parquet:    {parquet_path}")
    print(f"[CONFIG] Saida:      {saida_path}")
    print(f"[CONFIG] Servidor:   {args.url}")
    print(f"[CONFIG] max_tokens: {args.max_tokens}")
    print(f"[CONFIG] temp:       {args.temperature}")
    print(f"[CONFIG] timeout:    {args.timeout}s")
    print()

    # Etapa 1: identificar pendentes
    pendentes = identificar_pendentes(str(json_path))
    if not pendentes:
        print("[OK] Nenhuma nota pendente encontrada. Banco de dados completo!")
        return

    # Etapa 2: carregar textos do parquet
    hadm_ids = {p["paciente_id"] for p in pendentes}
    notas_texto = carregar_textos_parquet(str(parquet_path), hadm_ids)

    # Etapa 3: criar LLM com parametros otimizados para recuperacao
    llm = ChatOpenAI(
        base_url=args.url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        timeout=args.timeout,
    )

    # Etapa 4: reprocessar
    t_inicio = time.perf_counter()
    novos_resultados, ainda_falhos = reprocessar_pendentes(llm, notas_texto, pendentes)
    t_total = time.perf_counter() - t_inicio

    # Etapa 5: mesclar de volta ao JSON principal
    if novos_resultados:
        mesclar_resultados(str(json_path), novos_resultados, str(saida_path))

    # Resumo final
    recuperados_count = len(pendentes) - len(ainda_falhos)
    print(f"\n{'='*60}")
    print(f"  RESUMO - RECUPERACAO")
    print(f"{'='*60}")
    print(f"  Notas pendentes:    {len(pendentes)}")
    print(f"  Recuperadas:        {recuperados_count}")
    print(f"  Ainda sem extracao: {len(ainda_falhos)}")
    if ainda_falhos:
        print(f"  IDs nao recuperados: {ainda_falhos[:20]}{'...' if len(ainda_falhos) > 20 else ''}")
    print(f"  Tempo total:        {t_total:.0f}s ({t_total/60:.1f} min)")
    print(f"  Resultado salvo:    {saida_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
