# Notas Metodologicas — Evolucao do Grafo de Extracao

## Resumo da divergencia

A metodologia descrita no artigo (secao 3.3) referencia uma arquitetura com
4 agentes LLM: NER Extractor, SOAP Structurer, Completeness Auditor e Gap
Corrector. Porem, o grafo efetivamente usado na execucao das 749 notas contem
apenas **2 agentes LLM + 1 no de montagem**:

```
START -> extrair_ner ---+
START -> extrair_soap --+--> montar_resultado -> END
```

Os nos Auditor (Agent 3) e Corrector (Agent 4) foram removidos antes da
execucao em producao.

## Linha do tempo (baseada em git log)

| Data | Commit | Evento |
|------|--------|--------|
| 2026-03-30 | `b5898d0` | Grafo original implementado com 4 agentes (NER, SOAP, Validador, Corretor) + routing condicional |
| 2026-03-30 | `a6c4421` | Fix de sys.path para imports |
| 2026-03-31 | `abff60f` | **Grafo simplificado para 2 agentes** — removidos nos `validar` e `corrigir`, removidas importacoes de `ValidacaoResult`, `PROMPT_VALIDADOR`, `PROMPT_CORRETOR` |
| 2026-04-01 | `6233599` | Estado final: execucao batch de 749 notas com o grafo simplificado |

## Motivo tecnico da remocao

Durante os testes iniciais na DGX (2x NVIDIA H200, Qwen 3.5-122B-A10B via
vLLM 0.18.0 com FP8), o no Validador apresentou dois problemas criticos:

1. **Estouro de `max_tokens`**: O Qwen 3.5 possui um modo de "thinking"
   (chain-of-thought) que gera tokens de raciocinio antes de produzir o JSON
   de saida. Mesmo com `enable_thinking=False`, certas notas longas ou
   complexas ativavam esse comportamento residual. O Validador recebia como
   input o texto original + NER completo + SOAP completo (~3x o tamanho de
   um prompt normal), esgotando o budget de `max_tokens=32768` antes de gerar
   a resposta estruturada `ValidacaoResult`.

2. **Timeout**: Notas que ativavam o thinking mode demoravam 5+ minutos no
   no Validador (vs ~25s nos nos NER e SOAP), causando timeout de 120s na
   conexao HTTP com o vLLM. Os try/except capturavam a excecao silenciosamente,
   resultando em validacoes vazias que nao agregavam valor.

A decisao pragmatica foi remover Validador e Corretor para viabilizar a
execucao das 749 notas dentro do prazo disponivel na DGX (~10 horas total).

## Implicacoes para o dataset

O `banco_dados_iana_oficial.json` foi gerado **sem auditoria de completude
nem correcao de gaps**. A analise qualitativa posterior de 3 amostras
representativas revelou:

- Vazamento de categorias entre campos NER (ex: sintomas em disease_or_syndrome)
- Extracao de entidades a partir de testes negativos/pendentes
- Duplicacoes intra-categoria por sinonimos e traducao dupla
- Achados de imagem classificados como diagnosticos
- Mistura de idiomas (portugues + ingles residual)

Esses problemas motivaram a v3 do pipeline, que substitui os 2 agentes LLM
de auditoria por:
- Pos-processamento deterministico (Python puro, ~10ms)
- 1 unico agente LLM de auditoria semantica com schema enxuto
- Pos-processamento final de limpeza

## Filtro de cobertura textual

### Contexto

A seleção inicial de 749 notas foi baseada em códigos ICD (seq_num == 1 para
a doença alvo). Porém, a presença do código ICD no faturamento não garante
que a doença alvo seja mencionada textualmente no resumo de alta.

### Auditoria (2026-04-08)

O script `audit_text_coverage.py` aplicou regex case-insensitive específicas
por doença no corpo do texto (excluindo cabeçalhos adicionados pelo pipeline):

- **HIV**: termos como `hiv`, `aids`, `cd4`, `antiretroviral`, `haart`, nomes
  de antirretrovirais
- **Tuberculose**: `tuberculosis`, `tb`, `mycobacterium`, `ppd`, `quantiferon`,
  `afb`, `isoniazid`, `rifampin`, etc.
- **Sífilis**: `syphilis`, `treponema`, `rpr`, `vdrl`, `fta-abs`,
  `penicillin g benzathine`, etc.

### Resultados

| Métrica | Valor |
|---------|-------|
| Total de notas auditadas | 749 |
| Cobertura adequada (≥3 menções) | 735 (98.1%) |
| Cobertura mínima (1-2 menções) | 3 (0.4%) |
| Zero menções | 11 (1.5%) |

### Distribuição das 11 notas com zero menções

| Doença | Código ICD | Quantidade | Observação |
|--------|-----------|------------|------------|
| Sífilis | 0940 | 9 | Neurossífilis latente — código ICD presente mas texto fala sobre outras condições (ex: pé de Charcot, amputação) |
| Sífilis | A5216 | 1 | Idem |
| HIV | B20 | 1 | Código HIV presente mas texto não menciona HIV/AIDS |

### Decisão

- **Excluídas da produção**: 11 notas (IDs em `config/excluded_notes.py`)
- **Exceção**: nota `27306123` (Sífilis 0940, Amostra 2 da análise qualitativa)
  mantida no test set de 10 notas como caso patológico — espera-se que o
  pipeline v3 retorne listas vazias em vez de inventar entidades
- **Dataset final de produção**: 738 notas

### Distribuição final por doença

| Doença | Antes (749) | Excluídas | Depois (738) |
|--------|------------|-----------|-------------|
| HIV | ~proporção original | 1 | -1 |
| Tuberculose | ~proporção original | 0 | inalterado |
| Sífilis | ~proporção original | 10 | -10 |

*Nota: a distribuição exata por doença depende do parquet original. Os números
acima refletem apenas as exclusões.*

## Referência

- Commit do grafo original (4 agentes): `b5898d0`
- Commit da simplificação (2 agentes): `abff60f`
- Diff completo: `git diff b5898d0..abff60f -- experiments/graphs/extracao.py`
- Script de auditoria: `experiments/audit_text_coverage.py`
- Lista de exclusão: `experiments/config/excluded_notes.py`
