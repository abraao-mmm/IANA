# Projeto IANA: EHR Sentinel

**Área:** Engenharia de Dados & Processamento de Linguagem Natural (NLP) em Saúde
**Foco Clínico:** HIV, Tuberculose e Sífilis — Programa de Eliminação Tripla
**Publicação alvo:** MDPI Diagnostics

## Visão Geral

O Projeto IANA constrói uma pipeline de anotação semi-automatizada de notas clínicas usando um LLM de 122B parâmetros (Qwen 3.5-122B-A10B) como gerador de *silver standard*, seguido de validação por especialista médico e treinamento de modelos compactos (4B parâmetros) para extração de entidades clínicas (NER) e estruturação SOAP.

O dataset vem do MIMIC-IV (738 notas de alta em inglês, filtradas por CID-10 de HIV, TB e Sífilis). A saída é um JSON estruturado por nota contendo 6 categorias NER em português brasileiro + 6 campos SOAP expandidos.

## Pipeline v3.2

```
START → [NER ‖ SOAP] → Pós-processamento determinístico → Auditor LLM → Pós-processamento final → JSON
```

- **NER + SOAP** executam em paralelo (2 chamadas LLM)
- **Pós-processamento determinístico**: deduplicação, exclusividade mútua entre categorias, normalização canônica (~80 mapeamentos)
- **Auditor LLM**: correção semântica com fallback graceful em caso de estouro de tokens
- **Status por agente**: cada nota rastreia `ok`/`token_overflow`/`error` para NER, SOAP e Auditor

Infraestrutura: DGX com 2× NVIDIA H200 (FP8, tensor parallel), vLLM 0.18.0, LangGraph.

## Estrutura do Repositório

```
ehr-sentinel/
├── data/
│   ├── SemClinBr-xml-public-v1/       # Corpus SemClinBr (~1000 XMLs)
│   ├── download_mimic.py              # Download do MIMIC-IV via Google Drive
│   ├── extract_mimic.py               # Descompacta e converte CSV → Parquet
│   └── README.md                      # Documentação das bases de dados
├── docs/                              # Referências e amostras exploratórias
├── experiments/
│   ├── config/
│   │   ├── prompts.py                 # Prompts dos 3 agentes (NER, SOAP, Auditor)
│   │   ├── canonical_terms.py         # Dicionário de normalização (~80 termos)
│   │   └── excluded_notes.py          # 11 IDs excluídos por cobertura textual
│   ├── models/
│   │   └── schemas.py                 # EntidadeClinica, SOAP, AgentStatus
│   ├── graphs/
│   │   ├── extracao.py                # Grafo LangGraph v3.2 (6 nós)
│   │   └── batch.py                   # Processamento em lote (Map-Reduce)
│   ├── postprocess.py                 # Dedup, exclusividade mútua, canônico
│   ├── 04_inferencia_langgraph.py     # Script principal (amostras ou parquet)
│   ├── 05_recuperar_pendentes.py      # Recovery de notas com falha
│   ├── run_test_batch.py              # Test batch de 10 notas + validação
│   ├── audit_text_coverage.py         # Auditoria de cobertura textual
│   ├── select_test_samples.py         # Seleção estratificada de candidatas
│   ├── generate_medical_review.py     # Gera HTMLs para revisão médica
│   ├── validate_extraction_quality.py # 8 checks automáticos de qualidade
│   ├── PIPELINE_DOCUMENTATION.md      # Documentação completa (~8000 palavras)
│   ├── METHODOLOGY_NOTES.md           # Notas sobre evolução arquitetural
│   └── scripts/
│       └── start_vllm.sh             # Comando de inicialização do vLLM
├── requirements.txt
├── LICENSE.txt
└── README.md
```

## Como Executar

### 1. Ambiente

```bash
git clone <url-do-repositorio>
cd ehr-sentinel
pip install -r requirements.txt
```

### 2. Dados MIMIC-IV

```bash
python data/download_mimic.py   # Baixa o ZIP do Google Drive
python data/extract_mimic.py    # Descompacta e converte para Parquet
```

### 3. Servidor vLLM (DGX)

```bash
vllm serve Qwen/Qwen3.5-122B-A10B \
    --tensor-parallel-size 2 \
    --quantization fp8 \
    --max-model-len 65536 \
    --host 0.0.0.0 --port 8000 \
    --api-key iana-local-key \
    --gdn-prefill-backend triton
```

### 4. Pipeline

```bash
cd experiments

# Test batch (10 notas selecionadas)
python run_test_batch.py

# Produção (738 notas)
python 04_inferencia_langgraph.py --modo parquet

# Recovery de falhas
python 05_recuperar_pendentes.py
```

## Documentação Detalhada

A documentação completa da pipeline (curadoria, arquitetura, validação iterativa, decisões metodológicas, limitações) está em [`experiments/PIPELINE_DOCUMENTATION.md`](experiments/PIPELINE_DOCUMENTATION.md).

## Licença

Dados MIMIC-IV sob [PhysioNet Credentialed Health Data License v1.5.0](LICENSE.txt).
