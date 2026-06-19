# Pipeline de Treino — Benchmark de Modelos Compactos

## 1. Visão geral

Esta pasta contém a infraestrutura de treino para o benchmark de **4 modelos compactos** do Projeto IANA. O objetivo é treinar cada modelo no silver standard (738 notas pós-processadas) e avaliar contra o gold test set (30 notas validadas por especialista médico).

Os 4 modelos:

| # | Modelo | HF ID | Params | Cobertura |
|---|---|---|---|---|
| 1 | BioBERTpt-clin | `pucpr/biobertpt-clin` | 110M | NER apenas |
| 2 | MedGemma 4B | `google/medgemma-4b-it` | 4B | NER + SOAP |
| 3 | Gemma 4 E4B | `google/gemma-4-e4b-it` | 4B | NER + SOAP |
| 4 | Qwen3.5-4B | `Qwen/Qwen3.5-4B` | 4B | NER + SOAP |

## 2. Pré-requisitos

1. **Derrubar o vLLM** (libera GPUs 0 e 1):
   ```bash
   bash infra/stop_vllm.sh
   ```

2. **Instalar dependências** (na DGX):
   ```bash
   pip install transformers>=4.45 peft trl accelerate datasets evaluate seqeval huggingface-hub pyyaml
   ```

3. **Aceitar licenças no HuggingFace**:
   - MedGemma: aceitar termos HAI-DEF em https://huggingface.co/google/medgemma-4b-it
   - Gemma 4 E4B: aceitar termos em https://huggingface.co/google/gemma-4-e4b-it
   - Autenticar: `huggingface-cli login`

## 3. Verificação inicial

```bash
python infra/check_model_access.py --token $HF_TOKEN
```

Todos os 4 modelos devem retornar ✅ antes de prosseguir.

## 4. Preparação de dados

```bash
cd data_prep
python split_train_val.py
python format_bio_tagging.py --split train && python format_bio_tagging.py --split val
python format_chatml.py --split train && python format_chatml.py --split val
python format_gemma.py --split train && python format_gemma.py --split val
```

Smoke test (validação rápida com 3 notas):
```bash
python data_prep/format_chatml.py --smoke-test
python data_prep/format_gemma.py --smoke-test
python data_prep/format_bio_tagging.py --smoke-test
```

## 5. Execução dos treinos

Os treinos devem ser executados individualmente nesta primeira fase, para permitir validação isolada de cada modelo. A paralelização será implementada após o primeiro ciclo bem-sucedido.

### Sequência recomendada

1. Verificar acesso aos modelos:
   ```bash
   python infra/check_model_access.py
   ```

2. Derrubar o vLLM:
   ```bash
   bash infra/stop_vllm.sh
   ```

3. Preparar dados (ver seção 4).

4. Treinar BioBERTpt primeiro (mais rápido, valida pipeline):
   ```bash
   python train/train_biobertpt.py --config config/biobertpt.yaml --gpu 0
   ```

5. Avaliar BioBERTpt no gold:
   ```bash
   python eval/run_inference.py --model biobertpt --checkpoint checkpoints/biobertpt/best
   python eval/compute_metrics.py --model biobertpt
   ```

6. Repetir para os outros 3 modelos individualmente:
   ```bash
   python train/train_qwen35_4b.py --config config/qwen35_4b.yaml --gpu 0
   python train/train_medgemma.py --config config/medgemma.yaml --gpu 1
   python train/train_gemma4_e4b.py --config config/gemma4_e4b.yaml --gpu 2
   ```

7. Comparar resultados:
   ```bash
   python eval/compare_models.py
   ```

8. Reiniciar vLLM se necessário:
   ```bash
   bash infra/start_vllm.sh
   ```

### Dry-run (sem GPU)
```bash
python train/train_biobertpt.py --dry-run
python train/train_qwen35_4b.py --dry-run
```

## 6. Avaliação

Inferência e métricas:
```bash
python eval/run_inference.py --model biobertpt --checkpoint checkpoints/biobertpt/best
python eval/compute_metrics.py --model biobertpt
python eval/compare_models.py
python eval/error_analysis.py --model biobertpt
```

Saídas em `predictions/` (JSONs de inferência), `results/` (métricas + comparação).

## 7. Modelos considerados mas excluídos do benchmark

### GLiNER multi v2.1 (`urchade/gliner_multi-v2.1`)

**Motivo da exclusão:** incompatibilidade de alinhamento de spans entre o silver standard (português brasileiro) e o texto original do MIMIC-IV (inglês).

O GLiNER é um modelo generativo para NER baseado em spans — requer offsets `(start, end)` no texto para cada entidade. Porém:

- O silver standard produzido pelo Qwen 3.5-122B contém entidades já traduzidas para português brasileiro canônico (ex: "Hipertensão arterial sistêmica")
- O texto original do MIMIC-IV está em inglês (ex: "hypertension")
- O fuzzy matching no smoke test obteve match rate de apenas **13.7%**, muito abaixo do mínimo operacional de 80%

**Soluções consideradas mas não adotadas:**

1. **Dicionário bilíngue PT-BR ↔ EN**: exigiria curadoria manual extensiva de termos médicos e introduziria um ponto de falha adicional na pipeline. Não escala para os ~80 termos canônicos atuais muito menos para o vocabulário aberto de sintomas e achados.
2. **Re-traduzir o silver de volta para inglês**: comprometeria a consistência com o objetivo do projeto (NER em PT-BR) e adicionaria erros de round-trip translation.
3. **Usar versão do silver em inglês**: não existe — a extração original produz PT-BR direto.
4. **Annotation projection** (alinhar spans via tradutor neural): adiciona complexidade e mais uma dependência externa ao pipeline. Fora do escopo desta fase.

A decisão foi focar os recursos de treino e avaliação nos 4 modelos remanescentes, que operam em vocabulário aberto ou classificação de tokens sem exigir spans explícitos no texto original.

## 8. Troubleshooting

**OOM (Out of Memory):**
Reduza `per_device_train_batch_size` no YAML ou aumente `gradient_accumulation_steps`.

**Token overflow nos decoders:**
Reduza `max_seq_length` no YAML. Para notas muito longas, o chunking do `format_bio_tagging.py` já cuida da divisão.

**Licença não aceita (403):**
Rode `python infra/check_model_access.py` e siga os links de aceitação no HuggingFace.

**GPU não liberada após parar vLLM:**
```bash
nvidia-smi
# Se processos persistem:
kill -9 <PID>
```

## Estrutura de arquivos

```
training/
├── config/           # YAMLs de hiperparâmetros (4 modelos + splits)
├── data_prep/        # Conversores de formato (BIO, ChatML, Gemma)
├── train/            # Scripts de treino (1 por modelo)
│   └── shared/       # LoRA config e callbacks reutilizáveis
├── eval/             # Inferência, métricas, comparação, análise de erros
├── infra/            # check_model_access, stop/start vLLM, check GPUs
├── data/             # Outputs dos conversores (criados em runtime)
├── checkpoints/      # Modelos treinados (criados em runtime)
├── predictions/      # Predições no gold (criados em runtime)
├── results/          # Métricas e comparações (criados em runtime)
└── logs/             # Logs de treino JSON (criados em runtime)
```
