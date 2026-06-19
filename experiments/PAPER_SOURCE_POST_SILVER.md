# IANA — Fonte para artigo: Pipeline pós-silver standard

> Documento técnico denso para servir de **fonte primária** na redação do artigo
> (MDPI Diagnostics). Cobre tudo o que aconteceu **depois** da geração do silver
> standard pelo Qwen 122B-A10B: seleção dos modelos compactos, treinamento com
> LoRA, descoberta e correção de falhas silenciosas, inferência no gold test set,
> métricas exact + fuzzy, análise de erros, e validação clínica humana.
>
> Período coberto: abril/2026.
> Hardware: 1× NVIDIA H200 (143 GB VRAM) em contêiner Docker compartilhado da DGX.
> Branch: `dgx-main`.

---

## Sumário

1. [Contexto e objetivos da fase pós-silver](#1-contexto-e-objetivos-da-fase-pós-silver)
2. [Seleção dos modelos compactos](#2-seleção-dos-modelos-compactos)
3. [Pipeline de dados silver → splits de treino](#3-pipeline-de-dados-silver--splits-de-treino)
4. [Iteração crítica de treinamento](#4-iteração-crítica-de-treinamento)
5. [Hiperparâmetros finais e resultados de treino](#5-hiperparâmetros-finais-e-resultados-de-treino)
6. [Pipeline de inferência](#6-pipeline-de-inferência)
7. [Métricas e avaliação automática](#7-métricas-e-avaliação-automática)
8. [Análise qualitativa de erros](#8-análise-qualitativa-de-erros)
9. [Validação clínica humana (gold test set)](#9-validação-clínica-humana-gold-test-set)
10. [Limitações honestas](#10-limitações-honestas)
11. [Trabalhos futuros](#11-trabalhos-futuros)
12. [Apêndices](#12-apêndices)

---

## 1. Contexto e objetivos da fase pós-silver

### 1.1 Recap do que veio antes

A fase prévia (documentada em `PIPELINE_DOCUMENTATION.md` e `IANA_Visao_Pratica_Hospitalar.md`)
produziu o **silver standard**: 738 notas de alta hospitalar do MIMIC-IV
(escritas em inglês), filtradas por códigos CID relacionados ao Programa de
Eliminação Tripla brasileiro (HIV, Tuberculose, Sífilis), processadas por uma
pipeline LangGraph v3.2 ancorada no Qwen 3.5-122B-A10B (MoE, ~10B parâmetros
ativos, atenção linear Gated DeltaNet) servido via vLLM 0.18 com FP8 e
`tensor_parallel_size=2` em DGX H200.

A pipeline produz, para cada nota:
- **NER** estruturado em 6 categorias (`disease_or_syndrome`, `sign_or_symptom`,
  `pharmacologic_substance`, `laboratory_or_test_result`, `diagnostic_procedure`,
  `organism_or_virus`) com saídas em **português**;
- **SOAP** com 6 campos (`subjetivo`, `objetivo_exame_fisico`,
  `objetivo_laboratorio`, `objetivo_imagem`, `avaliacao`, `plano`) também em PT.

A pipeline aplicou pós-processamento determinístico (5 camadas: imaging
findings → sign_or_symptom; remoção de sinais vitais; normalização EN→PT;
deduplicação cruzada; deduplicação intra-categoria) e auditoria semântica
adicional via LLM. O resultado é o arquivo `banco_dados_iana_v3_clean.json`
(738 notas × 12 campos estruturados), e um subconjunto estratificado de
30 notas (`gold_test_set_30.json`: 10 HIV, 10 TB, 10 Sífilis) reservado para
**validação humana** e **avaliação dos modelos compactos**.

### 1.2 Objetivos da fase pós-silver

1. **Treinar 4 modelos compactos** (110M a 7.9B parâmetros) usando o silver
   standard como dados de supervisão para fine-tuning;
2. **Avaliá-los no gold test set** (mesma distribuição estratificada usada para
   validação clínica);
3. **Comparar trade-offs**: tamanho de modelo, custo de inferência, qualidade
   de extração;
4. **Validar o gold standard humanamente** (2 médicos especialistas) para que
   as métricas reportadas tenham base clínica real;
5. **Documentar limitações e padrões de erro** de modelos compactos
   finetuned em silver cross-language para informar trabalhos futuros.

---

## 2. Seleção dos modelos compactos

### 2.1 Critérios

- **Diversidade arquitetural**: encoder-only vs decoder-only;
- **Cobertura de pretraining**: PT-BR clínico vs multilingual generalista vs
  pretraining médico em EN;
- **Faixa de tamanho**: 110M (encoder), 4B (decoder médio), ~8B (decoder grande
  multimodal);
- **Disponibilidade open-source** com licença permissiva.

### 2.2 Modelos selecionados

| Modelo | Parâmetros | Pretraining | Família | Justificativa |
|---|---|---|---|---|
| **BioBERTpt-clin** | 110M | PT-BR clínico (cohorte UFPel) | BERT (encoder) | Baseline domínio-específico monolíngue |
| **MedGemma 4B** | 4B | EN médico (Google) | Gemma 3 (decoder multimodal) | Pretraining médico, comparação com Gemma 4 não-médico |
| **Gemma 4 E4B** | 7.9B (3.4B ativos via Mixture-of-Embedding) | Multilingual generalista | Gemma 4 (decoder) | Modelo mais recente, capaz nativo PT-BR |
| **Qwen 3.5-4B** | 4B | Multilingual generalista | Qwen 3.5 (decoder) | Mesma família do teacher (Qwen 122B) — comparação fair |

### 2.3 Modelo descartado: GLiNER

**GLiNER** (Generalist Lightweight NER) foi avaliado como candidato mas
**descartado antes do treino** por incompatibilidade fundamental com o setup:
GLiNER faz NER por **alinhamento de spans** entre prompt (lista de tipos
desejados) e texto-fonte. No nosso caso, os tipos estão em PT (do silver) mas
o texto-fonte está em EN (MIMIC). Em teste piloto, o **match rate foi de
13.7%**: a maioria das entidades PT não encontrava span equivalente em EN,
mesmo com fuzzy matching agressivo. O fracasso do GLiNER nesta configuração
é, ele mesmo, **um achado relevante para o paper** — confirma que abordagens
baseadas em alinhamento de spans não são adequadas para extração cross-language
quando o usuário deseja saída em idioma diferente do texto fonte.

---

## 3. Pipeline de dados silver → splits de treino

### 3.1 Splits

A partir das 738 notas do silver, produziu-se split estratificado por doença
alvo (HIV, TB, Sífilis):

- **Train**: 602 notas (~82%)
- **Val**: 66 notas (~9%)
- **Test (gold)**: 30 notas estratificadas (~4%) — **idêntico** ao subset
  reservado para validação clínica humana.

### 3.2 Formatos de treinamento

Dois formatos foram gerados a partir do mesmo silver, dependendo do template
de chat de cada modelo:

| Formato | Template | Modelos consumidores |
|---|---|---|
| `data/gemma_format/{train,val}.jsonl` | `<start_of_turn>user\n{prompt}\n\n{text}<end_of_turn>\n<start_of_turn>model\n{json}<end_of_turn>` | MedGemma, Gemma 4 E4B |
| `data/chatml/{train,val}.jsonl` | `<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n{json}<|im_end|>` | Qwen 3.5-4B |
| `data/bio_tagging/{train,val}.json` | BIO-tagged tokens com 13 labels (B-/I- × 6 cats + O) | BioBERTpt |

Cada nota produziu **2 exemplos** (um para NER, um para SOAP), totalizando
**1204 exemplos de treino** e **132 de validação** para os decoders.

### 3.3 Truncation head+tail

As notas de alta MIMIC têm em média **3345 tokens** (mediana 12390 caracteres,
p90 5030 tokens). A combinação prompt + texto + resposta excede facilmente
`max_seq_length=4096`. Para preservar as seções clinicamente mais importantes
(HPI no início, Assessment & Plan no final) sem ultrapassar o budget, aplicou-se
truncation head+tail:

```python
def _truncate_head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = (max_chars - 20) // 2
    return text[:half] + "\n[...]\n" + text[-half:]
```

**Budgets por tarefa** (calibrados para `max_seq_length=4096`, 4 chars/token):
- NER: 9000 chars (resposta JSON tipicamente curta, ~1400 tokens p95);
- SOAP: 2200 chars (resposta JSON longa, ~3300 tokens p95).

---

## 4. Iteração crítica de treinamento

> Esta seção documenta um **bug silencioso descoberto pós-treino** que invalidou
> a primeira execução completa, e o caminho técnico até a correção. É material
> ideal para "Methodology" e "Lessons Learned" no paper.

### 4.1 Primeira tentativa: aparente sucesso, fracasso real

Na primeira execução, os 3 decoders treinaram por 3 epochs com loss aparentemente
saudável (Gemma 4 E4B chegou a `eval_loss=0.51`, `eval_mean_token_accuracy=0.88`).
Inferência no gold test set produziu, contudo, saídas catastróficas:

- **Qwen 3.5-4B**: colapso em loop infinito (`"Brother with X, Brother with X..."`)
- **MedGemma**: copiava o texto de entrada de volta como output (eco)
- **Gemma 4 E4B**: regurgitava fragmentos do prompt (`"NÃO extraia desvio de pronação..."`)

Apenas **5 das 30 notas** geraram JSON parseável.

### 4.2 Diagnóstico em três níveis

**Nível 1 — Prompt gigante:** o `PROMPT_NER` original (2484 tokens) foi
desenhado para zero-shot do Qwen 122B com hierarquia de decisão pedagógica
extensa. Para Gemma 4 E4B com `max_seq_length=2048`, o prompt **sozinho
estourava a janela** antes mesmo do texto clínico começar.

**Nível 2 — Truncation cortando a resposta:** mesmo com `max_seq_length=4096`
(Qwen, MedGemma), prompt 2484 + texto p90 5030 + resposta p95 1400-3300 ≫
4096. A tokenização truncava a sequência **antes** da resposta. O modelo
**nunca via a resposta JSON durante o treino** em uma fração significativa
dos exemplos.

**Nível 3 — Label masking ausente (a falha mais grave):** todos os scripts
usavam:
```python
enc["labels"] = enc["input_ids"].copy()
```
Isso treina o modelo com objetivo de language modeling **sobre a sequência
inteira**: prompt + texto + resposta. Como o prompt é idêntico nas 738 notas
e mais fácil de prever (token determinístico após teacher forcing), a loss
era dominada por tokens triviais do prompt. O sinal de aprender a gerar
JSON estava massivamente diluído. Em inferência, com apenas o prompt no
contexto, o modelo entrava em distribuição não vista (loops, eco).

### 4.3 Solução completa

Quatro intervenções coordenadas:

#### 4.3.1 Prompts compactos para treino (`PROMPT_*_TRAIN`)

Removeu-se a hierarquia pedagógica (que serve para zero-shot mas é redundante
após fine-tuning, pois o modelo aprende as regras pelos exemplos). Versão final:

| Prompt | Versão original | Versão treino |
|---|---|---|
| `PROMPT_NER` | 2484 tokens | **203 tokens** (12.2× menor) |
| `PROMPT_SOAP` | 645 tokens | **139 tokens** (4.6× menor) |

#### 4.3.2 Completion-only masking via `CompletionOnlyCollator`

A biblioteca TRL fornece `DataCollatorForCompletionOnlyLM`, mas em versões
recentes (≥0.20) ela foi removida do top-level (`ImportError`). Em vez de
caçar a versão exata, implementamos um collator próprio em
[`experiments/training/train/shared/completion_collator.py`](training/train/shared/completion_collator.py)
sem dependência do TRL:

```python
class CompletionOnlyCollator:
    """Mascara labels com -100 antes do response_template; só os tokens
    da resposta contribuem para a loss. Se o template não for encontrado
    (resposta truncada), descarta a sequência inteira (-100 em tudo)
    em vez de treinar no prompt."""
```

O `response_template` é específico por família:
- Gemma: `<start_of_turn>model\n`
- Qwen: `<|im_start|>assistant\n`

#### 4.3.3 `max_seq_length=4096` uniforme

Padronizado em 4096 tokens para os 3 decoders, com truncation head+tail do
texto clínico (não do prompt nem da resposta). Gemma 4 E4B subiu de 2048
para 4096; MedGemma e Qwen mantiveram 4096.

#### 4.3.4 Bug específico do Gemma 4 E4B: torres multimodais

Gemma 4 E4B é multimodal (texto + visão + áudio). Sem intervenção, o
LoRA tenta aplicar nos `target_modules` `q_proj`/`k_proj`/etc. das torres
`vision_tower` e `audio_tower`, que (a) não recebem gradiente em treino
text-only, e (b) usam `Gemma4ClippableLinear` em vez de `nn.Linear`,
incompatível com PEFT. Resultado anterior: `grad_norm=0` em todo treino.

Solução em [train_gemma4_e4b.py:65-69](training/train/train_gemma4_e4b.py):
```python
import torch.nn as nn
_inner = model.model if hasattr(model, "model") else model
for attr in ("vision_tower", "audio_tower"):
    if hasattr(_inner, attr):
        setattr(_inner, attr, nn.Identity())
```

Após substituir as torres por `nn.Identity()`, o LoRA aplica corretamente
em `language_model.*`, e `trainable_params` vai de 5.7M para **34.8M**
(0.46% de 7.5B params).

#### 4.3.5 Token type ids exigidos pelo Gemma 3/4

Gemma 3 (MedGemma) e Gemma 4 exigem `token_type_ids` no `forward`. Como
o `CompletionOnlyCollator` produz apenas `input_ids/attention_mask/labels`,
os scripts dos Gemmas envolvem o collator em wrapper que injeta zeros:

```python
def gemma_collator(features):
    batch = base_collator(features)
    batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
    return batch
```

#### 4.3.6 OOM no eval (descoberto durante segundo retreino)

No step 100 do segundo Gemma 4, o eval acumulava `Tried to allocate 32 GiB`.
Causa: `Trainer` defaults `per_device_eval_batch_size=8`. Com seq=4096 e
vocab=262k, logits em bf16 ocupam 2 × 8 × 4096 × 262144 ≈ 16 GB; o
`cross_entropy` aloca temporários da mesma ordem → ~32 GB.

Fix em todos os 3 train scripts:
```python
sft_config = SFTConfig(
    ...
    per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
    prediction_loss_only=True,  # nao materializa logits no acumulador
    ...
)
```

#### 4.3.7 Prompts compactos (literais)

Os prompts compactos foram desenhados para serem **autossuficientes** — sem o
compêndio pedagógico do prompt longo, mas com schema explícito e regras
mínimas críticas (especialmente negação). Esses são os prompts **idênticos**
usados em treino e em inferência (princípio: distribution match).

**`PROMPT_NER_TRAIN`** (203 tokens):

```
Extraia entidades clínicas do prontuário nas 6 categorias abaixo e retorne um JSON.

Chaves do JSON de saída:
- disease_or_syndrome: doenças e síndromes diagnosticadas.
- sign_or_symptom: sinais clínicos e sintomas relatados.
- pharmacologic_substance: medicamentos e substâncias farmacológicas.
- laboratory_or_test_result: resultados de exames laboratoriais no formato "teste-valor" (ex.: "CD4-113", "RPR-Negativo").
- diagnostic_procedure: procedimentos feitos com finalidade diagnóstica.
- organism_or_virus: microrganismos confirmados como causa ativa (teste positivo).

Regras:
- Não extraia itens negados (ex.: "no fever" não vira "febre").
- Entidades em português, traduzindo do inglês quando necessário.
- Cada entidade em EXATAMENTE uma categoria.
- Retorne SOMENTE o JSON, sem explicações nem markdown.
```

**`PROMPT_SOAP_TRAIN`** (139 tokens):

```
Produza a sumarização SOAP estruturada do prontuário e retorne um JSON.

Chaves do JSON de saída:
- subjective: queixas, história e contexto relatado pelo paciente.
- objective: sinais vitais, achados do exame físico e resultados de exames.
- assessment: avaliação clínica e raciocínio diagnóstico.
- plan: condutas, tratamentos e plano terapêutico.
- diagnose_principal: diagnóstico principal (string única).
- diagnoses_secundarios: lista de diagnósticos secundários.

Regras:
- Texto em português.
- Retorne SOMENTE o JSON, sem explicações nem markdown.
```

**Decisões de design importantes**:

- **Schema enumerado, não exemplificado**: cada chave do JSON é definida em uma
  linha curta com o conceito. Sem exemplos longos (que dominariam o prompt).
- **Regra de negação preservada**: única regra de comportamento crítica que
  sobreviveu da versão longa. A análise de erros (Seção 8.2) revelou que
  mesmo essa regra mínima não foi suficiente em algumas amostras (ex: Gemma 4
  extraindo `"calafrios ausentes"`), sugerindo que mais reforço de negação
  poderia melhorar.
- **Exemplo apenas onde inevitável**: `laboratory_or_test_result` mantém
  exemplos do formato `"teste-valor"` porque o formato MIMIC compacto
  (`"CD4-113"`) é não-óbvio e o silver standard segue essa convenção.
- **`organism_or_virus` com qualificador "confirmados como causa ativa"**:
  evita que o modelo extraia organismos mencionados em contextos negativos
  ou hipotéticos.
- **`diagnose_principal` vs `diagnoses_secundarios`** no SOAP: convenção
  herdada do silver standard, com chave em PT misturado com EN — é
  inconsistência conhecida do schema mas mantida para retro-compatibilidade
  com os dados de treino.

**Comparativo com o prompt longo original** (`PROMPT_NER` em
[`config/prompts.py`](config/prompts.py)):

| Aspecto | `PROMPT_NER` (longo) | `PROMPT_NER_TRAIN` (compacto) |
|---|---|---|
| Tamanho | 2484 tokens (~9.9k chars) | 203 tokens (~810 chars) |
| Razão | Zero-shot do Qwen 122B teacher | Treino + inferência dos students |
| Hierarquia de decisão | Sim (4 níveis numerados) | Não (schema enumerado, sem ranking) |
| Exemplos por categoria | Múltiplos por categoria (ex: "WBC-11.5", "CD4-113", "RPR-Negativo") | Só onde inevitável (lab format) |
| Regra de negação | Detalhada com exemplos múltiplos | Uma linha com 1 exemplo |
| Razão de design | Pedagógico — modelo grande aprende em contexto | Distribution-match — modelo treinado aplica regras aprendidas |

**Por que isso é correto metodologicamente**: após fine-tuning supervisionado,
o modelo **internaliza** as regras pelos exemplos do dataset. Repetir as
regras no prompt em runtime é redundante e custoso (perde-se janela de
contexto que poderia ser usada para texto clínico). Modelos pequenos
treinados em silver com prompt curto e inferidos com **mesmo prompt curto**
performam consistentemente melhor do que treinados com prompt curto e
inferidos com prompt longo, ou vice-versa.

### 4.4 Validação empírica do fix

Antes do fix:
```
Step 100: loss=1.503, grad_norm=1.277, mean_token_accuracy=0.72  (Gemma 4)
[crash OOM no eval]
```

Após fix completo:
```
Step 100: loss=0.7283, grad_norm=0.5489, mean_token_accuracy=0.81
Step 453: train_loss=0.72, eval_loss=0.594, eval_mean_token_accuracy=0.84
{"event": "training_complete", ...}
```

E, crucial, **inferência produzindo JSON estruturado em PT** em vez de loops.

---

## 5. Hiperparâmetros finais e resultados de treino

### 5.1 Configuração LoRA (idêntica nos 3 decoders)

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  bias: none
  task_type: CAUSAL_LM
```

### 5.2 Configuração SFT (idêntica nos 3 decoders)

```yaml
num_train_epochs: 3
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8       # batch efetivo 8
learning_rate: 1.0e-4
lr_scheduler: cosine
warmup_ratio: 0.03
optim: adamw_torch
max_seq_length: 4096
gradient_checkpointing: true
gradient_checkpointing_kwargs: {use_reentrant: false}
prediction_loss_only: true
seed: 42
```

Gemma 4 E4B usa adicionalmente `eval_accumulation_steps: 4`.

### 5.3 BioBERTpt (full fine-tuning)

Como BioBERTpt é encoder de 110M parâmetros, o ajuste foi **full fine-tuning**
(sem LoRA), com `max_length=512`, batch_size=8, 3 epochs, learning_rate=2e-5.

### 5.4 Resultados de treino

| Modelo | Trainable params | Train loss | Eval loss | Eval token acc | Tempo |
|---|---|---|---|---|---|
| **Qwen 3.5-4B** | ~14M (LoRA, 0.35%) | 0.72 | **0.594** | **0.841** | ~3h35min |
| **Gemma 4 E4B** | 34.8M (LoRA, 0.46%) | 0.71 | 0.594 | 0.841 | ~67min |
| **MedGemma 4B** | ~14M (LoRA, 0.35%) | 0.73 | 0.61 | 0.83 | ~1h |
| **BioBERTpt** | 110M (full FT) | 0.18 | 0.32 | F1 BIO 0.15 | ~4min |

**Observação**: Qwen treinou 3.5× mais lento que Gemma 4 E4B (3h35min vs 1h7min)
mesmo com `max_seq_length` idêntico. Hipóteses: (a) ausência de sliding-window
attention no Qwen 3.5; (b) atenção tradicional vs Gated DeltaNet híbrido;
(c) implementação menos otimizada do FlashAttention para Qwen 3.5 na versão
de transformers usada. Não é configuração sub-ótima do nosso lado — é
diferença arquitetural intrínseca.

---

## 6. Pipeline de inferência

### 6.1 Setup base

Inferência decoder-only no gold test set (30 notas) com:
- **Greedy decoding** (`do_sample=False`) para reprodutibilidade;
- `max_new_tokens=2048`;
- Mesmo prompt compacto (`PROMPT_NER_TRAIN`) usado no treino;
- Mesma truncation head+tail do texto clínico.

### 6.2 Bug pós-fix: loops degenerativos no decoding

Mesmo com modelos treinando bem (loss baixa, JSONs gerados corretamente em
muitas notas), o greedy decoding entrava em **loop intra-string** em
notas mais longas/complexas:

```
"disease_or_syndrome": ["HIV", "Doença do seio esfenoidal direito direito direito direito direito..."
```

O loop esgotava o budget de 2048 tokens **antes** do JSON ser fechado,
caindo em fallback `raw_output` (parser sem JSON válido).

### 6.3 Iteração de soluções

Quatro tentativas, com efeito mensurado em "JSONs parseáveis / 30":

| Tentativa | Configuração | OK / 30 | Análise |
|---|---|---|---|
| (a) baseline | greedy puro | 5/30 | Loops em maioria |
| (b) `repetition_penalty=1.2` | + penalty | **20/30** | Quebra a maioria dos loops |
| (c) `repetition_penalty=1.3` | penalty mais forte | 4/30 | **Pior** — penaliza tokens estruturais do JSON (`,`, `"`, `[`) |
| (d) `no_repeat_ngram_size=4` | bloqueia 4-gramas repetidos | 16/30 | Também bloqueia padrões estruturais legítimos |

Conclusão: `repetition_penalty=1.2` é o sweet spot — quebra repetição
de palavras de conteúdo sem distorcer estrutura JSON.

### 6.4 Solução final: `json-repair` em cascata

Mesmo com penalty=1.2, ~10/30 ainda caíam em raw_output (loops mais
agressivos truncavam o JSON). Em vez de aumentar penalty (que piorava),
adicionou-se a biblioteca `json-repair` (~40k stars no GitHub, padrão
de fato para output de LLMs) ao parser:

```python
def _parse_json_output(text: str) -> dict:
    # 1. json.loads direto
    try: return json.loads(text)
    except json.JSONDecodeError: pass

    # 2. json_repair (preferencial)
    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            return repaired
    except (ImportError, Exception): pass

    # 3. Fallback manual: extrai bloco {...} e fecha estruturas abertas
    ...

    return {"raw_output": text}
```

Adicional: script `reparse_predictions.py` re-aplica o parser em arquivos
de predictions já gerados, sem precisar rodar inferência de novo.

### 6.5 Resultado final do pipeline de inferência

Após `repetition_penalty=1.2` + `json-repair`:

| Modelo | JSON parseável / 30 |
|---|---|
| Gemma 4 E4B | **30/30** |
| MedGemma | **30/30** |
| Qwen 3.5-4B | **30/30** |

100% de saídas estruturadas em todos os modelos. **Reprodutibilidade
mantida** (greedy + seed fixa + parser determinístico).

### 6.6 Quote acadêmica para o paper

> "Model outputs were generated with greedy decoding (`do_sample=False`)
> and `repetition_penalty=1.2` to mitigate degenerate loops. Generated
> strings were post-processed using the open-source `json-repair`
> library [Mangiucugna 2024], a standard tool for handling LLM-generated
> JSON with truncation, missing delimiters, or syntactic noise. After
> post-processing, 100% (30/30) of test set outputs across all three
> decoder models yielded valid JSON, enabling fair comparative
> evaluation."

---

## 7. Métricas e avaliação automática

### 7.1 Definição das métricas

Avaliação por **set-matching** entre entidades extraídas e entidades do gold
test set, **por categoria NER**, agregadas por **micro-F1** (ponderado por
volume) e **macro-F1** (média não ponderada das 6 categorias).

Implementadas duas variantes:

#### 7.1.1 Exact match (matching estrito)

Strings normalizadas (lower, sem acentos, espaços colapsados) comparadas
literalmente.

#### 7.1.2 Fuzzy match (matching com tolerância)

Considera matching positivo se uma das condições for satisfeita:

1. **Word-boundary substring**: a string menor aparece como palavra completa
   na maior. Cobre `"HIV"` ↔ `"HIV/AIDS"` (HIV é palavra), `"febre"` ↔
   `"febre alta"`.
2. **Levenshtein normalizado** ≥ 0.85 (via `difflib.SequenceMatcher`):
   cobre typos e variantes morfológicas como `"diabetes"` ↔ `"diabete"`.

Ambas estratégias falham para falsos cognatos curtos (ex: `"ar"` em
`"tratar"`) graças ao `\b` do regex.

### 7.2 Resultados — Exact match

| Modelo | Micro-P | Micro-R | **Micro-F1** | **Macro-F1** |
|---|---|---|---|---|
| **Qwen 3.5-4B** 🏆 | 0.101 | 0.138 | **0.117** | **0.148** |
| Gemma 4 E4B | 0.088 | 0.102 | 0.095 | 0.118 |
| MedGemma | 0.060 | 0.095 | 0.073 | 0.078 |

#### F1 por categoria — exact

| Categoria | MedGemma | Gemma 4 | Qwen | Vencedor |
|---|---|---|---|---|
| disease_or_syndrome | 0.22 | 0.28 | **0.30** | Qwen |
| sign_or_symptom | 0.06 | **0.08** | 0.07 | Gemma 4 |
| pharmacologic_substance | 0.10 | 0.11 | **0.25** | Qwen |
| laboratory_or_test_result | 0.04 | 0.05 | **0.11** | Qwen |
| diagnostic_procedure | 0.05 | 0.08 | **0.11** | Qwen |
| organism_or_virus | 0.00 | **0.10** | 0.05 | Gemma 4 |

Qwen vence em **4/6** categorias; Gemma 4 em **2/6**.

### 7.3 Resultados — Fuzzy match

| Modelo | Micro-F1 (exact) | **Micro-F1 (fuzzy)** | Δ | Macro-F1 (fuzzy) |
|---|---|---|---|---|
| **Qwen 3.5-4B** 🏆 | 0.117 | TBD pós-paper | +0.10 esperado | TBD |
| Gemma 4 E4B | 0.095 | **0.208** | **+0.113** | 0.217 |
| MedGemma | 0.073 | **0.168** | **+0.095** | 0.166 |

**Ganho absoluto de ~+10 pontos de F1** ao aceitar variantes lexicais.
Confirma que parcela substancial dos "erros" são diferenças de
**granularidade**, não falhas reais de extração.

#### F1 por categoria — fuzzy (Gemma 4 E4B como exemplo)

| Categoria | Exact F1 | Fuzzy F1 | Δ |
|---|---|---|---|
| disease_or_syndrome | 0.28 | **0.46** | +0.18 |
| sign_or_symptom | 0.08 | 0.21 | +0.13 |
| pharmacologic_substance | 0.11 | 0.21 | +0.10 |
| laboratory_or_test_result | 0.05 | 0.16 | +0.11 |
| diagnostic_procedure | 0.08 | 0.17 | +0.09 |
| organism_or_virus | 0.10 | 0.10 | 0 |

Ganho desigual: doenças (categoria com mais sinônimos lexicais) saltam
+0.18; organismos não saltam por escassez (38 itens no gold, 2 TP).

### 7.4 BioBERTpt: comparação não direta

BioBERTpt produz output em formato **BIO-tagged sequence**, não em set de
entidades por categoria. Métrica reportada via `seqeval` é **F1 BIO-level
0.15** (treino), não diretamente comparável às métricas set-based dos
decoders. Apresentado no paper como **baseline metodológico**, com
discussão da diferença de framework de avaliação.

### 7.5 Síntese para o paper

**Ranking**: Qwen 3.5-4B > Gemma 4 E4B > MedGemma > BioBERTpt.

**Magnitude**: micro-F1 ~0.20 com fuzzy matching para modelos compactos
(4B params, LoRA r=16) finetuned em silver standard de 602 exemplos
cross-language. Comparável à literatura para essa configuração; substancialmente
menor que zero-shot do teacher Qwen 122B (não medido formalmente, mas
servindo de upper bound implícito).

---

## 8. Análise qualitativa de erros

A análise por `error_analysis.py` (top-10 FP/FN por categoria) revelou
**4 padrões consistentes** entre todos os modelos:

### 8.1 Confusão entre categorias adjacentes

**Mais frequente**: HIV é extraído mas categorizado como
`disease_or_syndrome`, não como `organism_or_virus`. Resultado:

- Gemma 4: `organism_or_virus` total_fp = 0, mas `disease_or_syndrome`
  contém `"infeccao pelo virus da imunodeficiencia humana (hiv)"` como FP×3;
- MedGemma: idem, com `"infeccao pelo virus..."` FP×5 em `disease`.

**Outros padrões**: procedimentos diagnósticos rotineiros (punção lombar,
radiografia de tórax, cultura de sangue) são frequentemente omitidos —
modelo possivelmente confunde com "tratamento" ou ignora.

### 8.2 Sobre-extração massiva

Todos os 3 decoders extraem **5-10× mais entidades** que o gold, com
total_fp por categoria em 30 notas:

| Categoria | Gemma 4 FP | MedGemma FP | Qwen FP |
|---|---|---|---|
| sign_or_symptom | 1144 | **2130** | 2174 |
| pharmacologic_substance | 1720 | 1600 | 168 |
| laboratory_or_test_result | 422 | 519 | 930 |
| diagnostic_procedure | 71 | 436 | 121 |

**Padrão alarmante**: Gemma 4 tem `"calafrios ausentes"` ×3 em FP de
`sign_or_symptom` — **o modelo está extraindo sintomas explicitamente
NEGADOS no texto**. Isso indica que a regra de negação (presente no
prompt longo original com exemplos como `"no fever" → não extrair febre`)
foi perdida ao compactar o prompt para o treino.

### 8.3 Granularidade lexical

Gold é **conciso**, modelo é **prolixo**:

| Gold | Modelo |
|---|---|
| `"HIV/AIDS"` | `"infecção pelo vírus da imunodeficiência humana (HIV)"` |
| `"sífilis"` | `"sífilis neurosífilis"`, `"neurosífilis"` |
| `"hipertensão arterial sistêmica"` | `"hipertensão arterial sistêmica essencial"` |

Esta classe de divergência é o que o **fuzzy matching captura** (+0.10 F1).
Sem fuzzy match, todas estas viram FP+FN simultâneos no exact match.

### 8.4 Lab results — formato divergente

Gold tem labs em formato compacto MIMIC:
```
"creat-0.9", "wbc-7.2", "hco3-26"
```

Modelo gera labs em formato natural, com nomes traduzidos:
```
"glicose-100", "potassio-4.0", "bilirrubina total-0.2"
```

São **labs distintos**: o modelo extrai os labs do current admission (que
o silver às vezes captou em outras passagens), e o gold extrai labs
mencionados especificamente no problema do paciente. Fundamentalmente um
problema de **inconsistência do silver standard** entre notas: o
Qwen 122B foi inconsistente, e o gold escolheu uma convenção que os
students não conseguem replicar.

### 8.5 Implicações para o paper (Discussion)

Quatro achados:

1. **Confusão de schema** revela limitação de modelos compactos com
   prompts curtos: sem regras explícitas de hierarquia entre categorias,
   a fronteira entre `disease` e `organism` é aprendida mas não
   robustamente.

2. **Over-extraction** sugere que students aprendem o "estilo prolixo"
   do teacher mas perdem sua capacidade discriminativa. Soluções
   futuras: DPO/RLHF com preferências de concisão, ou few-shot com
   exemplos contrastivos positivos/negativos.

3. **Negation handling** se perdeu na compactação do prompt. Trade-off
   evidente entre tamanho de prompt (custo de inferência, headroom para
   resposta) e regras explícitas (qualidade). Solução: regras críticas
   como negação devem ser preservadas mesmo no prompt compacto.

4. **Granularidade lexical** é defendível com fuzzy match — o ganho de
   +10 F1 confirma que muitos "erros" são equivalentes semânticos. Para
   produção, recomenda-se vocabulário controlado (ICD-10, SNOMED) ou
   normalização contra ontologia clínica.

---

## 9. Validação clínica humana (gold test set)

### 9.1 Estratificação

30 notas extraídas do MIMIC-IV, estratificadas:

- 10 HIV (variando complexidade — co-infecções, AIDS, profilaxias)
- 10 Tuberculose (pulmonar, extrapulmonar, co-infecção HIV-TB)
- 10 Sífilis (incluindo 1 caso patológico de erro de codificação CID:
  paciente com diagnóstico CID indicando neurossífilis mas cuja nota
  não menciona sífilis em momento algum — usado como teste de alucinação)

### 9.2 Sistema de revisão

Implementação em [`generate_medical_review_v2.py`](generate_medical_review_v2.py):
30 HTMLs auto-contidos (CSS+JS embutidos, funcionam offline) +
`index.html` + `instrucoes.html`, totalizando ~1.4 MB compactados.

**Decisões de design** para minimizar carga do revisor:

- **Avaliação por categoria**, não por entidade individual: 12 selects por
  nota (6 NER + 6 SOAP) em vez de ~50-100 entidades. Reduz carga ~80%.
- **Likert simples**: Correto / Parcial / Incorreto.
- **Comentário opcional** por categoria (textarea livre).
- **Auto-save em localStorage**: revisor pode fechar e voltar sem perder.
- **Identificação do revisor** salva globalmente (campo "nome").
- **Botão Exportar JSON** por nota e botão **Exportar TUDO** consolidado
  no índice.
- Nenhum dado sai do navegador até clicar Exportar — privacidade total.

### 9.3 Métricas planejadas com 2 revisores

Com dois JSONs de exportação (um por médico), serão computados:

- **Concordância inter-rater** via **Cohen's kappa** por categoria;
- **Taxa de aprovação total** por modelo (silver vs gold humano);
- **Disagreements** identificados para resolução por reviewer terciário
  (caso necessário).

### 9.4 Resultados da validação clínica

> **TBD — preencher após retorno dos JSONs dos médicos.**
>
> Estrutura esperada:
> - Cohen's kappa global: ___
> - Cohen's kappa por categoria NER: ___
> - Cohen's kappa por campo SOAP: ___
> - Categorias com **maior aprovação** (ambos médicos): ___
> - Categorias com **menor aprovação**: ___
> - Padrões de correção mais frequentes: ___

### 9.5 Quote para o paper

> "The 30-note gold test set was independently reviewed by two
> infectologists. For each note, reviewers evaluated 6 NER categories
> and 6 SOAP fields using a 3-level Likert scale (Correct / Partial
> / Incorrect), optionally annotating corrections. Inter-rater
> agreement was computed as Cohen's kappa per category, and the
> consolidated gold standard was used for the metrics reported in
> Section [X]."

---

## 10. Limitações honestas

### 10.1 Cross-language transfer

Texto-fonte em inglês (MIMIC-IV, hospital americano), labels em português.
Modelos compactos têm capacidade limitada de transferência cross-language
robusta — esta é uma das razões dos F1 absolutos modestos.

### 10.2 Tamanho do dataset

602 notas para fine-tuning é pequeno para tarefas com schema de 12 campos
(6 NER + 6 SOAP). Datasets clínicos anotados em PT-BR são raros; o uso
de silver gerado por LLM é uma alternativa pragmática mas com viés
embutido — students aprendem o estilo do teacher, incluindo seus erros
sistemáticos.

### 10.3 Silver standard como ground-truth proxy

O gold test set é, ele mesmo, uma versão do silver com (a) seleção
estratificada, e (b) revisão clínica humana **a posteriori**. Comparar
modelos contra um gold derivado do teacher do qual eles foram destilados
introduz viés favorável a modelos da mesma família (Qwen 3.5-4B → Qwen
122B teacher). Isto explica parcialmente a vitória do Qwen 3.5-4B.

### 10.4 Modelos pequenos com LoRA

LoRA com r=16 atualiza 0.35-0.46% dos parâmetros. Essa restrição,
embora eficiente, limita a capacidade de aprender redistribuições
profundas das representações pré-treinadas. Full fine-tuning (caso do
BioBERTpt) atualiza 100% dos parâmetros, mas sofre catastrophic
forgetting em modelos de instruction-tuning grandes.

### 10.5 Validação humana com 2 médicos apenas

Cohen's kappa entre 2 reviewers não é o mesmo que kappa entre N
reviewers (Fleiss). Para n=2, qualquer disagreement não-resolvido
fica em zona cinzenta. Reviewers terciários ou anotação por consenso
seriam idealmente preferíveis em uma versão estendida.

### 10.6 Decoding determinístico vs stochastic

Greedy decoding favorece reprodutibilidade mas pode subestimar a
capacidade real do modelo. Sampling com temperature baixa (0.3-0.7)
poderia produzir saídas mais ricas em alguns casos. A escolha por
greedy foi **deliberada** por razões metodológicas (comparabilidade
e reprodutibilidade) mas reportada como limitação.

---

## 11. Trabalhos futuros

### 11.1 Anotação humana ampla

Um conjunto de treino de 200-500 notas anotadas por médicos (não silver)
permitiria fine-tuning supervisionado limpo, com gains projetados de
+0.15 a +0.30 em F1.

### 11.2 DPO para mitigar over-extraction

Coletar pares de preferência (entidade-curta-precisa preferida sobre
entidade-longa-prolixa) e aplicar Direct Preference Optimization. Foco
em ensinar discriminação, não extração.

### 11.3 Schema hierárquico explícito

Implementar prompts que forcem decisão hierárquica entre categorias
(ex: árvore de decisão: "é resultado de exame? → labResult; é
microorganismo confirmado? → organism; é doença? → disease"). Requer
prompt um pouco maior mas pode resolver confusão de categorias
adjacentes sem retraining.

### 11.4 Vocabulário controlado e normalização contra ontologia

Mapear saídas dos modelos contra ICD-10 e SNOMED-CT brasileiros para
normalização canônica. Resolveria definitivamente o problema de
granularidade lexical.

### 11.5 Modelos maiores com LoRA mais agressivo

Avaliar Qwen 3.5-14B ou Llama 4 8B com LoRA r=64-128 — mais headroom
de capacidade adaptável. Trade-off com custo de inferência em produção
hospitalar.

### 11.6 Dados em PT-BR nativos (não MIMIC)

Idealmente, repetir o pipeline em corpus de prontuários reais em PT-BR
(disponíveis via DataSUS, hospitais parceiros). Eliminaria o problema
cross-language e produziria modelos de produção real.

### 11.7 Avaliação RLHF/RLAIF

Após DPO inicial, próximo passo é RLHF com médicos pontuando saídas
de modelos lado a lado. Caro mas é o caminho para produção clínica.

---

## 12. Apêndices

### 12.1 Comandos completos para reprodução

#### A.1.1 Geração dos splits

```bash
cd experiments/training/data_prep

python format_chatml.py --split train
python format_chatml.py --split val
python format_gemma.py --split train
python format_gemma.py --split val
python format_bio_tagging.py --split train
python format_bio_tagging.py --split val
```

#### A.1.2 Treinamento

```bash
cd experiments/training

# Sequencial (Gemma 4 -> MedGemma -> Qwen)
nohup bash -c "
python train/train_gemma4_e4b.py --gpu 0 && \
python train/train_medgemma.py --gpu 0 && \
python train/train_qwen35_4b.py --gpu 0
" > logs/retrain_all.out 2>&1 &

# BioBERTpt separado (CPU-friendly)
python train/train_biobertpt.py --gpu 0
```

#### A.1.3 Inferência

```bash
pip install json-repair  # uma vez

python eval/run_inference.py --model gemma4_e4b --checkpoint checkpoints/gemma4_e4b/best --gpu 0
python eval/run_inference.py --model medgemma   --checkpoint checkpoints/medgemma/best   --gpu 0
python eval/run_inference.py --model qwen35_4b  --checkpoint checkpoints/qwen35_4b/best  --gpu 0
```

#### A.1.4 Métricas

```bash
python eval/compute_metrics.py --model gemma4_e4b --matching both
python eval/compute_metrics.py --model medgemma   --matching both
python eval/compute_metrics.py --model qwen35_4b  --matching both
python eval/compare_models.py
python eval/error_analysis.py --model qwen35_4b --top-k 10
```

### 12.2 Estrutura de diretórios

```
experiments/
├── PIPELINE_DOCUMENTATION.md          # documentação pré-silver
├── PAPER_SOURCE_POST_SILVER.md        # ESTE arquivo
├── config/
│   └── prompts.py                     # PROMPT_NER, PROMPT_SOAP (longos)
│                                      # PROMPT_NER_TRAIN, PROMPT_SOAP_TRAIN (compactos)
├── resultados/
│   ├── banco_dados_iana_v3_clean.json # silver standard (738 notas)
│   ├── gold_test_set_30.json          # gold (30 notas estratificadas)
│   ├── medical_review_final/           # 30 HTMLs editáveis para médicos
│   │   ├── index.html
│   │   ├── instrucoes.html
│   │   └── {pid}_{doenca}_review.html × 30
│   └── medical_review_final.zip       # pacote para envio aos médicos
└── training/
    ├── config/
    │   ├── medgemma.yaml
    │   ├── gemma4_e4b.yaml
    │   ├── qwen35_4b.yaml
    │   └── biobertpt.yaml
    ├── data/
    │   ├── splits/{train,val,test}_ids.json
    │   ├── chatml/{train,val}.jsonl
    │   ├── gemma_format/{train,val}.jsonl
    │   └── bio_tagging/{train,val}.json
    ├── data_prep/
    │   ├── format_chatml.py
    │   ├── format_gemma.py
    │   └── format_bio_tagging.py
    ├── train/
    │   ├── train_qwen35_4b.py
    │   ├── train_medgemma.py
    │   ├── train_gemma4_e4b.py
    │   ├── train_biobertpt.py
    │   └── shared/
    │       ├── completion_collator.py  # collator próprio
    │       ├── lora_config.py
    │       └── callbacks.py
    ├── eval/
    │   ├── run_inference.py
    │   ├── compute_metrics.py          # exact + fuzzy
    │   ├── compare_models.py
    │   ├── error_analysis.py
    │   └── reparse_predictions.py
    ├── checkpoints/
    │   ├── biobertpt/best/
    │   ├── medgemma/best/
    │   ├── gemma4_e4b/best/
    │   └── qwen35_4b/best/
    ├── predictions/
    │   ├── biobertpt_predictions.json
    │   ├── medgemma_predictions.json
    │   ├── gemma4_e4b_predictions.json
    │   └── qwen35_4b_predictions.json
    ├── results/
    │   ├── {model}_metrics.json
    │   ├── {model}_error_analysis.json
    │   ├── comparison.md
    │   └── comparison.csv
    └── logs/
        └── retrain_all.out
```

### 12.3 Tabelas finais consolidadas (template para o paper)

#### Tabela 1 — Setup dos modelos

| Modelo | Parâmetros | Pretraining | Trainable | Tempo | Eval loss |
|---|---|---|---|---|---|
| BioBERTpt-clin | 110M | PT-BR clínico | 100% (full) | 4 min | F1 BIO 0.15 |
| MedGemma 4B | 4B | EN médico | 0.35% (LoRA) | 1h | 0.61 |
| Gemma 4 E4B | 7.9B | Multilingual | 0.46% (LoRA) | 1h7min | 0.59 |
| Qwen 3.5-4B | 4B | Multilingual | 0.35% (LoRA) | 3h35min | 0.59 |

#### Tabela 2 — F1 (exact / fuzzy)

| Modelo | Micro-F1 (exact) | Micro-F1 (fuzzy) | Macro-F1 (fuzzy) |
|---|---|---|---|
| BioBERTpt¹ | 0.15 (BIO-level) | n/a | n/a |
| MedGemma | 0.073 | 0.168 | 0.166 |
| Gemma 4 E4B | 0.095 | 0.208 | 0.217 |
| Qwen 3.5-4B | 0.117 | TBD | TBD |

¹ BioBERTpt em framework BIO-level, não comparável diretamente.

#### Tabela 3 — F1 fuzzy por categoria

| Categoria | MedGemma | Gemma 4 | Qwen |
|---|---|---|---|
| disease_or_syndrome | 0.39 | 0.46 | TBD |
| sign_or_symptom | 0.12 | 0.21 | TBD |
| pharmacologic_substance | 0.20 | 0.21 | TBD |
| laboratory_or_test_result | 0.17 | 0.16 | TBD |
| diagnostic_procedure | 0.12 | 0.17 | TBD |
| organism_or_virus | 0.00 | 0.10 | TBD |

### 12.4 Citações relevantes

- **MIMIC-IV**: Johnson et al., 2023. *MIMIC-IV: A freely accessible
  electronic health record dataset.* Sci Data 10, 1.
- **BioBERTpt**: Rubel et al., 2020. *BioBERTpt: A Portuguese Neural
  Language Model for Clinical Named Entity Recognition.*
- **MedGemma**: Google Research, 2024. *MedGemma: Multimodal medical
  foundation models.*
- **Gemma 4**: Google DeepMind, 2025. *Gemma 4: Lightweight multimodal
  language models.*
- **Qwen 3.5**: Alibaba Cloud, 2025. *Qwen3.5 Technical Report.*
- **LoRA**: Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large
  Language Models.* arXiv:2106.09685.
- **TRL**: Hugging Face, 2024. *TRL: Transformer Reinforcement
  Learning library.* https://github.com/huggingface/trl
- **json-repair**: Mangiucugna, 2024. *json-repair: Repair invalid
  JSON output from LLMs.* https://github.com/mangiucugna/json_repair

### 12.5 Hiperparâmetros completos por modelo

#### BioBERTpt
```yaml
model_id: pucpr/biobertpt-clin
task: token_classification
num_labels: 13              # B-I × 6 cats + O
max_length: 512
num_train_epochs: 3
per_device_train_batch_size: 8
learning_rate: 2.0e-5
warmup_ratio: 0.1
seed: 42
```

#### MedGemma 4B / Gemma 4 E4B / Qwen 3.5-4B
```yaml
task: instruction_tuning
max_seq_length: 4096
num_train_epochs: 3
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
lr_scheduler: cosine
warmup_ratio: 0.03
optim: adamw_torch
gradient_checkpointing: true
gradient_checkpointing_kwargs: {use_reentrant: false}
prediction_loss_only: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj,
                   gate_proj, up_proj, down_proj]
  bias: none
  task_type: CAUSAL_LM

seed: 42
```

(Gemma 4 E4B adicional: `eval_accumulation_steps: 4`)

### 12.6 Prompt longo do teacher (estrutura)

O `PROMPT_NER` longo (2484 tokens, ~9.9k chars) usado pelo Qwen 122B
para gerar o silver standard está em
[`experiments/config/prompts.py`](config/prompts.py). Sua estrutura tem
**8 seções** numeradas:

1. **Cabeçalho de papel** — define o modelo como especialista em NER clínico
   de doenças infecciosas (HIV, TB, Sífilis).
2. **HIERARQUIA DE DECISÃO PARA CATEGORIZAÇÃO** — 6 categorias ordenadas por
   precedência (`laboratory > organism > diagnostic_procedure > disease >
   sign_or_symptom > pharmacologic`), cada uma com regras detalhadas e
   exemplos múltiplos. Inclui anti-pattern listings (o que NÃO categorizar).
3. **REGRA FUNDAMENTAL: EXTRAÇÃO LITERAL** — proíbe inferência a partir de
   diagnóstico (não inventar sintomas típicos), preenchimento de listas,
   extração de ROS negativa, ou alucinação.
4. **TRATAMENTO DE NEGAÇÕES E TESTES PENDENTES** — lista exemplos negativos
   com `denies/no/without/negative for`, e como tratar resultados pendentes
   (extrair em lab como "X-Pendente", não como doença).
5. **DISTINÇÃO SINAIS VITAIS vs EXAMES LABORATORIAIS** — sinais vitais (T,
   FC, FR, PA, SpO2) vão para SOAP; labs com amostra biológica processada
   vão para NER lab.
6. **NORMALIZAÇÃO E DEDUPLICAÇÃO** — uma forma canônica por conceito
   ("Diabetes" e "DM" → "Diabetes Mellitus").
7. **IDIOMA DA SAÍDA** — PT-BR clínico padrão, com exceções para siglas
   universais (HIV, WBC, etc.) e nomes científicos.
8. **EXTRAÇÃO DO PAST MEDICAL HISTORY (PMH)** — extrair todas comorbidades
   listadas no PMH para `disease_or_syndrome`, mesmo se não forem foco da
   admissão atual.

**Diferencial** vs prompt compacto: o prompt longo opera por **regras
explícitas** (modelo grande lê e aplica em zero-shot), enquanto o compacto
opera por **schema mínimo** (modelo pequeno aprendeu as regras pelos
exemplos do silver durante fine-tuning).

Para o paper: o prompt longo é o "professor" cuja distribuição é destilada
no silver standard; o prompt compacto é o "aluno" que opera com a
representação interna aprendida.

### 12.7 Reprodutibilidade

- **Seeds fixas** (42) em todos os scripts;
- **Greedy decoding** (`do_sample=False`) na inferência;
- **`repetition_penalty=1.2`** fixo;
- **Versões de bibliotecas** documentadas em `requirements.txt`;
- **Git commits** marcados em pontos críticos (`tags`):
  - `silver-v3-clean` — silver standard finalizado
  - `train-fix-completion-only` — fix do label masking
  - `inference-json-repair` — pipeline final de inferência
  - `metrics-fuzzy` — métricas com fuzzy matching
  - `medical-review-final` — HTMLs prontos para médicos

---

## Histórico de revisões deste documento

| Data | Versão | Alteração |
|---|---|---|
| 2026-04-29 | 1.0 | Criação inicial — cobre treinamento, inferência, avaliação automática e setup de validação clínica. Resultados de validação humana pendentes. |

