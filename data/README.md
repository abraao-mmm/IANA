# 📁 Diretório `data/`

Este diretório gerencia a obtenção e o pré-processamento inicial das bases de dados brutas utilizadas no projeto **EHR-Sentinel**.

> **Nota:** Os dados brutos do MIMIC-IV (`.csv`, `.parquet`) **nao** sao versionados por questoes de tamanho (~6.3GB). Use os scripts `download_mimic.py` e `extract_mimic.py` para obte-los localmente. A base SemClinBr (`.xml`, ~8MB) **e versionada** e ja esta incluida no repositorio.

---

## 🛠️ Scripts Utilitários

### `download_mimic.py`

Script automatizado para download da base **MIMIC-IV-Note** (v2.2) via um link direto do Google Drive. Utiliza a biblioteca `gdown`.

```bash
python data/download_mimic.py
```

- Realiza o download do arquivo ZIP contendo as anotações clinicas da PhysioNet para esta pasta (aproximadamente `3.10GB`).
- Exibe o progresso de transferência no console.

### `extract_mimic.py`

Script para o setup e otimização dos dados originais extraídos do MIMIC. Mapeia e executa três tarefas obrigatórias em sequência:

```bash
python data/extract_mimic.py
```

1. **Extração ZIP**: Identifica arquivos `.zip` baixados, efetua a descompactação das hierarquias e deleta o comprimido.
2. **Extração GZ**: Procura os CSVs distribuídos em `.csv.gz`, os descompacta localmente em `note/` e deleta as archives `.gz`.
3. **Conversão Parquet**: Faz a conversão 1-para-1 de todos os arquivos `*.csv` para o formato colunar `*.parquet` e os armazena dentro de `note_parquet/`. (Utiliza engine multithreaded do Polars).

> Os arquivos `.parquet` consumem tipicamente cerca de 5–10x menos memória em disco quando comparados com o formato `.csv`, acelerando o tempo de carregamento em Pandas/Polars exponencialmente em 5–20x.

---

## 📚 Bases de Dados

### 1. MIMIC-IV-Note v2.2 (Inglês)

> **Fonte:** [PhysioNet — MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/)
>
> Coleção exaustiva documentando notas clínicas desidentificadas inseridas em regime de texto livre associadas aos prontuários do MIMIC-IV no Beth Israel Deaconess Medical Center.

A estrutura esperada após o script de extração completo é:

```text
mimic-iv-note-deidentified-free-text-clinical-notes-2.2/
├── LICENSE.txt            # Licença PhysioNet Credentialed Health Data (v1.5.0)
├── SHA256SUMS.txt         # Hashes SHA-256 dos artefatos
├── note/                  # CSVs de Origem 
│   ├── discharge.csv          # Notas completas de Alta (Sumários)
│   ├── discharge_detail.csv   # Estrutura Metadados associados à Nota
│   ├── radiology.csv          # Laudos Integrados da Radiologia
│   └── radiology_detail.csv   # Estrutura Metadados vinculada ao Laudo
└── note_parquet/          # Transformação Aceleradora Parquet (`extract_mimic.py`)
    ├── discharge.parquet
    ├── discharge_detail.parquet
    ├── radiology.parquet
    └── radiology_detail.parquet
```

---

### 2. SemClinBr v1 (Português Brasileiro)

> **Fonte:** Corpus Público SemClinBr
>
> Corpus nacional pioneiro composto de documentações como notas evolutivas e prescrições que passaram por processos complexos de anotações gramaticais focadas em saúde, marcando relações lexicais e entidades biomédicas.

A estrutura esperada requer arquivos individuais brutos mapeando prontuários únicos:

```text
SemClinBr-xml-public-v1/
├── 8906.xml
├── 8907.xml
├── ...
└── 9935.xml     (≈ 1.000 arquivos originais .xml)
```

Cada iterador XML reflete um padrão gramatical:

```xml
<ANNOTATIONS>
  <TEXT>Texto clínico irrestrito oriundo de diários/enfermarias/consultórios...</TEXT>
  <TAGS>
    <annotation id="1" tag="Disease or Syndrome" start="0" end="12" text="OSTEOMIELITE" abbr="" />
  </TAGS>
  <RELATIONS>
    <rel annotation1="1" annotation2="2" reltype="associated_with" />
  </RELATIONS>
</ANNOTATIONS>
```

| Tag Principal | Escopo do Conteúdo Retornado |
| ------------- | ---------------------------- |
| `<TEXT>` | Transcrição primária, base do raciocínio sintático. |
| `<TAGS>` | Agrupa as entidades anotadas (`annotation`). Apresenta: `id`, `tag` (UMLS Mapping Ontology Equivalent), offsets geográficos (`start`/`end`), o trecho visual indexado e possíveis abreviaturas. |
| `<RELATIONS>`| O núcleo da extração de relacionamentos (`rel`); conecta entidades, definindo um predicado e um objeto (`reltype` ex: `associated_with`, `negation_of`). |
