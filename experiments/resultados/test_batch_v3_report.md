# Relatório — Test Batch v3 (10 notas)

**Data**: 2026-04-08 15:09

## 1. Tempos de processamento

| Métrica | Valor |
|---|---|
| Tempo total | 695.4s (11.6 min) |
| Tempo médio | 69.5s |
| Mediana | 39.8s |
| Mais rápida | 15.3s |
| Mais lenta | 326.6s |

### Detalhamento por nota

| hadm_id | Categoria | Entidades | Tempo (s) | Status | NER | SOAP | Audit |
|---|---|---|---|---|---|---|---|
| 22924630 | HIV complexa | 340 | 326.6 | ok | ok | ok | ok |
| 20248623 | TB simples | 69 | 25.6 | ok | ok | ok | ok |
| 23080963 | Sífilis adequada | 155 | 43.5 | ok | ok | ok | ok |
| 22413631 | HIV simples | 33 | 18.6 | ok | ok | ok | ok |
| 24918106 | HIV simples | 75 | 21.2 | ok | ok | ok | ok |
| 25557330 | HIV complexa | 167 | 46.4 | ok | ok | ok | ok |
| 27321074 | TB complexa | 137 | 48.4 | ok | ok | ok | ok |
| 27306123 | Sífilis zero (caso patológico) | 37 | 15.3 | ok | ok | ok | ok |
| 22978216 | Sífilis adequada | 146 | 36.1 | ok | ok | ok | ok |
| 20250010 | TB complexa | 0 | 113.6 | partial | token_overflow | ok | not_needed |

### Status agregado por agente

| Agente | ok | token_overflow | error | skipped | not_needed |
|---|---|---|---|---|---|
| NER | 9 | 1 | 0 | 0 | 0 |
| SOAP | 10 | 0 | 0 | 0 | 0 |
| Audit | 9 | 0 | 0 | 0 | 1 |

## 2. Métricas do auditor LLM

| Métrica | Valor |
|---|---|
| Notas auditadas | 9 |
| Duração média | 12.3s |
| % max_tokens médio | 6.0% |
| % max_tokens máximo | 14.0% |
| Notas com mudanças | 9/9 |
| Mudanças médias por nota | 60.1 |

### Detalhamento por nota

| hadm_id | Duração (s) | % max_tokens | Mudanças | Ent. antes | Ent. depois |
|---|---|---|---|---|---|
| 22924630 | 31.3 | 14.0% | 133 | 307 | 340 |
| 20248623 | 5.7 | 2.5% | 120 | 65 | 70 |
| 23080963 | 12.8 | 6.7% | 76 | 105 | 155 |
| 22413631 | 3.5 | 1.8% | 8 | 31 | 33 |
| 24918106 | 5.0 | 2.1% | 24 | 65 | 75 |
| 25557330 | 22.3 | 10.9% | 124 | 155 | 234 |
| 27321074 | 13.5 | 7.0% | 1 | 138 | 137 |
| 27306123 | 3.8 | 2.2% | 37 | 32 | 37 |
| 22978216 | 13.2 | 6.4% | 18 | 148 | 146 |

## 3. Validação de qualidade (8 checks)

| Métrica | Valor |
|---|---|
| Notas limpas (0 issues) | 8/10 |
| Notas com issues | 2 |
| Total de issues | 4 |

### Issues por check

| Check | Ocorrências |
|---|---|
| cross_category_duplication | ✅ |
| imaging_finding_in_disease | ✅ |
| language_violation | ⚠️ 4 |
| negative_test_leakage | ✅ |
| pending_test_leakage | ✅ |
| symptom_in_disease | ✅ |
| synonym_duplication | ✅ |
| vital_sign_in_lab | ✅ |

### Notas com issues

**22924630** (HIV) — 2 issues:

- `language_violation`: {"category": "disease_or_syndrome", "entity": "Rash hipopigmentado macular", "english_terms": ["Rash"]}
- `language_violation`: {"category": "sign_or_symptom", "entity": "Rash (negado no ROS, mas presente na história)", "english_terms": ["Rash"]}

**27321074** (Tuberculose) — 2 issues:

- `language_violation`: {"category": "disease_or_syndrome", "entity": "Reação medicamentosa (rash)", "english_terms": ["rash"]}
- `language_violation`: {"category": "sign_or_symptom", "entity": "Rash papular eritematoso (peito, abdômen superior, costas)", "english_terms": ["Rash"]}
