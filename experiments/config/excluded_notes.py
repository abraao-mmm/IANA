# Notas excluídas do dataset por falha no filtro de cobertura textual
# (doença alvo não aparece no corpo do texto, apenas no código ICD de faturamento)
# Critério: mention_count == 0 no audit_text_coverage.py
#
# Referência: auditoria de 2026-04-08 sobre 749 notas MIMIC-IV
# Resultado: 11 notas (1.5%) com zero menções — 10 sífilis + 1 HIV
#
# EXCEÇÃO: 27306123 é excluída da produção mas MANTIDA no test set de 10 notas
# para validar comportamento do pipeline em casos patológicos (deve retornar
# listas vazias em vez de inventar entidades).

EXCLUDED_PATIENT_IDS: set[str] = {
    "29440892",  # Sífilis 0940
    "21486510",  # Sífilis 0940
    "28210762",  # Sífilis A5216
    "22650574",  # Sífilis 0940
    "25347439",  # Sífilis 0940
    "28554991",  # Sífilis 0940
    "21336570",  # Sífilis 0940
    "27306123",  # Sífilis 0940 (Amostra 2 — mantida no test set)
    "29718503",  # Sífilis 0940
    "28746891",  # Sífilis 0940
    "25144784",  # HIV B20
}

EXCLUSION_REASON = "Doença alvo não mencionada textualmente no resumo de alta"
