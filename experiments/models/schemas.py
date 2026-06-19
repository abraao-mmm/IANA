"""
Projeto IANA - Modelos Pydantic compartilhados (v3).

Define os schemas de validação para extração clínica (NER + SOAP)
e os estados dos grafos LangGraph.

Changelog v3:
- Field descriptions enriquecidas com instruções negativas explícitas
- Hierarquia de exclusividade mútua documentada nos fields
"""

import operator
from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ============================================================
# 1. MODELOS DE SAÍDA DA LLM (Pydantic - Structured Output)
# ============================================================

class EntidadeClinica(BaseModel):
    """NER clínico baseado na taxonomia SemClinBR (v3 — com regras de exclusividade)."""

    disease_or_syndrome: List[str] = Field(
        default_factory=list,
        description=(
            "SOMENTE diagnósticos, comorbidades e condições clínicas FORMALMENTE "
            "estabelecidas no paciente. Fontes válidas: Discharge Diagnosis, "
            "Past Medical History, condições atribuídas no Brief Hospital Course. "
            "Cada conceito clínico deve aparecer UMA ÚNICA VEZ (forma canônica em português). "
            "NÃO INCLUA: sintomas (vão em sign_or_symptom), achados de imagem "
            "(vão em soap.objetivo_imagem), anormalidades laboratoriais descritas "
            "em linguagem clínica como 'leucocitose' ou 'PCR elevada' (extraia o valor "
            "numérico em laboratory_or_test_result), condições descartadas/negadas, "
            "condições com teste negativo, condições históricas tratadas e resolvidas."
        ),
    )
    sign_or_symptom: List[str] = Field(
        default_factory=list,
        description=(
            "SOMENTE sinais e sintomas AFIRMATIVAMENTE presentes: relatados pelo "
            "paciente ou observados no exame físico. "
            "NÃO INCLUA: sintomas negados pelo paciente ('denies fever' -> não extrair), "
            "diagnósticos formais (vão em disease_or_syndrome), valores laboratoriais "
            "(vão em laboratory_or_test_result), achados de imagem "
            "(vão em soap.objetivo_imagem). "
            "Se o mesmo termo aparece em disease_or_syndrome, NÃO duplique aqui."
        ),
    )
    pharmacologic_substance: List[str] = Field(
        default_factory=list,
        description=(
            "TODOS os medicamentos citados no prontuário: admissão, curso hospitalar e alta. "
            "Inclua nome e dosagem quando disponível (ex: Atenolol 50 mg, Prednisona 40 mg). "
            "Use forma canônica em português. Um medicamento = uma entrada (não duplique "
            "sinônimos como 'Bactrim' e 'Sulfametoxazol-Trimetoprima')."
        ),
    )
    laboratory_or_test_result: List[str] = Field(
        default_factory=list,
        description=(
            "TODOS os resultados de análises feitas em AMOSTRAS BIOLÓGICAS (sangue, urina, "
            "líquor, escarro, swab, tecido). Inclua NOME DO TESTE e RESULTADO juntos "
            "(ex: 'WBC-11.5', 'CD4-113', 'RPR-Negativo', 'Quantiferon Gold-Positivo'). "
            "Testes com resultado negativo ou pendente DEVEM ser incluídos aqui "
            "preservando o resultado (ex: 'HIV-Negativo', 'Brucella anticorpos-Pendente'). "
            "NÃO INCLUA: sinais vitais medidos à beira do leito (temperatura, PA, FC, FR, "
            "SpO2 por oximetria de pulso — esses vão em soap.objetivo_exame_fisico). "
            "Gasometria arterial (ABG) É laboratório e DEVE ser incluída aqui. "
            "NÃO extraia a mesma informação também como doença ou organismo: se "
            "'RPR-Negativo' está aqui, NÃO extraia 'Sífilis' em disease_or_syndrome "
            "nem 'Treponema pallidum' em organism_or_virus."
        ),
    )
    diagnostic_procedure: List[str] = Field(
        default_factory=list,
        description=(
            "SOMENTE intervenções FÍSICAS sobre o paciente para fins diagnósticos ou "
            "terapêuticos, que envolvem manipulação direta. "
            "Exemplos: lavagem broncoalveolar, biópsia, punção lombar, endoscopia, "
            "cateterismo, ecocardiograma, radiografia de tórax, tomografia, "
            "ressonância, cirurgias, indução de escarro. "
            "NÃO INCLUA: testes laboratoriais isolados cujo RESULTADO já está em "
            "laboratory_or_test_result. Se o resultado do teste está disponível, "
            "ele vai em laboratory_or_test_result; o PROCEDIMENTO de coleta só entra "
            "aqui se foi clinicamente notável (ex: BAL, biópsia pericárdica). "
            "NÃO duplique: se 'Quantiferon Gold-Positivo' está em lab, NÃO extraia "
            "'Teste Quantiferon Gold' aqui."
        ),
    )
    organism_or_virus: List[str] = Field(
        default_factory=list,
        description=(
            "SOMENTE microrganismos CONFIRMADOS como causa ativa ou suspeita ativa "
            "da condição do paciente. "
            "Nível 1 (extrair): organismo confirmado por teste positivo "
            "(ex: Pneumocystis jirovecii com BAL positivo). "
            "Nível 2 (NÃO extrair aqui): organismo testado e descartado — o resultado "
            "do teste vai em laboratory_or_test_result como 'AFB-Negativo'. "
            "Nível 3 (NÃO extrair em nenhuma categoria): organismo em hipótese "
            "descartada ou histórico tratado. "
            "NÃO confunda NOME DA DOENÇA com NOME DO ORGANISMO: "
            "'Hepatite B' é doença -> disease_or_syndrome; "
            "'Vírus da Hepatite B (HBV)' é organismo -> organism_or_virus (só se ativo). "
            "NUNCA extraia organismos de testes negativos, pendentes ou descartados."
        ),
    )


class SOAP(BaseModel):
    """Estrutura SOAP expandida com objetivo subdividido."""

    subjetivo: str = Field(
        default="",
        description=(
            "História da doença atual, queixa principal, relato do paciente, "
            "histórico médico relevante e revisão de sistemas. "
            "Inclua o que o paciente relata E o que nega."
        ),
    )
    objetivo_exame_fisico: str = Field(
        default="",
        description=(
            "Achados detalhados do exame físico: sinais vitais completos "
            "(Temperatura, PA, FC, FR, SpO2 por oximetria de pulso), "
            "inspeção, ausculta, palpação. "
            "INCLUA aqui SpO2/SaO2 medida por oximetria de pulso (sinal vital). "
            "Transcreva todos os achados relevantes."
        ),
    )
    objetivo_laboratorio: str = Field(
        default="",
        description=(
            "TODOS os resultados laboratoriais com valores numéricos: hemograma, "
            "bioquímica, sorologias, culturas, gasometria arterial (ABG), marcadores. "
            "Liste cada exame individualmente, não resuma."
        ),
    )
    objetivo_imagem: str = Field(
        default="",
        description=(
            "TODOS os achados de exames de imagem: raio-X, tomografia, ecocardiograma, etc. "
            "Inclua a impressão/conclusão de cada exame. "
            "Achados radiológicos como atelectasia, edema pulmonar, derrame pericárdico "
            "como achado de imagem pertencem AQUI, não em disease_or_syndrome do NER."
        ),
    )
    avaliacao: str = Field(
        default="",
        description=(
            "Raciocínio clínico completo: diagnóstico principal, diagnósticos diferenciais "
            "considerados, evolução do quadro durante a internação e conclusões da equipe médica."
        ),
    )
    plano: str = Field(
        default="",
        description=(
            "Conduta completa: medicações prescritas na alta (com dose e posologia), "
            "orientações ao paciente, encaminhamentos, exames de seguimento e retornos programados."
        ),
    )


class ValidacaoResult(BaseModel):
    """Resultado da auditoria de completude da extração (legacy, mantido para compatibilidade)."""

    completo: bool = Field(
        description="True se não há gaps significativos na extração."
    )
    gaps: List[str] = Field(
        default_factory=list,
        description=(
            "Lista de entidades ou informações PRESENTES no texto original "
            "mas AUSENTES na extração. Cada item deve ser específico "
            "(ex: 'Anemia de inflamação crônica', 'ALT-46', 'Levofloxacino')."
        ),
    )


class AgentStatus(BaseModel):
    """Status de execução de cada agente do pipeline."""

    ner_status: Literal["ok", "token_overflow", "error", "skipped"] = "ok"
    soap_status: Literal["ok", "token_overflow", "error", "skipped"] = "ok"
    audit_status: Literal["ok", "token_overflow", "error", "skipped", "not_needed"] = "ok"
    ner_error_message: Optional[str] = None
    soap_error_message: Optional[str] = None
    audit_error_message: Optional[str] = None


class RelatorioClinicoProcessado(BaseModel):
    """Contrato final de saída - um prontuário estruturado completo."""

    paciente_id: str = Field(description="ID da internação (hadm_id).")
    codigo_cid: str = Field(description="Código CID (ICD) associado.")
    doenca_alvo_identificada: str = Field(description="Doença principal (TB, HIV ou Sífilis).")
    ner: EntidadeClinica
    soap: SOAP
    agent_status: AgentStatus = Field(default_factory=AgentStatus)


# ============================================================
# 2. ESTADOS DOS GRAFOS LANGGRAPH (TypedDict)
# ============================================================

class EstadoExtracao(TypedDict):
    """Estado do grafo de extração para uma única nota clínica."""

    # Input
    hadm_id: str
    codigo_cid: str
    doenca_alvo: str
    texto_prontuario: str

    # Resultados dos agentes paralelos
    ner: Optional[EntidadeClinica]
    soap: Optional[SOAP]

    # Validação (legacy)
    validacao: Optional[ValidacaoResult]

    # Status por agente (preenchido pelos nós do grafo)
    ner_status: Optional[str]
    ner_error: Optional[str]
    soap_status: Optional[str]
    soap_error: Optional[str]
    audit_status: Optional[str]
    audit_error: Optional[str]

    # Output final
    resultado_json: Optional[dict]


class EstadoBatch(TypedDict):
    """Estado do grafo de processamento em lote."""

    notas: List[dict]
    resultados: Annotated[List[dict], operator.add]
