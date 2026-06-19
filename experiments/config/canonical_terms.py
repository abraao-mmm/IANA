"""
Projeto IANA - Dicionário de termos canônicos para normalização.

Mapeia variações comuns (sinônimos, abreviações, formas alternativas)
para a forma canônica em português brasileiro clínico.

Chaves: forma normalizada (lowercase, sem acentos) da variação.
Valores: forma canônica desejada na saída (COM acentuação correta).

Para adicionar novos termos: inclua a variação (lowercase, sem acentos)
como chave e a forma canônica como valor. A função de normalização aplica
strip + lowercase + remoção de acentos antes de consultar este dicionário.
"""

# ---------------------------------------------------------------------------
# Doenças e condições
# ---------------------------------------------------------------------------

CANONICAL_DISEASES: dict[str, str] = {
    # Hipertensão
    "hipertensao": "Hipertensão arterial sistêmica",
    "hipertensao arterial": "Hipertensão arterial sistêmica",
    "hipertensao arterial sistemica": "Hipertensão arterial sistêmica",
    "has": "Hipertensão arterial sistêmica",
    "htn": "Hipertensão arterial sistêmica",
    "hypertension": "Hipertensão arterial sistêmica",

    # Diabetes
    "diabetes": "Diabetes Mellitus",
    "diabetes mellitus": "Diabetes Mellitus",
    "dm": "Diabetes Mellitus",
    "diabetes mellitus tipo 2": "Diabetes Mellitus tipo 2",
    "dm2": "Diabetes Mellitus tipo 2",
    "diabetes tipo 2": "Diabetes Mellitus tipo 2",
    "diabetes mellitus tipo 1": "Diabetes Mellitus tipo 1",
    "dm1": "Diabetes Mellitus tipo 1",

    # Dislipidemia
    "hiperlipidemia": "Dislipidemia",
    "dislipidemia": "Dislipidemia",
    "hyperlipidemia": "Dislipidemia",

    # DRGE
    "drge": "Doença do Refluxo Gastroesofágico (DRGE)",
    "doenca do refluxo gastroesofagico": "Doença do Refluxo Gastroesofágico (DRGE)",
    "refluxo gastroesofagico": "Doença do Refluxo Gastroesofágico (DRGE)",
    "gerd": "Doença do Refluxo Gastroesofágico (DRGE)",

    # HIV/AIDS
    "aids": "AIDS (Síndrome da Imunodeficiência Adquirida)",
    "sindrome da imunodeficiencia adquirida": "AIDS (Síndrome da Imunodeficiência Adquirida)",
    "hiv/aids": "HIV/AIDS",
    "hiv": "HIV/AIDS",

    # Tuberculose
    "tuberculose": "Tuberculose",
    "tb": "Tuberculose",
    "tuberculose pulmonar": "Tuberculose pulmonar",

    # Sífilis
    "sifilis": "Sífilis",
    "syphilis": "Sífilis",
    "neurossifilis": "Neurossífilis",
    "neurosyphilis": "Neurossífilis",

    # Insuficiência renal
    "insuficiencia renal cronica": "Doença renal crônica",
    "irc": "Doença renal crônica",
    "doenca renal cronica": "Doença renal crônica",
    "drc": "Doença renal crônica",
    "ckd": "Doença renal crônica",
    "insuficiencia renal aguda": "Lesão renal aguda",
    "ira": "Lesão renal aguda",
    "lesao renal aguda": "Lesão renal aguda",

    # Insuficiência cardíaca
    "insuficiencia cardiaca": "Insuficiência cardíaca",
    "insuficiencia cardiaca congestiva": "Insuficiência cardíaca congestiva",
    "icc": "Insuficiência cardíaca congestiva",
    "chf": "Insuficiência cardíaca congestiva",

    # Fibrilação atrial
    "fibrilacao atrial": "Fibrilação atrial",
    "fa": "Fibrilação atrial",
    "afib": "Fibrilação atrial",

    # DPOC
    "dpoc": "Doença Pulmonar Obstrutiva Crônica (DPOC)",
    "doenca pulmonar obstrutiva cronica": "Doença Pulmonar Obstrutiva Crônica (DPOC)",
    "copd": "Doença Pulmonar Obstrutiva Crônica (DPOC)",

    # Raynaud
    "sindrome de raynaud": "Fenômeno de Raynaud",
    "fenomeno de raynaud": "Fenômeno de Raynaud",
    "doenca de raynaud": "Fenômeno de Raynaud",

    # Anemia
    "anemia": "Anemia",
    "anemia ferropriva": "Anemia ferropriva",
    "anemia de doenca cronica": "Anemia de doença crônica",
    "anemia de inflamacao cronica": "Anemia de doença crônica",

    # Hepatite
    "hepatite b": "Hepatite B",
    "hepatite c": "Hepatite C",
    "hepatite a": "Hepatite A",

    # Embolia pulmonar
    "embolia pulmonar": "Embolia pulmonar",
    "ep": "Embolia pulmonar",
    "tromboembolismo pulmonar": "Embolia pulmonar",
    "tep": "Embolia pulmonar",
    "pe": "Embolia pulmonar",

    # Outros
    "obesidade": "Obesidade",
    "obesidade morbida": "Obesidade mórbida",
    "hipotireoidismo": "Hipotireoidismo",
    "asma": "Asma",
    "osteoporose": "Osteoporose",
    "osteopenia": "Osteopenia",
    "gota": "Gota",
    "depressao": "Depressão",
    "ansiedade": "Transtorno de ansiedade",
}

# ---------------------------------------------------------------------------
# Medicamentos (sinônimos e nomes comerciais comuns)
# ---------------------------------------------------------------------------

CANONICAL_MEDICATIONS: dict[str, str] = {
    "bactrim": "Sulfametoxazol-Trimetoprima",
    "tmp-smx": "Sulfametoxazol-Trimetoprima",
    "sulfametoxazol-trimetoprima": "Sulfametoxazol-Trimetoprima",
    "cotrimoxazol": "Sulfametoxazol-Trimetoprima",

    "tylenol": "Paracetamol",
    "acetaminofeno": "Paracetamol",
    "paracetamol": "Paracetamol",

    "advil": "Ibuprofeno",
    "ibuprofeno": "Ibuprofeno",

    "lasix": "Furosemida",
    "furosemida": "Furosemida",

    "zofran": "Ondansetrona",
    "ondansetrona": "Ondansetrona",

    "heparina": "Heparina",
    "enoxaparina": "Enoxaparina",
    "lovenox": "Enoxaparina",

    "insulina": "Insulina",
    "insulina regular": "Insulina regular",
    "insulina nph": "Insulina NPH",
    "insulina glargina": "Insulina glargina",
    "lantus": "Insulina glargina",
}

# ---------------------------------------------------------------------------
# Organismos
# ---------------------------------------------------------------------------

CANONICAL_ORGANISMS: dict[str, str] = {
    "mycobacterium tuberculosis": "Mycobacterium tuberculosis",
    "m. tuberculosis": "Mycobacterium tuberculosis",
    "mtb": "Mycobacterium tuberculosis",

    "pneumocystis jirovecii": "Pneumocystis jirovecii",
    "pneumocystis carinii": "Pneumocystis jirovecii",
    "pcp": "Pneumocystis jirovecii",

    "treponema pallidum": "Treponema pallidum",

    "virus da hepatite b": "Vírus da Hepatite B (HBV)",
    "hbv": "Vírus da Hepatite B (HBV)",
    "virus da hepatite b (hbv)": "Vírus da Hepatite B (HBV)",

    "virus da hepatite c": "Vírus da Hepatite C (HCV)",
    "hcv": "Vírus da Hepatite C (HCV)",

    "staphylococcus aureus": "Staphylococcus aureus",
    "mrsa": "Staphylococcus aureus resistente à meticilina (MRSA)",
    "s. aureus": "Staphylococcus aureus",

    "escherichia coli": "Escherichia coli",
    "e. coli": "Escherichia coli",

    "candida albicans": "Candida albicans",
    "c. albicans": "Candida albicans",
}

# ---------------------------------------------------------------------------
# Índice unificado para busca rápida
# ---------------------------------------------------------------------------

ALL_CANONICAL: dict[str, str] = {
    **CANONICAL_DISEASES,
    **CANONICAL_MEDICATIONS,
    **CANONICAL_ORGANISMS,
}
