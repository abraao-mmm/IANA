"""
Projeto IANA — Utilitários de chunking para sequências longas.

Reutilizado pelos conversores de dados que precisam dividir
textos em chunks de 512 tokens com overlap.
"""


def chunk_sequence(
    items: list,
    max_len: int = 512,
    overlap: int = 50,
) -> list[list]:
    """Divide uma sequência em chunks com overlap.

    Args:
        items: Lista de itens (tokens, labels, etc.)
        max_len: Tamanho máximo de cada chunk
        overlap: Número de itens sobrepostos entre chunks

    Returns:
        Lista de chunks (sublistas)
    """
    if not items:
        return []

    chunks = []
    step = max(1, max_len - overlap)

    for start in range(0, len(items), step):
        end = min(start + max_len, len(items))
        chunk = items[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= len(items):
            break

    return chunks
