"""
Script para download da base de dados MIMIC a partir do Google Drive.

Requisitos:
    pip install gdown

Uso:
    python download_mimic.py
"""

import os
import sys

try:
    import gdown
except ImportError:
    print("Biblioteca 'gdown' não encontrada. Instalando...")
    os.system(f"{sys.executable} -m pip install gdown")
    import gdown


# ID do arquivo no Google Drive extraído da URL de compartilhamento
FILE_ID = "1ntV8pm0iwbKMxglszL8AfJ-7AP45WtvU"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Diretório de destino (mesmo diretório deste script)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "mimic_database.zip")


def main():
    print("=" * 60)
    print("  Download da Base de Dados MIMIC - Google Drive")
    print("=" * 60)
    print(f"\n  URL de origem : {DOWNLOAD_URL}")
    print(f"  Destino       : {OUTPUT_PATH}\n")

    if os.path.exists(OUTPUT_PATH):
        resp = input("O arquivo já existe. Deseja sobrescrever? (s/n): ").strip().lower()
        if resp != "s":
            print("Download cancelado.")
            return

    print("Iniciando download... (isso pode levar alguns minutos)")
    gdown.download(DOWNLOAD_URL, OUTPUT_PATH, quiet=False)

    if os.path.exists(OUTPUT_PATH):
        size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
        print(f"\n✔ Download concluído com sucesso!")
        print(f"  Arquivo: {OUTPUT_PATH}")
        print(f"  Tamanho: {size_mb:.2f} MB")
    else:
        print("\n✘ Erro: o download falhou. Verifique a URL e tente novamente.")
        sys.exit(1)


if __name__ == "__main__":
    main()
