"""
Script para descompactar e preparar todos os dados da base MIMIC.

Etapas:
  1. .zip  → extrai e apaga o .zip
  2. .gz   → descompacta (ex: arquivo.csv.gz → arquivo.csv) e apaga o .gz
  3. CSV   → converte cada .csv em .parquet na pasta note_parquet/

Requisitos:
    pip install polars

Uso:
    python data/extract_mimic.py
"""

import os
import sys
import gc
import gzip
import shutil
import zipfile
import glob
import time

# Diretório raiz dos dados MIMIC
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "mimic-iv-note-deidentified-free-text-clinical-notes-2.2")


def extract_zip(zip_path):
    """Extrai um arquivo .zip e o apaga em seguida."""
    extract_dir = os.path.dirname(zip_path)
    print(f"  ⏳ Extraindo ZIP: {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    os.remove(zip_path)
    print(f"  ✔  Extraído e removido: {os.path.basename(zip_path)}")


def extract_gz(gz_path):
    """Descompacta um arquivo .gz e o apaga em seguida."""
    out_path = gz_path[:-3]
    size_mb = os.path.getsize(gz_path) / (1024 * 1024)
    print(f"  ⏳ Descompactando GZ: {os.path.basename(gz_path)} ({size_mb:.1f} MB) ...")

    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(gz_path)
    out_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ✔  Descompactado: {os.path.basename(out_path)} ({out_size_mb:.1f} MB)")


def convert_csvs_to_parquet():
    """Converte todos os CSVs em note/ para Parquet em note_parquet/."""
    try:
        import polars as pl
    except ImportError:
        print("\n⚠️  Polars não instalado — pulando conversão para Parquet.")
        print("   Instale com: pip install polars")
        return

    note_dir = os.path.join(DATA_DIR, "note")
    parquet_dir = os.path.join(DATA_DIR, "note_parquet")

    csv_files = glob.glob(os.path.join(note_dir, "*.csv"))
    if not csv_files:
        print("\n📄 Nenhum CSV encontrado em note/ para converter.")
        return

    os.makedirs(parquet_dir, exist_ok=True)

    print(f"\n🔄 Convertendo {len(csv_files)} CSV(s) → Parquet...")
    print(f"   Destino: note_parquet/\n")

    for csv_path in csv_files:
        basename = os.path.basename(csv_path)
        parquet_name = basename.replace(".csv", ".parquet")
        parquet_path = os.path.join(parquet_dir, parquet_name)

        if os.path.exists(parquet_path):
            parquet_mb = os.path.getsize(parquet_path) / (1024 * 1024)
            print(f"  ⏭  {parquet_name} já existe ({parquet_mb:.1f} MB) — pulando")
            continue

        csv_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"  ⏳ {basename} ({csv_mb:.1f} MB) → {parquet_name} ...")

        t0 = time.perf_counter()
        df = pl.read_csv(csv_path)
        df.write_parquet(parquet_path)
        elapsed = time.perf_counter() - t0

        parquet_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        ratio = (1 - parquet_mb / csv_mb) * 100 if csv_mb > 0 else 0

        print(f"  ✔  {parquet_name} ({parquet_mb:.1f} MB, -{ratio:.0f}%) em {elapsed:.1f}s")

        del df
        gc.collect()


def main():
    print("=" * 60)
    print("  Extração e preparação dos dados MIMIC")
    print("=" * 60)

    if not os.path.isdir(DATA_DIR):
        print(f"\n✘ Diretório não encontrado: {DATA_DIR}")
        print("  Execute o download_mimic.py primeiro.")
        sys.exit(1)

    # --- Passo 1: Descompactar todos os .zip encontrados ---
    zip_files = glob.glob(os.path.join(DATA_DIR, "**", "*.zip"), recursive=True)
    if zip_files:
        print(f"\n📦 Encontrados {len(zip_files)} arquivo(s) .zip:")
        for zf in zip_files:
            extract_zip(zf)
    else:
        print("\n📦 Nenhum arquivo .zip encontrado.")

    # --- Passo 2: Descompactar todos os .gz encontrados ---
    gz_files = glob.glob(os.path.join(DATA_DIR, "**", "*.gz"), recursive=True)
    if gz_files:
        print(f"\n📦 Encontrados {len(gz_files)} arquivo(s) .gz:")
        for gf in gz_files:
            extract_gz(gf)
    else:
        print("\n📦 Nenhum arquivo .gz encontrado.")

    # --- Passo 3: Converter CSVs → Parquet ---
    convert_csvs_to_parquet()

    # --- Resumo final ---
    remaining_zip = glob.glob(os.path.join(DATA_DIR, "**", "*.zip"), recursive=True)
    remaining_gz = glob.glob(os.path.join(DATA_DIR, "**", "*.gz"), recursive=True)

    print("\n" + "=" * 60)
    if not remaining_zip and not remaining_gz:
        print("  ✔ Tudo descompactado! Nenhum .zip ou .gz restante.")
    else:
        print(f"  ⚠ Ainda restam: {len(remaining_zip)} .zip, {len(remaining_gz)} .gz")

    parquet_dir = os.path.join(DATA_DIR, "note_parquet")
    if os.path.isdir(parquet_dir):
        parquets = glob.glob(os.path.join(parquet_dir, "*.parquet"))
        print(f"  ✔ {len(parquets)} Parquet(s) gerados em note_parquet/")

    print("=" * 60)


if __name__ == "__main__":
    main()
