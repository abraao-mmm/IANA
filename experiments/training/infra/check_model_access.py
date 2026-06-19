#!/usr/bin/env python3
"""
Projeto IANA — Verifica acesso aos 5 modelos do benchmark via HuggingFace Hub.

Uso:
    python check_model_access.py
    python check_model_access.py --token $HF_TOKEN
"""

import argparse
import json
import logging
import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent.parent


class _JSONFormatter(logging.Formatter):
    def format(self, record):
        entry = {"ts": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
        if hasattr(record, "data"):
            entry["data"] = record.data
        return json.dumps(entry, ensure_ascii=False)

_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(_JSONFormatter())
log = logging.getLogger("check_access")
if not log.handlers:
    log.addHandler(_h)
log.setLevel(logging.INFO)


MODELS = [
    {"id": "pucpr/biobertpt-clin", "name": "BioBERTpt-clin", "license": "Apache 2.0", "notes": None},
    {"id": "google/medgemma-4b-it", "name": "MedGemma 4B", "license": "HAI-DEF",
     "notes": "Verificar se a licença HAI-DEF permite uso em pesquisa publicada"},
    {"id": "google/gemma-4-e4b-it", "name": "Gemma 4 E4B", "license": "Gemma License", "notes": None},
    {"id": "Qwen/Qwen3.5-4B", "name": "Qwen3.5-4B", "license": "Apache 2.0", "notes": None},
]


def check_access(model_id: str, token: str | None = None) -> tuple[str, str | None]:
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=model_id, filename="config.json", token=token)
        return "ok", None
    except Exception as e:
        error = str(e)
        if any(x in error.lower() for x in ["401", "403", "gated", "access", "restricted"]):
            return "license_pending", f"https://huggingface.co/{model_id}"
        return "error", error[:200]


def main():
    parser = argparse.ArgumentParser(description="IANA — Verifica acesso aos modelos")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    print(f"\n{'='*66}")
    print(f"  VERIFICAÇÃO DE ACESSO AOS MODELOS DO BENCHMARK IANA")
    print(f"{'='*66}")
    print(f"  {'Modelo':<25} {'Licença':<20} {'Status':<10}")
    print(f"  {'-'*25} {'-'*20} {'-'*10}")

    results = []
    ok_count = 0
    pending_count = 0

    for model in MODELS:
        status, detail = check_access(model["id"], token=args.token)
        if status == "ok":
            marker = "OK"
            ok_count += 1
        elif status == "license_pending":
            marker = f"Aceitar em: {detail}"
            pending_count += 1
        else:
            marker = f"Erro: {detail}"
        print(f"  {model['name']:<25} {model['license']:<20} {marker}")
        if model.get("notes") and status != "ok":
            print(f"  {'':>25}  {model['notes']}")
        results.append({"model_id": model["id"], "name": model["name"],
                        "license": model["license"], "status": status, "detail": detail})

    error_count = len(MODELS) - ok_count - pending_count
    print(f"  {'-'*25} {'-'*20} {'-'*10}")
    print(f"  {ok_count}/{len(MODELS)} acessíveis | {pending_count} pendentes | {error_count} erros")
    print(f"{'='*66}\n")

    log.info("model_access_check", extra={"data": {"ok": ok_count, "pending": pending_count,
                                                     "errors": error_count, "results": results}})

    logs_dir = _TRAINING_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    with open(logs_dir / "model_access_check.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    sys.exit(0 if ok_count == len(MODELS) else 1)


if __name__ == "__main__":
    main()
