#!/usr/bin/env python3
"""
Projeto IANA — Re-parse de predictions ja geradas.

Aplica o parser de _parse_json_output do run_inference.py em arquivos
predictions/{model}_predictions.json existentes, recuperando JSONs que
caíram em raw_output. Nao re-roda inferencia (cara) — só re-parseia.

Uso:
    python reparse_predictions.py --model gemma4_e4b
    python reparse_predictions.py --model gemma4_e4b --in-place
"""

import argparse
import json
import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent.parent

# Reusa o parser do run_inference
sys.path.insert(0, str(_TRAINING_DIR / "eval"))
from run_inference import _parse_json_output  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="IANA — Re-parse predictions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--predictions", default=None,
                        help="Path do predictions json (default: predictions/{model}_predictions.json)")
    parser.add_argument("--in-place", action="store_true",
                        help="Sobrescreve o arquivo original (default: salva como *_reparsed.json)")
    args = parser.parse_args()

    pred_path = Path(args.predictions or _TRAINING_DIR / "predictions" / f"{args.model}_predictions.json")

    with open(pred_path, encoding="utf-8") as f:
        data = json.load(f)

    before_raw = sum(1 for p in data if isinstance(p.get("predictions"), dict)
                     and "raw_output" in p["predictions"])

    recovered = 0
    for p in data:
        pred = p.get("predictions")
        if isinstance(pred, dict) and "raw_output" in pred:
            new_pred = _parse_json_output(pred["raw_output"])
            if "raw_output" not in new_pred:
                p["predictions"] = new_pred
                recovered += 1

    after_raw = sum(1 for p in data if isinstance(p.get("predictions"), dict)
                    and "raw_output" in p["predictions"])

    out_path = pred_path if args.in_place else pred_path.with_name(pred_path.stem + "_reparsed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "event": "reparse_complete",
        "model": args.model,
        "total": len(data),
        "raw_output_before": before_raw,
        "raw_output_after": after_raw,
        "recovered": recovered,
        "output": str(out_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
