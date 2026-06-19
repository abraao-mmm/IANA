#!/usr/bin/env python3
"""
Projeto IANA — Inferência de modelos treinados nas 30 notas gold.

Uso:
    python run_inference.py --model biobertpt --checkpoint ../checkpoints/biobertpt/best
    python run_inference.py --model qwen35_4b --checkpoint ../checkpoints/qwen35_4b/best
    python run_inference.py --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _TRAINING_DIR.parent


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_texts(parquet_path: Path, pids: set[str]) -> dict[str, str]:
    import polars as pl
    df = pl.read_parquet(str(parquet_path))
    df_f = df.filter(pl.col("hadm_id").cast(pl.Utf8).is_in(pids))
    return {str(row["hadm_id"]): row.get("text", "") for row in df_f.to_dicts()}


def main():
    parser = argparse.ArgumentParser(description="IANA — Inferência no gold test set")
    parser.add_argument("--model", required=True,
                        choices=["biobertpt", "medgemma", "gemma4_e4b", "qwen35_4b"])
    parser.add_argument("--checkpoint", required=True, help="Path do checkpoint treinado")
    parser.add_argument("--gold", default=str(_EXPERIMENTS_DIR / "resultados" / "gold_test_set_30.json"))
    parser.add_argument("--parquet", default=str(_EXPERIMENTS_DIR / "dados" / "mimic_filtrado_tb_hiv_sifilis.parquet"))
    parser.add_argument("--output-dir", default=str(_TRAINING_DIR / "predictions"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load gold notes
    with open(args.gold, encoding="utf-8") as f:
        gold_data = json.load(f)
    notes = gold_data.get("notes", gold_data) if isinstance(gold_data, dict) else gold_data

    if args.dry_run:
        print(json.dumps({"event": "dry_run", "model": args.model,
                          "gold_notes": len(notes), "checkpoint": args.checkpoint}))
        return

    pids = {n["paciente_id"] for n in notes}
    texts = _get_texts(Path(args.parquet), pids)

    predictions = []

    # Model-specific inference logic
    if args.model == "biobertpt":
        predictions = _infer_biobertpt(args.checkpoint, notes, texts)
    elif args.model in ("medgemma", "gemma4_e4b", "qwen35_4b"):
        predictions = _infer_decoder(args.checkpoint, args.model, notes, texts)

    # Save (converte numpy/torch scalars para Python nativo antes)
    def _to_json_safe(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: _to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.model}_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(predictions), f, indent=2, ensure_ascii=False)

    print(json.dumps({"event": "inference_complete", "model": args.model,
                      "predictions": len(predictions), "output": str(out_path)}))


def _infer_biobertpt(checkpoint: str, notes: list, texts: dict) -> list:
    """Inferência com BioBERTpt token classification."""
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    pipe = pipeline("ner", model=checkpoint, tokenizer=checkpoint, aggregation_strategy="simple")
    results = []
    for note in notes:
        text = texts.get(note["paciente_id"], "")
        if text:
            entities = pipe(text[:512])  # Trunca para max_length
            results.append({"paciente_id": note["paciente_id"], "predictions": entities})
    return results


def _parse_json_output(text: str) -> dict:
    """Parse robusto de JSON gerado por LLM.

    Estrategia em cascata:
    1. json.loads direto (caso ideal)
    2. json_repair (lib estado-da-arte para LLM JSON: trata truncacao,
       brackets faltando, quotes invalidas, comentarios, etc.)
    3. fallback manual: extracao + fechamento heuristico
    4. raw_output (ultima tentativa fracassou)

    json-repair ref: https://github.com/mangiucugna/json_repair
    """
    import re

    # 1. Parse direto
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. json_repair (preferencial - mainstream para LLM JSON)
    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            return repaired
    except ImportError:
        pass
    except Exception:
        pass

    # 3. Fallback manual: extrai bloco { ... } e fecha estruturas abertas
    brace_pos = text.find("{")
    if brace_pos >= 0:
        candidate = text[brace_pos:]
        # Tenta extrair primeiro bloco bem-formado
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        # Fecha strings, listas e dict raiz
        if candidate.count('"') % 2 == 1:
            candidate += '"'
        open_brackets = candidate.count("[") - candidate.count("]")
        candidate += "]" * max(0, open_brackets)
        open_braces = candidate.count("{") - candidate.count("}")
        candidate += "}" * max(0, open_braces)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {"raw_output": text}


def _infer_decoder(checkpoint: str, model_name: str, notes: list, texts: dict) -> list:
    """Inferência com modelos decoder (Qwen, Gemma, MedGemma)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    # Load base + LoRA
    config_path = _TRAINING_DIR / "config" / f"{model_name}.yaml"
    cfg = _load_yaml(str(config_path))
    base_id = cfg["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                                       trust_remote_code=True, device_map="auto")

    # Gemma 4 E4B: remove towers vision/audio igual no treino (senão PeftModel
    # não encontra as camadas onde o LoRA foi aplicado)
    if model_name == "gemma4_e4b":
        import torch.nn as nn
        _inner = base_model.model if hasattr(base_model, "model") else base_model
        for attr in ("vision_tower", "audio_tower"):
            if hasattr(_inner, attr):
                setattr(_inner, attr, nn.Identity())

    model = PeftModel.from_pretrained(base_model, checkpoint)
    model.eval()

    if str(_EXPERIMENTS_DIR) not in sys.path:
        sys.path.insert(0, str(_EXPERIMENTS_DIR))
    from config.prompts import PROMPT_NER_TRAIN

    # Mesmo orçamento de truncagem usado no data prep (format_gemma/chatml).
    TEXT_BUDGET_NER = 9000

    def _truncate_head_tail(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        half = (max_chars - 20) // 2
        return text[:half] + "\n[...]\n" + text[-half:]

    # Templates de prompt por família de modelo — IDÊNTICOS aos usados no treino.
    is_gemma = model_name in ("medgemma", "gemma4_e4b")

    def format_prompt(text: str) -> str:
        text_trunc = _truncate_head_tail(text, TEXT_BUDGET_NER)
        if is_gemma:
            return (f"<start_of_turn>user\n{PROMPT_NER_TRAIN}\n\n{text_trunc}<end_of_turn>\n"
                    f"<start_of_turn>model\n")
        # Qwen ChatML
        return (f"<|im_start|>system\n{PROMPT_NER_TRAIN}<|im_end|>\n"
                f"<|im_start|>user\n{text_trunc}<|im_end|>\n"
                f"<|im_start|>assistant\n")

    max_len = cfg.get("max_seq_length", 4096)

    results = []
    for note in notes:
        text = texts.get(note["paciente_id"], "")
        if not text:
            continue
        prompt = format_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = _parse_json_output(decoded)
        results.append({"paciente_id": note["paciente_id"], "predictions": parsed})

    return results


if __name__ == "__main__":
    main()
