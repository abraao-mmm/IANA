#!/usr/bin/env python3
"""
Projeto IANA — Treino do MedGemma 4B com LoRA para NER + SOAP.

v2: completion-only masking via DataCollatorForCompletionOnlyLM (TRL).
    Antes: labels=input_ids.copy() treinava LM na sequência inteira → outputs
    degenerados. Agora só os tokens após "<start_of_turn>model\\n" contribuem
    para a loss. Collator wrapper injeta token_type_ids=0 (exigido pelo Gemma 3).

Uso:
    python train_medgemma.py --config ../config/medgemma.yaml --gpu 0
    python train_medgemma.py --dry-run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent.parent


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="IANA — Treino MedGemma LoRA")
    parser.add_argument("--config", default=str(_TRAINING_DIR / "config" / "medgemma.yaml"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = _load_yaml(args.config)
    model_id = cfg["model_id"]
    seed = cfg.get("seed", 42)
    lora_cfg = cfg.get("lora", {})

    print(json.dumps({"event": "config_loaded", "model": model_id, "gpu": args.gpu}))

    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    import torch
    from shared.completion_collator import CompletionOnlyCollator

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    )

    from shared.lora_config import create_lora_config
    lora = create_lora_config(lora_cfg)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    if args.dry_run:
        dummy = tokenizer("Teste", return_tensors="pt")
        with torch.no_grad():
            out = model(**dummy)
        print(json.dumps({"event": "dry_run_ok", "logits_shape": list(out.logits.shape)}))
        return

    _dd = Path(cfg.get("data_dir", "data/gemma_format"))
    data_dir = _dd if _dd.is_absolute() else _TRAINING_DIR / _dd
    train_ds = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=str(data_dir / "val.jsonl"), split="train")

    max_len = cfg.get("max_seq_length", 4096)

    def tokenize(example):
        enc = tokenizer(example["text"], truncation=True, max_length=max_len)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize, remove_columns=val_ds.column_names)

    # Completion-only masking: só tokens após "<start_of_turn>model\n" contam.
    base_collator = CompletionOnlyCollator(
        tokenizer=tokenizer,
        response_template="<start_of_turn>model\n",
    )

    # Wrapper: Gemma 3 exige token_type_ids no forward. Injeta zeros após o
    # collate do base (que só produz input_ids/attention_mask/labels).
    def gemma_collator(features):
        batch = base_collator(features)
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch

    _od = Path(cfg.get("output_dir", "checkpoints/medgemma"))
    output_dir = _od if _od.is_absolute() else _TRAINING_DIR / _od
    _ld = Path(cfg.get("log_dir", "logs"))
    log_dir = _ld if _ld.is_absolute() else _TRAINING_DIR / _ld

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 1e-4),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        optim=cfg.get("optim", "adamw_torch"),
        max_length=max_len,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        prediction_loss_only=True,
        seed=seed,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        load_best_model_at_end=True,
        logging_dir=str(log_dir),
        report_to="none",
    )

    from shared.callbacks import JSONLoggingCallback
    json_cb = JSONLoggingCallback(
        log_path=log_dir / f"medgemma_{int(time.time())}.jsonl",
        model_name="medgemma",
    )

    trainer = SFTTrainer(
        model=model, args=sft_config,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer, callbacks=[json_cb],
        data_collator=gemma_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(output_dir / "best"))
    print(json.dumps({"event": "training_complete", "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
