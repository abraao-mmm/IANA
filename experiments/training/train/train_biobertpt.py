#!/usr/bin/env python3
"""
Projeto IANA — Treino do BioBERTpt-clin para NER clínico.

Usa HuggingFace Trainer com AutoModelForTokenClassification.

Uso:
    python train_biobertpt.py --config ../config/biobertpt.yaml --gpu 0
    python train_biobertpt.py --config ../config/biobertpt.yaml --dry-run
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
    parser = argparse.ArgumentParser(description="IANA — Treino BioBERTpt")
    parser.add_argument("--config", default=str(_TRAINING_DIR / "config" / "biobertpt.yaml"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = _load_yaml(args.config)
    model_id = cfg["model_id"]
    seed = cfg.get("seed", 42)

    print(json.dumps({"event": "config_loaded", "model": model_id, "gpu": args.gpu, "dry_run": args.dry_run}))

    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        TrainingArguments, Trainer, set_seed,
        DataCollatorForTokenClassification,
    )
    from datasets import load_dataset
    import evaluate

    set_seed(seed)

    # Label mapping
    label_list = ["O", "B-DISEASE", "I-DISEASE", "B-SYMPTOM", "I-SYMPTOM",
                  "B-MEDICATION", "I-MEDICATION", "B-LAB", "I-LAB",
                  "B-PROCEDURE", "I-PROCEDURE", "B-ORGANISM", "I-ORGANISM"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Model
    model = AutoModelForTokenClassification.from_pretrained(
        model_id, num_labels=len(label_list),
        label2id=label2id, id2label=id2label,
    )

    if args.dry_run:
        import torch
        dummy = tokenizer("Paciente com HIV e tuberculose", return_tensors="pt",
                          padding="max_length", max_length=32, truncation=True)
        dummy["labels"] = torch.zeros(1, 32, dtype=torch.long)
        with torch.no_grad():
            output = model(**dummy)
        print(json.dumps({"event": "dry_run_ok", "loss": output.loss.item(),
                          "logits_shape": list(output.logits.shape)}))
        return

    # Load data (resolve relativos a _TRAINING_DIR)
    _dd = Path(cfg.get("data_dir", "data/bio_tagging"))
    data_dir = _dd if _dd.is_absolute() else _TRAINING_DIR / _dd
    train_ds = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=str(data_dir / "val.jsonl"), split="train")

    # Converte tokens (strings) para input_ids e labels (strings) para IDs numéricos
    def prepare_features(examples):
        input_ids = [tokenizer.convert_tokens_to_ids(toks) for toks in examples["tokens"]]
        labels = [[label2id.get(l, 0) for l in seq] for seq in examples["labels"]]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    cols_to_remove = [c for c in train_ds.column_names if c not in ("input_ids", "labels")]
    train_ds = train_ds.map(prepare_features, batched=True, remove_columns=cols_to_remove)
    val_ds = val_ds.map(prepare_features, batched=True, remove_columns=cols_to_remove)

    # DataCollator faz padding dinâmico dos batches
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    # Training args (resolve relativos a _TRAINING_DIR)
    _od = Path(cfg.get("output_dir", "checkpoints/biobertpt"))
    output_dir = _od if _od.is_absolute() else _TRAINING_DIR / _od
    _ld = Path(cfg.get("log_dir", "logs"))
    log_dir = _ld if _ld.is_absolute() else _TRAINING_DIR / _ld

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 16),
        learning_rate=cfg.get("learning_rate", 2e-5),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        weight_decay=cfg.get("weight_decay", 0.01),
        eval_strategy=cfg.get("eval_strategy", cfg.get("evaluation_strategy", "steps")),
        eval_steps=cfg.get("eval_steps", 100),
        save_steps=cfg.get("save_steps", 100),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "f1"),
        seed=seed,
        logging_dir=str(log_dir),
        report_to="none",
    )

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        import numpy as np
        predictions = np.argmax(predictions, axis=2)
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
        true_preds = [[id2label[p] for p, l in zip(pred, label) if l != -100]
                      for pred, label in zip(predictions, labels)]
        results = seqeval.compute(predictions=true_preds, references=true_labels)
        return {"precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"]}

    from shared.callbacks import JSONLoggingCallback
    json_cb = JSONLoggingCallback(
        log_path=log_dir / f"biobertpt_{int(time.time())}.jsonl",
        model_name="biobertpt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[json_cb],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(output_dir / "best"))
    print(json.dumps({"event": "training_complete", "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
