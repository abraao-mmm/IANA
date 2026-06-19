#!/usr/bin/env python3
"""
Projeto IANA — Split estratificado train/val/gold.

Remove as 30 notas gold do pool, filtra notas com NER/SOAP fallback,
e faz split 90/10 preservando proporção por doença.

Uso:
    python split_train_val.py
    python split_train_val.py --config ../config/splits.yaml
    python split_train_val.py --dry-run
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _TRAINING_DIR.parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))


class _JSONFormatter(logging.Formatter):
    def format(self, record):
        entry = {"ts": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
        if hasattr(record, "data"):
            entry["data"] = record.data
        return json.dumps(entry, ensure_ascii=False)

_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(_JSONFormatter())
log = logging.getLogger("split")
if not log.handlers:
    log.addHandler(_h)
log.setLevel(logging.INFO)


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_disease(raw: str) -> str:
    key = raw.strip().lower()
    aliases = {"hiv": "HIV", "tuberculose": "Tuberculose", "sifilis": "Sifilis",
               "syphilis": "Sifilis", "tuberculosis": "Tuberculose"}
    return aliases.get(key, raw)


def main():
    parser = argparse.ArgumentParser(description="IANA — Split train/val/gold")
    parser.add_argument("--config", default=str(_TRAINING_DIR / "config" / "splits.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    seed = cfg.get("seed", 42)
    val_ratio = cfg.get("val_ratio", 0.10)
    random.seed(seed)

    # Resolve paths
    silver_path = _EXPERIMENTS_DIR / cfg["silver_clean_file"]
    gold_path = _EXPERIMENTS_DIR / cfg["gold_test_file"]
    output_dir = _TRAINING_DIR / cfg.get("output_dir", "data/splits/")

    # Load data
    with open(silver_path, encoding="utf-8") as f:
        silver = json.load(f)
    with open(gold_path, encoding="utf-8") as f:
        gold_data = json.load(f)

    gold_notes = gold_data.get("notes", gold_data) if isinstance(gold_data, dict) else gold_data
    gold_ids = {n["paciente_id"] for n in gold_notes}

    log.info("Dados carregados", extra={"data": {
        "silver_total": len(silver), "gold_total": len(gold_ids)}})

    # Filter: remove gold + failed agents
    pool = []
    excluded = {"gold": 0, "ner_fail": 0, "soap_fail": 0}
    for r in silver:
        pid = r["paciente_id"]
        if pid in gold_ids:
            excluded["gold"] += 1
            continue
        st = r.get("agent_status", {})
        if st.get("ner_status", "ok") != "ok":
            excluded["ner_fail"] += 1
            continue
        if st.get("soap_status", "ok") != "ok":
            excluded["soap_fail"] += 1
            continue
        pool.append(r)

    log.info("Filtros aplicados", extra={"data": {
        "pool_size": len(pool), "excluded": excluded}})

    # Stratified split
    by_disease: dict[str, list[str]] = defaultdict(list)
    for r in pool:
        d = _normalize_disease(r.get("doenca_alvo_identificada", ""))
        by_disease[d].append(r["paciente_id"])

    train_ids: list[str] = []
    val_ids: list[str] = []

    for disease, ids in sorted(by_disease.items()):
        random.shuffle(ids)
        n_val = max(1, int(len(ids) * val_ratio))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    # Stats
    train_diseases = Counter()
    val_diseases = Counter()
    gold_diseases = Counter()
    pid_to_disease = {r["paciente_id"]: _normalize_disease(r.get("doenca_alvo_identificada", "")) for r in silver}

    for pid in train_ids:
        train_diseases[pid_to_disease.get(pid, "?")] += 1
    for pid in val_ids:
        val_diseases[pid_to_disease.get(pid, "?")] += 1
    for pid in gold_ids:
        gold_diseases[pid_to_disease.get(pid, "?")] += 1

    stats = {
        "seed": seed,
        "train": {"total": len(train_ids), "by_disease": dict(train_diseases)},
        "val": {"total": len(val_ids), "by_disease": dict(val_diseases)},
        "gold": {"total": len(gold_ids), "by_disease": dict(gold_diseases)},
        "overlap_check": {
            "train_val": len(set(train_ids) & set(val_ids)),
            "train_gold": len(set(train_ids) & gold_ids),
            "val_gold": len(set(val_ids) & gold_ids),
        },
    }

    log.info("Split concluído", extra={"data": stats})

    if args.dry_run:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_ids.json", "w", encoding="utf-8") as f:
        json.dump(train_ids, f, indent=2)
    with open(output_dir / "val_ids.json", "w", encoding="utf-8") as f:
        json.dump(val_ids, f, indent=2)
    with open(output_dir / "gold_ids.json", "w", encoding="utf-8") as f:
        json.dump(sorted(gold_ids), f, indent=2)

    log.info("Arquivos salvos", extra={"data": {"dir": str(output_dir)}})


if __name__ == "__main__":
    main()
