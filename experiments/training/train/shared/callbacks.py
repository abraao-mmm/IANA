"""
Projeto IANA — Callbacks de logging e checkpoint para treino.
"""

import json
import logging
import time
from pathlib import Path
from transformers import TrainerCallback


class JSONLoggingCallback(TrainerCallback):
    """Loga métricas de treino em formato JSON estruturado."""

    def __init__(self, log_path: Path, model_name: str):
        self.log_path = log_path
        self.model_name = model_name
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8")
        self._start_time = time.time()

    def _log(self, event: str, data: dict):
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model_name,
            "event": event,
            "elapsed_s": round(time.time() - self._start_time, 1),
            **data,
        }
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self._log("metrics", {k: v for k, v in logs.items() if isinstance(v, (int, float))})

    def on_save(self, args, state, control, **kwargs):
        self._log("checkpoint", {"step": state.global_step, "epoch": state.epoch})

    def on_train_end(self, args, state, control, **kwargs):
        self._log("train_end", {
            "total_steps": state.global_step,
            "total_epochs": state.num_train_epochs,
            "best_metric": state.best_metric,
        })
        self._file.close()
