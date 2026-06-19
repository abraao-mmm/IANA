"""
Projeto IANA — Collator de completion-only masking sem depender do TRL.

Motivação:
  trl.DataCollatorForCompletionOnlyLM foi renomeado/removido do top-level em
  versões recentes. Esse módulo implementa a mesma lógica (mascarar com -100
  tudo antes do response_template no campo labels) com dependência apenas
  do torch + tokenizer HF. Funciona em qualquer versão do TRL.

Uso:
    from shared.completion_collator import CompletionOnlyCollator

    collator = CompletionOnlyCollator(
        tokenizer=tokenizer,
        response_template="<start_of_turn>model\\n",
    )
"""

from typing import Any

import torch


class CompletionOnlyCollator:
    """Pad a batch e mascara com -100 tudo antes do response_template nos labels.

    - features: lista de dicts com 'input_ids' e 'attention_mask'.
    - Se o template não for achado numa sequência, essa sequência inteira fica
      mascarada (labels=-100 em tudo) e não contribui para a loss — melhor do
      que treinar no prompt todo.
    """

    def __init__(self, tokenizer, response_template: str):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        # Tokeniza o template sem special tokens para buscar a sequência de
        # ids dentro do input_ids tokenizado.
        self.response_token_ids = tokenizer(
            response_template, add_special_tokens=False
        )["input_ids"]
        if not self.response_token_ids:
            raise ValueError(
                f"response_template tokenizou para vazio: {response_template!r}"
            )

    def _find_response_start(self, input_ids: list[int]) -> int:
        """Retorna o índice do primeiro token APÓS o response_template. -1 se não achar."""
        tpl = self.response_token_ids
        n = len(tpl)
        if n == 0 or len(input_ids) < n:
            return -1
        for i in range(len(input_ids) - n + 1):
            if input_ids[i : i + n] == tpl:
                return i + n
        return -1

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_attn = []
        batch_labels = []
        for f in features:
            input_ids = list(f["input_ids"])
            attn = list(f.get("attention_mask", [1] * len(input_ids)))
            labels = list(input_ids)
            resp_start = self._find_response_start(input_ids)
            if resp_start == -1:
                # template não achado (resposta foi truncada) — descarta toda a seq
                labels = [-100] * len(input_ids)
            else:
                for j in range(resp_start):
                    labels[j] = -100
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.pad_token_id] * pad_len
                attn = attn + [0] * pad_len
                labels = labels + [-100] * pad_len
            batch_input_ids.append(input_ids)
            batch_attn.append(attn)
            batch_labels.append(labels)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
