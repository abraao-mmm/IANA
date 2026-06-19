"""
Projeto IANA — Configuração LoRA reutilizável para modelos decoder.

Cria LoraConfig a partir de dicionário YAML padronizado.
"""

from peft import LoraConfig, TaskType


def create_lora_config(cfg: dict) -> LoraConfig:
    """Cria LoraConfig a partir do bloco 'lora' do YAML.

    Args:
        cfg: Dicionário com chaves r, alpha, dropout, target_modules, bias, task_type

    Returns:
        LoraConfig pronta para uso com get_peft_model()
    """
    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
    }

    return LoraConfig(
        r=cfg.get("r", 16),
        lora_alpha=cfg.get("alpha", 32),
        lora_dropout=cfg.get("dropout", 0.05),
        target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=cfg.get("bias", "none"),
        task_type=task_type_map.get(cfg.get("task_type", "CAUSAL_LM"), TaskType.CAUSAL_LM),
    )
