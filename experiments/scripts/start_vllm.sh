#!/bin/bash
# ============================================================
# Projeto IANA - Inicializacao do servidor vLLM na DGX
#
# Uso:
#   chmod +x scripts/start_vllm.sh
#   ./scripts/start_vllm.sh
#
# Requisitos:
#   pip install vllm>=0.17.1
#   2x NVIDIA H200 (ou GPUs com VRAM total >= 140GB)
# ============================================================

set -euo pipefail

MODEL="Qwen/Qwen3.5-122B-A10B"
TP=2                    # Tensor Parallelism: 2 GPUs
QUANT="fp8"             # FP8: metade da VRAM do BF16, qualidade proxima
MAX_MODEL_LEN=65536     # 64K tokens de contexto (prontuarios cabem com folga)
PORT=8000
API_KEY="iana-local-key"

echo "============================================================"
echo "  IANA - Servidor vLLM"
echo "============================================================"
echo "  Modelo:     ${MODEL}"
echo "  GPUs:       ${TP}x (tensor parallel)"
echo "  Quantizacao: ${QUANT}"
echo "  Contexto:   ${MAX_MODEL_LEN} tokens"
echo "  Porta:      ${PORT}"
echo "============================================================"
echo ""
echo "Iniciando servidor... (primeira execucao pode baixar o modelo)"
echo ""

vllm serve "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --quantization "${QUANT}" \
    --dtype auto \
    --max-model-len "${MAX_MODEL_LEN}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --trust-remote-code \
    2>&1 | tee vllm_server.log
