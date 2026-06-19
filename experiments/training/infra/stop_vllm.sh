#!/bin/bash
# Projeto IANA — Para o vLLM e libera as GPUs 0 e 1
set -e
echo "Parando vLLM..."
pkill -f "vllm serve" 2>/dev/null || echo "vLLM não estava rodando"
sleep 5
echo "Aguardando liberação de memória..."
sleep 10
echo "Estado das GPUs:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo "GPUs liberadas."
