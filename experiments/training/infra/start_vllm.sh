#!/bin/bash
# Projeto IANA — Reinicia o vLLM nas configurações originais
set -e
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
nohup vllm serve Qwen/Qwen3.5-122B-A10B \
    --tensor-parallel-size 2 \
    --quantization fp8 \
    --max-model-len 65536 \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key iana-local-key \
    --gdn-prefill-backend triton \
    >> ~/vllm.log 2>&1 &
echo "vLLM iniciado, PID $!"
echo "Acompanhe com: tail -f ~/vllm.log"
