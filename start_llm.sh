#!/bin/bash

echo "=========================================="
echo "   SFEDUCASTLING - Qwen3-8B Запуск"
echo "=========================================="

# Переходим в папку проекта (на всякий случай)
cd "$(dirname "$0")"

# Активируем виртуальное окружение
source ../llm_env/bin/activate

echo "✅ Окружение активировано"
echo "🚀 Запускаем Qwen3-8B"

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  --max-model-len 16384 \
  --served-model-name qwen3-8b \
  --trust-remote-code

