#!/bin/sh
set -e
ollama serve &
SERVE_PID=$!
until ollama list >/dev/null 2>&1; do
  echo "waiting for ollama server..."
  sleep 1
done
echo "pulling ${OLLAMA_MODEL:-hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:UD-Q4_K_XL} ..."
ollama pull "${OLLAMA_MODEL:-hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:UD-Q4_K_XL}"
echo "model ready."
wait "$SERVE_PID"
