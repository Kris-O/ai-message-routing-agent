#!/bin/sh
set -e
ollama serve &
SERVE_PID=$!
until ollama list >/dev/null 2>&1; do
  echo "waiting for ollama server..."
  sleep 1
done
echo "pulling ${OLLAMA_MODEL:-qwen2.5:3b} ..."
ollama pull "${OLLAMA_MODEL:-qwen2.5:3b}"
echo "model ready."
wait "$SERVE_PID"
