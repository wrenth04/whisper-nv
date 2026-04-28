#!/bin/sh
set -eu

PORT="${PORT:-8000}"
RIVA_MAX_MESSAGE_LENGTH="${RIVA_MAX_MESSAGE_LENGTH:-16777216}"
FUNCTION_ID="${FUNCTION_ID:-b702f636-f60c-4a3d-a6f4-f3568c13bd7d}"
RIVA_SERVER="${RIVA_SERVER:-grpc.nvcf.nvidia.com:443}"
RIVA_USE_SSL="${RIVA_USE_SSL:-true}"

if [ -n "${NVIDIA_API_KEY:-}" ] && [ -z "${RIVA_METADATA:-}" ]; then
  export RIVA_METADATA="function-id:${FUNCTION_ID},authorization:Bearer ${NVIDIA_API_KEY}"
fi

export RIVA_SERVER
export RIVA_USE_SSL
export RIVA_MAX_MESSAGE_LENGTH

exec uvicorn openai_server:app --host 0.0.0.0 --port "${PORT}"
