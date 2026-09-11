#!/bin/bash
# Polls localhost:8000 until the vLLM OpenAI-compatible server responds, or times out.
for i in $(seq 1 120); do
    if curl -s -m 2 http://localhost:8000/v1/models > /tmp/models_check.json 2>/dev/null; then
        echo "READY after ${i} tries"
        cat /tmp/models_check.json
        exit 0
    fi
    sleep 5
done
echo "NOT READY after timeout"
exit 1
