#!/bin/bash
# Lightweight Gemma 4 E4B (4B effective params) on port 8081 — for opencode.
# Run alongside serve.sh (12B on 8080) only if you have headroom (>20GB unified
# recommended). On 16GB Macs, run this OR serve.sh, not both.

mlx_vlm.server \
    --model mlx-community/gemma-4-e4b-it-OptiQ-4bit \
    --host 127.0.0.1 \
    --port 8081 \
    --prefill-step-size 4096 \
    --max-kv-size 65536
