#!/bin/bash
# Qwen2.5-Coder 1.5B (base, FIM) on port 8082 — for VSCode autocomplete via Continue.
# Standard text arch, so it runs on mlx_lm.server (NOT mlx_vlm like the Gemma servers).
# Tiny (~1GB at 4-bit) and fast — safe to run alongside serve.sh (12B on 8080).
# Use the BASE model, not -Instruct: fill-in-the-middle completion wants raw FIM tokens.

mlx_lm.server \
    --model mlx-community/Qwen2.5-Coder-1.5B-4bit \
    --host 127.0.0.1 \
    --port 8082
