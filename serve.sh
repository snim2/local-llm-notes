#!/bin/bash

mlx_vlm.server \
    --model mlx-community/gemma-4-12B-it-OptiQ-4bit \
    --host 127.0.0.1 \
    --port 8080 \
    --prefill-step-size 4096 \
    --max-kv-size 65536
#    --kv-bits 8 --kv-quant-scheme uniform
#    --log-level DEBUG
