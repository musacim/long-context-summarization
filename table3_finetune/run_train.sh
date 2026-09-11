#!/bin/bash
set -e
DATA=${1:?sft jsonl}; OUT=${2:?output dir}; EP=${3:-2}
NGPU=${NGPU:-4}
export MAX_LEN=${MAX_LEN:-38000}
torchrun --standalone --nproc_per_node=$NGPU train_lora.py \
    --model "${MODEL:?set MODEL to an HF id or local path}" \
    --data "$DATA" --out "$OUT" --epochs "$EP"
