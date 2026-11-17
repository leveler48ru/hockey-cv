#!/bin/bash
# Upload a training script to the GPU server and launch it inside docker.
# Environment variables:
#   GPU_SSH_KEY  - path to the SSH key for the GPU server
#   GPU_HOST     - GPU server address (user@host)
set -e
HK="${GPU_SSH_KEY:?set GPU_SSH_KEY to your SSH key path}"
HOST="${GPU_HOST:?set GPU_HOST to your GPU server, e.g. root@gpu.example.com}"
scp -o StrictHostKeyChecking=no -i "$HK" train_v4.sh "$HOST:/root/train_v4.sh"
ssh -o StrictHostKeyChecking=no -i "$HK" "$HOST" "chmod +x /root/train_v4.sh && bash /root/train_v4.sh && sleep 30 && docker ps --format \"{{.Names}} {{.Status}}\" | grep v4; nvidia-smi --query-gpu=memory.used --format=csv,noheader"
