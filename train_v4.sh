#!/bin/bash
# v4: fine-tuning players from the v2 weights.
# Difference from v3: patience=0 (don't give up too early), lr0=0.005 and freeze=10
# (v3's best result came at epoch 1 and afterwards it only spoiled the weights — a sign
#  of too high a learning rate for fine-tuning).
set -e
docker rm -f train_players_v4 >/dev/null 2>&1 || true
docker run -d --gpus all --shm-size=16g \
  -v /root/dataset_real_all:/data -v /root/runs:/runs \
  --name train_players_v4 ultralytics/ultralytics:latest \
  yolo segment train \
    model=/runs/rink_players_v2/weights/best.pt \
    data=/data/data_vm.yaml \
    epochs=30 imgsz=1088 batch=8 \
    project=/runs name=rink_players_v4 exist_ok=False \
    patience=0 workers=4 device=0 amp=True \
    lr0=0.005 freeze=10 \
    seed=42 deterministic=True
echo STARTED
