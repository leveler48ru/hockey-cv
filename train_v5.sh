#!/bin/bash
# v5: referees. Start from the v4 weights, dataset dataset_real_v5 (class8: train 95 / val 38).
# Recipe v4 proven: freeze=10, patience=0, all 30 epochs.
set -e
docker rm -f train_referees_v5 >/dev/null 2>&1 || true
docker run -d --gpus all --shm-size=16g \
  -v /root/dataset_real_v5:/data -v /root/runs:/runs \
  --name train_referees_v5 ultralytics/ultralytics:latest \
  yolo segment train \
    model=/runs/rink_players_v4/weights/best.pt \
    data=/data/data_vm.yaml \
    epochs=30 imgsz=1088 batch=8 \
    project=/runs name=rink_referees_v5 exist_ok=False \
    patience=0 workers=4 device=0 amp=True \
    freeze=10 \
    seed=42 deterministic=True
echo STARTED
