#!/bin/bash
# v7: team classes. Start from v6 (9 classes), dataset dataset_teams (10 classes).
set -e
docker rm -f train_teams_v7 >/dev/null 2>&1 || true
docker run -d --gpus all --shm-size=16g \
  -v /root/dataset_teams:/data -v /root/runs:/runs \
  --name train_teams_v7 ultralytics/ultralytics:latest \
  yolo segment train \
    model=/runs/rink_all_v6/weights/best.pt \
    data=/data/data.yaml \
    epochs=30 imgsz=1088 batch=8 \
    project=/runs name=rink_teams_v7 exist_ok=False \
    patience=0 workers=4 device=0 amp=True \
    freeze=10 \
    seed=42 deterministic=True
echo STARTED
