#!/bin/bash
# v8: referees from manual labeling (9 classes, class8: train 141 / val 77).
# Start from the v6 weights (same class scheme). Recipe proven by v4..v7: freeze=10, patience=0.
set -e
docker rm -f train_ref_v8 >/dev/null 2>&1 || true
docker run -d --gpus all --shm-size=16g \
  -v /root/dataset_ref_v8:/data -v /root/runs:/runs \
  --name train_ref_v8 ultralytics/ultralytics:latest \
  yolo segment train \
    model=/runs/rink_all_v6/weights/best.pt \
    data=/data/data_vm.yaml \
    epochs=30 imgsz=1088 batch=8 \
    project=/runs name=rink_ref_v8 exist_ok=False \
    patience=0 workers=4 device=0 amp=True \
    freeze=10 \
    seed=42 deterministic=True
echo STARTED
