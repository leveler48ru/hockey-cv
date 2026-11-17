#!/bin/bash
# v6: all-in-one — all classes + arena show-negatives 1 + 10 frames from the second rink.
# Start from v5 (referees), recipe v4 proven: freeze=10, patience=0, 30 epochs.
set -e
docker rm -f train_all_v6 >/dev/null 2>&1 || true
docker run -d --gpus all --shm-size=16g \
  -v /root/dataset_real_v6:/data -v /root/runs:/runs \
  --name train_all_v6 ultralytics/ultralytics:latest \
  yolo segment train \
    model=/runs/rink_referees_v5/weights/best.pt \
    data=/data/data_vm.yaml \
    epochs=30 imgsz=1088 batch=8 \
    project=/runs name=rink_all_v6 exist_ok=False \
    patience=0 workers=4 device=0 amp=True \
    freeze=10 \
    seed=42 deterministic=True
echo STARTED
