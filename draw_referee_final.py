#!/usr/bin/env python3
"""Draws referee boxes on frames for final inspection."""
import cv2, numpy as np
from pathlib import Path

REF_FRAMES = Path("referee_frames")
TRAIN_LABELS = Path("dataset_real_all/labels/train")
OUT_DIR = Path("referee_check_final")
OUT_DIR.mkdir(exist_ok=True)

W, H = 1920, 1080
REFEREE_CLASS = 8

# Take 6 frames for inspection
check_frames = ["match_01200", "m1_01200", "m2_00900", "match_02700", "m1_03000", "m2_01800"]

for base in check_frames:
    img_path = REF_FRAMES / f"{base}.jpg"
    label_path = TRAIN_LABELS / f"{base}.txt"
    if not img_path.exists() or not label_path.exists():
        continue
    img = cv2.imread(str(img_path))
    for line in label_path.read_text().strip().split("\n"):
        parts = line.strip().split()
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        if cls == REFEREE_CLASS and len(coords) >= 8:
            # Polygon: x1 y1 x2 y2 x3 y3 x4 y4
            pts = [(int(coords[i]*W), int(coords[i+1]*H)) for i in range(0, len(coords), 2)]
            cv2.polylines(img, [np.array(pts)], True, (0, 255, 0), 3)
            cv2.putText(img, "referee", pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif cls == 7 and len(coords) >= 8:
            pts = [(int(coords[i]*W), int(coords[i+1]*H)) for i in range(0, len(coords), 2)]
            cv2.polylines(img, [np.array(pts)], True, (255, 0, 0), 2)
    out_path = OUT_DIR / f"{base}_check.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"Saved {out_path}")

print("Done")
