#!/usr/bin/env python3
"""Visualization of automatic referee markup on a few frames."""
import cv2, numpy as np, glob, random
from pathlib import Path

BASE = Path(".")
WIDE = BASE / "wide"
REF_MASKS = BASE / "referee_masks"
OUT = BASE / "referee_check.jpg"

# Take 6 random frames with markup
mask_files = sorted(glob.glob(str(REF_MASKS / "*.txt")))
random.seed(42)
samples = random.sample(mask_files, min(6, len(mask_files)))

frames = []
for mf in samples:
    base = Path(mf).stem
    img_path = WIDE / f"{base}.jpg"
    if not img_path.exists():
        continue
    
    frame = cv2.imread(str(img_path))
    h, w = frame.shape[:2]
    
    # Draw referee boxes
    with open(mf) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:9]))
            # polygon -> bbox for drawing
            xs = [coords[i] * w for i in range(0, 8, 2)]
            ys = [coords[i] * h for i in range(1, 8, 2)]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ref", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    frames.append(frame)

if frames:
    # 2x3 contact sheet
    rows = []
    for i in range(0, len(frames), 3):
        row = frames[i:i+3]
        while len(row) < 3:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    contact = np.vstack(rows)
    cv2.imwrite(str(OUT), contact)
    print(f"Saved {OUT} ({contact.shape[1]}x{contact.shape[0]})")
else:
    print("No frames to visualize")
