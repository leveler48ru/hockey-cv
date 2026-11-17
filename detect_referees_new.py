#!/usr/bin/env python3
"""Automatic referee detection on NEW wide shots (wide_new: v_k2, v_sb1, v_award).
Same HSV method as detect_referees.py, but without the good-frame qc filter and without
the scoreboard panel mask (panel coordinates are not defined for the new broadcasts).
Output: referee_masks_new/{base}.txt (YOLO seg format, class 8)."""
import cv2, numpy as np
from pathlib import Path

BASE = Path(".")
WIDE = BASE / "wide_new"
OUT = BASE / "referee_masks_new"
OUT.mkdir(exist_ok=True)

MIN_AREA, MAX_AREA = 300, 50000
MIN_HEIGHT, MAX_HEIGHT = 25, 350
ASPECT_MIN, ASPECT_MAX = 0.15, 1.2
REF_CLASS = 8


def detect_referees(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    stripe_mask = cv2.bitwise_and(edges, black_mask)
    combined = cv2.bitwise_or(black_mask, stripe_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if bh < MIN_HEIGHT or bh > MAX_HEIGHT:
            continue
        if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
            continue
        cx = (x + bw / 2) / w; cy = (y + bh / 2) / h
        nw = bw / w; nh = bh / h
        x1, y1 = cx - nw / 2, cy - nh / 2
        x2, y2 = cx + nw / 2, cy - nh / 2
        x3, y3 = cx + nw / 2, cy + nh / 2
        x4, y4 = cx - nw / 2, cy + nh / 2
        lines.append(f"{REF_CLASS} {x1:.5f} {y1:.5f} {x2:.5f} {y2:.5f} {x3:.5f} {y3:.5f} {x4:.5f} {y4:.5f}")
    return lines


done = total = skipped = 0
for fpath in sorted(WIDE.glob("*.jpg")):
    base = fpath.stem
    of = OUT / f"{base}.txt"
    if of.exists():
        skipped += 1
        continue
    frame = cv2.imread(str(fpath))
    if frame is None:
        continue
    refs = detect_referees(frame)
    if refs:
        of.write_text("\n".join(refs) + "\n")
        total += len(refs)
        done += 1

print(f"frames with candidates: {done}, objects: {total}, skipped (already present): {skipped}")
