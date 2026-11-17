#!/usr/bin/env python3
"""Automatic referee markup on wide shots of hockey games.
Method: HSV filtering on black/striped + morphology.
Referees = dark/striped blobs on the light ice, vertical, size > threshold.
Output: referee_masks/{base}.txt (YOLO seg format: class x1 y1 x2 y2 x3 y3 x4 y4)
"""
import cv2, numpy as np, glob, os, json
from pathlib import Path

BASE = Path(".")
WIDE = BASE / "wide"
OUT = BASE / "referee_masks"
OUT.mkdir(exist_ok=True)

# Load the list of good frames
qc = json.load(open(BASE / "qc" / "propagate_all.json"))
good_frames = {r['frame'] for r in qc if r.get('inliers', 0) >= 60}

# Scoreboard panel (masked so it isn't picked up as a referee)
SB = {'match': (40, 25, 650, 160), 'm1': (55, 30, 385, 115), 'm2': (50, 25, 720, 175)}

# Referee detection parameters
MIN_AREA = 300       # minimum contour area (px^2)
MAX_AREA = 50000     # maximum
MIN_HEIGHT = 25      # minimum bbox height
MAX_HEIGHT = 350     # maximum
ASPECT_MIN = 0.15    # minimum w/h ratio (referees are vertical)
ASPECT_MAX = 1.2     # maximum

REF_CLASS = 8  # referee class in data.yaml

def detect_referees(frame, vid):
    """Referee detection via HSV (black + striped)."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Black mask (low brightness)
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
    
    # Striped mask (alternation of black and white/yellow)
    # Simplified: take the black regions and check for light pixels nearby
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    stripe_mask = cv2.bitwise_and(edges, black_mask)
    
    # Combine the masks
    combined = cv2.bitwise_or(black_mask, stripe_mask)
    
    # Morphology: remove noise, merge nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    # Mask out the scoreboard panel
    if vid in SB:
        x, y, bw, bh = SB[vid]
        combined[y:y+bh, x:x+bw] = 0
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
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
        
        # Normalize the coordinates
        cx = (x + bw/2) / w
        cy = (y + bh/2) / h
        nw = bw / w
        nh = bh / h
        
        # Convert to segment format (rectangle)
        x1 = cx - nw/2
        y1 = cy - nh/2
        x2 = cx + nw/2
        y2 = cy - nh/2
        x3 = cx + nw/2
        y3 = cy + nh/2
        x4 = cx - nw/2
        y4 = cy + nh/2
        
        bboxes.append(f"{REF_CLASS} {x1:.5f} {y1:.5f} {x2:.5f} {y2:.5f} {x3:.5f} {y3:.5f} {x4:.5f} {y4:.5f}")
    
    return bboxes

# Process all good frames
processed = 0
total_refs = 0

for vid in ['match', 'm1', 'm2']:
    frames = sorted(glob.glob(str(WIDE / f"{vid}_*.jpg")))
    for fpath in frames:
        base = Path(fpath).stem
        if base not in good_frames:
            continue
        
        frame = cv2.imread(fpath)
        if frame is None:
            continue
        
        refs = detect_referees(frame, vid)
        if refs:
            out_file = OUT / f"{base}.txt"
            with open(out_file, 'w') as fp:
                fp.write('\n'.join(refs) + '\n')
            total_refs += len(refs)
            processed += 1

print(f"Processed: {processed} frames with referees")
print(f"Total referee detections: {total_refs}")
print(f"Output: {OUT}")
