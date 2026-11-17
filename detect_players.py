#!/usr/bin/env python3
"""Automatic player markup on wide shots of hockey games.
Method: background subtraction (median background from N frames) -> contours -> filtering.
Players = dark/colored blobs on the light ice, size > threshold.
Output: player_masks/{base}.npz (bboxes in normalized YOLO format coordinates).
"""
import cv2, numpy as np, glob, os, json

BASE = '.'
WIDE = f'{BASE}/wide'
OUT = f'{BASE}/player_masks'
os.makedirs(OUT, exist_ok=True)

# Load the list of good frames (only they have accurate geometry)
qc = json.load(open(f'{BASE}/qc/propagate_all.json'))
good_frames = {r['frame'] for r in qc if r.get('inliers', 0) >= 60}

SB = {'match': (40, 25, 650, 160), 'm1': (55, 30, 385, 115), 'm2': (50, 25, 720, 175)}

# Detection parameters
MIN_AREA = 400       # minimum contour area (px^2)
MAX_AREA = 80000     # maximum (cut off artifacts)
MIN_HEIGHT = 20      # minimum bbox height
MAX_HEIGHT = 400     # maximum height
ASPECT_MIN = 0.15    # minimum w/h ratio (players are vertical)
ASPECT_MAX = 1.5     # maximum

def detect_players(frame, vid):
    """Player detection via HSV + morphology."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Ice = light, low saturation
    ice = cv2.inRange(hsv, (0, 0, 140), (180, 60, 255))
    
    # Scoreboard panel mask
    sx1, sy1, sx2, sy2 = SB[vid]
    ice[sy1:sy2, sx1:sx2] = 0
    ice[:int(h*0.18), :] = 0  # ceiling
    
    # Players = NOT ice (dark or colored on the ice)
    non_ice = cv2.bitwise_not(ice)
    
    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    non_ice = cv2.morphologyEx(non_ice, cv2.MORPH_OPEN, kernel)
    non_ice = cv2.morphologyEx(non_ice, cv2.MORPH_CLOSE, kernel)
    
    # Frame edges and the scoreboard are not players
    non_ice[sy1:sy2, sx1:sx2] = 0
    non_ice[:int(h*0.20), :] = 0
    non_ice[int(h*0.95):, :] = 0
    non_ice[:, :20] = 0
    non_ice[:, w-20:] = 0
    
    # Contour analysis
    contours, _ = cv2.findContours(non_ice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh < MIN_HEIGHT or bh > MAX_HEIGHT:
            continue
        aspect = bw / max(bh, 1)
        if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
            continue
        # Normalize to YOLO format: cx cy w h (all in [0,1])
        cx = (x + bw/2) / w
        cy = (y + bh/2) / h
        nw = bw / w
        nh = bh / h
        bboxes.append((cx, cy, nw, nh))
    
    return bboxes

# Process all good frames
files = sorted(glob.glob(f'{WIDE}/*.jpg'))
n_processed = 0
n_with_players = 0
total_players = 0

for f in files:
    base = os.path.splitext(os.path.basename(f))[0]
    if base not in good_frames:
        continue
    
    img = cv2.imread(f)
    if img is None:
        continue
    
    vid = base.split('_')[0]
    bboxes = detect_players(img, vid)
    
    if bboxes:
        with open(f'{OUT}/{base}.txt', 'w') as fp:
            for cx, cy, nw, nh in bboxes:
                fp.write(f'0 {cx:.5f} {cy:.5f} {nw:.5f} {nh:.5f}\n')
        n_with_players += 1
        total_players += len(bboxes)
    
    n_processed += 1
    if n_processed % 200 == 0:
        print(f'  {n_processed}/{len(good_frames)} processed, '
              f'with players: {n_with_players}, total detections: {total_players}', flush=True)

print(f'TOTAL: {n_processed} frames, {n_with_players} with players, {total_players} detections', flush=True)
