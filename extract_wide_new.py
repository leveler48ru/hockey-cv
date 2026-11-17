#!/usr/bin/env python3
"""Extract wide-shot frames from the new videos (opposite side of the rink)."""
import cv2, os
from pathlib import Path

OUT = Path("wide_new")
OUT.mkdir(exist_ok=True)

videos = {
    "v_k2": "./media/Vegas_-_Khrystal_12092026_-_2nd_period.mp4",
    "v_award": "./media/Vegas_-_Khrystal_12092026_-_Award_ceremony.mp4",
    "v_sb1": "./media/Vegas_-_Sneghnye_Barsy_23052026_-_1st_period.mp4",
}

INTERVAL_SEC = 5  # one frame every 5 seconds

for prefix, vpath in videos.items():
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"FAIL: {prefix}")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * INTERVAL_SEC)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval_frames == 0:
            t_sec = frame_idx / fps
            out_path = OUT / f"{prefix}_{t_sec:05.0f}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1
    
    cap.release()
    print(f"{prefix}: {saved} frames extracted ({frame_idx} total, {fc} expected)")

print(f"\nTotal files: {len(list(OUT.glob('*.jpg')))}")
