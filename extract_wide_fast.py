#!/usr/bin/env python3
"""Fast frame extraction: seek to the target frame instead of sequential reading."""
import cv2, sys
from pathlib import Path

OUT = Path("wide_new")
OUT.mkdir(exist_ok=True)

videos = {
    "v_k2": "./media/Vegas_-_Khrystal_12092026_-_2nd_period.mp4",
    "v_award": "./media/Vegas_-_Khrystal_12092026_-_Award_ceremony.mp4",
    "v_sb1": "./media/Vegas_-_Sneghnye_Barsy_23052026_-_1st_period.mp4",
}

INTERVAL_SEC = 5

for prefix, vpath in videos.items():
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"FAIL open {prefix}", flush=True)
        continue
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps * INTERVAL_SEC)
    saved = 0
    for idx in range(0, fc, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        t_sec = idx / fps
        cv2.imwrite(str(OUT / f"{prefix}_{t_sec:05.0f}.jpg"), frame)
        saved += 1
        if saved % 50 == 0:
            print(f"{prefix}: {saved} frames (t={t_sec:.0f}s/{fc/fps:.0f}s)", flush=True)
    cap.release()
    print(f"DONE {prefix}: {saved} frames", flush=True)

print(f"TOTAL: {len(list(OUT.glob('*.jpg')))} files", flush=True)
