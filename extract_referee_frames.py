#!/usr/bin/env python3
"""Extracts referee frames at the specified timestamps."""
import cv2, shutil
from pathlib import Path

WIDE = Path("wide")
OUT = Path("referee_frames")
OUT.mkdir(exist_ok=True)

# Referee frames (specified by Alexander)
referee_times = {
    'match': [1200, 1500, 1800, 2100, 2400, 2700, 3000],
    'm1': [1200, 1800, 2400, 2700, 3000, 3300, 3600],
    'm2': [60, 600, 900, 1500, 1800, 2700],
}

extracted = 0
for vid, times in referee_times.items():
    for t in times:
        src = WIDE / f"{vid}_{t:05d}.jpg"
        dst = OUT / f"{vid}_{t:05d}.jpg"
        if src.exists():
            shutil.copy2(src, dst)
            extracted += 1
            print(f"  {src.name} -> {dst.name}")
        else:
            print(f"  MISSING: {src.name}")

print(f"\nExtracted {extracted} frames to {OUT}/")
