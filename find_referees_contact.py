#!/usr/bin/env python3
"""Contact sheets from various parts of the matches, for finding referees."""
import cv2, numpy as np, glob, random
from pathlib import Path

WIDE = Path("wide")
OUT_DIR = Path("referee_search")
OUT_DIR.mkdir(exist_ok=True)

# Frame sampling from different parts of each match
samples = {
    'match': [60, 180, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000],
    'm1': [60, 180, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600],
    'm2': [60, 180, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700],
}

for vid, times in samples.items():
    frames = []
    for t in times:
        fpath = WIDE / f"{vid}_{t:05d}.jpg"
        if fpath.exists():
            frame = cv2.imread(str(fpath))
            if frame is not None:
                # Add a timestamp label
                cv2.putText(frame, f"{vid} t={t}s", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                frames.append(frame)
    
    if frames:
        # Contact sheet 4x3 or smaller
        rows = []
        for i in range(0, len(frames), 4):
            row = frames[i:i+4]
            while len(row) < 4:
                row.append(np.zeros_like(row[0]))
            rows.append(np.hstack(row))
        contact = np.vstack(rows)
        
        out_path = OUT_DIR / f"{vid}_referee_search.jpg"
        cv2.imwrite(str(out_path), contact)
        print(f"Saved {out_path} ({contact.shape[1]}x{contact.shape[0]}, {len(frames)} frames)")

print("Done. Check referee_search/ for contact sheets.")
