#!/usr/bin/env python3
import numpy
import cv2, sys
from pathlib import Path
base = sys.argv[1]
img = cv2.imread(f"referee_frames/{base}.jpg")
h, w = img.shape[:2]
for line in open(f"referee_masks_manual/{base}.txt"):
    p = line.strip().split()
    cls = int(p[0]); coords = list(map(float, p[1:]))
    pts = [(int(coords[i]*w), int(coords[i+1]*h)) for i in range(0, len(coords), 2)]
    cv2.polylines(img, [np.array(pts) if False else __import__('numpy').array(pts)], True, (0,0,255), 3)
    cv2.putText(img, "referee", pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
cv2.imwrite(f"referee_check_{base}.jpg", img)
print("ok")
