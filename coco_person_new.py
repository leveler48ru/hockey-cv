#!/usr/bin/env python3
"""COCO person pseudo-labels for the new frames (v_k2, v_sb1)."""
import cv2, json, os
from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolov8x-seg.pt")  # COCO-pretrained
IMG_DIR = Path("dataset_real_all/images/train")
OUT_DIR = Path("coco_person_new")
OUT_DIR.mkdir(exist_ok=True)

# Only the new frames (v_k2_*, v_sb1_*)
new_files = sorted([f for f in IMG_DIR.glob("*.jpg") if f.name.startswith(("v_k2_", "v_sb1_"))])
print(f"Processing {len(new_files)} new frames", flush=True)

total_detections = 0
saved = 0

for i, img_path in enumerate(new_files):
    results = model(str(img_path), conf=0.25, verbose=False)
    boxes = []
    for r in results:
        if r.masks is None:
            continue
        for mask, cls, conf in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if int(cls) != 0:  # person only
                continue
            # Convert the mask to a polygon
            mask_uint8 = (mask * 255).astype('uint8')
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) < 3:
                continue
            h, w = mask.shape
            poly = cnt.reshape(-1, 2).tolist()
            boxes.append({'class': 7, 'poly': poly})  # class 7 = player
    
    if boxes:
        out_path = OUT_DIR / img_path.name.replace('.jpg', '.txt')
        with open(out_path, 'w') as f:
            for b in boxes:
                # YOLO seg format: class x1 y1 x2 y2 ... xn yn (normalized)
                coords = []
                for px, py in b['poly']:
                    coords.append(f"{px/w:.6f}")
                    coords.append(f"{py/h:.6f}")
                f.write(f"7 {' '.join(coords)}\n")
        saved += 1
        total_detections += len(boxes)
    
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(new_files)}: saved={saved}, detections={total_detections}", flush=True)

print(f"DONE: {saved} label files, {total_detections} person detections", flush=True)
