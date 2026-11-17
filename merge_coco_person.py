#!/usr/bin/env python3
"""Merges COCO person labels into dataset_real_all (class 7 = player)."""
import shutil
from pathlib import Path

DATASET = Path("dataset_real_all")
COCO_MASKS = Path("coco_person_masks")
WIDE = Path("wide")
TRAIN_LABELS = DATASET / "labels" / "train"
VAL_LABELS = DATASET / "labels" / "val"
TRAIN_IMAGES = DATASET / "images" / "train"
VAL_IMAGES = DATASET / "images" / "val"

PLAYER_CLASS = 7
added_new = 0
merged_existing = 0
skipped = 0

for cm_file in sorted(COCO_MASKS.glob("*.txt")):
    base = cm_file.stem
    
    # Check whether the frame exists in train or val
    if (TRAIN_LABELS / f"{base}.txt").exists():
        target_label = TRAIN_LABELS / f"{base}.txt"
    elif (VAL_LABELS / f"{base}.txt").exists():
        target_label = VAL_LABELS / f"{base}.txt"
    else:
        # New frame - add it to train
        wide_img = WIDE / f"{base}.jpg"
        if not wide_img.exists():
            skipped += 1
            continue
        target_label = TRAIN_LABELS / f"{base}.txt"
        target_image = TRAIN_IMAGES / f"{base}.jpg"
        shutil.copy2(wide_img, target_image)
        added_new += 1
    
    # Read the COCO person labels
    coco_lines = cm_file.read_text().strip()
    if not coco_lines:
        continue
    
    # Append to the existing labels
    if target_label.exists():
        existing = target_label.read_text().strip()
        target_label.write_text(existing + "\n" + coco_lines + "\n")
        merged_existing += 1
    else:
        target_label.write_text(coco_lines + "\n")
        added_new += 1

print(f"Done: added_new={added_new}, merged_existing={merged_existing}, skipped={skipped}")
print(f"Total train labels: {len(list(TRAIN_LABELS.glob('*.txt')))}")
print(f"Total val labels: {len(list(VAL_LABELS.glob('*.txt')))}")
