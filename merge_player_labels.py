#!/usr/bin/env python3
"""Adds player boxes (class 7) to existing annotation labels in dataset_real_all."""
import os, shutil
from pathlib import Path

DATASET = Path("dataset_real_all")
PLAYER_MASKS = Path("player_masks")
TRAIN_LABELS = DATASET / "labels" / "train"
VAL_LABELS = DATASET / "labels" / "val"
TRAIN_IMAGES = DATASET / "images" / "train"
VAL_IMAGES = DATASET / "images" / "val"

# player class = 7 (after marking 0-6)
PLAYER_CLASS = 7

added = 0
merged = 0
skipped = 0

for pm_file in sorted(PLAYER_MASKS.glob("*.txt")):
    base = pm_file.stem  # e.g. m1_00090
    
    # Check whether this frame exists in train or val
    if (TRAIN_LABELS / f"{base}.txt").exists():
        target_label = TRAIN_LABELS / f"{base}.txt"
        target_image = TRAIN_IMAGES / f"{base}.jpg"
    elif (VAL_LABELS / f"{base}.txt").exists():
        target_label = VAL_LABELS / f"{base}.txt"
        target_image = VAL_IMAGES / f"{base}.jpg"
    else:
        # Frame is in player_masks but not in dataset_real_all
        # Add to train (if the image exists in wide/)
        wide_img = Path(f"wide/{base}.jpg")
        if not wide_img.exists():
            skipped += 1
            continue
        target_label = TRAIN_LABELS / f"{base}.txt"
        target_image = TRAIN_IMAGES / f"{base}.jpg"
        shutil.copy2(wide_img, target_image)
        added += 1
    
    # Read the player boxes
    player_lines = []
    with open(pm_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Replace class 0 with PLAYER_CLASS
                player_lines.append(f"{PLAYER_CLASS} {' '.join(parts[1:])}\n")
    
    if not player_lines:
        continue
    
    # If the label file already exists - append the player boxes at the end
    if target_label.exists():
        with open(target_label, "a") as f:
            f.writelines(player_lines)
        merged += 1
    else:
        with open(target_label, "w") as f:
            f.writelines(player_lines)
        added += 1

print(f"Done: added={added}, merged={merged}, skipped={skipped}")
print(f"Total train labels: {len(list(TRAIN_LABELS.glob('*.txt')))}")
print(f"Total val labels: {len(list(VAL_LABELS.glob('*.txt')))}")
