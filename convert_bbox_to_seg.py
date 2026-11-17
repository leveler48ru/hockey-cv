#!/usr/bin/env python3
"""Converts player bbox (class cx cy w h) to segment (class x1 y1 x2 y2 x3 y3 x4 y4)."""
from pathlib import Path

DATASET = Path("dataset_real_all")
PLAYER_CLASS = 7

converted = 0
for split in ["train", "val"]:
    labels_dir = DATASET / "labels" / split
    for lbl_file in sorted(labels_dir.glob("*.txt")):
        lines = lbl_file.read_text().strip().split("\n")
        new_lines = []
        changed = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls == PLAYER_CLASS and len(parts) == 5:
                # bbox -> rect polygon
                cx, cy, w, h = map(float, parts[1:])
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy - h/2
                x3 = cx + w/2
                y3 = cy + h/2
                x4 = cx - w/2
                y4 = cy + h/2
                new_lines.append(f"{cls} {x1:.5f} {y1:.5f} {x2:.5f} {y2:.5f} {x3:.5f} {y3:.5f} {x4:.5f} {y4:.5f}")
                changed = True
            else:
                new_lines.append(line)
        if changed:
            lbl_file.write_text("\n".join(new_lines) + "\n")
            converted += 1

print(f"Converted {converted} files")
