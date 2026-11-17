"""
Converts referee boxes from annotate/data/boxes/*.json
into YOLO-seg polygons of class 8 (referee) and adds them to the dataset.

Box -> rectangular polygon of 4 points.
If the label file already exists — append the lines at the end.
If not — create a new one (empty for the other classes).
"""
import json, os, glob
from pathlib import Path

ANNO_DIR = Path("annotate/data/boxes")
DATASET_LABELS = Path("dataset_real_all/labels/train")
DATASET_IMAGES = Path("dataset_real_all/images/train")
REF_CLASS = 8  # referee

def box_to_yolo_seg(x1, y1, x2, y2, img_w, img_h):
    """Rectangular box -> 4-point polygon in YOLO-seg format."""
    pts = [
        (x1 / img_w, y1 / img_h),
        (x2 / img_w, y1 / img_h),
        (x2 / img_w, y2 / img_h),
        (x1 / img_w, y2 / img_h),
    ]
    coords = " ".join(f"{px:.6f} {py:.6f}" for px, py in pts)
    return f"{REF_CLASS} {coords}"

converted = 0
skipped_empty = 0
files_touched = set()

for jf in sorted(ANNO_DIR.glob("*.json")):
    frame_name = jf.stem  # e.g. match_01200
    
    with open(jf) as f:
        boxes = json.load(f)
    
    if not boxes:
        skipped_empty += 1
        continue
    
    # Find the matching image for its dimensions
    img_path = None
    for ext in ['.jpg', '.png']:
        p = DATASET_IMAGES / (frame_name + ext)
        if p.exists():
            img_path = p
            break
    
    if not img_path:
        # Try annotate/images/boxes
        for ext in ['.jpg', '.png']:
            p = Path("annotate/images/boxes") / (frame_name + ext)
            if p.exists():
                img_path = p
                break
    
    if not img_path:
        print(f"  ⚠ {frame_name}: image not found, skipping")
        continue
    
    # Read the dimensions
    from PIL import Image
    im = Image.open(img_path)
    img_w, img_h = im.size
    
    # Generate YOLO-seg lines
    lines = []
    for b in boxes:
        line = box_to_yolo_seg(b['x1'], b['y1'], b['x2'], b['y2'], img_w, img_h)
        lines.append(line)
    
    # Write into the label file
    label_path = DATASET_LABELS / (frame_name + ".txt")
    
    # If the file exists — read it, drop old class 8 entries, add the new ones
    existing = []
    if label_path.exists():
        with open(label_path) as f:
            for l in f:
                l = l.strip()
                if l and not l.startswith(f"{REF_CLASS} "):
                    existing.append(l)
    
    all_lines = existing + lines
    with open(label_path, 'w') as f:
        f.write("\n".join(all_lines) + "\n")
    
    files_touched.add(frame_name)
    converted += len(lines)
    print(f"  ✓ {frame_name}: {len(lines)} referee boxes -> seg ({img_w}x{img_h})")

print(f"\nTotal: {converted} referee polygons across {len(files_touched)} frames")
print(f"Skipped empty: {skipped_empty}")
print(f"Total train labels: {len(list(DATASET_LABELS.glob('*.txt')))}")
