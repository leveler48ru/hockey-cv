#!/usr/bin/env python3
"""Batch of frames for referee annotation.
Input: referee_masks/*.txt (wide: match,m1,m2) + referee_masks_new/*.txt (wide_new: v_k2,v_sb1,v_award).
Input label format: YOLO seg (class x1 y1 ... x4 y4, normalized).
Junk filter: pixel bbox >= 20x30 (platform minimums), <= 350 in height.
Selection: uniform over time within each video, quotas below.
Output:
  annotate/images/boxes/<frame>.jpg   — image
  annotate/data/prefill/<frame>.json  — SUGGESTED boxes [{x1,y1,x2,y2}] (not the annotation!)
  annotate/data/prefill/_manifest.json — batch composition
Already annotated frames (data/boxes/*.json) are NOT taken into the batch.
"""
import json, shutil
from pathlib import Path
from PIL import Image

BASE = Path(".")

QUOTA = {  # frame diversity across 6 broadcasts (batch N2, 08.11 evening)
    "m1": 40, "match": 40, "m2": 30,
    "v_k2": 45, "v_sb1": 45, "v_award": 6,
}

SRC_DIRS = {"wide": BASE / "referee_masks", "wide_new": BASE / "referee_masks_new"}

ANNO_IMG = BASE / "annotate" / "images" / "boxes"
ANNO_DONE = BASE / "annotate" / "data" / "boxes"
PREFILL = BASE / "annotate" / "data" / "prefill"
PREFILL.mkdir(parents=True, exist_ok=True)


def seg_to_boxes(txt_path: Path, img_w: int, img_h: int):
    boxes = []
    for line in txt_path.read_text().splitlines():
        p = line.split()
        if len(p) < 9:
            continue
        xs = [float(p[i]) for i in range(1, 9, 2)]
        ys = [float(p[i]) for i in range(2, 9, 2)]
        x1, x2 = min(xs) * img_w, max(xs) * img_w
        y1, y2 = min(ys) * img_h, max(ys) * img_h
        w, h = x2 - x1, y2 - y1
        if w < 20 or h < 30:
            continue
        # a little breathing room around the silhouette: the HSV detector cuts off arms/legs
        pad_x, pad_y = w * 0.15, h * 0.08
        boxes.append({
            "x1": max(0, int(x1 - pad_x)), "y1": max(0, int(y1 - pad_y)),
            "x2": min(img_w, int(x2 + pad_x)), "y2": min(img_h, int(y2 + pad_y)),
        })
    return boxes


done_frames = {f.stem for f in ANNO_DONE.glob("*.json")}
by_video = {}
for src_name, mask_dir in SRC_DIRS.items():
    img_dir = BASE / src_name
    for mf in sorted(mask_dir.glob("*.txt")):
        stem = mf.stem
        if stem in done_frames:
            continue
        vid = stem.rsplit("_", 1)[0]
        img = img_dir / f"{stem}.jpg"
        if not img.exists():
            continue
        iw, ih = Image.open(img).size
        boxes = seg_to_boxes(mf, iw, ih)
        if not boxes or len(boxes) > 10:  # >10 — almost certainly junk/a crowd by the boards
            continue
        try:
            t = float(stem.rsplit("_", 1)[1])
        except ValueError:
            t = 0.0
        by_video.setdefault(vid, []).append((t, stem, boxes, img))

manifest = {"created": "2026-11-08 N2", "note": "prefill = auto-detector suggestions, need editing", "frames": {}}
total = 0
for vid, cand in by_video.items():
    cand.sort()
    quota = QUOTA.get(vid, 0)
    if not quota:
        continue
    step = max(1, len(cand) // quota)
    picked = cand[::step][:quota]
    for t, stem, boxes, img in picked:
        shutil.copy2(img, ANNO_IMG / f"{stem}.jpg")
        (PREFILL / f"{stem}.json").write_text(json.dumps(boxes, ensure_ascii=False))
        manifest["frames"][stem] = {"video": vid, "t_sec": t, "candidates": len(boxes)}
        total += 1

(PREFILL / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
per = {}
for f in manifest["frames"].values():
    per[f["video"]] = per.get(f["video"], 0) + 1
print(f"batch: {total} frames, by video: {per}")
print(f"total candidates were: {{{', '.join(f'{v}: {len(c)}' for v, c in by_video.items())}}}")
