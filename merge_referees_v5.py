#!/usr/bin/env python3
"""Merge manual referee annotations (annotate/data/boxes) into the v5 dataset.

dataset_real_v5 = copy of dataset_real_all where, for EVERY annotated
frame class 8 is rewritten from the manual boxes (idempotent):
  - boxes present   -> class8 rows built from them (polygon TL,TR,BR,BL, normalized);
  - empty json      -> frame confirmed as 'no referees', class8 rows removed;
  - frame absent from the dataset (v_award) -> skipped, not trained on.
Some frames with >=2 referees move from train to val - otherwise the
referee metric cannot be measured (in v3 validation for class 8 was empty).
"""
import json, shutil, os, collections
from pathlib import Path
from PIL import Image

BASE = Path(".")
SRC_DS = BASE / "dataset_real_all"
DST_DS = BASE / "dataset_real_v5"
BOXES = BASE / "annotate" / "data" / "boxes"

MOVE_TO_VAL_MIN_BOXES = 2
MOVE_TO_VAL_TARGET = 15

def box_to_seg(b, W, H):
    x1, y1, x2, y2 = (max(0, b["x1"]), max(0, b["y1"]), min(W, b["x2"]), min(H, b["y2"]))
    a = [x1 / W, y1 / H, x2 / W, y1 / H, x2 / W, y2 / H, x1 / W, y2 / H]
    return "8 " + " ".join(f"{v:.5f}" for v in a)

# 1. fresh copy of the dataset
if DST_DS.exists():
    shutil.rmtree(DST_DS)
shutil.copytree(SRC_DS, DST_DS)

# where each frame is located
loc = {}
for split in ("train", "val"):
    for p in (DST_DS / "images" / split).glob("*.jpg"):
        loc[p.stem] = split

merged = collections.Counter()
skipped_no_frame = []
size_mismatch = []
ref_classes = collections.defaultdict(list)   # name -> seg lines

for jf in sorted(BOXES.glob("*.json")):
    name = jf.stem
    if name not in loc:
        skipped_no_frame.append(name)
        continue
    boxes = json.loads(jf.read_text())
    # sizes of the annotated frame (platform) and the dataset one must match
    anno_img = BASE / "annotate" / "images" / "boxes" / f"{name}.jpg"
    ds_img = DST_DS / "images" / loc[name] / f"{name}.jpg"
    W, H = Image.open(anno_img).size
    W2, H2 = Image.open(ds_img).size
    if (W, H) != (W2, H2):
        size_mismatch.append((name, (W, H), (W2, H2)))
        W, H = W2, H2  # normalize against the dataset image
    lines = [box_to_seg(b, W, H) for b in boxes]
    ref_classes[name] = lines
    merged[loc[name]] += 1
    merged["objs"] += len(lines)

# rewrite labels: drop old class8 rows, append the manual ones
for name, lines in list(ref_classes.items()) + [(n, []) for n in
        [jf.stem for jf in BOXES.glob("*.json") if jf.stem in loc and not json.loads(jf.read_text())]
        if n not in ref_classes]:
    split = loc[name]
    lp = DST_DS / "labels" / split / f"{name}.txt"
    keep = []
    if lp.exists():
        keep = [l for l in lp.read_text().splitlines() if l and not l.startswith("8 ")]
    new = keep + lines
    tmp = lp.with_suffix(".tmp")
    tmp.write_text("\n".join(new) + ("\n" if new else ""))
    os.replace(tmp, lp)

# 2. move some referee frames to val (2+ referees, spread evenly across videos)
cands = [(n, len(l)) for n, l in ref_classes.items() if len(l) >= MOVE_TO_VAL_MIN_BOXES and loc[n] == "train"]
cands.sort(key=lambda x: x[0].rsplit("_", 1)[0])
by_vid = collections.defaultdict(list)
for n, c in cands:
    by_vid[n.rsplit("_", 1)[0]].append(n)
move = []
i = 0
while len(move) < MOVE_TO_VAL_TARGET and any(by_vid[v] for v in by_vid):
    vids = [v for v in sorted(by_vid) if by_vid[v]]
    v = vids[i % len(vids)]
    move.append(by_vid[v].pop(0))
    i += 1
for n in move:
    for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
        src = DST_DS / kind / "train" / f"{n}{ext}"
        dst = DST_DS / kind / "val" / f"{n}{ext}"
        src.rename(dst)
    loc[n] = "val"

# 3. honest summary for class 8
report = {"train_frames": 0, "val_frames": 0, "train_objs": 0, "val_objs": 0}
for split in ("train", "val"):
    for lp in (DST_DS / "labels" / split).glob("*.txt"):
        n = sum(1 for l in lp.read_text().splitlines() if l.startswith("8 "))
        if n:
            report[f"{split}_frames"] += 1
            report[f"{split}_objs"] += n

print(json.dumps({
    "merged frames": sum(merged[k] for k in ("train", "val")),
    "class8 objects": merged["objs"],
    "moved to val": len(move),
    "class8 now": report,
    "skipped (not in dataset)": skipped_no_frame,
    "size mismatches": size_mismatch[:5],
    "suspicious (>350px tall)": sorted(
        jf.stem for jf in BOXES.glob("*.json")
        for b in json.loads(jf.read_text()) if b["y2"] - b["y1"] > 350),
}, ensure_ascii=False, indent=1))
