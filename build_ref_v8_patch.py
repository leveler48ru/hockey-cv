#!/usr/bin/env python3
"""v8: manual referee class on top of v5 without rebuilding the images.

 dataset_real_v5 (9 classes: 0-6 geometry, 7 player, 8 referee) is NOT modified.
 This computes ONLY a class 8 label patch from the platform's manual annotation
 (annotate/data/boxes/*.json) plus the apply list for the VM.

Rules (idempotent, like merge_referees_v5):
  - frame annotated, boxes present  -> class8 = manual boxes (4-point polygon);
  - frame annotated, json empty     -> "no referees": the old class8 is removed;
  - frame not in the dataset (v_award) -> skip, don't train (separate list);
  - frame NOT annotated             -> label left untouched (in v5 that class8 = 0,
                                      the auto-HSV legacy was already dropped in v5).

Output:
  ref_v8_patch/labels/<split>/<frame>.txt  — only the changed files
  ref_v8_patch/meta.json                   — summary for verification
"""
import json, os, shutil, collections
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parent
SRC = BASE / "dataset_real_v5"
BOXES = BASE / "annotate" / "data" / "boxes"
OUT = BASE / "ref_v8_patch"
REF = "8"


def box_to_seg(b, W, H):
    x1, y1 = max(0, b["x1"]), max(0, b["y1"])
    x2, y2 = min(W, b["x2"]), min(H, b["y2"])
    if x2 - x1 < 4 or y2 - y1 < 6:
        return None
    a = [x1 / W, y1 / H, x2 / W, y1 / H, x2 / W, y2 / H, x1 / W, y2 / H]
    return f"{REF} " + " ".join(f"{v:.5f}" for v in a)


def main():
    boxes = {jf.stem: json.loads(jf.read_text()) for jf in sorted(BOXES.glob("*.json"))}
    loc = {}
    for split in ("train", "val"):
        for p in (SRC / "images" / split).glob("*.jpg"):
            loc[p.stem] = split

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "labels" / "train").mkdir(parents=True)
    (OUT / "labels" / "val").mkdir(parents=True)

    stats = collections.Counter()
    per_class_new = {}
    for name, blist in boxes.items():
        split = loc.get(name)
        if split is None:
            stats["not in dataset (skipped)"] += 1
            continue
        lp = SRC / "labels" / split / f"{name}.txt"
        old = lp.read_text().splitlines() if lp.exists() else []
        old_ref = [l for l in old if l and l.split()[0] == REF]
        old_other = [l for l in old if l and l.split()[0] != REF]
        W, H = Image.open(SRC / "images" / split / f"{name}.jpg").size
        new_ref = [l for l in (box_to_seg(b, W, H) for b in blist) if l]
        merged = old_other + new_ref
        # write only if something actually changed
        if merged != [l for l in old if l]:
            (OUT / "labels" / split / f"{name}.txt").write_text(
                "\n".join(merged) + ("\n" if merged else ""))
            stats["files updated"] += 1
        stats["class8 before"] += len(old_ref)
        stats["class8 after"] += len(new_ref)
        stats["removed as negatives"] += max(0, len(old_ref) - len(new_ref)) if not blist else 0
        if new_ref:
            stats["frames with referees"] += 1
            stats[f"frames with referees in {split}"] += 1
            stats[f"objects in {split}"] += len(new_ref)
        stats["droppedDegraded"] += len(blist) - len(new_ref)

    meta = {
        "source_ds": SRC.name,
        "boxes_total": len(boxes),
        "stats": dict(stats),
        "changed_files": {
            s: sorted(p.stem for p in (OUT / "labels" / s).glob("*.txt"))
            for s in ("train", "val")
        },
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta["stats"], ensure_ascii=False, indent=1))
    print("changed files:", {k: len(v) for k, v in meta["changed_files"].items()})


if __name__ == "__main__":
    main()
