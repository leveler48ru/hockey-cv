#!/usr/bin/env python3
"""Overlay manual referee annotations onto dataset_real_v6 labels (9 classes).

Only class 8 (referee): in annotated frames it is rewritten from the manual
boxes; in unconfirmed frames it stays as is. 145 degenerate polygons of the
base (zero area) are left untouched: v8 must remain comparable with v6 on the same base.

Input: --labels (directory with v6 train/val labels, downloaded from the VM)
Output: --out (v8 labels) + data_vm.yaml + report.json
"""
import json, os, glob, collections, argparse
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parent
BOXES = BASE / "annotate" / "data" / "boxes"
ANNO_IMG = BASE / "annotate" / "images" / "boxes"
REF = "8"


def box_to_seg(b, W, H):
    x1, y1 = max(0, b["x1"]), max(0, b["y1"])
    x2, y2 = min(W, b["x2"]), min(H, b["y2"])
    if x2 - x1 < 4 or y2 - y1 < 6:
        return None
    a = [x1 / W, y1 / H, x2 / W, y1 / H, x2 / W, y2 / H, x1 / W, y2 / H]
    return f"{REF} " + " ".join(f"{v:.5f}" for v in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="base labels directory: train/ val/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-move", type=int, default=15)
    args = ap.parse_args()
    src, out = Path(args.labels), Path(args.out)
    for s in ("train", "val"):
        (out / s).mkdir(parents=True, exist_ok=True)

    boxes = {jf.stem: json.loads(jf.read_text()) for jf in sorted(BOXES.glob("*.json"))}
    stats = collections.Counter()
    written = {}
    for split in ("train", "val"):
        for p in sorted((src / split).glob("*.txt")):
            name = p.stem
            raw = [l for l in p.read_text().splitlines() if l]
            if name not in boxes:
                written.setdefault(split, {})[name] = raw
                stats["unannotated, copied as is"] += 1
                stats["class8 in unannotated"] += sum(1 for l in raw if l.split()[0] == REF)
                continue
            im = ANNO_IMG / f"{name}.jpg"
            if not im.exists():
                written.setdefault(split, {})[name] = raw
                stats["no frame in annotate (skipped)"] += 1
                continue
            W, H = Image.open(im).size
            new = [l for l in raw if l.split()[0] != REF]
            stats["old class8 removed"] += len(raw) - len(new)
            segs = [s for s in (box_to_seg(b, W, H) for b in boxes[name]) if s]
            new += segs
            stats["new class8 added"] += len(segs)
            if boxes[name]:
                stats["frames with referees"] += 1
            else:
                stats["confirmed negatives"] += 1
            written.setdefault(split, {})[name] = new

    # val balance: frames with 2+ referees move from train to val (spread evenly across videos)
    per_vid = collections.defaultdict(list)
    for name, lines in written["train"].items():
        if sum(1 for l in lines if l.split()[0] == REF) >= 2:
            per_vid[name.rsplit("_", 1)[0]].append(name)
    move, i = [], 0
    while len(move) < args.val_move and any(per_vid.values()):
        vids = [v for v in sorted(per_vid) if per_vid[v]]
        v = vids[i % len(vids)]
        move.append(per_vid[v].pop(0))
        i += 1
    for n in move:
        written["val"][n] = written["train"].pop(n)
    stats["moved to val"] = len(move)

    for split in ("train", "val"):
        for name, lines in written[split].items():
            (out / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    names = ['red_line', 'blue_line', 'center_circle', 'faceoff_spot', 'faceoff_circle',
             'goal_crease', 'board', 'player', 'referee']
    (out / "data_vm.yaml").write_text(
        "path: /data\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names}\n")

    rep = {}
    for split in ("train", "val"):
        fr = ob = 0
        per = collections.Counter()
        for name, lines in written[split].items():
            c = sum(1 for l in lines if l.split()[0] == REF)
            if c:
                fr += 1
                ob += c
                per[name.rsplit("_", 1)[0]] += c
        rep[split] = {"labels": len(written[split]), "frames_with_referees": fr,
                      "referee_objects": ob, "per_video": dict(sorted(per.items()))}
    report = {"base": str(src), "stats": dict(stats), "referee_now": rep,
              "moved_to_val": sorted(move)}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
