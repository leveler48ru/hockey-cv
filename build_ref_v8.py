#!/usr/bin/env python3
"""v8: manual referee class on top of dataset_real_v6. We change ONLY the class 8 labels.

v6 class scheme (9): 0-6 geometry, 7 player, 8 referee.
Base: /tmp/v6labels/labels (downloaded from the VM, images stay on the VM).
Annotation: annotate/data/boxes/*.json (manual, 406 frames, coordinates in pixels of the
native frame resolution) + sizes from annotate/images/boxes.

Rules:
  - frame annotated, boxes present  -> class8 = manual boxes (4-point polygon);
  - frame annotated, json empty     -> "no referees": class8 is removed;
  - frame not annotated             -> label left untouched;
  - v_award (not in the base)       -> skipped, we train on what exists;
  - degenerate base polygons (zero area) are not touched: those are classes 0/1/6,
    kept as is so v8 and v6 stay comparable on geometry.

Output: dataset_ref_v8/labels/{train,val}/*.txt + data_vm.yaml + report.json
"""
import json, os, glob, shutil, collections, argparse
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parent
BOXES = BASE / "annotate" / "data" / "boxes"
ANNO_IMG = BASE / "annotate" / "images" / "boxes"
REF = "8"
DUP_SAFE = ("m1_", "m2_", "match_", "v_k2_", "v_sb1_", "v_award_")


def box_to_seg(b, W, H, sx=1.0, sy=1.0):
    x1, y1 = max(0, b["x1"] * sx), max(0, b["y1"] * sy)
    x2, y2 = min(W, b["x2"] * sx), min(H, b["y2"] * sy)
    if x2 - x1 < 4 or y2 - y1 < 6:
        return None
    a = [x1 / W, y1 / H, x2 / W, y1 / H, x2 / W, y2 / H, x1 / W, y2 / H]
    return f"{REF} " + " ".join(f"{v:.5f}" for v in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/tmp/v6labels/labels")
    ap.add_argument("--sizes-from", default=str(BASE / "dataset_real_v5/images"),
                    help="directory with the base images (for W,H); if missing, take from annotate")
    ap.add_argument("--out", default=str(BASE / "dataset_ref_v8"))
    ap.add_argument("--val-move-target", type=int, default=18,
                    help="how many frames with 2+ referees to move into val")
    args = ap.parse_args()
    src = Path(args.labels)
    out = Path(args.out)
    sizes_dir = Path(args.sizes_from)

    boxes = {jf.stem: json.loads(jf.read_text()) for jf in sorted(BOXES.glob("*.json"))}
    base_files = {s: {p.stem for p in (src / s).glob("*.txt")} for s in ("train", "val")}
    print(f"v6 base: train {len(base_files['train'])} val {len(base_files['val'])} | "
          f"manual annotation: {len(boxes)} frames, {sum(len(v) for v in boxes.values())} objects")

    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val"):
        (out / "labels" / s).mkdir(parents=True)

    stats = collections.Counter()
    ref_lines = {}
    for name, blist in boxes.items():
        splits = [s for s in ("train", "val") if name in base_files[s]]
        if not splits:
            stats["not in base (skipped)"] += 1
            continue
        if len(splits) > 1:
            stats["stem in both splits (skipped)"] += 1
            continue
        split = splits[0]
        ip = sizes_dir / split / f"{name}.jpg"
        ap_ = ANNO_IMG / f"{name}.jpg"
        if ip.exists():
            W, H = Image.open(ip).size
            aw, ah = Image.open(ap_).size if ap_.exists() else (W, H)
            sx, sy = W / aw, H / ah          # safety net in case of different resolutions
            if (round(sx, 3), round(sy, 3)) != (1.0, 1.0):
                stats["annotation scale != 1"] += 1
        else:
            W, H = Image.open(ap_).size
            sx = sy = 1.0
            stats["no base image, sizes taken from annotate"] += 1
        ref_lines[name] = (split, [l for l in (box_to_seg(b, W, H, sx, sy) for b in blist) if l])

    moved = collections.Counter()
    train_candidates = []
    for split in ("train", "val"):
        for name in sorted(base_files[split]):
            raw = [l for l in (src / split / f"{name}.txt").read_text().splitlines() if l]
            rl = ref_lines.get(name)
            if rl and rl[0] == split:
                keep = [l for l in raw if l.split()[0] != REF]
                stats["old class8 removed"] += len(raw) - len(keep)
                new = keep + rl[1]
                stats["new class8 added"] += len(rl[1])
                if not json.loads((BOXES / f"{name}.json").read_text()):
                    stats["confirmed as negative"] += 1
            else:
                new = raw
                stats["labels untouched"] += 1
                stats["class8 in untouched"] += sum(1 for l in raw if l.split()[0] == REF)
            if split == "train" and len([l for l in new if l.split()[0] == REF]) >= 2:
                train_candidates.append(name)
            (out / "labels" / split / f"{name}.txt").write_text(
                "\n".join(new) + ("\n" if new else ""))

    # val balance: some frames with 2+ referees go to val, spread evenly across videos,
    # otherwise the referee metric can't be measured (v6 had only 38 objects in val)
    by_vid = collections.defaultdict(list)
    for n in sorted(train_candidates):
        by_vid[next(v for v in DUP_SAFE if n.startswith(v))].append(n)
    move, i = [], 0
    while len(move) < args.val_move_target and any(by_vid.values()):
        vids = [v for v in sorted(by_vid) if by_vid[v]]
        v = vids[i % len(vids)]
        move.append(by_vid[v].pop(0))
        i += 1
    for n in move:
        (out / "labels" / "train" / f"{n}.txt").rename(out / "labels" / "val" / f"{n}.txt")
    stats["moved to val"] = len(move)

    names = ['red_line', 'blue_line', 'center_circle', 'faceoff_spot', 'faceoff_circle',
             'goal_crease', 'board', 'player', 'referee']
    (out / "data_vm.yaml").write_text(
        "path: /data\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names}\n")

    rep = {}
    for split in ("train", "val"):
        fr = ob = imgs = frames_any = 0
        per = collections.Counter()
        for p in (out / "labels" / split).glob("*.txt"):
            imgs += 1
            ls = [l for l in p.read_text().splitlines() if l]
            c = sum(1 for l in ls if l.split()[0] == REF)
            if c:
                fr += 1
                ob += c
                per[p.stem.rsplit("_", 1)[0]] += c
            frames_any += 1
        rep[split] = {"labels": imgs, "frames_with_referees": fr,
                     "referee_objects": ob, "by_video": dict(per)}
    report = {"stats": dict(stats), "referee_now": rep,
              "moved_to_val": sorted(move)}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
