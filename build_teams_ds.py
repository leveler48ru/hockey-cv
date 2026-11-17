#!/usr/bin/env python3
"""Building the dataset with team classes.

Class scheme (11 classes -> 10):
  0 red_line 1 blue_line 2 center_circle 3 faceoff_spot 4 faceoff_circle
  5 goal_crease 6 board 7 player_red 8 player_blue 9 referee

Source: dataset_real_v5 (9 classes: 7=player, 8=referee) + teams.json
(team annotation by jersey color).

Rules:
  - a frame goes into the dataset only if teams.json provided player labels
    (otherwise players are unclassified and the "player" class disappears);
  - players outside red/blue (black gear, gray, helmets) — dropped,
    they have no team class;
  - class 8 (referee) from v5 -> class 9.
"""
import glob, os, json, shutil, collections, argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "dataset_real_v5"
DST = BASE / "dataset_teams"
TEAMS = BASE / "teams.json"

GEOM = range(0, 7)      # geometry classes stay unchanged
PLAYER_OLD = "7"
REF_OLD = "8"
RED, BLUE, REF_NEW = "7", "8", "9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DST))
    args = ap.parse_args()
    dst = Path(args.out)
    teams = json.load(open(TEAMS))

    if dst.exists():
        shutil.rmtree(dst)
    for s in ("train", "val"):
        (dst / "images" / s).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / s).mkdir(parents=True, exist_ok=True)

    stats = collections.Counter()
    per_class = collections.Counter()
    skipped = []

    for split in ("train", "val"):
        for lp in sorted((SRC / "labels" / split).glob("*.txt")):
            name = lp.stem
            t = teams.get(name)
            if not t:
                stats["frame without teams"] += 1
                continue
            src_img = SRC / "images" / split / f"{name}.jpg"
            if not src_img.exists():
                stats["missing image"] += 1
                continue
            lines_in = []
            pidx = 0
            for line in lp.read_text().splitlines():
                p = line.split()
                if not p:
                    continue
                cls, coords = p[0], p[1:]
                if cls == PLAYER_OLD:
                    red_idx = set(t["red"]); blue_idx = set(t["blue"])
                    if pidx in red_idx:
                        lines_in.append(f"{RED} {coords}")
                        per_class[RED] += 1
                    elif pidx in blue_idx:
                        lines_in.append(f"{BLUE} {coords}")
                        per_class[BLUE] += 1
                    else:
                        stats["player without team dropped"] += 1
                    pidx += 1
                elif cls == REF_OLD:
                    lines_in.append(f"{REF_NEW} {coords}")
                    per_class[REF_NEW] += 1
                else:
                    lines_in.append(f"{cls} {coords}")
                    per_class[cls] += 1

            shutil.copy2(src_img, dst / "images" / split / f"{name}.jpg")
            (dst / "labels" / split / f"{name}.txt").write_text(
                "\n".join(lines_in) + ("\n" if lines_in else ""))
            stats[f"frames in {split}"] += 1

    names = ['red_line', 'blue_line', 'center_circle', 'faceoff_spot', 'faceoff_circle',
             'goal_crease', 'board', 'player_red', 'player_blue', 'referee']
    yaml = (f"path: /root/{dst.name}\ntrain: images/train\nval: images/val\n"
            f"nc: {len(names)}\nnames: {names}\n")
    (dst / "data.yaml").write_text(yaml)

    print(json.dumps(dict(stats), ensure_ascii=False, indent=1))
    print("objects by class:", dict(per_class))
    print("total frames:", sum(v for k, v in stats.items() if k.startswith("frames in")))


if __name__ == "__main__":
    main()
