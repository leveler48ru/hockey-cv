#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""track_m1.py — track the whole m1 match with the puck_v2 model (4 classes: goaltender,
player, puck, referee) using ByteTrack. One video pass, JSONL track log.

Writes /root/track_m1.jsonl.gz:
  {"f": frame, "t": seconds, "d": [[cls, tid, x1,y1,x2,y2, conf], ...]}
Progress log every 1200 frames (~40 s) — to shell out based on facts, not feelings.
"""
import gzip, json, sys, time
from ultralytics import YOLO

SRC = "/work/m1.mp4"          # /root on the host is mounted as /work inside the container
OUT = "/work/track_m1.jsonl.gz"
LOG = "/work/track_m1.log"
FPS = 29.96
NAMES = {0: "goaltender", 1: "player", 2: "puck", 3: "referee"}

def main():
    t0 = time.time()
    m = YOLO("/work/puck_v2_best.pt")
    gen = m.track(source=SRC, stream=True, persist=True,
                  tracker="/work/tracker.yaml",
                  classes=[0, 1, 2, 3], conf=0.25, iou=0.5,
                  imgsz=1280, device=0, verbose=False, half=True)
    n = 0
    with gzip.open(OUT, "wt") as fo, open(LOG, "w") as fl:
        for r in gen:
            dets = []
            if r.boxes is not None and len(r.boxes):
                b = r.boxes.xyxy.cpu().numpy()
                c = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                idn = r.boxes.id
                ids = idn.cpu().numpy().astype(int) if idn is not None else [-1] * len(b)
                for i in range(len(b)):
                    dets.append([int(c[i]), int(ids[i]),
                                 round(float(b[i][0]), 1), round(float(b[i][1]), 1),
                                 round(float(b[i][2]), 1), round(float(b[i][3]), 1),
                                 round(float(cf[i]), 3)])
            fo.write(json.dumps({"f": n, "d": dets}, separators=(",", ":")) + "\n")
            if n % 1200 == 0:
                el = time.time() - t0
                eta = (126076 - n) / max(n / max(el, .001), .001)
                msg = f"f={n} t={n / FPS:.0f}s el={el:.0f}s fps_eff={n / max(el, .001):.1f} eta={eta / 60:.0f}m"
                print(msg, flush=True)
                fl.write(msg + "\n"); fl.flush()
            n += 1
        msg = f"DONE frames={n} elapsed={time.time() - t0:.0f}s out={OUT}"
        print(msg, flush=True); fl.write(msg + "\n")

if __name__ == "__main__":
    main()
