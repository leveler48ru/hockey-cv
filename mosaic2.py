#!/usr/bin/env python3
"""Whole-rink panorama: median (player-free) key frames -> cv2.Stitcher.
Diagnostics: stitch status, number of matches found, canvas size.
Output: match/panorama_raw.jpg, match/med_{c,l,r}.jpg"""
import cv2, numpy as np, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
M = BASE + "/match"
F = "./media/9281020430926.mp4"
FPS = 30


def median_frame(sec, half=2.0, n=13):
    """median over n frames around sec: moving players disappear"""
    cap = cv2.VideoCapture(F)
    frs = []
    for s in np.linspace(sec - half, sec + half, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * FPS))
        ok, fr = cap.read()
        if ok:
            frs.append(fr.astype(np.uint16))
    cap.release()
    if len(frs) < 5:
        raise RuntimeError(f"too few frames for sec={sec}")
    return np.median(np.stack(frs), axis=0).astype(np.uint8)


def main():
    print("median frames...", flush=True)
    med = {}
    med["l"] = median_frame(240)
    med["c"] = median_frame(3)
    med["r"] = median_frame(2570)
    for k, v in med.items():
        cv2.imwrite(f"{M}/med_{k}.jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("medians ready", flush=True)

    st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    imgs = [med["l"], med["c"], med["r"]]
    status, pano = st.stitch(imgs)
    print("stitch status:", status, flush=True)
    if status != 0 or pano is None:
        # fallback: center+right and center+left pairs only
        for name, pair in [("cr", [med["c"], med["r"]]), ("lc", [med["l"], med["c"]])]:
            s2, p2 = cv2.Stitcher_create(cv2.Stitcher_PANORAMA).stitch(pair)
            print(f"  {name}: status={s2}", "shape=" + str(p2.shape) if p2 is not None else "")
        sys.exit("full panorama not assembled")
    cv2.imwrite(f"{M}/panorama_raw.jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("PANO shape:", pano.shape, flush=True)


if __name__ == "__main__":
    main()
