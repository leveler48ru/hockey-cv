#!/usr/bin/env python3
"""Extract wide shots from figure-skating roll videos (3 files from Alexander).
One frame every 5 seconds. Output: wide_fg/{fs1,fs2,fs3}_{sec:05d}.jpg + contact sheet."""
import cv2
from pathlib import Path
from PIL import Image, ImageDraw

SRC = {
    "fs1": "./media/video_2026-11-08_13-21-51.mp4",
    "fs2": "./media/video_2026-11-08_13-22-06.mp4",
    "fs3": "./media/video_2026-11-08_13-22-13.mp4",
}
OUT = Path("wide_fg")
OUT.mkdir(exist_ok=True)
INTERVAL_SEC = 5

counts = {}
for prefix, path in SRC.items():
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), path
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(round(fps * INTERVAL_SEC))
    idx = saved = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                cv2.imwrite(str(OUT / f"{prefix}_{idx / fps:05.0f}.jpg"), frame)
                saved += 1
        idx += 1
    cap.release()
    counts[prefix] = saved
print("frames:", counts, "total:", sum(counts.values()))

# Contact sheet: 12 frames evenly spaced across all three videos
all_fr = sorted(OUT.glob("*.jpg"))
pick = all_fr[:: max(1, len(all_fr) // 12)][:12]
TW, TH, cols = 480, 270, 4
rows = (len(pick) + cols - 1) // cols
sheet = Image.new("RGB", (cols * TW, rows * (TH + 20)), (18, 18, 18))
d = ImageDraw.Draw(sheet)
for i, p in enumerate(pick):
    im = Image.open(p).resize((TW, TH))
    r, c = divmod(i, cols)
    sheet.paste(im, (c * TW, r * (TH + 20)))
    d.text((c * TW + 6, r * (TH + 20) + TH + 4), p.stem, fill=(255, 210, 120))
sheet.save(OUT.parent / "fs_contact_sheet.jpg", quality=88)
print("sheet:", OUT.parent / "fs_contact_sheet.jpg")
