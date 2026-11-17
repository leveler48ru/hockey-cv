# -*- coding: utf-8 -*-
"""Generate numbered sheets for manual labeling of crops (path 3, teams).

Input: /work/reid/meta.json + /work/reid/crops/<pseudo>/<crop>
Output: /work/sheets/sheet_NN.jpg + /work/sheets/index.json
Label codes (ASCII, so cv2 doesn't break on Cyrillic):
  K red / D dark / W light / R referee / J junk
"""
import cv2, json, os, random, numpy as np

OUT = '/work/sheets'
os.makedirs(OUT, exist_ok=True)
meta = json.load(open('/work/reid/meta.json'))
random.seed(42)
N = 200
COLS, ROWS = 4, 5
TW, TH = 96, 144          # tile with room for the index number

# uniform sampling over the whole pool (natural distribution, no "pretty crop" filtering)
sample = random.sample(meta, min(N, len(meta)))

recs = []
for i, r in enumerate(sample):
    p = f"/work/reid/crops/{r['pseudo']}/{r['crop']}"
    im = cv2.imread(p)
    if im is None:
        im = np.zeros((TH - 24, TW, 3), np.uint8)
        im[:] = 60
    h, w = im.shape[:2]
    s = min((TW - 8) / w, (TH - 32) / h)
    im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
    tile = np.zeros((TH, TW, 3), np.uint8)
    y0 = 24
    x0 = (TW - im.shape[1]) // 2
    tile[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    cv2.putText(tile, str(i), (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    recs.append(dict(idx=i, crop=r['crop'], pseudo=r['pseudo'], split=r['split'],
                     src=r['src'], conf=r['conf']))
    # into the grid
    k = len(recs) - 1
    per = COLS * ROWS
    sheet_i, pos = divmod(k, per)
    cv2.imwrite(f'{OUT}/_t_{k:03d}.jpg', tile)

per = COLS * ROWS
nsheets = (len(recs) + per - 1) // per
for si in range(nsheets):
    tiles = []
    for j in range(si * per, (si + 1) * per):
        if j < len(recs):
            tiles.append(cv2.imread(f'{OUT}/_t_{j:03d}.jpg'))
        else:
            tiles.append(np.zeros((TH, TW, 3), np.uint8))
    rows = [np.hstack(tiles[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
    grid = np.vstack(rows)
    bar = np.zeros((26, grid.shape[1], 3), np.uint8)
    cv2.putText(bar, f'sheet {si+1}/{nsheets}   K=red D=dark W=light R=referee J=junk',
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 230, 255), 1)
    cv2.imwrite(f'{OUT}/sheet_{si:02d}.jpg', np.vstack([bar, grid]))
    print('wrote sheet', si)

json.dump(recs, open(f'{OUT}/index.json', 'w'), ensure_ascii=False, indent=1)
print('N_LABELLED', len(recs), 'SHEETS', nsheets)
for f in os.listdir(OUT):
    if f.startswith('_t_'):
        os.remove(os.path.join(OUT, f))
