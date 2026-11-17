#!/usr/bin/env python3
"""Sheet of 12 referee candidates with the FILE NAME LABELED on each tile.
Coordinates come from refscan.json (class 8, conf from the original label).
Output: media/qc/ref_cands_names.jpg"""
import json, os
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(BASE)
d = json.load(open(os.path.join(BASE, 'refscan.json')))
TILES = [('1.1', 'v_k2_00210', 0.83), ('1.2', 'v_k2_00130', 0.82), ('1.3', 'v_k2_00519', 0.80),
         ('2.1', 'm2_02640', 0.70), ('2.2', 'v_sb1_00938', 0.68), ('2.3', 'v_sb1_01013', 0.64),
         ('3.1', 'v_k2_00359', 0.61), ('3.2', 'match_01840', 0.56), ('3.3', 'm1_01200', 0.56),
         ('4.1', 'v_k2_00958', 0.54), ('4.2', 'v_k2_00698', 0.53), ('4.3', 'm1_03000', 0.52)]
TW, TH = 480, 360
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
f_big = ImageFont.truetype(FONT, 22)
f_small = ImageFont.truetype(FONT, 17)

sheet = Image.new('RGB', (3 * TW, 4 * TH + 4 * 4), (18, 18, 18))
draw = ImageDraw.Draw(sheet)
for i, (pos, name, conf) in enumerate(TILES):
    src = None
    for sub in ('wide_new', 'wide'):
        p = os.path.join(BASE, sub, name + '.jpg')
        if os.path.exists(p):
            src = p; break
    if not src:
        continue
    im = Image.open(src).convert('RGB')
    W, H = im.size
    hit = None
    for b in d.get(name) or []:
        if b['c'] == 8 and abs(b['conf'] - conf) <= 0.012:
            hit = b
    if hit is None:
        c8 = [b for b in (d.get(name) or []) if b['c'] == 8]
        hit = max(c8, key=lambda b: b['conf']) if c8 else None
    if hit:
        draw_ = ImageDraw.Draw(im)
        sc = 3.0
        for k in range(-1, 2):
            for j in range(-1, 2):
                draw_.rectangle([hit['x1'] + k, hit['y1'] + j, hit['x2'] + k, hit['y2'] + j],
                                outline=(0, 0, 0), width=2)
        draw_.rectangle([hit['x1'], hit['y1'], hit['x2'], hit['y2']], outline=(255, 230, 0), width=2)
    im = im.resize((TW, TH), Image.LANCZOS)
    r, c = divmod(i, 3)
    x, y = c * TW, r * (TH + 4)
    sheet.paste(im, (x, y))
    dr = ImageDraw.Draw(sheet)
    cap = f"{pos}  {name}"
    dr.rectangle([x + 2, y + 2, x + 2 + len(cap) * 11 + 12, y + 30], fill=(0, 0, 0))
    dr.text((x + 8, y + 5), cap, font=f_big, fill=(255, 255, 0))
    if hit:
        info = f"conf {hit['conf']:.2f}  {round(hit['x2']-hit['x1'])}x{round(hit['y2']-hit['y1'])}px  frame {W}x{H}"
        cut = [s for s, ok in (('bottom', hit['y2'] >= H - 4), ('top', hit['y1'] <= 4),
                               ('right', hit['x2'] >= W - 4), ('left', hit['x1'] <= 4)) if ok]
        if cut:
            info += "  CROPPED:" + ",".join(cut)
        dr.rectangle([x + 2, y + TH - 26, x + 2 + len(info) * 9 + 12, y + TH - 2], fill=(0, 0, 0))
        dr.text((x + 8, y + TH - 23), info, font=f_small, fill=(255, 255, 255))
out = os.path.join(WS, 'media/qc/ref_cands_names.jpg')
sheet.save(out, quality=90)
print(out, sheet.size)
