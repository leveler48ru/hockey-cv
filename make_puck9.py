#!/usr/bin/env python3
"""HockeyAI SHL -> YOLO dataset 'puck9': figure objects + puck classes only.
Source: hockey/datasets/SHL/{frames/*.jpg, annotations/*.txt} (7 classes, cls5=puck).
Mapping: 3->0 goaltender, 4->1 player, 5->2 PUCK, 6->3 referee (centerIce/faceoff/goal - dropped).
85/15 split, fixed seed; symlink structure, tar for transfer to the GPU server."""
import os, random, shutil, tarfile, json

SRC = 'datasets/SHL'
OUT = 'datasets/puck9'
MAP = {3: 0, 4: 1, 5: 2, 6: 3}
NAMES = ['goaltender', 'player', 'puck', 'referee']

if os.path.exists(OUT):
    shutil.rmtree(OUT)
for d in ('images/train', 'images/val', 'labels/train', 'labels/val'):
    os.makedirs(f'{OUT}/{d}', exist_ok=True)

frames = sorted(f for f in os.listdir(f'{SRC}/frames') if f.endswith('.jpg'))
random.seed(13)
order = frames[:]
random.shuffle(order)
nval = int(len(order) * 0.15)
valset = set(o[:-4] for o in order[:nval])

stats = {'train': 0, 'val': 0, 'puck_train': 0, 'puck_val': 0, 'dropped': 0}
for fn in frames:
    stem = fn[:-4]
    src_ann = f'{SRC}/annotations/{stem}.txt'
    lines = []
    for l in open(src_ann):
        p = l.split()
        if not p: continue
        c = int(p[0])
        if c in MAP:
            lines.append(f"{MAP[c]} " + ' '.join(p[1:]))
        else:
            stats['dropped'] += 1
    npuck = sum(1 for l in lines if l.startswith('2 '))
    split = 'val' if stem in valset else 'train'
    shutil.copyfile(f'{SRC}/frames/{fn}', f'{OUT}/images/{split}/{fn}')
    with open(f'{OUT}/labels/{split}/{stem}.txt', 'w') as fh:
        fh.write('\n'.join(lines) + ('\n' if lines else ''))
    stats[split] += 1
    stats[f'puck_{split}'] += npuck

yaml = (f"train: images/train\nval: images/val\nnc: 4\nnames: {NAMES}\n")
open(f'{OUT}/data.yaml', 'w').write(yaml)
json.dump(stats, open(f'{OUT}/stats.json', 'w'), indent=1)
print(stats)

TAR = f'{OUT}.tar'
with tarfile.open(TAR, 'w') as tf:
    tf.add(OUT, arcname='puck9')
print('tar MB', round(os.path.getsize(TAR) / 1e6, 1))
