# Team_clf inference over all 14.8k crops -> select 200 "hard" ones for the second annotation batch.
# Criteria: low top1-top2 margin OR predicted W/D (weak classes) AND not in the first sample.
import json, os, shutil
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT = '/work/team_clf_cv.pt'
CLASSES = ['K', 'D', 'W', 'R']

tf = transforms.Compose([
    transforms.Resize((160, 120)),
    transforms.CenterCrop((144, 108)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

m = models.resnet50(weights=None)
m.fc = torch.nn.Linear(m.fc.in_features, len(CLASSES))
ck = torch.load(CKPT, map_location=dev)
m.load_state_dict(ck['state'])
m.to(dev).eval()
print('classes', ck['classes'])

meta = json.load(open('/work/reid/meta.json'))
human = set(json.load(open('/work/human_labels.json')).keys())  # already annotated - skip
paths = []
for r in meta:
    p = f"/work/reid/crops/{r['pseudo']}/{r['crop']}"
    if r['crop'][:-4] not in human and os.path.exists(p):
        paths.append((p, r))
print('to score:', len(paths))

rows = []
B = 256
with torch.no_grad():
    for i in range(0, len(paths), B):
        xs = []
        batch = paths[i:i+B]
        for p, r in batch:
            try:
                xs.append(tf(Image.open(p).convert('RGB')))
            except Exception:
                pass
        x = torch.stack(xs).to(dev)
        pr = torch.softmax(m(x), 1).cpu().numpy()
        for j, (p, r) in enumerate(batch):
            if j >= len(pr): continue
            s = np.sort(pr[j])[::-1]
            rows.append(dict(crop=r['crop'], pseudo=r['pseudo'], src=r['src'],
                             pred=CLASSES[int(pr[j].argmax())], p1=float(s[0]),
                             margin=float(s[0] - s[1])))
np.save('/work/preds.npy', np.array(rows, dtype=object), allow_pickle=True)
json.dump(rows, open('/work/preds.json', 'w'))
import collections
print('pred dist', dict(collections.Counter(r['pred'] for r in rows)))

# selection: minimal margin, but not total garbage; quotas per victim classes
cand = [r for r in rows]
cand.sort(key=lambda r: r['margin'])
hard = cand[:120]
rest = [r for r in cand[120:] if r['pred'] in ('W', 'D')]
import random; random.seed(7); random.shuffle(rest)
take = hard + rest[:80]
# dedup by src (no more than 3 crops from one frame)
cnt = collections.Counter(); sel = []
for r in take:
    if cnt[r['src']] >= 3: continue
    cnt[r['src']] += 1; sel.append(r)
sel = sel[:200]
print('selected', len(sel), dict(collections.Counter(r['pred'] for r in sel)))

dst = '/work/round2'
os.makedirs(dst, exist_ok=True)
for r in sel:
    src_p = f"/work/reid/crops/{r['pseudo']}/{r['crop']}"
    shutil.copy(src_p, f"{dst}/{r['crop']}")
json.dump(sel, open('/work/round2.json', 'w'))
print('copied to /work/round2')
