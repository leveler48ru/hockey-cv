#!/usr/bin/env python3
"""Path 3: ReID embeddings of player crops + clustering.
1. resnet50 (IMAGENET1K_V2, fc stripped) -> 2048-d L2-normalized embeddings of all crops.
2. KMeans k=8, silhouette, cluster sizes.
3. Reference sheets per cluster (12 crops, src/conf caption on each — derived from the file name).
Output: /root/reid/embed.npz, /root/reid/clusters.json, /root/reid/cla_<k>_<i>.jpg"""
import os, json, glob
import cv2, numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.cluster import KMeans
import sklearn.metrics as sm

OUT = '/work/reid'
files = sorted(glob.glob(OUT + '/crops/*/*.jpg'))
print('crops', len(files))

class DS(Dataset):
    def __init__(self, files, tf):
        self.f, self.tf = files, tf
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        im = cv2.imread(self.f[i])
        if im is None:
            im = np.zeros((128, 64, 3), np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return self.tf(im), i

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Resize((224, 224), antialias=True),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
m.fc = torch.nn.Identity(); m.to(dev).eval()

emb = np.zeros((len(files), 2048), np.float32)
with torch.no_grad():
    for bi, (x, idx) in enumerate(DataLoader(DS(files, tf), batch_size=256, num_workers=4)):
        v = m(x.to(dev)).cpu().numpy()
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        emb[idx.numpy()] = v
        if bi % 10 == 0: print('batch', bi, flush=True)

np.savez_compressed(OUT + '/embed.npz', emb=emb, files=np.array(files))
X = emb / (emb.std(0) + 1e-6)
km = KMeans(8, n_init=10, random_state=0)
lab = km.fit_predict(X)
print('silhouette_k8', round(sm.silhouette_score(emb, lab, metric='cosine'), 3))
meta = {os.path.basename(p): p for p in files}
sizes = np.bincount(lab)
print('sizes', sizes.tolist())
cl = {}
for k in range(8):
    idx = np.where(lab == k)[0]
    cl[int(k)] = [os.path.basename(files[i]) for i in idx]
    tiles = []
    for i in idx[:12]:
        im = cv2.imread(files[i])
        if im is None: continue
        tiles.append(cv2.resize(im, (64, 96)))
    if tiles:
        while len(tiles) < 12: tiles.append(np.zeros_like(tiles[0]))
        sheet = np.hstack(tiles)
        cv2.imwrite(f'{OUT}/cla_{k}.jpg', sheet)
json.dump({'sizes': sizes.tolist(), 'members': cl}, open(OUT + '/clusters.json', 'w'))
print('DONE')
