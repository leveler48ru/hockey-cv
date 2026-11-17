# Fine-tune the resnet50 team classifier on 162 manual labels.
# Fairness: same StratifiedKFold(5) as the linear probe. The winner gets an all-data model.
import json, os, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

CLASSES = ['K', 'D', 'W', 'R']
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('dev', dev)

z = np.load('/work/reid/embed.npz')
files = [str(f) for f in z['files']]
lab = json.load(open('/work/human_labels.json'))
samples = []  # (path, class_idx)
for f in files:
    stem = f.split('/')[-1][:-4]
    c = lab.get(stem)
    if c in CLASSES:
        samples.append((f, CLASSES.index(c)))
print('samples', len(samples))
paths = [s[0] for s in samples]; ys = np.array([s[1] for s in samples])

tf_train = transforms.Compose([
    transforms.Resize((160, 120)),
    transforms.RandomResizedCrop((144, 108), scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.15, 0.03),  # gentle: color is signal
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
tf_val = transforms.Compose([
    transforms.Resize((160, 120)),
    transforms.CenterCrop((144, 108)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class DS(torch.utils.data.Dataset):
    def __init__(self, idxs, tf):
        self.idxs, self.tf = idxs, tf
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        p, c = samples[self.idxs[i]]
        im = Image.open(p).convert('RGB')
        return self.tf(im), c

def make_model():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in m.parameters(): p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m.to(dev)

def train_eval(tr_idx, va_idx, epochs=15):
    m = make_model()
    # unfreeze layer4+fc — with 162 examples, going deeper = overfitting
    for p in m.layer4.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([
        {'params': [p for n, p in m.named_parameters() if p.requires_grad and 'layer4' not in n], 'lr': 1e-3},
        {'params': m.layer4.parameters(), 'lr': 1e-4},
    ], weight_decay=0.05)
    dl = DataLoader(DS(tr_idx, tf_train), batch_size=16, shuffle=True, num_workers=2)
    va = DS(va_idx, tf_val)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        m.train()
        for x, y in dl:
            x, y = x.to(dev), torch.tensor(y).to(dev)
            opt.zero_grad(); loss = crit(m(x), y); loss.backward(); opt.step()
        m.eval(); ok = n = 0
        with torch.no_grad():
            for i in va_idx:
                p, c = samples[i]
                im = tf_val(Image.open(p).convert('RGB')).to(dev)[None]
                ok += int(m(im).argmax(1).item() == c); n += 1
        acc = ok / n
        best = max(best, acc)
    return best

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
accs = []
for k, (tr, va) in enumerate(skf.split(paths, ys)):
    a = train_eval(list(tr), list(va))
    accs.append(a)
    print(f'fold{k} acc {a:.3f}', flush=True)
print(f'CV_mean {np.mean(accs):.3f} +- {np.std(accs):.3f}')

# final model trained on all data
m = make_model()
for p in m.layer4.parameters(): p.requires_grad = True
opt = torch.optim.AdamW([
    {'params': [p for n, p in m.named_parameters() if p.requires_grad and 'layer4' not in n], 'lr': 1e-3},
    {'params': m.layer4.parameters(), 'lr': 1e-4},
], weight_decay=0.05)
dl = DataLoader(DS(list(range(len(samples))), tf_train), batch_size=16, shuffle=True, num_workers=2)
crit = nn.CrossEntropyLoss()
m.train()
for ep in range(20):
    for x, y in dl:
        x, y = x.to(dev), torch.tensor(y).to(dev)
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
torch.save({'state': m.state_dict(), 'classes': CLASSES, 'tf': 'resize160x120>crop144x108>norm_imagenet'},
           '/work/team_clf_cv.pt')
print('saved /work/team_clf_cv.pt')
