# Linear probe over ResNet embeddings on human labels (200 crops).
# Honest 5-fold CV estimate: accuracy, balanced acc, per-class F1, confusion matrix.
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

z = np.load('/work/reid/embed.npz')
emb, files = z['emb'], z['files']
lab = json.load(open('/work/human_labels.json'))

stems = [str(f).split('/')[-1][:-4] for f in files]
X, y, keep = [], [], []
for i, s in enumerate(stems):
    c = lab.get(s)
    if c in ('K', 'D', 'W', 'R'):
        X.append(emb[i]); y.append(c); keep.append(s)
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
print('samples:', len(y), dict(zip(*[l.tolist() for l in np.unique(y, return_counts=True)])))

for cls in (['K', 'D', 'W'], ['K', 'D', 'W', 'R']):
    m = np.isin(y, cls)
    Xs, ys = X[m], y[m]
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    best = None
    for C in (0.01, 0.1, 1.0):
        lr = LogisticRegression(C=C, max_iter=3000, class_weight='balanced')
        pred = cross_val_predict(lr, Xs, ys, cv=skf)
        acc = (pred == ys).mean()
        if best is None or acc > best[0]:
            best = (acc, C, pred)
    acc, C, pred = best
    print(f'=== classes {cls} best C={C} acc={acc:.3f}')
    print(classification_report(ys, pred, digits=3))
    print(confusion_matrix(ys, pred, labels=cls))
