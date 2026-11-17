#!/usr/bin/env python3
"""Team demo: YOLOv8-seg (player/referee) + team classifier on crops.
Player boxes are colored by team: K=red, D=blue (dark), W=white; referee - cyan REF.
Output /work/demo_teams.mp4, 8 s starting at DEMO_T0."""
import cv2, os, numpy as np, torch
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image

SRC = '/work/m1.mp4'
T0 = float(os.environ.get('DEMO_T0', 2118))
SECS = 8.0
OUT = '/work/demo_teams.mp4'
CONF = float(os.environ.get('DEMO_CONF', 0.25))

CLASSES = ['K', 'D', 'W', 'R']
COL = {'K': (0, 0, 255), 'D': (255, 60, 0), 'W': (255, 255, 255), 'R': (0, 255, 255)}
NAME = {'K': 'RED', 'D': 'DARK', 'W': 'LIGHT', 'R': 'REF'}

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO('/runs/rink_ref_v8/weights/best.pt')

tc = models.resnet50(weights=None)
tc.fc = torch.nn.Linear(tc.fc.in_features, len(CLASSES))
ck = torch.load('/work/team_clf_cv.pt', map_location=dev)
tc.load_state_dict(ck['state']); tc.to(dev).eval()

tf = transforms.Compose([
    transforms.Resize((160, 120)),
    transforms.CenterCrop((144, 108)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def team_of(crops_bgr):
    if not crops_bgr:
        return []
    x = torch.stack([tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops_bgr]).to(dev)
    with torch.no_grad():
        pr = torch.softmax(tc(x), 1)
        conf, pred = pr.max(1)
    return [(CLASSES[p], float(c)) for p, c in zip(pred.tolist(), conf.tolist())]

cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.set(cv2.CAP_PROP_POS_MSEC, T0 * 1000)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale = 1280.0 / W
dim = (1280, int(H * scale))
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), 30, dim)

n = int(fps * SECS); k = 0
stat = {'K': 0, 'D': 0, 'W': 0, 'R': 0}
while k < n:
    ok, frame = cap.read()
    if not ok:
        break
    k += 1
    if k % 2:
        continue
    r = yolo.predict(frame, imgsz=1088, conf=CONF, verbose=False)[0]
    img = cv2.resize(frame, dim)
    crops, metas = [], []
    if r.boxes is not None:
        xyxy = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
        for b, cid, c in zip(xyxy, cls, cf):
            cid = int(cid)
            x1, y1, x2, y2 = [int(v) for v in b]
            if cid == 8:
                X1, Y1, X2, Y2 = [int(v * scale) for v in b]
                cv2.rectangle(img, (X1, Y1), (X2, Y2), COL['R'], 2)
                cv2.putText(img, f'REF {c:.2f}', (X1, max(20, Y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL['R'], 2)
                stat['R'] += 1
            elif cid == 7 and x2 - x1 >= 24 and y2 - y1 >= 30:
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)
                metas.append((b, c))
    labs = team_of(crops)
    for (b, c), (lab, lc) in zip(metas, labs):
        X1, Y1, X2, Y2 = [int(v * scale) for v in b]
        col = COL[lab]
        cv2.rectangle(img, (X1, Y1), (X2, Y2), col, 2)
        cv2.putText(img, f'{NAME[lab]} {lc:.2f}', (X1, max(16, Y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        stat[lab] += 1
    cnt = img.shape[1]
    cv2.putText(img, f'm1 t={T0 + k / fps:6.1f}s  RED:{stat["K"]} DARK:{stat["D"]} LIGHT:{stat["W"]} REF:{stat["R"]}',
                (14, img.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    vw.write(img)
    if (k // 2) % 20 == 0:
        print('frame', k // 2, stat, flush=True)
cap.release(); vw.release()
print('WROTE', OUT, stat)
