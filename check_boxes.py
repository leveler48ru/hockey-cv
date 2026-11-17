import os, cv2
from ultralytics import YOLO
m = YOLO('/root/runs/rink26/weights/best.pt')
img = cv2.imread('/root/test_600.png')
print('shape:', img.shape)
r = m.predict('/root/test_600.png', conf=0.05, device=0, verbose=False)[0]
for i, (b, c, cf) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())):
    x1,y1,x2,y2 = [round(v) for v in b]
    print(f'{i}: {m.names[int(c)]:12s} conf={cf:.2f} box=({x1},{y1})-({x2},{y2}) size={x2-x1}x{y2-y1}')
