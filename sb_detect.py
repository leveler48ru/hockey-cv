import cv2, numpy as np, glob
files = sorted(glob.glob('./media/*.mp4'))
def sb_bbox(frame):
    h,w = frame.shape[:2]
    roi = frame[0:int(h*0.30), 0:int(w*0.60)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv,(0,110,70),(12,255,255)) | cv2.inRange(hsv,(168,110,70),(180,255,255))
    dark = cv2.inRange(hsv,(0,0,0),(180,255,55))
    m = cv2.morphologyEx(red|dark, cv2.MORPH_CLOSE, np.ones((21,21),np.uint8))
    cs,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None
    c = max(cs, key=cv2.contourArea)
    x,y,ww,hh = cv2.boundingRect(c)
    return (x,y,x+ww,y+hh) if ww>80 and hh>25 else None
for f in files:
    cap = cv2.VideoCapture(f)
    fps = cap.get(cv2.CAP_PROP_FPS)
    res=[]
    for ts in (30, 300, 900, 1800, 2700):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts*fps))
        ok, fr = cap.read()
        if not ok: continue
        bb = sb_bbox(fr)
        res.append((ts, fr.shape[1], fr.shape[0], bb))
    cap.release()
    print(f.split('/')[-1])
    for ts,w,h,bb in res:
        print(f'   t={ts:5d} {w}x{h} scoreboard={bb}')
