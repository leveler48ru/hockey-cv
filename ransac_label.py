#!/usr/bin/env python3
"""RANSAC cleanup of marking lines from color masks + face-off circles + boards.
Input: wide/*.jpg (2174 frames)
Output: labeled/{vid}_{t}.jpg (visualization) + masks/{vid}_{t}.npz (masks for YOLO)
"""
import cv2, numpy as np, os, glob, random

WIDE = 'wide'
OUT_LABELED = 'labeled'
OUT_MASKS = 'masks'
os.makedirs(OUT_LABELED, exist_ok=True)
os.makedirs(OUT_MASKS, exist_ok=True)

# Scoreboard region per video
SB = {
    'match': (40, 25, 650, 160),
    'm1':    (55, 30, 385, 115),
    'm2':    (50, 25, 720, 175),
}

def ransac_line(pts, n_iter=200, thresh=3.0):
    """RANSAC straight-line fit through points. Returns (a,b,c) where ax+by+c=0, or None."""
    if len(pts) < 10:
        return None, []
    best_inliers = []
    best_line = None
    for _ in range(n_iter):
        idx = np.random.choice(len(pts), 2, replace=False)
        p1, p2 = pts[idx[0]], pts[idx[1]]
        # line through two points
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        if abs(dx) < 1 and abs(dy) < 1:
            continue
        # normal
        a, b = dy, -dx
        c = -(a*p1[0] + b*p1[1])
        norm = np.sqrt(a*a + b*b)
        if norm < 1e-6:
            continue
        a, b, c = a/norm, b/norm, c/norm
        # distances
        dists = np.abs(a*pts[:,0] + b*pts[:,1] + c)
        inliers = np.where(dists < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = (a, b, c)
    return best_line, best_inliers

def fit_lines_from_mask(mask, color_name, min_pts=50):
    """Find straight lines in a binary mask via RANSAC."""
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pts:
        return []
    pts = np.column_stack([xs, ys]).astype(np.float32)
    lines = []
    remaining = np.ones(len(pts), dtype=bool)
    for _ in range(5):  # at most 5 lines per color
        sub = pts[remaining]
        if len(sub) < min_pts:
            break
        line, inliers = ransac_line(sub, n_iter=300, thresh=4.0)
        if line is None or len(inliers) < min_pts:
            break
        lines.append((line, len(inliers)))
        remaining[np.where(remaining)[0][inliers]] = False
    return lines

def detect_board(frame, sb):
    """Boards = upper boundary of the ice. Look for a sharp light→dark transition vertically."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # scoreboard mask
    x1, y1, x2, y2 = sb
    gray_masked = gray.copy()
    gray_masked[y1:y2, x1:x2] = 128  # neutral value
    
    # for each column find the upper ice boundary (first pixel >200 after a dark zone)
    board_pts = []
    for x in range(0, w, 10):
        col = gray_masked[:, x]
        # look for transition: dark (<100) → light (>180)
        for y in range(int(h*0.15), int(h*0.6)):
            if col[y] < 100 and y+5 < h and col[y+5] > 180:
                board_pts.append((x, y+5))
                break
    return board_pts

def detect_faceoff_circles(frame, sb):
    """Face-off circles: HoughCircles on the red channel after subtracting lines."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    x1, y1, x2, y2 = sb
    red[y1:y2, x1:x2] = 0
    # morphology for cleanup
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    circles = cv2.HoughCircles(red, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                param1=50, param2=25, minRadius=8, maxRadius=40)
    if circles is not None:
        return [(int(x), int(y), int(r)) for x, y, r in circles[0]]
    return []

def process_frame(path, vid):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    sb = SB[vid]
    
    # Color masks
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
    red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    
    # Scoreboard mask
    x1, y1, x2, y2 = sb
    blue_mask[y1:y2, x1:x2] = 0
    red_mask[y1:y2, x1:x2] = 0
    
    # RANSAC lines
    blue_lines = fit_lines_from_mask(blue_mask, 'blue')
    red_lines = fit_lines_from_mask(red_mask, 'red')
    
    # Boards
    board_pts = detect_board(img, sb)
    
    # Face-off circles
    circles = detect_faceoff_circles(img, sb)
    
    # Visualization
    vis = img.copy()
    # Blue lines
    for (a, b, c), n_inl in blue_lines:
        # draw the line
        if abs(b) > 0.1:
            x0, x1_ = 0, w
            y0 = int(-(a*x0 + c) / b)
            y1_ = int(-(a*x1_ + c) / b)
            cv2.line(vis, (x0, y0), (x1_, y1_), (255, 0, 0), 2)
        else:
            x_ = int(-c / a)
            cv2.line(vis, (x_, 0), (x_, h), (255, 0, 0), 2)
    # Red lines
    for (a, b, c), n_inl in red_lines:
        if abs(b) > 0.1:
            x0, x1_ = 0, w
            y0 = int(-(a*x0 + c) / b)
            y1_ = int(-(a*x1_ + c) / b)
            cv2.line(vis, (x0, y0), (x1_, y1_), (0, 0, 255), 2)
        else:
            x_ = int(-c / a)
            cv2.line(vis, (x_, 0), (x_, h), (0, 0, 255), 2)
    # Boards
    for i in range(len(board_pts)-1):
        cv2.line(vis, board_pts[i], board_pts[i+1], (0, 255, 0), 2)
    # Circles
    for cx, cy, r in circles:
        cv2.circle(vis, (cx, cy), r, (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 2, (0, 255, 255), -1)
    
    # Save masks for YOLO
    mask_data = {
        'blue_lines': np.array([(a, b, c) for (a, b, c), _ in blue_lines], dtype=np.float32) if blue_lines else np.zeros((0, 3), dtype=np.float32),
        'red_lines': np.array([(a, b, c) for (a, b, c), _ in red_lines], dtype=np.float32) if red_lines else np.zeros((0, 3), dtype=np.float32),
        'board_pts': np.array(board_pts, dtype=np.int32) if board_pts else np.zeros((0, 2), dtype=np.int32),
        'circles': np.array(circles, dtype=np.int32) if circles else np.zeros((0, 3), dtype=np.int32),
    }
    
    return vis, mask_data

# Process all frames
files = sorted(glob.glob(f'{WIDE}/*.jpg'))
print(f'Total frames: {len(files)}')

processed = 0
for i, f in enumerate(files):
    vid = os.path.basename(f).split('_')[0]
    result = process_frame(f, vid)
    if result is None:
        continue
    vis, mask_data = result
    base = os.path.splitext(os.path.basename(f))[0]
    cv2.imwrite(f'{OUT_LABELED}/{base}.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    np.savez_compressed(f'{OUT_MASKS}/{base}.npz', **mask_data)
    processed += 1
    if processed % 200 == 0:
        print(f'  processed {processed}/{len(files)}', flush=True)

print(f'DONE: {processed} frames labeled')