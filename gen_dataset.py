#!/usr/bin/env python3
"""Synthetic ice hockey rink dataset generator for YOLOv8-seg.

Classes:
  0: red_line      (red lines: center + goal lines)
  1: blue_line     (zone blue lines)
  2: center_circle (center circle)
  3: faceoff_spot  (faceoff spots)
  4: goal_crease   (goal creases)
  5: board         (boards)

Output: dataset/images/ + dataset/labels/ in YOLO-seg format
        dataset.yaml for training
"""
import cv2, numpy as np, os, json, random, math

OUT = os.environ.get("DATASET", "/root/dataset")
N = int(os.environ.get("N", 600))
W, H = 1920, 1080
os.makedirs(f"{OUT}/images/train", exist_ok=True)
os.makedirs(f"{OUT}/labels/train", exist_ok=True)

# World: rink 60x30 m, origin at the bottom-left corner
# X: 0..60 (along the long side), Y: 0..30 (across)
RINK_W, RINK_H = 60.0, 30.0

CLASSES = ["red_line", "blue_line", "center_circle", "faceoff_spot", "goal_crease", "board"]

def world_to_pix(pts_w, H_mat):
    """pts_w: Nx2 array of world coordinates -> Nx2 pixel coords"""
    pts_h = np.column_stack([pts_w, np.ones(len(pts_w))])
    q = (H_mat @ pts_h.T).T
    return q[:, :2] / q[:, 2:3]

def random_homography():
    """Generates a random homography imitating a camera from the stands."""
    # 4 rink corners in the world
    corners_w = np.array([[0, 0], [60, 0], [60, 30], [0, 30]], float)
    
    # Random perspective: camera somewhere to the side/above
    # Emulated via random corner offsets
    cx, cy = W // 2, H // 2
    
    # Base scaling: the rink occupies 60-90% of the frame
    scale = random.uniform(0.6, 0.9) * min(W / RINK_W, H / RINK_H)
    
    # Random yaw (camera rotation)
    yaw = random.uniform(-25, 25)  # degrees
    cos_y, sin_y = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    
    # Projection with tilt
    tilt = random.uniform(0.3, 0.7)  # perspective coefficient
    
    corners_p = []
    for x, y in corners_w:
        # Center relative to the rink center
        dx = (x - 30) * scale
        dy = (y - 15) * scale
        # Rotation
        rx = dx * cos_y - dy * sin_y
        ry = dx * sin_y + dy * cos_y
        # Perspective: the farther (larger ry), the stronger the compression
        persp = 1.0 / (1.0 + tilt * (ry / (H * 0.5)))
        px = cx + rx * persp + random.gauss(0, 20)
        py = cy + ry * persp * 0.8 + random.gauss(0, 15)  # 0.8 = vertical compression
        corners_p.append([px, py])
    
    corners_p = np.array(corners_p, float)
    
    # Check: all corners inside the frame?
    if np.any(corners_p < -100) or np.any(corners_p > [W + 100, H + 100]):
        return None
    
    H_mat = cv2.getPerspectiveTransform(corners_w.astype(np.float32), corners_p.astype(np.float32))
    if abs(np.linalg.det(H_mat)) < 1e-6:
        return None
    return H_mat

def draw_rink(img, H_mat):
    """Draws rink markings and returns YOLO-seg annotations."""
    annotations = []
    
    def w2p(pts):
        return world_to_pix(np.array(pts, float), H_mat)
    
    def poly_to_yolo(pts_px, cls_id):
        """Converts a polygon to YOLO-seg format: class x1 y1 x2 y2 ..."""
        pts = np.clip(pts_px, 0, [W - 1, H - 1]).astype(int)
        if len(pts) < 3:
            return None
        coords = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in pts)
        return f"{cls_id} {coords}"
    
    def line_poly(x_start, x_end, y, width_m, n_pts=40):
        """Polygon for a horizontal line"""
        xs = np.linspace(x_start, x_end, n_pts)
        top = list(zip(xs, [y + width_m / 2] * n_pts))
        bot = list(zip(xs[::-1], [y - width_m / 2] * n_pts))
        return top + bot
    
    def vline_poly(x, y_start, y_end, width_m, n_pts=40):
        """Polygon for a vertical line"""
        ys = np.linspace(y_start, y_end, n_pts)
        left = list(zip([x - width_m / 2] * n_pts, ys))
        right = list(zip([x + width_m / 2] * n_pts, ys[::-1]))
        return left + right
    
    LINE_W = 0.15  # line width in meters
    
    # === BOARDS (class 5) ===
    board_w = 1.2  # board width
    # Bottom board (y=0)
    pts = line_poly(0, 60, board_w / 2, board_w, 60)
    ann = poly_to_yolo(w2p(pts), 5)
    if ann: annotations.append(ann)
    # Top board (y=30)
    pts = line_poly(0, 60, 30 - board_w / 2, board_w, 60)
    ann = poly_to_yolo(w2p(pts), 5)
    if ann: annotations.append(ann)
    # Left board (x=0)
    pts = vline_poly(board_w / 2, 0, 30, board_w, 40)
    ann = poly_to_yolo(w2p(pts), 5)
    if ann: annotations.append(ann)
    # Right board (x=60)
    pts = vline_poly(60 - board_w / 2, 0, 30, board_w, 40)
    ann = poly_to_yolo(w2p(pts), 5)
    if ann: annotations.append(ann)
    
    # === RED LINES (class 0) ===
    # Center red line (x=30)
    pts = vline_poly(30, 0, 30, LINE_W, 50)
    ann = poly_to_yolo(w2p(pts), 0)
    if ann: annotations.append(ann)
    # Goal lines (x=4, x=56)
    for gx in [4.0, 56.0]:
        pts = vline_poly(gx, 0, 30, LINE_W, 50)
        ann = poly_to_yolo(w2p(pts), 0)
        if ann: annotations.append(ann)
    
    # === BLUE LINES (class 1) ===
    for bx in [22.4, 37.6]:
        pts = vline_poly(bx, 0, 30, LINE_W, 50)
        ann = poly_to_yolo(w2p(pts), 1)
        if ann: annotations.append(ann)
    
    # === CENTER CIRCLE (class 2) ===
    r = 4.5
    angles = np.linspace(0, 2 * np.pi, 60)
    outer = list(zip(30 + (r + LINE_W / 2) * np.cos(angles), 15 + (r + LINE_W / 2) * np.sin(angles)))
    inner = list(zip(30 + (r - LINE_W / 2) * np.cos(angles[::-1]), 15 + (r - LINE_W / 2) * np.sin(angles[::-1])))
    pts = outer + inner
    ann = poly_to_yolo(w2p(pts), 2)
    if ann: annotations.append(ann)
    
    # === FACEOFF SPOTS (class 3) ===
    spots = [(22.4, 7.1), (22.4, 22.9), (37.6, 7.1), (37.6, 22.9),  # neutral zone
             (6.0, 7.1), (6.0, 22.9), (54.0, 7.1), (54.0, 22.9)]    # defensive zones
    spot_r = 0.3
    for sx, sy in spots:
        angles = np.linspace(0, 2 * np.pi, 20)
        pts = list(zip(sx + spot_r * np.cos(angles), sy + spot_r * np.sin(angles)))
        ann = poly_to_yolo(w2p(pts), 3)
        if ann: annotations.append(ann)
    
    # === GOAL CREASE (class 4) ===
    crease_r = 1.83
    for gx in [4.0, 56.0]:
        angles = np.linspace(-np.pi / 2, np.pi / 2, 30)
        outer = list(zip(gx + (crease_r + LINE_W / 2) * np.cos(angles), 15 + (crease_r + LINE_W / 2) * np.sin(angles)))
        inner = list(zip(gx + (crease_r - LINE_W / 2) * np.cos(angles[::-1]), 15 + (crease_r - LINE_W / 2) * np.sin(angles[::-1])))
        pts = outer + inner
        ann = poly_to_yolo(w2p(pts), 4)
        if ann: annotations.append(ann)
    
    return annotations

def make_ice_texture(w, h):
    """Generates an ice texture."""
    ice = np.full((h, w, 3), 220, dtype=np.uint8)
    # Light noise
    noise = np.random.randint(-8, 8, (h, w, 3), dtype=np.int16)
    ice = np.clip(ice.astype(np.int16) + noise, 200, 240).astype(np.uint8)
    # Skate marks (random thin lines)
    for _ in range(random.randint(20, 60)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = x1 + random.randint(-200, 200), y1 + random.randint(-200, 200)
        cv2.line(ice, (x1, y1), (x2, y2), (random.randint(190, 210),) * 3, 1)
    return ice

def add_advertising(img, H_mat):
    """Adds advertising banners on the boards."""
    colors = [(200, 50, 50), (50, 50, 200), (50, 200, 50), (200, 200, 50), (200, 50, 200)]
    for i in range(random.randint(3, 8)):
        x = random.uniform(5, 55)
        w_ad = random.uniform(2, 6)
        pts = [(x, 0.3), (x + w_ad, 0.3), (x + w_ad, 1.0), (x, 1.0)]
        pts_px = world_to_pix(np.array(pts, float), H_mat).astype(int)
        col = random.choice(colors)
        cv2.fillPoly(img, [pts_px], col)
        # Same on the top board
        pts_top = [(x, 29.7), (x + w_ad, 29.7), (x + w_ad, 29.0), (x, 29.0)]
        pts_px = world_to_pix(np.array(pts_top, float), H_mat).astype(int)
        cv2.fillPoly(img, [pts_px], col)

def add_players(img, H_mat, n_players=None):
    """Adds placeholder players."""
    if n_players is None:
        n_players = random.randint(0, 12)
    for _ in range(n_players):
        wx = random.uniform(2, 58)
        wy = random.uniform(2, 28)
        pt = world_to_pix(np.array([[wx, wy]], float), H_mat)[0]
        px, py = int(pt[0]), int(pt[1])
        if 0 <= px < W and 0 <= py < H:
            h_player = random.randint(30, 60)
            w_player = h_player // 3
            color = random.choice([(200, 200, 200), (50, 50, 150), (150, 50, 50)])
            cv2.ellipse(img, (px, py), (w_player, h_player // 2), 0, 0, 360, color, -1)

def add_overlay(img):
    """Adds a TV overlay (score, time)."""
    if random.random() < 0.4:
        # Score bar at the bottom
        h_bar = random.randint(30, 60)
        y_bar = random.choice([0, H - h_bar])
        cv2.rectangle(img, (0, y_bar), (W, y_bar + h_bar), (0, 0, 0), -1)
        # Placeholder text
        cv2.putText(img, "HOME 2 - 1 AWAY", (W // 4, y_bar + h_bar // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

print(f"Generating {N} images...")
generated = 0
for i in range(N * 3):  # attempts with a margin (some homographies are invalid)
    if generated >= N:
        break
    
    H_mat = random_homography()
    if H_mat is None:
        continue
    
    # Background: ice
    img = make_ice_texture(W, H)
    
    # Advertising
    add_advertising(img, H_mat)
    
    # Markings
    annotations = draw_rink(img, H_mat)
    
    if len(annotations) < 5:
        continue  # too few objects visible
    
    # Players
    add_players(img, H_mat)
    
    # Overlay
    add_overlay(img)
    
    # Lighting noise
    brightness = random.uniform(0.8, 1.2)
    img = np.clip(img.astype(float) * brightness, 0, 255).astype(np.uint8)
    
    # Save
    name = f"rink_{generated:05d}"
    cv2.imwrite(f"{OUT}/images/train/{name}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(f"{OUT}/labels/train/{name}.txt", "w") as f:
        f.write("\n".join(annotations) + "\n")
    
    generated += 1
    if generated % 50 == 0:
        print(f"  {generated}/{N}")

print(f"Generated {generated} images")

# dataset.yaml
yaml_content = f"""path: {OUT}
train: images/train
val: images/train

nc: {len(CLASSES)}
names: {CLASSES}
"""
with open(f"{OUT}/dataset.yaml", "w") as f:
    f.write(yaml_content)

print("dataset.yaml saved")
print(f"Classes: {CLASSES}")
print("DONE")
