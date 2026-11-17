#!/usr/bin/env python3
"""Converts referee boxes into YOLO-seg format (4-point polygon) and adds them to the dataset."""
import json, shutil
from pathlib import Path

DATASET = Path("dataset_real_all")
REF_FRAMES = Path("referee_frames")
TRAIN_LABELS = DATASET / "labels" / "train"
VAL_LABELS = DATASET / "labels" / "val"
TRAIN_IMAGES = DATASET / "images" / "train"

# JSON with referee detections (from a previous run on the VM)
ref_data = {
    "m1_01200": [{"box": [909, 309, 1010, 562], "sat": 28, "stripe": 1.59, "contrast": 70}],
    "m1_01800": [{"box": [201, 356, 243, 474], "sat": 35, "stripe": 0.36, "contrast": 95}],
    "m1_02400": [{"box": [483, 284, 581, 504], "sat": 30, "stripe": 0.3, "contrast": 65}, {"box": [661, 265, 704, 408], "sat": 35, "stripe": 0.34, "contrast": 54}],
    "m1_02700": [{"box": [538, 332, 613, 508], "sat": 47, "stripe": 1.18, "contrast": 46}, {"box": [874, 330, 937, 455], "sat": 35, "stripe": 1.16, "contrast": 41}, {"box": [600, 348, 644, 489], "sat": 47, "stripe": 1.0, "contrast": 37}, {"box": [1003, 341, 1049, 474], "sat": 42, "stripe": 0.62, "contrast": 66}],
    "m1_03000": [{"box": [909, 347, 964, 467], "sat": 38, "stripe": 0.54, "contrast": 81}, {"box": [1003, 364, 1144, 695], "sat": 41, "stripe": 0.74, "contrast": 63}, {"box": [1034, 397, 1140, 688], "sat": 39, "stripe": 0.69, "contrast": 71}],
    "m1_03300": [{"box": [1183, 316, 1221, 431], "sat": 41, "stripe": 2.69, "contrast": 37}],
    "m1_03600": [{"box": [227, 339, 289, 448], "sat": 46, "stripe": 0.48, "contrast": 104}, {"box": [18, 339, 53, 433], "sat": 30, "stripe": 2.0, "contrast": 30}, {"box": [611, 352, 663, 511], "sat": 52, "stripe": 0.56, "contrast": 39}],
    "m2_00060": [{"box": [1129, 271, 1207, 450], "sat": 49, "stripe": 0.38, "contrast": 31}, {"box": [38, 249, 165, 463], "sat": 35, "stripe": 0.71, "contrast": 64}],
    "m2_00600": [{"box": [1595, 269, 1687, 411], "sat": 41, "stripe": 0.32, "contrast": 63}, {"box": [820, 222, 878, 374], "sat": 34, "stripe": 0.75, "contrast": 31}, {"box": [1705, 256, 1799, 462], "sat": 46, "stripe": 0.47, "contrast": 34}, {"box": [536, 291, 592, 413], "sat": 53, "stripe": 0.53, "contrast": 40}],
    "m2_00900": [{"box": [1152, 270, 1249, 444], "sat": 19, "stripe": 0.46, "contrast": 32}, {"box": [127, 276, 229, 478], "sat": 34, "stripe": 1.62, "contrast": 94}, {"box": [1270, 313, 1380, 506], "sat": 29, "stripe": 0.68, "contrast": 68}, {"box": [94, 419, 304, 800], "sat": 28, "stripe": 0.43, "contrast": 52}],
    "m2_01500": [{"box": [1500, 379, 1573, 550], "sat": 70, "stripe": 0.82, "contrast": 69}, {"box": [866, 399, 972, 623], "sat": 24, "stripe": 0.56, "contrast": 74}, {"box": [977, 376, 1049, 552], "sat": 45, "stripe": 0.83, "contrast": 80}],
    "m2_01800": [{"box": [80, 246, 243, 517], "sat": 79, "stripe": 0.92, "contrast": 82}, {"box": [1514, 296, 1598, 474], "sat": 65, "stripe": 0.36, "contrast": 45}, {"box": [1238, 256, 1317, 452], "sat": 25, "stripe": 0.38, "contrast": 85}, {"box": [887, 269, 949, 424], "sat": 17, "stripe": 0.71, "contrast": 74}, {"box": [1240, 302, 1306, 450], "sat": 56, "stripe": 0.45, "contrast": 146}],
    "m2_02700": [{"box": [1650, 252, 1699, 410], "sat": 34, "stripe": 2.73, "contrast": 44}, {"box": [757, 262, 811, 396], "sat": 43, "stripe": 1.11, "contrast": 76}, {"box": [822, 269, 877, 394], "sat": 19, "stripe": 1.62, "contrast": 34}, {"box": [926, 266, 971, 400], "sat": 50, "stripe": 0.65, "contrast": 40}],
    "match_01200": [{"box": [1256, 373, 1356, 610], "sat": 27, "stripe": 1.47, "contrast": 52}, {"box": [944, 365, 985, 472], "sat": 45, "stripe": 1.03, "contrast": 47}, {"box": [1657, 372, 1743, 580], "sat": 24, "stripe": 0.34, "contrast": 70}, {"box": [1555, 376, 1593, 477], "sat": 34, "stripe": 0.77, "contrast": 32}],
    "match_01500": [{"box": [1470, 443, 1554, 570], "sat": 25, "stripe": 0.36, "contrast": 139}, {"box": [1652, 412, 1712, 573], "sat": 25, "stripe": 0.5, "contrast": 60}, {"box": [1157, 446, 1210, 557], "sat": 46, "stripe": 1.35, "contrast": 50}, {"box": [537, 441, 609, 586], "sat": 18, "stripe": 0.42, "contrast": 42}],
    "match_01800": [{"box": [943, 374, 1066, 559], "sat": 27, "stripe": 0.36, "contrast": 50}, {"box": [1510, 337, 1568, 461], "sat": 42, "stripe": 2.0, "contrast": 34}, {"box": [109, 351, 185, 490], "sat": 26, "stripe": 0.58, "contrast": 40}],
    "match_02100": [{"box": [1190, 312, 1251, 492], "sat": 22, "stripe": 2.44, "contrast": 34}, {"box": [1490, 361, 1666, 667], "sat": 46, "stripe": 0.34, "contrast": 72}, {"box": [858, 364, 915, 490], "sat": 64, "stripe": 0.77, "contrast": 31}],
    "match_02400": [{"box": [924, 378, 988, 569], "sat": 26, "stripe": 0.45, "contrast": 66}, {"box": [763, 384, 846, 556], "sat": 40, "stripe": 1.93, "contrast": 41}],
    "match_02700": [{"box": [764, 419, 900, 715], "sat": 31, "stripe": 1.09, "contrast": 75}, {"box": [945, 411, 1060, 636], "sat": 27, "stripe": 0.39, "contrast": 58}, {"box": [332, 437, 456, 694], "sat": 24, "stripe": 0.36, "contrast": 69}],
    "match_03000": [{"box": [569, 374, 646, 538], "sat": 25, "stripe": 0.57, "contrast": 62}, {"box": [1429, 374, 1506, 592], "sat": 27, "stripe": 1.51, "contrast": 59}, {"box": [347, 399, 417, 619], "sat": 23, "stripe": 0.42, "contrast": 58}, {"box": [907, 377, 984, 525], "sat": 30, "stripe": 0.38, "contrast": 40}, {"box": [1797, 435, 1915, 622], "sat": 45, "stripe": 0.38, "contrast": 65}],
}

W, H = 1920, 1080
REFEREE_CLASS = 8
added = 0

for base, dets in ref_data.items():
    # Copy the image into train if it isn't there yet
    src_img = REF_FRAMES / f"{base}.jpg"
    dst_img = TRAIN_IMAGES / f"{base}.jpg"
    if not dst_img.exists() and src_img.exists():
        shutil.copy2(src_img, dst_img)
    
    # Build the labels
    label_file = TRAIN_LABELS / f"{base}.txt"
    existing_lines = []
    if label_file.exists():
        existing_lines = [l.strip() for l in label_file.read_text().strip().split("\n") if l.strip()]
    
    new_lines = []
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        # Normalize to YOLO format (center, width, height)
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        # Convert to a 4-point polygon (YOLO-seg requires a segment, not a bbox)
        # 4 corners: top-left, top-right, bottom-right, bottom-left
        poly = f"{x1/W:.6f} {y1/H:.6f} {x2/W:.6f} {y1/H:.6f} {x2/W:.6f} {y2/H:.6f} {x1/W:.6f} {y2/H:.6f}"
        new_lines.append(f"{REFEREE_CLASS} {poly}")
    
    all_lines = existing_lines + new_lines
    label_file.write_text("\n".join(all_lines) + "\n")
    added += len(new_lines)

print(f"Added {added} referee labels to {len(ref_data)} frames")
print(f"Total train labels: {len(list(TRAIN_LABELS.glob('*.txt')))}")
