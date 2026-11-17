#!/usr/bin/env python3
"""Unified annotation platform: boxes mode (referees/boxes) and calib mode (homography points).
LAN only: 127.0.0.1:8090."""
import http.server
import json
import os
import re
import cgi
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
# Sizes are read dynamically from the frame file; fallback only if the file is not found
DEFAULT_IMG_W, DEFAULT_IMG_H = 1280, 720

# Real coordinates of reference points (meters, origin at center ice).
# IIHF (standard for our arenas): 60x30 m; goal line 4 m from the boards → x=±26;
# blue lines 15 m apart → x=±7.5; zone faceoff dots: 6 m from the goal line (x=±20),
# 7 m from the boards (y=±8); center dots — 1.5 m from the red line; circles r=4.5 m.
RINK_POINTS = {
    "center": [0, 0],                       # center of the rink
    "red_near": [0, -15],                   # red line — near board (bottom of frame)
    "red_far": [0, 15],                     # red line — far board (top of frame)
    "blue_L_near": [-7.5, -15],             # left blue line — near board
    "blue_L_far": [-7.5, 15],               # left blue line — far board
    "blue_R_near": [7.5, -15],              # right blue line — near board
    "blue_R_far": [7.5, 15],                # right blue line — far board
    "faceoff_C_left": [-1.524, 0],          # center circle: dot slightly left of the red line (10 ft between dots)
    "faceoff_C_right": [1.524, 0],          # center circle: dot slightly right of the red line
    "faceoff_L_near": [-19.904, -8.294],    # left zone — near board (20 ft from goal line, 22 ft from the boards)
    "faceoff_L_far": [-19.904, 8.294],      # left zone — far board
    "faceoff_R_near": [19.904, -8.294],     # right zone — near board
    "faceoff_R_far": [19.904, 8.294],       # right zone — far board
    "goal_L": [-26, 0],                     # left goal
    "goal_R": [26, 0],                      # right goal
}

# Human-readable labels (keys are technical, they go into JSON).
# Orientation is the frame itself: "left/right" = frame edges, "near/far board" = bottom/top of frame.
RINK_LABELS = {
    "center": "Center ice",
    "red_near": "Red line — near board (bottom of frame)",
    "red_far": "Red line — far board (top of frame)",
    "blue_L_near": "Left blue line — near board",   # left edge of frame
    "blue_L_far": "Left blue line — far board",
    "blue_R_near": "Right blue line — near board",  # right edge of frame
    "blue_R_far": "Right blue line — far board",
    "faceoff_C_left": "Center circle — dot left of the red line",
    "faceoff_C_right": "Center circle — dot right of the red line",
    "faceoff_L_near": "Left zone — dot by the near board",
    "faceoff_L_far": "Left zone — dot by the far board",
    "faceoff_R_near": "Right zone — dot by the near board",
    "faceoff_R_far": "Right zone — dot by the far board",
    "goal_L": "Left goal — center",
    "goal_R": "Right goal — center",
}

# Where exactly to click (shown in the hint when picking a point)
RINK_HINTS = {
    "center": "Intersection of the red line and the longitudinal axis (dead center of the ice)",
    "red_near": "Where the red line meets the board nearest to the camera (bottom of frame)",
    "red_far": "Where the red line meets the far board (top of frame)",
    "blue_L_near": "Where the blue line at the LEFT edge of the frame reaches the near board",
    "blue_L_far": "The same blue line at the left edge — where it reaches the far board",
    "blue_R_near": "Where the blue line at the RIGHT edge of the frame reaches the near board",
    "blue_R_far": "The same blue line at the right edge — where it reaches the far board",
    "faceoff_C_left": "Faceoff dot in the center circle, slightly left of the red line (1.5 m from center)",
    "faceoff_C_right": "Faceoff dot in the center circle, slightly right of the red line",
    "faceoff_L_near": "Faceoff dot in the left zone of the frame, by the near board",
    "faceoff_L_far": "Faceoff dot in the left zone of the frame, by the far board",
    "faceoff_R_near": "Faceoff dot in the right zone of the frame, by the near board",
    "faceoff_R_far": "Faceoff dot in the right zone of the frame, by the far board",
    "goal_L": "Left goal: midpoint of the line between the posts (if no goal — center of the crease trapezoid/semicircle)",
    "goal_R": "Right goal: midpoint of the line between the posts (if no goal — center of the crease trapezoid/semicircle)",
}

# Short labels for chips on the phone (long ones don't fit)
RINK_SHORT = {
    "center": "Center",
    "red_near": "Red ↓bottom",
    "red_far": "Red ↑top",
    "blue_L_near": "Blue-L ↓",
    "blue_L_far": "Blue-L ↑",
    "blue_R_near": "Blue-R ↓",
    "blue_R_far": "Blue-R ↑",
    "faceoff_C_left": "Ctr-circle ←",
    "faceoff_C_right": "Ctr-circle →",
    "faceoff_L_near": "L-zone ↓",
    "faceoff_L_far": "L-zone ↑",
    "faceoff_R_near": "R-zone ↓",
    "faceoff_R_far": "R-zone ↑",
    "goal_L": "Goal left",
    "goal_R": "Goal right",
}

MODES = {
    "boxes": {"dir": "images/boxes", "data": "data/boxes", "kind": "boxes"},
    "calib": {"dir": "images/calib", "data": "data/calib", "kind": "points"},
    "crops": {"dir": "images/crops", "data": "data/crops", "kind": "label"},
    # Puck: same boxes, but small-object validation (puck is ~10-15 px at 1280 width)
    "puck": {"dir": "images/puck", "data": "data/puck", "kind": "puckboxes"},
    # Audio annotation: listening to candidates (whistle/siren). mp3 files, label is a single code.
    "audio": {"dir": "images/audio", "data": "data/audio", "kind": "sound"},
    # Player numbers: c_*.jpg crops, label is a number 0..99 or "X" (illegible).
    # data/numbers may also hold prefill/pconf from OCR — they are preserved on edit.
    "numbers": {"dir": "images/numbers", "data": "data/numbers", "kind": "number"},
}

# Audio label codes (key is ASCII in JSON, label is human-readable).
SOUND_CODES = ["W", "WS", "S", "M", "N", "X"]
SOUND_LABELS = {
    "W": "whistle only",
    "WS": "whistle + siren (goal)",
    "S": "siren without whistle",
    "M": "music/announcement",
    "N": "none of these (noise)",
    "X": "can't tell",
}

# Crop annotation: one letter per whole crop (team/type). Codes are ASCII, labels human-readable.
TEAM_CODES = ["K", "D", "W", "R", "J", "X"]
TEAM_LABELS = {
    "K": "red",
    "D": "dark",
    "W": "light/goie",
    "R": "referee",
    "J": "junk",
    "X": "unsure",
}

def valid_number(v):
    """Number label: 'X' (illegible) or a digit string 0..99. Returns the canonical form or None."""
    if v == "X":
        return "X"
    if isinstance(v, str) and v.isdigit() and len(v) <= 2 and 0 <= int(v) <= 99:
        return str(int(v))
    return None


# Build number: the page compares it against the server's and complains if the client is on a cached old version
BUILD = "2026-11-16-0830"


def mode_cfg(q):
    return MODES.get(q.get("mode", ["boxes"])[0], MODES["boxes"])


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

    def _json(self, obj, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def _form(self):
        ctype = self.headers.get('Content-Type', '')
        if 'multipart/form-data' in ctype:
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': ctype})
            return form.getvalue('frame', ''), form.getvalue('payload', '')
        cl = int(self.headers.get('Content-Length', 0))
        p = parse_qs(self.rfile.read(cl).decode('utf-8'))
        return p.get('frame', [''])[0], p.get('payload', [''])[0]

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/save":
            self.send_error(404); return
        q = parse_qs(u.query)
        cfg = mode_cfg(q)
        frame, payload = self._form()
        frame = re.sub(r'[^A-Za-z0-9_\-]', '', frame)
        if not frame:
            self._json({'ok': False, 'error': 'no frame'}, 400); return
        try:
            data = json.loads(payload or ('[]' if cfg['kind'] in ('boxes', 'puckboxes') else '{}'))
        except json.JSONDecodeError as e:
            self._json({'ok': False, 'error': f'bad json: {e}'}, 400); return

        if cfg['kind'] == 'puckboxes' and isinstance(data, dict) and data.get('none'):
            out = DATA_DIR / cfg['data'].split('/', 1)[1] / f"{frame}.json"
            out.write_text(json.dumps({'none': True}))
            self._json({'ok': True, 'saved': 0}); return

        if cfg['kind'] in ('boxes', 'puckboxes'):
            small = cfg['kind'] == 'puckboxes'
            # Read the actual image size for this frame
            img_path = BASE_DIR / cfg['dir'] / f"{frame}.jpg"
            if img_path.exists():
                try:
                    from PIL import Image as _Img
                    _im = _Img.open(img_path)
                    IMG_W, IMG_H = _im.size
                except Exception:
                    IMG_W, IMG_H = DEFAULT_IMG_W, DEFAULT_IMG_H
            else:
                IMG_W, IMG_H = DEFAULT_IMG_W, DEFAULT_IMG_H
            clean = []
            for b in data if isinstance(data, list) else []:
                if not all(k in b for k in ('x1', 'y1', 'x2', 'y2')):
                    continue
                x1 = max(0, min(IMG_W, int(b['x1'])))
                y1 = max(0, min(IMG_H, int(b['y1'])))
                x2 = max(0, min(IMG_W, int(b['x2'])))
                y2 = max(0, min(IMG_H, int(b['y2'])))
                w, h = x2 - x1, y2 - y1
                if small:
                    if w < 5 or h < 5 or w > 240 or h > 240:
                        continue
                else:
                    if w < 20 or h < 30:
                        continue
                    if h < 50 and w > h * 4:
                        continue
                    if w > IMG_W * 0.5 or h > IMG_H * 0.8:
                        continue
                clean.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
            out = DATA_DIR / cfg['data'].split('/', 1)[1] / f"{frame}.json"
            out.write_text(json.dumps(clean, indent=2))
            self._json({'ok': True, 'saved': len(clean)})
        elif cfg['kind'] == 'label':
            code = data.get('label') if isinstance(data, dict) else None
            clean = {'label': code if code in TEAM_CODES else None}
            out = DATA_DIR / cfg['data'].split('/', 1)[1] / f"{frame}.json"
            out.write_text(json.dumps(clean, indent=2))
            self._json({'ok': True, 'saved': 1 if clean['label'] else 0})
        elif cfg['kind'] == 'number':
            raw = data.get('label') if isinstance(data, dict) else None
            code = valid_number(raw) if raw is not None else None
            if raw is not None and code is None:
                self._json({'ok': False, 'error': 'bad number label'}, 400); return
            out = DATA_DIR / cfg['data'].split('/', 1)[1] / f"{frame}.json"
            clean = {'label': code}
            # keep the OCR hint if it was present in the file
            if out.exists():
                try:
                    old = json.loads(out.read_text())
                    for k in ('prefill', 'pconf'):
                        if isinstance(old, dict) and k in old:
                            clean[k] = old[k]
                except Exception:
                    pass
            out.write_text(json.dumps(clean, indent=2))
            self._json({'ok': True, 'saved': 1 if clean['label'] is not None else 0})
        elif cfg['kind'] == 'sound':
            code = data.get('label') if isinstance(data, dict) else None
            note = (data.get('note') or '')[:200] if isinstance(data, dict) else ''
            clean = {'label': code if code in SOUND_CODES else None, 'note': note}
            out = DATA_DIR / cfg['data'].split('/', 1)[1] / f"{frame}.json"
            out.write_text(json.dumps(clean, indent=2))
            self._json({'ok': True, 'saved': 1 if clean['label'] else 0})
        else:
            clean = {}
            if isinstance(data, dict):
                # For calib, also read the actual image size
                img_path_c = BASE_DIR / cfg['dir'] / f"{frame}.jpg"
                if img_path_c.exists():
                    try:
                        from PIL import Image as _ImgC
                        _imc = _ImgC.open(img_path_c)
                        IMG_W, IMG_H = _imc.size
                    except Exception:
                        IMG_W, IMG_H = DEFAULT_IMG_W, DEFAULT_IMG_H
                else:
                    IMG_W, IMG_H = DEFAULT_IMG_W, DEFAULT_IMG_H
                for k, v in data.items():
                    if isinstance(v, (list, tuple, dict)):
                        if isinstance(v, dict) and 'x' in v and 'y' in v:
                            clean[k] = {'x': max(0, min(IMG_W, int(v['x']))),
                                        'y': max(0, min(IMG_H, int(v['y'])))}
                        elif len(v) == 2:
                            clean[k] = {'x': max(0, min(IMG_W, int(v[0]))),
                                        'y': max(0, min(IMG_H, int(v[1])))}
            out = DATA_DIR / cfg['data'].split('/', 1)[1] / f"{frame}.json"
            out.write_text(json.dumps(clean, indent=2))
            self._json({'ok': True, 'saved': len(clean)})

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path
        if path == "/config":
            self._json({'build': BUILD, 'modes': {
                'boxes': {'label': 'Boxes (referees)', 'kind': 'boxes'},
                'calib': {'label': 'Points (calibration)', 'kind': 'points'},
                'crops': {'label': 'Crops (teams)', 'kind': 'label'},
                'puck': {'label': 'Puck (boxes)', 'kind': 'puckboxes'},
                'audio': {'label': 'Audio (whistles/sirens)', 'kind': 'sound'},
                'numbers': {'label': 'Player numbers', 'kind': 'number'},
            }, 'rink_points': RINK_POINTS, 'rink_labels': RINK_LABELS, 'rink_short': RINK_SHORT, 'rink_hints': RINK_HINTS,
                'teams': {'codes': TEAM_CODES, 'labels': TEAM_LABELS},
                'sounds': {'codes': SOUND_CODES, 'labels': SOUND_LABELS}})
        elif path == "/frames":
            cfg = mode_cfg(q)
            base = DATA_DIR / cfg['data'].split('/', 1)[1]
            exts = ('*.mp3',) if cfg['kind'] == 'sound' else ('*.jpg',)
            files = []
            for e in exts:
                files += sorted((BASE_DIR / cfg['dir']).glob(e))
            frames = []
            for img in files:
                dfile = base / f"{img.stem}.json"
                n = 0
                if dfile.exists():
                    try:
                        d = json.loads(dfile.read_text())
                        if cfg['kind'] in ('label', 'sound'):
                            n = 1 if isinstance(d, dict) and d.get('label') else 0
                        elif cfg['kind'] == 'number':
                            n = 1 if isinstance(d, dict) and d.get('label') is not None else 0
                        else:
                            n = len(d)
                    except Exception:
                        pass
                frames.append({'name': img.stem, 'done': dfile.exists(), 'count': n})
            self._json(frames)
        elif path.startswith("/data/"):
            cfg = mode_cfg(q)
            base = DATA_DIR / cfg['data'].split('/', 1)[1]
            fname = re.sub(r'[^A-Za-z0-9_\-\.]', '', path[6:])
            fp = base / fname
            if not fp.exists() and cfg['kind'] == 'boxes':
                # no manual annotation yet — serve the auto-detector prefill (edited and saved into data/boxes)
                pf = DATA_DIR / 'prefill' / fname
                if pf.exists():
                    fp = pf
            if fp.exists() and fp.suffix == '.json':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(fp.read_bytes())
            else:
                self.send_error(404)
        elif path == "/all":
            # All annotations of a mode at once — needed for the "how many frames have this point" counter
            cfg = mode_cfg(q)
            base = DATA_DIR / cfg['data'].split('/', 1)[1]
            out = {}
            for f in sorted(base.glob('*.json')):
                try:
                    out[f.stem] = json.loads(f.read_text())
                except Exception:
                    out[f.stem] = {}
            self._json(out)
        else:
            super().do_GET()


if __name__ == "__main__":
    for d in ["data/boxes", "data/calib", "data/crops", "data/puck", "data/audio", "data/numbers",
              "images/boxes", "images/calib", "images/crops", "images/puck", "images/audio", "images/numbers"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
    bind = os.environ.get("BIND", "0.0.0.0")
    port = int(os.environ.get("PORT", "8090"))
    s = http.server.ThreadingHTTPServer((bind, port), Handler)
    print(f"Annotate platform on {bind}:{port} (boxes + calib + crops)")
    s.serve_forever()
