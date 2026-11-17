#!/usr/bin/env python3
"""Ice homography calibration: placing rink marking anchor points on a single video frame."""
import http.server
import json
import re
from pathlib import Path
import cgi

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Standard IIHF rink: 60x30 m. Coordinates in meters, origin at the ice center.
# Rink marking anchor points (real coordinates in meters):
# classes: red_line(0), blue_line(1), center_circle(2), faceoff_spot(3), faceoff_circle(4), goal_crease(5), board(6)
RINK_POINTS = {
    "center": [0, 0],                    # rink center
    "red_left": [-0.15, -15],            # red line left edge (thickness ~0.3 m, taking the center)
    "red_right": [-0.15, 15],            # red line right edge
    "blue_near_left": [-17.5, -15],      # near blue line left
    "blue_near_right": [-17.5, 15],      # near blue line right
    "blue_far_left": [17.5, -15],        # far blue line left
    "blue_far_right": [17.5, 15],        # far blue line right
    "faceoff_center_top": [0, -4.5],     # faceoff spot center top
    "faceoff_center_bot": [0, 4.5],      # faceoff spot center bottom
    "faceoff_near_left": [-22, -7],      # near zone left
    "faceoff_near_right": [-22, 7],      # near zone right
    "faceoff_far_left": [22, -7],        # far zone left
    "faceoff_far_right": [22, 7],        # far zone right
    "goal_near": [-28.5, 0],             # near goal center
    "goal_far": [28.5, 0],               # far goal center
}

class CalibHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        if path == "/save":
            ctype = self.headers.get('Content-Type', '')
            if 'multipart/form-data' in ctype:
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': ctype})
                frame = form.getvalue('frame', '')
                points_json = form.getvalue('points', '{}')
            else:
                cl = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(cl).decode('utf-8')
                from urllib.parse import parse_qs
                p = parse_qs(body)
                frame = p.get('frame', [''])[0]
                points_json = p.get('points', ['{}'])[0]

            frame = re.sub(r'[^A-Za-z0-9_\-]', '', frame)
            if not frame:
                self.send_error(400, "Invalid frame"); return
            try:
                pts = json.loads(points_json)
            except:
                self.send_error(400, "Bad JSON"); return

            out = DATA_DIR / f"{frame}.json"
            out.write_text(json.dumps(pts, indent=2))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True, 'saved': len(pts)}).encode())
        else:
            self.send_error(404)

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == "/frames":
            frames = []
            for img in sorted(IMAGES_DIR.glob("*.jpg")):
                base = img.stem
                df = DATA_DIR / f"{base}.json"
                done = df.exists()
                n = 0
                if done:
                    try: n = len(json.loads(df.read_text()))
                    except: pass
                frames.append({'name': base, 'done': done, 'points': n})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(frames).encode())
        elif path == "/rink_points":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(RINK_POINTS).encode())
        elif path.startswith("/data/"):
            fname = re.sub(r'[^A-Za-z0-9_\-\.]', '', path[6:])
            fp = DATA_DIR / fname
            if fp.exists() and fp.suffix == '.json':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(fp.read_bytes())
            else:
                self.send_error(404)
        else:
            super().do_GET()

if __name__ == "__main__":
    s = http.server.HTTPServer(('127.0.0.1', 8091), CalibHandler)
    print("Calibration running on http://127.0.0.1:8091")
    s.serve_forever()
