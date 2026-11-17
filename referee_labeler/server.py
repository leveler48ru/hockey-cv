#!/usr/bin/env python3
"""Internal HTTP server for referee labeling (LAN only)."""
import http.server
import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qs
import cgi

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

class RefereeHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def end_headers(self):
        # cache disable: Telegram-WebView aggressively caches HTML/JS
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        if path == "/save":
            ctype = self.headers.get('Content-Type', '')
            if 'multipart/form-data' in ctype:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST',
                             'CONTENT_TYPE': ctype})
                frame = form.getvalue('frame', '')
                boxes_json = form.getvalue('boxes', '[]')
            else:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8')
                params = parse_qs(body)
                frame = params.get('frame', [''])[0]
                boxes_json = params.get('boxes', ['[]'])[0]
            
            # Sanitize the frame name
            frame = re.sub(r'[^A-Za-z0-9_\-]', '', frame)
            if not frame:
                self.send_error(400, "Invalid frame name")
                return
            
            try:
                boxes = json.loads(boxes_json)
                if not isinstance(boxes, list):
                    raise ValueError("boxes must be array")
            except (json.JSONDecodeError, ValueError) as e:
                self.send_error(400, f"Invalid JSON: {e}")
                return
            
            # Sanitize coordinates
            clean = []
            for b in boxes:
                if all(k in b for k in ('x1', 'y1', 'x2', 'y2')):
                    x1 = max(0, min(1280, int(b['x1'])))
                    y1 = max(0, min(720, int(b['y1'])))
                    x2 = max(0, min(1280, int(b['x2'])))
                    y2 = max(0, min(720, int(b['y2'])))
                    w = x2 - x1
                    h = y2 - y1
                    # drop junk: zero/too-small geometry and
                    # flat wide strips (artifacts of horizontal swipes)
                    if w < 20 or h < 30:
                        continue
                    if h < 50 and w > h * 4:
                        continue
                    clean.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
            
            out_path = DATA_DIR / f"{frame}.json"
            out_path.write_text(json.dumps(clean, indent=2))
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True, 'saved': len(clean)}).encode())
        else:
            self.send_error(404, "Not found")

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == "/frames":
            frames = []
            for img in sorted(IMAGES_DIR.glob("*.jpg")):
                base = img.stem
                data_file = DATA_DIR / f"{base}.json"
                done = data_file.exists()
                count = 0
                if done:
                    try:
                        count = len(json.loads(data_file.read_text()))
                    except:
                        pass
                frames.append({'name': base, 'done': done, 'boxes': count})
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(frames).encode())
        elif path.startswith("/data/"):
            # Serve the labeling JSON
            fname = path[6:]
            fname = re.sub(r'[^A-Za-z0-9_\-\.]', '', fname)
            fpath = DATA_DIR / fname
            if fpath.exists() and fpath.suffix == '.json':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(fpath.read_bytes())
            else:
                self.send_error(404, "Not found")
        else:
            super().do_GET()

if __name__ == "__main__":
    server = http.server.HTTPServer(('127.0.0.1', 8090), RefereeHandler)
    print("Referee labeler running on http://127.0.0.1:8090")
    server.serve_forever()
