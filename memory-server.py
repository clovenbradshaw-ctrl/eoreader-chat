#!/usr/bin/env python3
"""
EOReader Memory Server — thin HTTP wrapper for conversation memory.

Provides:
  GET /extract?url=<URL>     → { "text": "extracted content..." }
  GET /memory/list           → { "files": ["part_001.txt", ...] }
  GET /memory/<filename>     → raw text content
  POST /memory/save          → saves a summary file (JSON body: {filename, content})

Run:
  python3 memory-server.py [--port 8080] [--dir ./memory]

Then point the chat app at http://localhost:8080
"""

import argparse
import json
import os
import re
import hashlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

MEMORY_DIR = Path(__file__).parent / "memory"
MEMORY_DIR.mkdir(exist_ok=True)


def extract_text_from_url(url: str) -> str:
    """Fetch a URL and extract readable text. Falls back to raw content."""
    if not HAS_REQUESTS:
        return f"[memory-server] 'requests' library not installed. Run: pip install requests\nFetched URL: {url}"

    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "EOReader-Memory/1.0"
        })
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")

        if "text/html" in content_type or "application/xhtml" in content_type:
            return _strip_html(resp.text)
        elif "text/plain" in content_type or "text/markdown" in content_type:
            return resp.text
        else:
            return resp.text[:5000]
    except Exception as e:
        return f"[memory-server] Error fetching {url}: {e}"


def _strip_html(html: str) -> str:
    """Minimal HTML-to-text: strip tags, collapse whitespace."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class MemoryHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        if path == "/extract":
            self._handle_extract(params)
        elif path == "/memory/list":
            self._handle_list()
        elif path.startswith("/memory/"):
            self._handle_serve(path)
        elif path == "/" or path == "":
            self._send_json({"status": "ok", "service": "eoreader-memory-server", "endpoints": ["/extract?url=", "/memory/list", "/memory/<file>"]})
        else:
            self._send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/memory/save":
            self._handle_save()
        else:
            self._send_error(404, "Not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _handle_extract(self, params):
        url = params.get("url", [None])[0]
        if not url:
            self._send_error(400, "Missing ?url= parameter")
            return
        url = unquote(url)
        text = extract_text_from_url(url)
        self._send_json({"text": text, "url": url, "length": len(text)})

    def _handle_list(self):
        files = sorted([f.name for f in MEMORY_DIR.iterdir() if f.is_file()])
        self._send_json({"files": files, "count": len(files)})

    def _handle_serve(self, path):
        filename = path.split("/memory/", 1)[-1]
        filename = unquote(filename)
        filepath = MEMORY_DIR / filename
        if not filepath.exists() or not filepath.is_file():
            self._send_error(404, f"Memory file '{filename}' not found")
            return
        content = filepath.read_text(encoding="utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _handle_save(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
            return
        filename = data.get("filename", "")
        content = data.get("content", "")
        if not filename or not content:
            self._send_error(400, "Missing filename or content")
            return
        filename = re.sub(r"[^a-zA-Z0-9_\-.]", "_", filename)
        filepath = MEMORY_DIR / filename
        filepath.write_text(content, encoding="utf-8")
        self._send_json({"saved": filename, "length": len(content)})

    def _send_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, message):
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        print(f"[memory] {args[0]}")


def main():
    global MEMORY_DIR
    parser = argparse.ArgumentParser(description="EOReader Memory Server")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--dir", type=str, default=str(MEMORY_DIR), help="Memory directory")
    args = parser.parse_args()

    MEMORY_DIR = Path(args.dir)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    server = HTTPServer(("0.0.0.0", args.port), MemoryHandler)
    print(f"EOReader Memory Server listening on http://0.0.0.0:{args.port}")
    print(f"Memory directory: {MEMORY_DIR}")
    print(f"requests library: {'available' if HAS_REQUESTS else 'NOT installed (pip install requests)'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
