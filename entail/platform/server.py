"""server: `entail serve`, the platform's local web server (LIBRARY_DESIGN.md 13.2, 13.5; ROADMAP product track P2).

It reads the record files of one log folder (entail_logs/ in the current folder by default) and serves:
  /                         the node view (static/index.html, app.js, style.css)
  /api/runs                 the launches, newest first
  /api/graph?run=R          a launch's workflow: nodes with state and progress, flows, where meaning broke
  /api/node?run=R&node=N    one node's decisions (declared, chosen, rule, resolution, note), counts and timing
  /api/locate?run=R         where meaning broke (record.locate, LIBRARY_DESIGN.md 12)
  /api/events?run=R         server-sent events: the launch's graph again whenever lines are added
It is read-only and bound to 127.0.0.1: it never writes, and it never runs inside an engine, so it adds nothing to an
engine's cost. A request whose Host is not this server's own address is refused (a page on another site could
otherwise reach it through a name that resolves to 127.0.0.1). The Python standard library only.
"""
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from . import graph

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
FILES = {"/": ("index.html", "text/html; charset=utf-8"), "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
         "/static/style.css": ("style.css", "text/css; charset=utf-8")}
POLL_S = 0.5        # how often the event stream looks for new lines
KEEPALIVE_S = 15.0  # a comment line so proxies and the browser keep the stream open


class Store:
    """The record lines of a folder, read incrementally: each file from where the last read stopped, whole lines
    only (a line still being written waits for its newline)."""

    def __init__(self, folder: str):
        self.folder = folder
        self.lock = threading.Lock()
        self.offsets: Dict[str, int] = {}
        self.lines: List[Tuple[str, dict]] = []
        self.version = 0
        self._groups = None

    def refresh(self) -> int:
        """Read what was added since the last call; returns the store's version (it grows when lines came in)."""
        with self.lock:
            added = False
            for path in graph.record_files(self.folder):
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                start = self.offsets.get(path, 0)
                if size < start:                      # the file was replaced: read it again from the start
                    self.lines = [(p, o) for p, o in self.lines if p != path]
                    start = 0
                if size == start:
                    continue
                with open(path, "rb") as f:
                    f.seek(start)
                    data = f.read(size - start)
                end = data.rfind(b"\n")
                if end < 0:
                    continue
                for raw in data[:end].split(b"\n"):
                    try:
                        obj = json.loads(raw.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(obj, dict):
                        self.lines.append((path, obj))
                        added = True
                self.offsets[path] = start + end + 1
            if added:
                self.version += 1
                self._groups = None
            return self.version

    def groups(self) -> Dict[str, List[dict]]:
        with self.lock:
            if self._groups is None:
                self._groups = graph.launches(self.lines)
            return self._groups

    def run(self, key: Optional[str]) -> Optional[List[dict]]:
        groups = self.groups()
        if key:
            return groups.get(key)
        if not groups:
            return None
        newest = graph.summaries(groups)[0]["run"]
        return groups[newest]


class Handler(BaseHTTPRequestHandler):
    server_version = "entail"
    store: Store = None          # set by make_server
    allowed_hosts: Tuple[str, ...] = ()

    def log_message(self, fmt, *args):   # the console is the user's; requests are not worth a line each
        pass

    def _host_ok(self) -> bool:
        return (self.headers.get("Host") or "") in self.allowed_hosts

    def _send(self, status, body: bytes, ctype: str):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=HTTPStatus.OK):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):   # noqa: N802 - the http.server interface
        if not self._host_ok():
            return self._send(HTTPStatus.FORBIDDEN, b"forbidden: not this server's address", "text/plain")
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path in FILES:
            name, ctype = FILES[url.path]
            try:
                with open(os.path.join(STATIC, name), "rb") as f:
                    return self._send(HTTPStatus.OK, f.read(), ctype)
            except OSError:
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        if not url.path.startswith("/api/"):
            return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        self.store.refresh()
        if url.path == "/api/runs":
            return self._json({"folder": self.store.folder, "runs": graph.summaries(self.store.groups())})
        lines = self.store.run(q.get("run"))
        if lines is None:
            return self._json({"error": "no such launch", "run": q.get("run")}, HTTPStatus.NOT_FOUND)
        if url.path == "/api/graph":
            return self._json(graph.graph(lines))
        if url.path == "/api/node":
            return self._json(graph.node_detail(lines, q.get("node", "")))
        if url.path == "/api/locate":
            return self._json(graph.graph(lines)["locate"])
        if url.path == "/api/events":
            return self._events(q.get("run"))
        return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def _events(self, run: Optional[str]):
        """Server-sent events: the launch's graph now, then again whenever lines are added; a comment line keeps
        the stream open. Ends when the client goes away or the server stops."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seen, quiet = None, time.monotonic()
        try:
            while not getattr(self.server, "stopping", False):
                version = self.store.refresh()
                if version != seen:
                    seen = version
                    lines = self.store.run(run)
                    if lines is not None:
                        data = json.dumps(graph.graph(lines), ensure_ascii=False)
                        self.wfile.write(f"event: graph\ndata: {data}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        quiet = time.monotonic()
                elif time.monotonic() - quiet > KEEPALIVE_S:
                    self.wfile.write(b": still here\n\n")
                    self.wfile.flush()
                    quiet = time.monotonic()
                time.sleep(POLL_S)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


def make_server(folder: str, port: int = 8765) -> ThreadingHTTPServer:
    """A server over `folder`, bound to 127.0.0.1 (port 0: any free port). Not started: call serve_forever()."""
    store = Store(folder)
    handler = type("EntailHandler", (Handler,), {"store": store})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    real = server.server_address[1]
    handler.allowed_hosts = (f"127.0.0.1:{real}", f"localhost:{real}")
    store.refresh()
    return server


def serve(folder: str, port: int = 8765, open_browser: bool = False) -> int:
    """`entail serve`: run the server until Ctrl-C."""
    try:
        server = make_server(folder, port)
    except OSError as e:
        print(f"[entail] could not listen on 127.0.0.1:{port}: {e}; choose another with --port")
        return 2
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[entail] serving {folder} at {url} (Ctrl-C stops; read-only, this machine only)", flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stopping = True
        server.server_close()
    return 0
