"""server: `entail serve`, the platform's local web server (LIBRARY_DESIGN.md 13.2, 13.5; ROADMAP product track P2).

It reads the record files of one log folder (entail_logs/ in the current folder by default) and serves:
  /                         the node view (static/index.html, app.js, style.css)
  /api/runs                 the launches, newest first
  /api/graph?run=R          a launch's workflow: nodes with state and progress, flows, where meaning broke
  /api/node?run=R&node=N    one node's decisions (declared, chosen, rule, resolution, note), counts and timing
  /api/locate?run=R         where meaning broke (record.locate, LIBRARY_DESIGN.md 12)
  /api/events?run=R         server-sent events: the launch's graph again whenever lines are added
  /api/safe-mode            the safety mode the next start uses (safe_mode.json) and the selective safe path's state
  POST /api/safe-mode       a write: {"mode": "off" | "auto" | "all"} into <folder>/safe_mode.json
  /api/nodes                the custom nodes turned off (nodes.json; entail/nodes.py)
  POST /api/nodes           a write: {"node": "rag.answer", "on": false} into <folder>/nodes.json
It is bound to 127.0.0.1 and never runs inside an engine, so it adds nothing to an engine's cost. It writes two files of
the log folder and nothing else - the safety mode of the next start, and which custom nodes are off (a running program
reads that within a second) - and touches no engine (LIBRARY_DESIGN.md 13.2; the second write came with P5). A request
whose Host is not this server's own address is refused (a page on another site could otherwise reach it through a name
that resolves to 127.0.0.1). A write also needs this server's current token - printed when it starts, put in its own
page, and a new one after each write - and an Origin that is this server, so that a page on another site cannot change
anything. The Python standard library only.
"""
import hmac
import json
import os
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from . import graph
from ..nodes import valid_node
from ..safe_mode import MODES, STORE

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
FILES = {"/": ("index.html", "text/html; charset=utf-8"), "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
         "/static/style.css": ("style.css", "text/css; charset=utf-8")}
POLL_S = 0.5        # how often the event stream looks for new lines
KEEPALIVE_S = 15.0  # a comment line so proxies and the browser keep the stream open
MAX_BODY = 4096     # the one write's body is a few bytes
TOKEN_SLOT = b"__ENTAIL_TOKEN__"   # where the page gets this server's token


class Store:
    """The record lines of a folder, read incrementally: each file from where the last read stopped, whole lines
    only (a line still being written waits for its newline). A record file it cannot read (a broken link, no
    permission) is kept in `unreadable` with the reason, and said by the API and the page, never skipped quietly."""

    def __init__(self, folder: str):
        self.folder = folder
        self.lock = threading.Lock()
        self.offsets: Dict[str, int] = {}
        self.lines: List[Tuple[str, dict]] = []
        self.version = 0
        self._groups = None
        self.unreadable: Dict[str, str] = {}

    def refresh(self) -> int:
        """Read what was added since the last call; returns the store's version (it grows when lines came in)."""
        with self.lock:
            added = False
            for path in graph.record_files(self.folder):
                if not os.path.isfile(path):              # a broken link (it follows it), or not a file at all
                    self.unreadable[path] = "not a readable file (a broken link, or not a file)"
                    continue
                try:
                    size = os.path.getsize(path)
                    start = self.offsets.get(path, 0)
                    if size < start:                      # the file was replaced: read it again from the start
                        self.lines = [(p, o) for p, o in self.lines if p != path]
                        start = 0
                    if size == start:
                        self.unreadable.pop(path, None)
                        continue
                    with open(path, "rb") as f:
                        f.seek(start)
                        data = f.read(size - start)
                    self.unreadable.pop(path, None)
                except OSError as e:
                    self.unreadable[path] = f"{type(e).__name__}: {e.strerror or e}"
                    continue
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


class SafeWrite:
    """The platform's one write (LIBRARY_DESIGN.md 13.2, 13.6): the safety mode of the next start, in
    <folder>/safe_mode.json, where safe_mode.mode() reads it when ENTAIL_SAFE is not set. A write needs the current
    token; each write replaces it, so a token works once."""

    def __init__(self, folder: str):
        self.folder = folder
        self.path = os.path.join(folder, "safe_mode.json")
        self.lock = threading.Lock()
        self.token = secrets.token_urlsafe(24)

    def mode(self) -> str:
        try:
            with open(self.path, encoding="utf-8") as f:
                m = json.load(f).get("mode")
            return m if m in MODES else "auto"
        except (OSError, ValueError, AttributeError):
            return "auto"

    def state(self) -> dict:
        """The mode the file says, and the selective safe path's configurations (safe_paths.json), newest first."""
        paths = []
        try:
            with open(os.path.join(self.folder, STORE), encoding="utf-8") as f:
                store = json.load(f)
            for key, e in (store.items() if isinstance(store, dict) else ()):
                if isinstance(e, dict):
                    paths.append({"key": key, **{k: e.get(k) for k in ("engine", "model", "status", "pairs",
                                                                       "candidates", "tried", "off", "updated")}})
        except (OSError, ValueError):
            pass
        paths.sort(key=lambda e: -(e.get("updated") or 0))
        return {"mode": self.mode(), "file": self.path, "set": os.path.exists(self.path), "paths": paths}

    def write(self, token: str, mode: str) -> Optional[str]:
        """Writes the mode when `token` is the current one; returns the next token, or None (not the token)."""
        return self._write(token, self.path, lambda: {"mode": mode})

    def nodes_state(self) -> dict:
        """The custom nodes turned off (nodes.json)."""
        path = os.path.join(self.folder, "nodes.json")
        try:
            with open(path, encoding="utf-8") as f:
                off = json.load(f).get("off", [])
        except (OSError, ValueError, AttributeError):
            off = []
        return {"off": sorted({n for n in off if isinstance(n, str)}), "file": path}

    def write_node(self, token: str, node: str, on: bool) -> Optional[str]:
        """Turns one custom node on or off when `token` is the current one; returns the next token, or None."""
        def content():
            off = set(self.nodes_state()["off"])
            (off.discard if on else off.add)(node)
            return {"off": sorted(off)}

        return self._write(token, os.path.join(self.folder, "nodes.json"), content)

    def _write(self, token: str, path: str, content) -> Optional[str]:
        with self.lock:
            if not hmac.compare_digest(token.encode("utf-8"), self.token.encode("utf-8")):
                return None
            os.makedirs(self.folder, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({**content(), "by": "entail serve", "t": round(time.time(), 3)}, f)
            os.replace(tmp, path)
            self.token = secrets.token_urlsafe(24)
            return self.token


class Handler(BaseHTTPRequestHandler):
    server_version = "entail"
    store: Store = None          # set by make_server
    safe: SafeWrite = None
    allowed_hosts: Tuple[str, ...] = ()
    allowed_origins: Tuple[str, ...] = ()

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
                    body = f.read()
            except OSError:
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            if name == "index.html":     # the page carries the token for the one write (another site cannot read it)
                body = body.replace(TOKEN_SLOT, self.safe.token.encode("ascii"))
            return self._send(HTTPStatus.OK, body, ctype)
        if not url.path.startswith("/api/"):
            return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        if url.path == "/api/safe-mode":
            return self._json(self.safe.state())
        if url.path == "/api/nodes":
            return self._json(self.safe.nodes_state())
        self.store.refresh()
        if url.path == "/api/runs":
            return self._json({"folder": self.store.folder, "folder_found": os.path.isdir(self.store.folder),
                               "unreadable": [{"file": os.path.basename(p), "why": w}
                                              for p, w in sorted(self.store.unreadable.items())],
                               "runs": graph.summaries(self.store.groups())})
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

    def do_POST(self):   # noqa: N802 - the http.server interface
        """The writes: POST /api/safe-mode {"mode": ...} and POST /api/nodes {"node": ..., "on": ...}, each with header
        X-Entail-Token."""
        if not self._host_ok():
            return self._send(HTTPStatus.FORBIDDEN, b"forbidden: not this server's address", "text/plain")
        where = urlparse(self.path).path
        if where not in ("/api/safe-mode", "/api/nodes"):
            return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        if (self.headers.get("Origin") or "") not in self.allowed_origins:
            return self._json({"error": "the request must come from this server's page (Origin)"},
                              HTTPStatus.FORBIDDEN)
        try:
            size = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            size = -1
        if size < 0 or size > MAX_BODY:
            return self._json({"error": f"the body must be at most {MAX_BODY} bytes"},
                              HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        try:
            body = json.loads(self.rfile.read(size).decode("utf-8"))
            body = body if isinstance(body, dict) else {}
        except (ValueError, UnicodeDecodeError):
            body = {}
        given = self.headers.get("X-Entail-Token") or ""
        if where == "/api/safe-mode":
            if body.get("mode") not in MODES:
                return self._json({"error": f"mode must be one of {', '.join(MODES)}"}, HTTPStatus.BAD_REQUEST)
            write, answer = (lambda: self.safe.write(given, body["mode"])), self.safe.state
        else:
            if not valid_node(body.get("node")) or not isinstance(body.get("on"), bool):
                return self._json({"error": "give a custom node's name (dotted lower-case letters, digits, '_' and "
                                            "'-') and on: true or false"}, HTTPStatus.BAD_REQUEST)
            write, answer = (lambda: self.safe.write_node(given, body["node"], body["on"])), self.safe.nodes_state
        try:
            token = write()
        except OSError as e:
            return self._json({"error": f"could not write in {self.safe.folder}: {e}"},
                              HTTPStatus.INTERNAL_SERVER_ERROR)
        if token is None:
            return self._json({"error": "not this server's current token: it changes after each write (reload the "
                                        "page, or use the one the last write returned)"}, HTTPStatus.FORBIDDEN)
        return self._json({**answer(), "token": token})

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
    handler = type("EntailHandler", (Handler,), {"store": store, "safe": SafeWrite(folder)})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    real = server.server_address[1]
    handler.allowed_hosts = (f"127.0.0.1:{real}", f"localhost:{real}")
    handler.allowed_origins = (f"http://127.0.0.1:{real}", f"http://localhost:{real}")
    server.safe = handler.safe
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
    print(f"[entail] serving {folder} at {url} (Ctrl-C stops; this machine only)", flush=True)
    # what the token is for, so that nobody wonders what to do with it (field test, entail#7): nothing, on the page
    print(f"[entail] it reads the records; the page itself writes its two settings there (the safety mode of the next "
          f"start, which custom nodes are off) - nothing to do here. Only a script that changes them needs this "
          f"token, which works once: {server.safe.token}", flush=True)
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
