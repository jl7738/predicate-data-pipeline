"""CONCEPT TEST: the downloader against a local fake FDA server (404s, flaky 503s, HTML-with-200, resume)."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import make_cfg
from predx.fetch import folder_candidates, run_fetch
from predx.io import last_by_key

PDF = b"%PDF-1.4\n%fake\n"
HITS: dict = {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        HITS[self.path] = HITS.get(self.path, 0) + 1
        p = self.path
        if p.endswith("/pdf21/K210001.pdf") or p.endswith("/pdf5/K050002.pdf") or p.endswith("/pdf/K960003.pdf"):
            return self._send(200, PDF)
        if p.endswith("/pdf21/K210004.pdf"):                                   # flaky: 503, 503, then OK
            return self._send(200, PDF) if HITS[p] > 2 else self._send(503, b"busy")
        if p.endswith("K210005.pdf"):
            return self._send(200, b"<html>Page not found</html>")             # soft-404 served as 200
        self._send(404, b"nope")

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/cdrh_docs"
    srv.shutdown()


def test_folder_candidates_by_year():
    assert folder_candidates("K210001")[0] == "pdf21"
    assert folder_candidates("K050002")[0] == "pdf5" and "pdf05" in folder_candidates("K050002")
    assert folder_candidates("K960003")[0] == "pdf"


def test_downloads_retries_and_classifies(tmp_path, server):
    cfg, paths = make_cfg(tmp_path, [f"fetch.base_url={server}", "fetch.requests_per_second=1000", "fetch.max_retries=4"])
    ks = ["K210001", "K050002", "K960003", "K210004", "K210005", "K210006"]
    counts = run_fetch(cfg, paths, ks)
    st = last_by_key(paths.fetch_status)
    assert st["K210001"]["status"] == "ok" and paths.pdf("K210001").read_bytes() == PDF
    assert st["K050002"]["status"] == "ok" and st["K050002"]["folder"] == "pdf5"      # fell through to the right spelling
    assert st["K960003"]["status"] == "ok" and st["K960003"]["folder"] == "pdf"
    assert st["K210004"]["status"] == "ok"                                              # survived two 503s
    assert st["K210005"]["status"] == "not_found"                                       # HTML page with 200 is NOT a PDF
    assert st["K210006"]["status"] == "not_found"
    assert counts == {"ok": 4, "not_found": 2, "error": 0}


def test_resume_skips_done_and_known_404s(tmp_path, server):
    cfg, paths = make_cfg(tmp_path, [f"fetch.base_url={server}", "fetch.requests_per_second=1000"])
    run_fetch(cfg, paths, ["K210001", "K210006"])
    HITS.clear()
    again = run_fetch(cfg, paths, ["K210001", "K210006"])
    assert again == {"ok": 0, "not_found": 0, "error": 0} and not HITS                  # nothing re-requested
    run_fetch(cfg, paths, ["K210006"], retry_not_found=True)
    assert HITS                                                                          # explicit retry does hit the server
