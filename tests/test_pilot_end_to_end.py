"""CONCEPT TEST: the real `meta` + `pilot` commands, against a local fake FDA (openFDA bulk file + PDF server).

This is the closest thing to the real run that works offline: metadata comes from a zipped openFDA-format JSON,
PDFs are fetched over HTTP, parsed from real PDF files, predicates are expanded one hop, and the audit is produced.
"""
import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from conftest import make_cfg
from predx.cli import main
from predx.fetch import folder_candidates
from predx.io import load_pages, read_jsonl
from predx.metadata import download_meta, normalise_record
from predx.plan import plan_closure, plan_pilot, plan_subjects, read_queue
from predx.synth import make_synthetic
from predx.synth_pdf import render_pdf

DEMO = Path(__file__).resolve().parents[1] / "config" / "demo.yaml"
PILOT = Path(__file__).resolve().parents[1] / "config" / "pilot.yaml"


def to_openfda(m):
    return {"k_number": m["k"], "device_name": m["device_name"], "applicant": m["applicant"], "product_code": m["product_code"],
            "review_advisory_committee": m["panel"], "decision_date": m["decision_date"].replace("-", ""), "date_received": m["date_received"],
            "decision_code": m["decision_code"], "decision_description": m["decision_description"], "clearance_type": m["clearance_type"],
            "statement_or_summary": "Summary" if m["statement_or_summary"] == "SUMMARY" else "Statement"}


@pytest.fixture(scope="module")
def fake_fda(tmp_path_factory):
    src = tmp_path_factory.mktemp("fda_side")
    cfg, paths = make_cfg(src)
    make_synthetic(src, n=500, seed=21)
    meta = list(read_jsonl(paths.meta))
    web = tmp_path_factory.mktemp("web")
    n_pdf = 0
    for f in paths.text_dir.glob("*/*.txt.gz"):
        k = f.name[:-7]
        dest = web / "cdrh_docs" / folder_candidates(k)[0] / f"{k}.pdf"
        render_pdf(load_pages(f), dest, scanned=False)
        n_pdf += 1
    part = io.BytesIO()
    with zipfile.ZipFile(part, "w") as z:
        z.writestr("device-510k-0001-of-0001.json", json.dumps({"meta": {}, "results": [to_openfda(m) for m in meta]}))
    (web / "part.zip").write_bytes(part.getvalue())

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/download.json":
                body = json.dumps({"results": {"device": {"510k": {"partitions": [{"file": f"http://127.0.0.1:{srv.server_address[1]}/part.zip"}]}}}}).encode()
            else:
                p = web / self.path.lstrip("/")
                if not p.is_file():
                    self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
                body = p.read_bytes()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield {"base": base, "meta": meta, "n_pdf": n_pdf, "truth": json.loads((src / "truth.json").read_text())}
    srv.shutdown()


def overrides(base, root):
    return ["--config", str(DEMO), "--config", str(PILOT), "--set", f"paths.data_dir={root}", "--set", f"openfda.download_index={base}/download.json",
            "--set", f"fetch.base_url={base}/cdrh_docs", "--set", "fetch.requests_per_second=1000", "--set", "fetch.trust_env=false",
            "--set", "text.workers=2", "--set", "pilot.n_subjects=40", "--set", "pilot.panel=RA"]


def test_normalise_openfda_record():
    r = normalise_record({"k_number": "k211140", "decision_date": "20211009", "statement_or_summary": "Summary", "product_code": "qas", "decision_code": "sese"})
    assert r["k"] == "K211140" and r["decision_date"] == "2021-10-09" and r["year"] == 2021
    assert r["statement_or_summary"] == "SUMMARY" and r["product_code"] == "QAS" and r["decision_code"] == "SESE"
    assert normalise_record({"decision_date": "2021-01-01"}) is None


def test_meta_download_plan_and_pilot_end_to_end(tmp_path, fake_fda):
    root = tmp_path / "pipe"
    ov = overrides(fake_fda["base"], root)
    cfg, paths = make_cfg(root, ["openfda.download_index=" + fake_fda["base"] + "/download.json"])
    assert download_meta(cfg, paths) == 500
    meta = {m["k"]: m for m in read_jsonl(paths.meta)}
    assert meta == {m["k"]: m for m in fake_fda["meta"]}                        # round trip through openFDA format is lossless

    main(ov + ["pilot", "--no-tokenizers"])

    q = read_queue(paths, "pilot_subjects")
    assert 0 < len(q) <= 40 and all(meta[k]["panel"] == "RA" for k in q)
    assert all(a <= meta[k]["decision_date"] <= b for k in q for a, b in [("2015-01-01", "2026-12-31")])
    closure = read_queue(paths, "pilot_closure")
    assert closure and not set(closure) & set(q)                                # predicates were fetched too (1 hop)
    assert paths.text(q[0]).exists() and paths.pdf(q[0]).exists()
    md = (paths.reports / "audit.md").read_text()
    assert "## 9. Projection to the full run" in md and "projected subjects" in md   # pilot knows it is a pilot
    rows = list(read_jsonl(paths.dataset / "device_status.jsonl.gz"))
    assert sum(r["status"] == "not_processed" for r in rows) > 300               # most of the universe was never touched
    # subjects only come from the pilot queue; everything else may appear as a candidate (closure) but not as a subject
    subj = {x["k"] for s in ("train", "dev", "test") for x in read_jsonl(paths.dataset / f"queries_{s}.jsonl.gz")}
    assert subj <= set(q)


def test_closure_contains_exactly_the_missing_cited_predicates(tmp_path, fake_fda):
    root = tmp_path / "pipe2"
    ov = overrides(fake_fda["base"], root)
    cfg, paths = make_cfg(root, ["openfda.download_index=" + fake_fda["base"] + "/download.json", "fetch.base_url=" + fake_fda["base"] + "/cdrh_docs",
                                 "fetch.requests_per_second=1000", "text.workers=2"])
    download_meta(cfg, paths)
    plan_pilot(cfg, paths, "RA", 30)
    main(ov + ["fetch", "--queue", "pilot_subjects"])
    main(ov + ["text", "--queue", "pilot_subjects"])
    main(ov + ["parse"])
    ck = set(plan_closure(cfg, paths, "pilot_subjects", "c"))
    truth, meta = fake_fda["truth"], {m["k"]: m for m in fake_fda["meta"]}
    want = set()
    for k in read_queue(paths, "pilot_subjects"):
        t = truth[k]
        if t["status"] != "ok" or not t["primary"]:
            continue
        for p in [t["primary"], *t["additional"]]:
            m = meta.get(p)
            if m and m["decision_code"] == "SESE" and m["statement_or_summary"] == "SUMMARY" and not paths.text(p).exists():
                want.add(p)
    assert ck == want and ck
