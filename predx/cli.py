"""Command line:  python -m predx.cli [--config FILE ...] [--set a.b=c ...] <command> [options]

  meta          download openFDA 510(k) metadata
  plan          build a queue of K-numbers: --stage pilot|subjects|closure|full
  fetch         download PDFs for a queue
  text          PDFs -> cached text (OCR fallback)
  parse         cached text -> sections + predicates
  build         assemble corpus / queries / splits / train pairs
  validate      schema + invariants on the built dataset
  audit         full audit report (data/reports/audit.md)
  label-sample / label-eval   100-summary hand-check of the extraction
  pilot         meta -> pilot queue -> fetch/text/parse -> predicate closure -> build -> audit   (use --config config/pilot.yaml)
  run-all       the full run, stage by stage, resumable
  demo          offline end-to-end run on synthetic data (conceptual tests; use --config config/demo.yaml)
  status        what exists on disk
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from .config import Paths, load_config


def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def _process(cfg, paths, ks):
    from .fetch import run_fetch
    from .parse_stage import run_parse
    from .text_stage import run_text

    run_fetch(cfg, paths, ks)
    run_text(cfg, paths, ks)
    run_parse(cfg, paths)


def _print_summary(rep):
    print(f"\nOVERALL: {rep['overall']}")
    for c in rep["checks"]:
        if c["status"] != "PASS":
            v = c["value"]
            v = f"{v:.3f}" if isinstance(v, float) else v
            print(f"  [{c['status']}] {c['description']}  value={v} threshold={c['threshold']} {c['note']}")
    print(f"  ({sum(c['status'] == 'PASS' for c in rep['checks'])}/{len(rep['checks'])} checks PASS)")


def _build_validate_audit(cfg, paths, args):
    from .audit import run_audit, write_report
    from .dataset import build_dataset, validate_dataset

    build_dataset(cfg, paths, getattr(args, "mask_level", None), getattr(args, "subjects_queue", None))
    val = validate_dataset(paths, cfg)
    if val["problems"]:
        print("VALIDATION PROBLEMS:", json.dumps(val["problems"], indent=1)[:2000])
    rep = run_audit(cfg, paths, baselines=not getattr(args, "no_baselines", False), tokenizers=not getattr(args, "no_tokenizers", False))
    write_report(rep, paths)
    _print_summary(rep)
    print(f"\nreport: {paths.reports / 'audit.md'}")
    return rep


def main(argv=None):
    _setup_logging()
    ap = argparse.ArgumentParser(prog="predx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[], help="YAML file(s) layered over config/default.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. --set fetch.requests_per_second=2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("meta").add_argument("--force", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("--stage", required=True, choices=["pilot", "subjects", "closure", "full"])
    p.add_argument("--of", help="closure: only predicates cited by subjects in this queue")
    p.add_argument("--name", default=None)
    p.add_argument("--panel")
    p.add_argument("--n", type=int)
    p = sub.add_parser("fetch")
    p.add_argument("--queue", required=True)
    p.add_argument("--retry-not-found", action="store_true")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("text")
    p.add_argument("--queue", default="all", help="queue name, or 'all' = every PDF on disk")
    p.add_argument("--force", action="store_true")
    sub.add_parser("parse")
    p = sub.add_parser("build")
    p.add_argument("--mask-level", choices=["none", "standard", "strict"])
    p.add_argument("--subjects-queue", help="restrict subjects (queries) to this queue, e.g. pilot_subjects")
    sub.add_parser("validate")
    p = sub.add_parser("audit")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--no-tokenizers", action="store_true")
    p = sub.add_parser("label-sample")
    p.add_argument("--n", type=int)
    p.add_argument("--no-fetch", action="store_true")
    p = sub.add_parser("label-eval")
    p.add_argument("--csv")
    p = sub.add_parser("pilot")
    p.add_argument("--panel")
    p.add_argument("--n", type=int)
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--no-tokenizers", action="store_true")
    p = sub.add_parser("run-all")
    p.add_argument("--stages", default="subjects,closure,full,build,audit")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--no-tokenizers", action="store_true")
    p.add_argument("--mask-level", choices=["none", "standard", "strict"])
    p = sub.add_parser("demo")
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--tiny", action="store_true", help="60 devices: should FAIL the size checks (shows what insufficient data looks like)")
    p.add_argument("--with-pdfs", type=int, default=0, help="render N documents as real PDFs (1/3 scanned) and run the PDF/OCR stage on them")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--keep", action="store_true")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--no-tokenizers", action="store_true")
    p = sub.add_parser("probe", help="run the real pipeline on a few PDFs and print full diagnostics (first contact with real FDA PDFs)")
    p.add_argument("ks", nargs="*", help="K-numbers to download from FDA and probe")
    p.add_argument("--ks-file", help="text file with one K-number per line (# comments allowed)")
    p.add_argument("--pdf", action="append", default=[], help="a local PDF (file name should contain the K-number)")
    p.add_argument("--pdf-dir", help="probe every *.pdf in this folder")
    p.add_argument("--out", default="data_probe")
    p.add_argument("--no-fetch", action="store_true", help="do not download; only use PDFs already in --out/pdf")
    sub.add_parser("sizing", help="planning model: expected gold pairs, queries, recall ceiling, compute (uses real counts after `meta`)")
    p = sub.add_parser("ocr-stress", help="OCR robustness curve on synthetic PDFs with increasing scan degradation")
    p.add_argument("--n", type=int, default=12, help="documents per degradation level")
    p.add_argument("--levels", default="0,1,2,3")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default="data_stress")
    sub.add_parser("status")

    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    paths = Paths(cfg)
    c = args.cmd

    if c == "meta":
        from .metadata import download_meta

        print(download_meta(cfg, paths, args.force), "records")
    elif c == "plan":
        from . import plan as pl

        if args.stage == "pilot":
            pl.plan_pilot(cfg, paths, args.panel, args.n)
        elif args.stage == "subjects":
            pl.plan_subjects(cfg, paths)
        elif args.stage == "closure":
            pl.plan_closure(cfg, paths, args.of, args.name or "closure")
        else:
            pl.plan_full(cfg, paths)
    elif c == "fetch":
        from .fetch import run_fetch
        from .plan import read_queue

        print(run_fetch(cfg, paths, read_queue(paths, args.queue), args.retry_not_found, args.limit))
    elif c == "text":
        from .plan import read_queue
        from .text_stage import run_text

        ks = [f.stem for f in paths.pdf_dir.glob("*/*.pdf")] if args.queue == "all" else read_queue(paths, args.queue)
        print(run_text(cfg, paths, ks, args.force))
    elif c == "parse":
        from .parse_stage import run_parse

        run_parse(cfg, paths)
    elif c == "build":
        from .dataset import build_dataset

        build_dataset(cfg, paths, args.mask_level, args.subjects_queue)
    elif c == "validate":
        from .dataset import validate_dataset

        v = validate_dataset(paths, cfg)
        print(json.dumps(v, indent=1))
        sys.exit(1 if v["problems"] else 0)
    elif c == "audit":
        from .audit import run_audit, write_report

        rep = run_audit(cfg, paths, baselines=not args.no_baselines, tokenizers=not args.no_tokenizers)
        write_report(rep, paths)
        _print_summary(rep)
        print(f"\nreport: {paths.reports / 'audit.md'}")
    elif c == "label-sample":
        from .labeling import make_label_sample

        print(make_label_sample(cfg, paths, args.n, not args.no_fetch))
    elif c == "label-eval":
        from .labeling import eval_labels, format_label_eval

        print(format_label_eval(eval_labels(cfg, paths, args.csv)))
    elif c == "pilot":
        from . import plan as pl
        from .metadata import download_meta

        download_meta(cfg, paths)
        ks = pl.plan_pilot(cfg, paths, args.panel, args.n)
        _process(cfg, paths, ks)
        if cfg["pilot"]["expand_predicates"]:
            ck = pl.plan_closure(cfg, paths, "pilot_subjects", "pilot_closure")
            _process(cfg, paths, ck)
        args.subjects_queue = "pilot_subjects"
        _build_validate_audit(cfg, paths, args)
        print("\nNext: python -m predx.cli label-sample   (then hand-label ~100 subjects; see README)")
    elif c == "run-all":
        from . import plan as pl
        from .metadata import download_meta

        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        download_meta(cfg, paths)
        if "subjects" in stages:
            _process(cfg, paths, pl.plan_subjects(cfg, paths))
        if "closure" in stages:
            _process(cfg, paths, pl.plan_closure(cfg, paths))
        if "full" in stages:
            _process(cfg, paths, pl.plan_full(cfg, paths))
        if "build" in stages or "audit" in stages:
            _build_validate_audit(cfg, paths, args)
    elif c == "demo":
        from .parse_stage import run_parse
        from .synth import make_synthetic
        from .text_stage import run_text

        if paths.root.exists() and not args.keep:
            shutil.rmtree(paths.root)
        n = 60 if args.tiny else args.n
        info = make_synthetic(paths.root, n=n, seed=args.seed, with_pdfs=args.with_pdfs)
        if info["pdfs"]:
            print(run_text(cfg, paths, info["pdfs"]))
        run_parse(cfg, paths)
        _build_validate_audit(cfg, paths, args)
    elif c == "probe":
        from .probe import print_summary, run_probe

        ks = list(args.ks)
        if args.ks_file:
            ks += [ln.split("#")[0].strip() for ln in open(args.ks_file) if ln.split("#")[0].strip()]
        pdfs = [Path(x) for x in args.pdf]
        if args.pdf_dir:
            pdfs += sorted(Path(args.pdf_dir).expanduser().glob("*.pdf"))
        if not ks and not pdfs:
            ap.error("give K-numbers, --ks-file, --pdf or --pdf-dir")
        known = None
        if Path(paths.meta).exists():
            from .metadata import load_meta

            known = set(load_meta(paths).keys())
        recs = run_probe(cfg, ks, pdfs, Path(args.out), fetch=not args.no_fetch, known=known)
        print_summary(recs)
        print(f"\nreport: {Path(args.out) / 'probe_report.md'}   full text: {Path(args.out) / 'text'}/<K>.txt")
    elif c == "sizing":
        from .sizing import render_markdown, run_sizing, write_report

        rep = run_sizing(cfg, paths)
        out = write_report(rep, paths)
        print(render_markdown(rep))
        print(f"\nreport: {out}")
    elif c == "ocr-stress":
        from .stress import run_stress

        run_stress(cfg, [int(x) for x in args.levels.split(",")], args.n, args.seed, Path(args.out))
    elif c == "status":
        print(json.dumps({
            "meta": paths.meta.exists(),
            "queues": {q.stem: sum(1 for _ in open(q)) for q in paths.queues.glob("*.txt")} if paths.queues.exists() else {},
            "pdfs": sum(1 for _ in paths.pdf_dir.glob("*/*.pdf")) if paths.pdf_dir.exists() else 0,
            "text_cached": sum(1 for _ in paths.text_dir.glob("*/*.txt.gz")) if paths.text_dir.exists() else 0,
            "parsed": paths.parsed.exists(),
            "dataset": (paths.dataset / "dataset_card.json").exists(),
            "report": (paths.reports / "audit.md").exists(),
        }, indent=1))


if __name__ == "__main__":
    main()
