"""Synthetic 510(k) world with PLANTED structure, for offline conceptual tests (no network, no FDA data).

What is planted (and recorded in truth.json, so tests can check the pipeline recovers it exactly):
  * product-code families with shared vocabulary and predicate lineages (so BM25 has real signal)
  * citation graph with preferential attachment; primary / additional / reference roles; 3 predicate-list styles
  * failure modes: statement-only, non-SE decision, PDF 404, text-extraction failure, missing description section
  * citations to unknown K-numbers, PMA and De Novo numbers, and a few that point forward in time
  * leakage: some descriptions name a predicate or quote its K-number
  * heading-wording variants, FDA indications form page present/absent
Synthetic data is only for testing the PIPELINE. Never report metrics from it.
"""
from __future__ import annotations

import json
import math
import random
import textwrap
from datetime import date, timedelta
from pathlib import Path

from .io import append_jsonl, save_pages, write_jsonl

FAMILIES = {
    "QAS": ("RA", "computer-aided triage software", "chest ct pulmonary nodule radiologist triage dicom algorithm lung flag worklist notification".split()),
    "LLZ": ("RA", "image analysis software", "mammography breast lesion detection screening density segmentation workstation reader annotation".split()),
    "JAK": ("RA", "computed tomography x-ray system", "gantry detector tube rotation slice reconstruction kvp dose collimator scanner".split()),
    "DQY": ("CV", "percutaneous catheter", "guidewire vascular introducer lumen hydrophilic sheath femoral radial angiography hub".split()),
    "NLF": ("CV", "pulse oximeter", "oximetry spo2 sensor finger perfusion saturation probe wavelength photodiode cable".split()),
    "DXN": ("CV", "blood pressure monitor", "cuff oscillometric systolic diastolic arm bladder inflation transducer hypertension".split()),
    "GEX": ("SU", "surgical laser system", "laser wavelength fiber handpiece tissue ablation coagulation pulse energy aiming".split()),
    "FRO": ("SU", "wound dressing", "hydrocolloid absorbent adhesive exudate wound moist foam silver dressing border".split()),
    "OZO": ("OR", "bone fixation plate", "titanium screw plate locking fracture cortical osteosynthesis torque implant fixation".split()),
    "HWC": ("OR", "intramedullary nail", "nail tibia femur humerus reaming interlocking guide proximal distal canal".split()),
    "MNR": ("AN", "infusion pump", "infusion flow rate occlusion syringe bolus alarm keypad volume cassette".split()),
    "BZD": ("AN", "ventilator", "ventilation tidal pressure peep breath circuit flow trigger oxygen mode".split()),
}
BRANDS = "Avalen Corvia Medrix Zentra Lumora Vitaris Orbion Kelvra Nuvanta Syntral Hexaro Praxor Olmeda Tervia Qualix Brenox Dylara Fenwick Gyrona Halcyr".split()
APPLICANTS = [b + " Medical Inc." for b in BRANDS]
SETTINGS = ["hospital", "clinic", "emergency department", "operating room", "home care setting", "intensive care unit"]
POPS = ["adult patients", "pediatric and adult patients", "patients over 18 years", "neonatal patients", "all patient populations"]


def _words(rng, vocab, n):
    return [rng.choice(vocab) for _ in range(n)]


def _ifu(rng, name, fam, lineage):
    cat, vocab = fam[1], fam[2]
    w = _words(rng, vocab + lineage, 6)
    t = [
        f"The {name} is a {cat} intended for use in the {rng.choice(SETTINGS)} to assist with {w[0]} and {w[1]} in {rng.choice(POPS)}.",
        f"It is indicated for {w[2]} {w[3]} where {w[4]} monitoring by a trained clinician is required.",
        f"The device is not intended for {w[5]} use outside of a professional healthcare facility.",
    ]
    return " ".join(t[: rng.choice([1, 2, 3])]) if rng.random() < 0.9 else t[0]


def _desc(rng, name, fam, lineage, n_words):
    vocab = fam[2] + lineage * 3
    sents = []
    total = 0
    while total < n_words:
        w = _words(rng, vocab, 5)
        s = rng.choice([
            f"The {name} consists of a {w[0]} module, a {w[1]} assembly and a {w[2]} interface.",
            f"The {w[0]} component communicates with the {w[1]} using a {w[2]} protocol and reports {w[3]} status.",
            f"Materials in patient-contacting parts include {w[0]} and {w[1]}; the {w[2]} is supplied sterile.",
            f"Operation relies on {w[0]} data processed by the {w[1]} to produce {w[2]} output for the {w[3]}.",
        ])
        sents.append(s)
        total += len(s.split())
    return " ".join(sents)


def _knum(year, seq):
    return f"K{year % 100:02d}{seq:04d}"


def _render_letter(dev):
    return "\n".join([
        "Food and Drug Administration", "10903 New Hampshire Avenue", "Silver Spring, MD 20993", dev["date"], dev["applicant"],
        f"Re: {dev['k']}", f"Trade/Device Name: {dev['name']}", "Regulation Number: 21 CFR 892.2080", "Regulatory Class: Class II",
        f"Product Code: {dev['pc']}", "Dear Sir or Madam:",
        "We have reviewed your Section 510(k) premarket notification of intent to market the device referenced above and have",
        "determined the device is substantially equivalent (for the indications for use stated in the enclosure) to legally",
        "marketed predicate devices marketed in interstate commerce prior to May 28, 1976.", "Sincerely,",
        f"{dev['k']} - {dev['applicant']} Page 2",
    ])


def _render_form(dev, ifu):
    return "\n".join([
        "DEPARTMENT OF HEALTH AND HUMAN SERVICES", "Food and Drug Administration", "Indications for Use", "Form Approved: OMB No. 0910-0120",
        "510(k) Number (if known)", dev["k"], "Device Name", dev["name"], "Indications for Use (Describe)", *textwrap.wrap(ifu, 90),
        "Type of Use (Select one or both, as applicable)", "☒ Prescription Use (Part 21 CFR 801 Subpart D)   ☐ Over-The-Counter Use (21 CFR 801 Subpart C)",
        "CONTINUE ON A SEPARATE PAGE IF NEEDED.", "This section applies only to requirements of the Paperwork Reduction Act of 1995.",
    ])


IFU_HEADS = ["INDICATIONS FOR USE", "Indications for Use:", "V. INDICATIONS FOR USE", "Intended Use / Indications for Use:", "Indications for Use Statement"]
DESC_HEADS = ["DEVICE DESCRIPTION", "Device Description:", "III. Device Description", "Description of the Device"]
PRED_HEADS = ["PREDICATE DEVICE", "Predicate Device(s):", "II. PREDICATE DEVICES", "Legally Marketed Predicate Device"]


def _pred_block(rng, style, head, prim, adds, refs, names):
    L = [head]
    if style == "a":
        L.append(f"Predicate Device: {names[prim]} ({prim})")
        for a in adds:
            L.append(f"Additional Predicate Device: {names[a]} ({a})")
        for r in refs:
            L.append(f"Reference Device: {names[r]} ({r})")
    elif style == "b":
        L.append("Predicate Devices: " + ", ".join([prim] + adds))
        for r in refs:
            L.append(f"Reference Device: {names[r]} ({r})")
    else:
        L.append(f"{names[prim]}, {prim} (primary predicate)")
        for a in adds:
            L.append(f"{names[a]}, {a} (additional predicate)")
        for r in refs:
            L.append(f"{names[r]}, {r} (reference device)")
    return "\n".join(L)


def _render_summary(rng, dev, ifu, desc, pred_block, style_order, ifu_head, desc_head, with_ifu_in_summary):
    parts = ["510(k) SUMMARY", "I. SUBMITTER", dev["applicant"], "Contact Person: Regulatory Affairs", f"Date Prepared: {dev['date']}",
             "II. DEVICE", f"Trade Name: {dev['name']}", f"Common Name: {dev['cat']}", f"Product Code: {dev['pc']}"]
    blocks = {
        "pred": pred_block,
        "desc": (desc_head + "\n" + "\n".join(textwrap.wrap(desc, 90))) if desc else None,
        "ifu": (ifu_head + "\n" + "\n".join(textwrap.wrap(ifu, 90))) if with_ifu_in_summary else None,
    }
    for key in style_order:
        if blocks[key]:
            parts.append(blocks[key])
    parts += [
        "VI. COMPARISON OF TECHNOLOGICAL CHARACTERISTICS",
        "The subject device has the same intended use and similar technological characteristics as the predicate device.",
        "VII. PERFORMANCE DATA", "Bench testing was performed and all acceptance criteria were met.",
        "VIII. CONCLUSION", "The subject device is substantially equivalent to the predicate device.",
    ]
    return "\n".join(parts)


def make_synthetic(root, n: int = 1500, seed: int = 7, start: str = "2005-01-01", end: str = "2026-10-31", with_pdfs: int = 0) -> dict:
    from .config import Paths

    rng = random.Random(seed)
    cfg_stub = {"paths": {"data_dir": str(root)}}
    paths = Paths(cfg_stub, root=root)
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    span = (d1 - d0).days
    codes = list(FAMILIES)

    devs = []
    for i in range(n):
        dd = d0 + timedelta(days=rng.randrange(span))
        pc = rng.choice(codes)
        devs.append({"date": dd, "pc": pc})
    devs.sort(key=lambda x: x["date"])
    seq: dict[int, int] = {}
    for i, d in enumerate(devs):
        rec = d["date"] - timedelta(days=rng.randint(30, 200))
        seq[rec.year] = seq.get(rec.year, 0) + 1
        d["k"] = _knum(rec.year, seq[rec.year])
        d["received"] = rec
        fam = FAMILIES[d["pc"]]
        brand = rng.choice(BRANDS)
        d["name"] = f"{brand} {rng.choice(['X', 'Pro', 'Flex', 'Neo'])}{rng.randint(100, 999)} {fam[1].title()}"
        d["cat"] = fam[1]
        d["applicant"] = rng.choice(APPLICANTS)
        d["panel"] = fam[0]
        d["date"] = d["date"].isoformat()
        r = rng.random()
        yr = int(d["date"][:4])
        d["status"] = ("nse" if r < 0.02 else "statement" if r < 0.10 else "pdf404" if r < (0.15 if yr < 2010 else 0.13)
                       else "text_fail" if r < (0.17 if yr < 2010 else 0.15) else "no_desc" if r < 0.20 else "ok")
        d["lineage"] = rng.sample(fam[2], 3)
    names = {d["k"]: d["name"] for d in devs}
    by_k = {d["k"]: d for d in devs}
    cited: dict[str, int] = {}
    used_unknown = set()

    def rand_unknown():
        while True:
            k = _knum(rng.randint(1996, 2024), rng.randint(9000, 9999))
            if k not in by_k and k not in used_unknown:
                used_unknown.add(k)
                return k

    truth = {}
    for i, d in enumerate(devs):
        earlier = devs[:i]
        t = {"k": d["k"], "status": d["status"], "primary": None, "additional": [], "reference": [], "types": {}, "leak": None}
        truth[d["k"]] = t
        if i < 15 or rng.random() < 0.04 or d["status"] in ("statement", "nse", "pdf404", "text_fail"):
            continue  # no predicates (first of a kind) or no document anyway
        r = rng.random()
        n_pred = 1 if r < 0.55 else 2 if r < 0.80 else 3 if r < 0.92 else 4
        chosen = []
        for _ in range(n_pred):
            u = rng.random()
            if u < 0.04:
                k, ty = rand_unknown(), "unknown"
            elif u < 0.06:
                k, ty = f"P{rng.randint(100000, 199999)}", "pma"
            elif u < 0.07:
                k, ty = f"DEN{rng.randint(150000, 249999)}", "den"
            elif u < 0.08 and i < n - 5:
                k, ty = rng.choice(devs[i + 1:])["k"], "future"
            else:
                w = [((1 + cited.get(e["k"], 0)) * (6 if e["pc"] == d["pc"] else 1)) for e in earlier]
                k, ty = rng.choices(earlier, weights=w)[0]["k"], "real"
            if k not in chosen and k != d["k"]:
                chosen.append(k)
                t["types"][k] = ty
        if not chosen:
            continue
        for k in chosen:
            cited[k] = cited.get(k, 0) + 1
        t["primary"], t["additional"] = chosen[0], chosen[1:]
        if rng.random() < 0.15:
            rk = rng.choice(earlier)["k"]
            if rk not in chosen:
                t["reference"] = [rk]
                t["types"][rk] = "reference"
        # lineage: children inherit vocabulary from their primary predicate
        if t["types"].get(chosen[0]) == "real":
            par = by_k[chosen[0]]["lineage"]
            d["lineage"] = par[:2] + d["lineage"][:1]

    # ---- write files ---------------------------------------------------------------------------------------------
    meta_rows = []
    for d in devs:
        meta_rows.append({
            "k": d["k"], "device_name": d["name"], "applicant": d["applicant"], "product_code": d["pc"], "panel": d["panel"],
            "decision_date": d["date"], "date_received": d["received"].isoformat(), "year": int(d["date"][:4]),
            "decision_code": "NSE" if d["status"] == "nse" else "SESE", "decision_description": "Substantially Equivalent",
            "clearance_type": "Traditional", "statement_or_summary": "STATEMENT" if d["status"] == "statement" else "SUMMARY",
        })
    write_jsonl(paths.meta, meta_rows)

    pdf_budget = with_pdfs
    pdf_ks = []
    for d in devs:
        t = truth[d["k"]]
        k = d["k"]
        if d["status"] in ("statement", "nse"):
            continue
        if d["status"] == "pdf404":
            append_jsonl(paths.fetch_status, {"k": k, "status": "not_found", "url": None, "folder": None, "bytes": 0, "error": None})
            continue
        append_jsonl(paths.fetch_status, {"k": k, "status": "ok", "url": f"synthetic/{k}.pdf", "folder": f"pdf{k[1:3]}", "bytes": rng.randint(150_000, 2_500_000), "error": None})
        if d["status"] == "text_fail":
            append_jsonl(paths.text_stats, {"k": k, "n_pages": 0, "n_pages_read": 0, "ocr_pages": 0, "n_chars": 0, "seconds": 0.0, "error": "ValueError('unreadable PDF (0 pages)')"})
            continue
        fam = FAMILIES[d["pc"]]
        ifu = _ifu(rng, d["name"], fam, d["lineage"])
        n_words = int(min(700, max(30, rng.lognormvariate(math.log(120), 0.6))))
        desc = _desc(rng, d["name"], fam, d["lineage"], n_words) if d["status"] != "no_desc" else None
        t["ifu"], t["description"] = ifu, desc
        prim, adds, refs = t["primary"], t["additional"], t["reference"]
        # planted leakage inside the description
        if desc and prim and rng.random() < 0.08:
            if rng.random() < 0.5 and t["types"].get(prim) == "real":
                desc += f" The {names[prim]} was used as a design starting point."
                t["leak"] = "name"
            else:
                desc += f" This design builds on {prim}."
                t["leak"] = "k"
        pages = [_render_letter(d)]
        form = rng.random() < 0.7
        if form:
            pages.append(_render_form(d, ifu))
        style = rng.choice("abc")
        if prim:
            allnames = {**names, **{k2: "Legacy Device" for k2 in [prim, *adds, *refs] if k2 not in names}}
            pred_block = _pred_block(rng, style, rng.choice(PRED_HEADS), prim, adds, refs, allnames)
        else:
            pred_block = rng.choice(PRED_HEADS) + "\nNo predicate device was identified for this submission." if rng.random() < 0.5 else None
        order = rng.sample(["pred", "desc", "ifu"], 3)
        summ = _render_summary(rng, d, ifu, desc, pred_block, order, rng.choice(IFU_HEADS), rng.choice(DESC_HEADS), with_ifu_in_summary=not form or rng.random() < 0.5)
        for chunk in range(0, len(summ.split("\n")), 45):
            pages.append("\n".join(summ.split("\n")[chunk: chunk + 45]))
        ocr = (rng.random() < 0.15) if int(d["date"][:4]) < 2012 else False
        n_pages = len(pages) + rng.randint(0, 8)
        if pdf_budget > 0 and d["status"] == "ok" and t["primary"]:      # PDFs only for documents that plant a predicate
            pdf_budget -= 1
            pdf_ks.append((k, pages, ocr or (len(pdf_ks) % 3 == 2)))
            continue  # text cache will be produced by the text stage from the real PDF
        save_pages(paths.text(k), pages)
        nocr = rng.randint(3, 9) if ocr else 0
        append_jsonl(paths.text_stats, {"k": k, "n_pages": n_pages, "n_pages_read": len(pages), "ocr_pages": nocr,
                                        "n_chars": sum(len(p) for p in pages), "seconds": round(1.0 + 4.0 * nocr + rng.random(), 2), "error": None})

    if pdf_ks:
        from .synth_pdf import render_pdf

        for k, pages, scanned in pdf_ks:
            render_pdf(pages if not scanned else pages[:4], paths.pdf(k), scanned=scanned)
            truth[k]["pdf"] = "scanned" if scanned else "digital"
    (Path(root) / "truth.json").write_text(json.dumps(truth, indent=1))
    return {"n": n, "pdfs": [k for k, _, _ in pdf_ks]}
