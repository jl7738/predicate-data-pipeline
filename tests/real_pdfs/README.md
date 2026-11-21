Drop REAL FDA 510(k) PDFs here (file name must contain the K-number, e.g. `K243175.pdf`) and run

    python -m pytest tests/test_real_pdfs.py -s

Each PDF goes through the real pipeline; the test fails on extraction errors or near-empty text, and prints the warnings.
Once you have checked a document by hand, record what is right in `expected.json` and it becomes a regression test:

    {"K243175": {"primary": "K200134", "additional": [], "reference": [], "ifu_contains": "urethral"}}

All fields are optional. Unlike `probe` (which only reports), this will fail when a later rule change breaks a document.
