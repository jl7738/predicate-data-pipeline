"""Token-length measurement and the truncation / chunking policy.

Reviewer: "Check the token lengths of your two input sections and define how you will handle truncation/chunking
across models." -> measure with each model's OWN tokenizer (EmbeddingGemma and ModernBERT differ), count the prompt
and field headers too, and compare truncation policies at several max lengths.

If a tokenizer cannot be loaded (offline, gated model without HF_TOKEN) a clearly-labelled APPROXIMATE tokenizer is used;
the audit prints APPROX next to every affected number. Re-run on the VM with a token for the real numbers.
"""
from __future__ import annotations

import re


class _Approx:
    """Deterministic stand-in: words/punctuation, words longer than 6 chars split into 6-char pieces."""

    pat = re.compile(r"\w+|[^\w\s]")

    def encode(self, text: str) -> list[str]:
        toks: list[str] = []
        for w in self.pat.findall(text or ""):
            if len(w) <= 6 or not w.isalnum():
                toks.append(w)
            else:
                pieces = [w[i: i + 6] for i in range(0, len(w), 6)]
                toks.append(pieces[0])
                toks += ["##" + p for p in pieces[1:]]
        return toks

    def decode(self, toks) -> str:
        out = ""
        for t in toks:
            out += t[2:] if t.startswith("##") else ((" " if out else "") + t)
        return out


class Tok:
    def __init__(self, name: str, impl, approx: bool, n_special: int, reason: str = ""):
        self.name, self.impl, self.approx, self.n_special, self.reason = name, impl, approx, n_special, reason

    def ids(self, text: str) -> list:
        if self.approx:
            return self.impl.encode(text)
        return self.impl(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids) -> str:
        if self.approx:
            return self.impl.decode(ids)
        return self.impl.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def count_many(self, texts: list[str], batch: int = 2000) -> list[int]:
        if self.approx:
            return [len(self.impl.encode(t)) for t in texts]
        out: list[int] = []
        for i in range(0, len(texts), batch):
            enc = self.impl(texts[i: i + batch], add_special_tokens=False)["input_ids"]
            out += [len(e) for e in enc]
        return out


def load_tokenizer(mcfg: dict, allow_approx: bool = True) -> Tok:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(mcfg["hf_id"])
        tok.model_max_length = 10**9  # silence "sequence longer than max" warnings; we do our own truncation
        n_special = len(tok("x")["input_ids"]) - len(tok("x", add_special_tokens=False)["input_ids"])
        return Tok(mcfg["name"], tok, False, n_special)
    except Exception as e:
        if not allow_approx:
            raise
        return Tok(mcfg["name"], _Approx(), True, 2, reason=repr(e)[:160])


def overhead_tokens(tok: Tok, prefix: str, tmpl: dict) -> int:
    text = prefix + tmpl["ifu_header"] + tmpl["separator"] + tmpl["desc_header"]
    return len(tok.ids(text)) + tok.n_special


def allocate(n_ifu: int, n_desc: int, avail: int, policy: str, ifu_budget: int) -> tuple[int, int]:
    """Tokens kept from (indications, description) given `avail` tokens.
    head   : plain left-to-right truncation of 'indications + description' (a long indications eats the description).
    budget : indications limited to `ifu_budget`; leftover goes to the description; any spare goes back to indications."""
    avail = max(0, avail)
    if policy == "head":
        ki = min(n_ifu, avail)
        kd = min(n_desc, avail - ki)
        return ki, kd
    ki = min(n_ifu, ifu_budget, avail)
    kd = min(n_desc, avail - ki)
    spare = avail - ki - kd
    if spare > 0 and ki < n_ifu:
        ki += min(spare, n_ifu - ki)
    return ki, kd


def build_input(ifu: str, desc: str, tok: Tok, mcfg: dict, tmpl: dict, policy: str, ifu_budget: int, role: str = "doc"):
    """The exact string to feed a model: prompt prefix + headers + (possibly truncated) fields, within max_len tokens."""
    prefix = mcfg["doc_prefix"] if role == "doc" else mcfg["query_prefix"]
    avail = mcfg["max_len"] - overhead_tokens(tok, prefix, tmpl)
    ids_i, ids_d = tok.ids(ifu or ""), tok.ids(desc or "")
    ki, kd = allocate(len(ids_i), len(ids_d), avail, policy, ifu_budget)
    ifu_t = tok.decode(ids_i[:ki]) if ki < len(ids_i) else (ifu or "")
    desc_t = tok.decode(ids_d[:kd]) if kd < len(ids_d) else (desc or "")
    text = prefix + tmpl["ifu_header"] + ifu_t + tmpl["separator"] + tmpl["desc_header"] + desc_t
    info = {"n_ifu": len(ids_i), "n_desc": len(ids_d), "kept_ifu": ki, "kept_desc": kd, "avail": avail,
            "ifu_truncated": ki < len(ids_i), "desc_truncated": kd < len(ids_d)}
    return text, info


def chunk_windows(n_tokens: int, size: int, stride: int) -> list[tuple[int, int]]:
    """Sliding windows over a token sequence; document score = max over chunk scores (fallback if truncation hurts)."""
    if n_tokens <= size:
        return [(0, n_tokens)]
    out, s = [], 0
    while True:
        e = min(s + size, n_tokens)
        out.append((s, e))
        if e == n_tokens:
            break
        s += stride
    return out
