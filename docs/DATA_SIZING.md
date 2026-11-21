# Data sufficiency: how much gold-standard data, what shape is it, and is the plan realistic?

_Written 2026-11-19. The **counts** are real openFDA numbers (api.fda.gov, retrieved 2026-11-19; they sum to within 0.2% of the proposal's 94,143 summaries / 176,529 decisions).
The **extraction yields** (how many PDFs parse, how many cite K-numbers) are assumptions until the pilot replaces them: re-run
`python -m predx.cli sizing` after `meta` (it then recomputes the counts from your download) and override the yields with
`--set sizing.scenarios.mid.usable_subject=0.9 --set sizing.scenarios.mid.pred_rate=0.85 ...`. Nothing below was measured on real PDFs._

## Verdict

| question | answer |
|---|---|
| Enough gold pairs to fine-tune EmbeddingGemma-300M with InfoNCE? | **Yes.** Mid case ≈ 20k training queries / 31k (subject, predicate) pairs (range 15.5k–23.4k queries). That is a normal size for adapting a retrieval-pretrained encoder, but roughly 20x smaller than SPECTER's training set (SPECTER2's 6M triplets are described as 10x SPECTER's, i.e. about 0.6M), so I would expect gains but cannot promise their size. |
| Is ModernBERT-base a fair test of "does retrieval pretraining matter"? | **Only partly.** ModernBERT-base is an MLM encoder with no embedding training; SPECTER used roughly 0.6M triplets to turn SciBERT into a document embedder. With ~31k pairs it will probably be data-starved, so a gap to EmbeddingGemma would mix retrieval pretraining, architecture, size and data volume (the reviewer's point). Add the retrieval-trained `gte-modernbert-base` (149M, same ModernBERT backbone) as the matched control (already in `config/default.yaml`, commented out). |
| Is the test set big enough? | **Yes for headline numbers, marginal for fine comparisons.** ≈4.5k retrievable test queries: Recall@10 95% CI ±1.2–1.6 pts; smallest reliably detectable gap between two models ≈1.3–2.6 pts. Seed-to-seed training variance is not included: run ≥3 seeds for any comparison you plan to claim. |
| Slices (product code, specialty, prior clearances)? | Rare slices will be noisy: n=100 gives ±8–10 pts. Report n next to every slice and merge small ones. |
| Compute and budget? | Fits. Rough: ~14M tokens per epoch in-batch; with one hard negative per pair ≈7 min/epoch on an A100 or ≈23 min on an L4 (assumed throughput), ~2–7 min to embed the corpus. 15–20 runs is well inside $100. |
| What actually threatens the project? | Not data volume: (1) the recall **ceiling** (~80–85% of cited predicates have a usable record) and many acceptable but uncited predicates, so absolute Recall@10 will look low; (2) **extraction on old scanned PDFs** (time and K-number digit errors); (3) a **popularity shortcut** (fine-tuning may mostly learn which devices get cited); (4) correlated queries (same product code / same company) make real CIs wider than the independent-query numbers here. |

## What the data looks like

* **Query** (subject device): *Indications for Use* (usually one to a few sentences) + *Device Description* (typically hundreds of words), from the subject's own 510(k) summary; K-numbers and predicate names masked. Token lengths and truncation are measured by the audit (§3); EmbeddingGemma accepts up to 2,048 tokens and ModernBERT-based models 8,192, so `max_len` is a cost choice, not a hard limit.
* **Positives**: the subject's primary + additional predicates. Published samples of traditional 510(k)s average about 1.9 predicates per clearance (393 and 369 predicates in two 200-clearance samples); a JAMA study of recalled devices found a median of 1 (IQR 1–2). The model here assumes 1.5 / 1.8 / 2.1 (low / mid / high). Reference devices are excluded.
* **Candidate pool**: every earlier cleared device with a usable summary: about 51–59k at the start of 2015, 79–91k by the end of 2026. Random Recall@10 is therefore about 0.0001.
* **Time structure**: predicates are mostly recent (median age about 4.4 years, mean about 6.2 years in the FDA Law Blog samples; FDA reported that about one in five 510(k)s cleared 2015–2018 relied on a predicate older than 10 years), so a 2025–26 test subject mostly cites devices from roughly 2019–2025. About 16% of test gold was decided in 2024 or later, so it cannot appear as a positive in training; the rest can, which is what makes popularity memorisation possible (audit §6 reports % seen in train).
* **Same product code**: not measured here. For AI/ML devices roughly one in five predicates is under a different product code, so most (not all) true predicates share the subject's code. That makes same-code negatives hard (many are acceptable predicates that happen not to be cited) and makes the product-code-popularity baseline strong. The audit reports it for the full set.
* **Coverage ceiling**: a cited predicate is retrievable only if its own summary exists and parses. Summary share by period: ≈10% (1976–95), 65% (1996–2000), 81% (2001–05), 90% (2006–10), 93% (2011–14), 94–96% (2015+). Older predicates are the problem; the table in §5 shows what each corpus start year costs.

## Sources of the published numbers

* Predicate age and count per clearance: FDA Law Blog, "Does the 510(k) Program Need Predicate Modernization?" (2019), https://www.thefdalawblog.com/2019/01/does-the-510k-program-need-predicate-modernization/
* Predicates per recalled device (median 1, IQR 1–2): JAMA 2023;329(2):136–143, https://pmc.ncbi.nlm.nih.gov/articles/PMC9857464/
* Predicates under a different product code (AI/ML devices): Innolitics, https://innolitics.com/articles/how-far-can-a-k-predicate-stretch/
* SPECTER2 uses 6M triplets, ten times SPECTER's: https://github.com/allenai/SPECTER2 ; SPECTER's replication used 311,860 unique papers: https://arxiv.org/pdf/2202.06671
* gte-modernbert-base (149M, ModernBERT backbone): https://huggingface.co/Alibaba-NLP/gte-modernbert-base ; EmbeddingGemma model card (320B-token training, no float16): https://ai.google.dev/gemma/docs/embeddinggemma/model_card

---

# Planning-model output (`python -m predx.cli sizing`)


_Planning model. Counts: **built-in openFDA counts retrieved 2026-11-19**. Extraction yields are ASSUMED until you replace them with the pilot's measured rates (`--set sizing.scenarios.mid.pred_rate=0.9` ...)._

## 1. Counted (openFDA): decisions by period

| period | summary | statement | blank | summary share |
|---|---|---|---|---|
| 1976-1995 | 7,684 | 12,384 | 55,409 | 10.2% |
| 1996-2000 | 12,643 | 6,398 | 457 | 64.8% |
| 2001-2005 | 13,934 | 3,245 | 45 | 80.9% |
| 2006-2010 | 13,629 | 1,421 | 22 | 90.4% |
| 2011-2014 | 11,619 | 811 | 83 | 92.9% |
| 2015-2023 | 25,987 | 1,400 | 270 | 94.0% |
| 2024-2024 | 2,935 | 147 | 47 | 93.8% |
| 2025-2026 | 5,860 | 193 | 60 | 95.9% |

Total summaries 94,291. Splits: train 2015-2023, dev 2024, test 2025-2026 (by the subject's decision date).

## 2. Expected gold-standard data (subject -> cited predicate pairs)

Scenario assumptions: usable_subject = IFU+description extracted; pred_rate = at least one K-number found; preds_per_doc = primary+additional per document that cites any; usable_candidate = a cited predicate's own PDF parses. `pair_coverage` also folds in how many *older* predicates have a published summary at all.

**low** (usable_subject=0.85, pred_rate=0.8, preds_per_doc=1.5, usable_candidate=0.85)

| split | subjects w/ summary | usable | cite a predicate | gold pairs | pair coverage | pairs in corpus | **retrievable queries** | gold decided 2024+ |
|---|---|---|---|---|---|---|---|---|
| train | 25,727 | 21,868 | 17,494 | 26,242 | 77% | 20,091 | **15,509** | 0% |
| dev | 2,906 | 2,470 | 1,976 | 2,964 | 79% | 2,327 | **1,779** | 0% |
| test | 5,801 | 4,931 | 3,945 | 5,917 | 79% | 4,664 | **3,561** | 16% |

**mid** (usable_subject=0.92, pred_rate=0.88, preds_per_doc=1.8, usable_candidate=0.92)

| split | subjects w/ summary | usable | cite a predicate | gold pairs | pair coverage | pairs in corpus | **retrievable queries** | gold decided 2024+ |
|---|---|---|---|---|---|---|---|---|
| train | 25,727 | 23,669 | 20,829 | 37,492 | 83% | 31,068 | **19,959** | 0% |
| dev | 2,906 | 2,673 | 2,352 | 4,234 | 85% | 3,598 | **2,275** | 0% |
| test | 5,801 | 5,337 | 4,697 | 8,454 | 85% | 7,213 | **4,548** | 16% |

**high** (usable_subject=0.97, pred_rate=0.95, preds_per_doc=2.1, usable_candidate=0.97)

| split | subjects w/ summary | usable | cite a predicate | gold pairs | pair coverage | pairs in corpus | **retrievable queries** | gold decided 2024+ |
|---|---|---|---|---|---|---|---|---|
| train | 25,727 | 24,955 | 23,708 | 49,786 | 87% | 43,498 | **23,400** | 0% |
| dev | 2,906 | 2,818 | 2,678 | 5,623 | 90% | 5,037 | **2,654** | 0% |
| test | 5,801 | 5,627 | 5,346 | 11,227 | 90% | 10,099 | **5,303** | 16% |

## 3. Search space and test-set precision

| scenario | candidate pool at 2015 / 2020 / 2025 / end-2026 | Recall@10 CI half-width at 0.2 / 0.4 / 0.6 (test) | smallest detectable A-vs-B gap, pts (discordant 10% / 20% / 30%) |
|---|---|---|---|
| low | 51,292 / 63,441 / 75,647 / 79,346 | ±1.3 pts / ±1.6 pts / ±1.6 pts | 1.5 / 2.1 / 2.6 |
| mid | 55,516 / 68,665 / 81,877 / 85,880 | ±1.2 pts / ±1.4 pts / ±1.4 pts | 1.3 / 1.9 / 2.3 |
| high | 58,533 / 72,397 / 86,327 / 90,548 | ±1.1 pts / ±1.3 pts / ±1.3 pts | 1.2 / 1.7 / 2.1 |

Random Recall@10 over a pool of ~80,000 is about 0.0001, so every non-trivial number is a large multiple of chance; the *comparisons that matter* (fine-tuned vs zero-shot vs BM25) are between methods that both beat chance by a wide margin.

## 4. Rough compute (ASSUMED throughput: A100 50k tokens/s, L4 15k tokens/s, 220 tokens per text)

| scenario | M tokens / epoch (in-batch) | minutes / epoch with 1 hard negative per pair (A100 / L4) | minutes to embed the corpus (A100 / L4) |
|---|---|---|---|
| low | 8.8 | 4 / 15 | 2 / 6 |
| mid | 13.7 | 7 / 23 | 2 / 7 |
| high | 19.1 | 10 / 32 | 2 / 7 |

## 5. What starting the corpus later would cost (mid scenario)

| corpus starts | summaries to download+parse | train pair coverage | test pair coverage | test ceiling vs full corpus |
|---|---|---|---|---|
| all years | 94,291 | 83% | 85% | 100% |
| 1996 | 86,607 | 83% | 85% | 100% |
| 2001 | 73,964 | 81% | 85% | 99% |
| 2006 | 60,030 | 77% | 84% | 98% |
| 2011 | 46,401 | 65% | 81% | 95% |

Older documents are the ones most likely to be scanned (OCR is the expensive and least accurate step), so the rows above are the trade-off between recall ceiling and extraction risk.
