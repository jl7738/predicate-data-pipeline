import argparse, json
import numpy as np
from predx.metrics import gold_ranks_from_scores, per_query_metrics, summarize
from retrieval.data import load


def rank_dev(encode, q, c, limit=None):
    c = c.sort_values('date', kind='stable').reset_index(drop=True)
    row_of = {k: i for i, k in enumerate(c['k'])}
    dates = c['date'].values
    d = q[q['split'] == 'dev']
    if limit:
        d = d.sample(limit, random_state=0)

    D = encode(c['text'].tolist())
    Q = encode(d['query'].tolist())

    primary_ranks, all_ranks = [], []
    for i, (qdate, prim, positives) in enumerate(zip(d['date'], d['primary'], d['positives'])):
        n_cand = int(np.searchsorted(dates, qdate, side='left'))
        scores = D[:n_cand] @ Q[i]
        gold = [row_of[k] for k in positives if k in row_of and row_of[k] < n_cand]
        all_ranks.append(gold_ranks_from_scores(scores, n_cand, gold))
        if isinstance(prim, str) and prim in row_of and row_of[prim] < n_cand:
            primary_ranks.append(gold_ranks_from_scores(scores, n_cand, [row_of[prim]]))
        else:
            primary_ranks.append([np.inf])
    return d, primary_ranks, all_ranks


def report(primary_ranks, all_ranks, has_primary):
    prim = per_query_metrics(primary_ranks, recall_ks=[10], hit_ks=[1, 10])
    allp = per_query_metrics(all_ranks, recall_ks=[10], hit_ks=[1, 10, 100])
    out = {
        'primary (all dev queries)': summarize({k: prim[k] for k in ['hit@1', 'hit@10', 'mrr']}),
        'primary (queries with a primary)': summarize({k: prim[k][has_primary] for k in ['hit@1', 'hit@10', 'mrr']}),
        'all predicates': summarize({k: allp[k] for k in ['recall@10', 'hit@1', 'hit@10', 'hit@100', 'mrr']}),
    }
    for name, res in out.items():
        print(f'\n{name}  (n={res["n_queries"]})')
        for m, v in res.items():
            if m != 'n_queries':
                print(f'  {m:10s} {100*v["mean"]:5.1f}  [{100*v["lo"]:.1f}, {100*v["hi"]:.1f}]')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='BAAI/bge-base-en-v1.5')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    encode = lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True,
                                        convert_to_numpy=True, show_progress_bar=True)
    q, c = load()
    d, primary_ranks, all_ranks = rank_dev(encode, q, c, args.limit)
    res = report(primary_ranks, all_ranks, d['primary'].notna().values)
    if args.out:
        json.dump(res, open(args.out, 'w'), indent=2)
