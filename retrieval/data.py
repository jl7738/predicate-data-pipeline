import pandas as pd

DATA_DIR = 'data'

def load():
    q = pd.read_parquet(f'{DATA_DIR}/queries.parquet')
    c = pd.read_parquet(f'{DATA_DIR}/corpus.parquet')
    return q, c

def build_pairs(q, c, split='train'):
    text_of = dict(zip(c['k'], c['text']))
    s = q[(q['split'] == split) & q['primary'].notna()]
    s = s[s['primary'].isin(text_of)]
    return pd.DataFrame({
        'anchor': s['query'].values,
        'positive': s['primary'].map(text_of).values,
        'subject_k': s['k'].values,
        'primary_k': s['primary'].values,
    })

if __name__ == '__main__':
    q, c = load()
    pairs = build_pairs(q, c)
    print('train pairs:', len(pairs))
    print('distinct primaries:', pairs['primary_k'].nunique())
    print('most-cited primaries:'); print(pairs['primary_k'].value_counts().head(5))
    for col in ['anchor', 'positive']:
        print(col, 'chars: median', int(pairs[col].str.len().median()),
              '| 90th pct', int(pairs[col].str.len().quantile(0.9)))
    print('anchor still has a K-number:', pairs['anchor'].str.contains(r'K\d{6}').mean().round(4))
    print(pairs.iloc[0]['anchor'][:300]); print('---'); print(pairs.iloc[0]['positive'][:300])
