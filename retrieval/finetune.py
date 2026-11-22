import argparse
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from retrieval.data import load, build_pairs

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='BAAI/bge-base-en-v1.5')
    ap.add_argument('--out', default='models/bge-base-primary-1ep')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--epochs', type=float, default=1)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    q, c = load()
    pairs = build_pairs(q, c, split='train')
    if args.limit:
        pairs = pairs.sample(args.limit, random_state=0)
    train = Dataset.from_pandas(pairs[['anchor', 'positive']], preserve_index=False)
    print('training pairs:', len(train))

    model = SentenceTransformer(args.model)
    loss = MultipleNegativesRankingLoss(model)

    targs = SentenceTransformerTrainingArguments(
        output_dir=args.out + '-ckpt',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=50,
        save_strategy='no',
        report_to='none',
        seed=4705,
    )
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=train, loss=loss).train()
    model.save(args.out)
    print('saved to', args.out)
