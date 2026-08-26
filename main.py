"""Kronos RAG — command line interface.

Pipeline:
  1. fetch          download ticker minute bars from Alpaca (needs ALPACA_API_KEY)
  2. build-windows  turn fetched frames into normalized train/val ArrayRecords
  3. build-db       embed the train windows with Kronos and build the faiss store
  4. evaluate       score RAG direction calls on the val windows vs baseline
  5. example        plot one query window against its retrieved neighbors

The model is loaded on demand from HuggingFace and transferred into the Flax
port (see transfer.py). Choose the variant with --model {base,small,mini}.

Example:
  python main.py --model base fetch
  python main.py --model base build-windows --lookback 90 --predict 10
  python main.py --model base build-db
  python main.py --model base evaluate
  python main.py --model base example --batch 0 --sample 0
"""

import argparse
import logging
import os
import pickle
from datetime import datetime, timedelta

from fetch_data import NY_TZ, SP_TICKERS, TRAIN_FRAC, fetch_tickers, write_windows_arrayrecord

logger = logging.getLogger("kronos_rag")


def get_models(args):
    import transfer
    return transfer.load_models(args.model, kronos_path=args.kronos_path)


def cmd_fetch(args):
    end = datetime.now(NY_TZ)
    start = end - timedelta(days=int(args.years * 365.25))
    frames = fetch_tickers(args.tickers, start, end)
    with open(args.out, "wb") as f:
        pickle.dump(frames, f)
    logger.info("[fetch] saved %d tickers to %s", len(frames), args.out)


def cmd_build_windows(args):
    with open(args.frames, "rb") as f:
        frames = pickle.load(f)
    write_windows_arrayrecord(frames, args.train_frac, args.lookback, args.predict, args.out, clip=args.clip)


def cmd_build_db(args):
    import dataset as ds
    import rag as ragmod

    train, _ = ds.make_dataloaders(
        os.path.join(args.splits, "train.arrayrecord"),
        os.path.join(args.splits, "val.arrayrecord"),
    )
    tokenizer, model = get_models(args)
    ragmod.build_db(train, args.lookback, args.store, tokenizer, model)


def cmd_evaluate(args):
    import dataset as ds
    import rag as ragmod

    _, val = ds.make_dataloaders(
        os.path.join(args.splits, "train.arrayrecord"),
        os.path.join(args.splits, "val.arrayrecord"),
    )
    tokenizer, model = get_models(args)
    ragmod.rag_profit_probability(
        val, args.store, args.lookback, tokenizer, model,
        k=args.k, tau=args.tau, sim_threshold=args.sim_threshold,
        max_queries=args.max_queries,
    )


def cmd_example(args):
    import dataset as ds
    import rag as ragmod

    _, val = ds.make_dataloaders(
        os.path.join(args.splits, "train.arrayrecord"),
        os.path.join(args.splits, "val.arrayrecord"),
    )
    tokenizer, model = get_models(args)
    ragmod.rag_single_example(
        val, args.store, args.lookback, tokenizer, model,
        k=args.k, batch_idx=args.batch, sample_idx=args.sample,
    )


def main():
    parser = argparse.ArgumentParser(description="Kronos RAG pipeline")
    parser.add_argument("--model", choices=["base", "small", "mini"], default="base",
                        help="Kronos variant to load from HuggingFace")
    parser.add_argument("--kronos-path", default="Kronos",
                        help="path to the cloned upstream Kronos repo (for its PyTorch classes)")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download ticker bars from Alpaca")
    p.add_argument("--tickers", nargs="+", default=SP_TICKERS)
    p.add_argument("--years", type=float, default=2)
    p.add_argument("--out", default="frames.pkl")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("build-windows", help="write normalized windows to ArrayRecords")
    p.add_argument("--frames", default="frames.pkl")
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--predict", type=int, default=10)
    p.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--out", default="splits")
    p.set_defaults(func=cmd_build_windows)

    p = sub.add_parser("build-db", help="embed train windows and build the faiss store")
    p.add_argument("--splits", default="splits")
    p.add_argument("--store", default="store")
    p.add_argument("--lookback", type=int, default=90)
    p.set_defaults(func=cmd_build_db)

    p = sub.add_parser("evaluate", help="score RAG direction calls on val windows")
    p.add_argument("--splits", default="splits")
    p.add_argument("--store", default="store")
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--sim-threshold", type=float, default=0.98)
    p.add_argument("--max-queries", type=int, default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("example", help="plot one query window against its neighbors")
    p.add_argument("--splits", default="splits")
    p.add_argument("--store", default="store")
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--sample", type=int, default=0)
    p.set_defaults(func=cmd_example)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
