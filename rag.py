"""RAG over Kronos: vector store, embedding extraction, retrieval and evaluation.

The pipeline:
  1. build_db: encode every train window through the Kronos predictor and store
     the final hidden state (last position) in a faiss IndexFlatIP, along with
     the normalized future candles, the past series and the ticker.
  2. Query time (rag_profit_probability / rag_single_example): embed a window,
     find the k most similar historical windows (cosine similarity), and either
     score the direction using the similarity-weighted neighbor futures, or
     plot the query against its retrieved neighbors.

Extracted from kronos_rag.ipynb.
"""

import logging
import os

import faiss
import h5py
import jax
import numpy as np
from flax import nnx

logger = logging.getLogger(__name__)


class RagDB:
    """faiss IndexFlatIP for L2-normalized embedding vectors + HDF5 payloads.

    Payload per window: the normalized future candles, the past series and the
    ticker symbol. `lookback` / `future_len` / `dim` are inferred on first add
    and persisted as HDF5 attributes.
    """

    def __init__(self):
        self.dim = None
        self.future_len = None
        self.lookback = None
        self.index = None
        self.futures = []
        self.tickers = []
        self.series = []

    def add(self, vectors, futures, tickers, series):
        """Add a batch of windows. All four args are (B, ...) arrays."""
        vectors = np.asarray(vectors, dtype=np.float32)
        futures = np.asarray(futures, dtype=np.float32)
        tickers = np.asarray(tickers, dtype=str)
        series = np.asarray(series, dtype=np.float32)

        if self.index is None:
            self.dim = vectors.shape[-1]
            self.future_len = futures.shape[1]
            self.lookback = series.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)
        elif vectors.shape[-1] != self.dim or futures.shape[1] != self.future_len or series.shape[1] != self.lookback:
            raise ValueError("batch shape mismatch with existing store")

        normalized = np.array(vectors, dtype=np.float32, copy=True)
        faiss.normalize_L2(normalized)
        self.index.add(normalized)

        self.futures.extend(futures)
        self.tickers.extend(tickers.tolist())
        self.series.extend(series)

    def search(self, query, k):
        """k-NN by cosine similarity. Returns (sims, idxs, tickers, futures, series)."""
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        sims, idxs = self.index.search(query, k)
        idxs = idxs[0]
        futures = np.stack([self.futures[i] for i in idxs])
        series = np.stack([self.series[i] for i in idxs])
        tickers = [self.tickers[i] for i in idxs]
        return sims[0], idxs, tickers, futures, series

    def __len__(self):
        return len(self.tickers)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with h5py.File(os.path.join(path, "payloads.h5"), "w") as f:
            f.attrs["dim"] = self.dim
            f.attrs["lookback"] = self.lookback
            f.attrs["future_len"] = self.future_len
            f.create_dataset("ticker", data=np.array(self.tickers, dtype="S"), dtype=h5py.string_dtype("utf-8"))
            f.create_dataset("future", data=np.stack(self.futures))
            f.create_dataset("series", data=np.stack(self.series))

    @classmethod
    def load(cls, path):
        with h5py.File(os.path.join(path, "payloads.h5"), "r") as f:
            dim = int(f.attrs["dim"])
            future_len = int(f.attrs["future_len"])
            lookback = int(f.attrs["lookback"])
            tickers = [str(t) for t in f["ticker"][:]]
            futures = f["future"][:]
            series = f["series"][:]

        db = cls()
        db.dim = dim
        db.future_len = future_len
        db.lookback = lookback
        db.index = faiss.read_index(os.path.join(path, "index.faiss"))
        db.futures = [futures[i] for i in range(len(tickers))]
        db.series = [series[i] for i in range(len(tickers))]
        db.tickers = tickers
        return db


@nnx.jit
def predict_embeddings(model, s1_ids, s2_ids, stamp):
    """Run the predictor and return the final hidden state at the last position."""
    model.eval()
    _, _, hidden = model(s1_ids, s2_ids, stamp)
    return hidden[:, -1, :]


def build_db(dataset, lookback, path, tokenizer, model):
    """Embed every window in `dataset` and store vectors + payloads in a RagDB."""
    db = RagDB()
    for i, batch in enumerate(dataset):
        candle, stamp = batch["x"][:, :lookback, :], batch["stamps"][:, :lookback, :]
        futures = batch["x"][:, lookback:, :]
        tickers = batch["symbol"]

        s1_ids, s2_ids = tokenizer.encode(candle, half=True)
        hidden = predict_embeddings(model, s1_ids, s2_ids, stamp)
        hidden = np.asarray(jax.device_get(hidden))
        db.add(hidden, futures, tickers, candle)

        if i % 10 == 0:
            logger.info("[build_db] processed batch %d (%d windows)", i, len(db))

    db.save(path)
    logger.info("[build_db] saved %d windows to %s", len(db), path)
    return db


def rag_profit_probability(dataset, db_path, lookback, tokenizer, model,
                           k=10, tau=0.1, sim_threshold=0.98, max_queries=None):
    """Score RAG direction calls on a dataset of single-sample batches.

    For every query window whose top-1 cosine similarity clears `sim_threshold`,
    the direction is the sign of the similarity-weighted mean neighbor move
    (last future close vs last past close, z-scored space). Reported:
    hit rate vs the unconditional up-move baseline of the evaluated queries.

    Returns a dict of metrics.
    """
    db = RagDB.load(db_path)

    total = 0
    evaluated = 0
    positive = 0
    actual_ups = 0

    for i, batch in enumerate(dataset):
        if max_queries is not None and total >= max_queries:
            break
        total += 1

        x = batch["x"][0]
        stamps = batch["stamps"][0]
        ticker = batch["symbol"][0]

        candle = x[:lookback][None, ...]
        stamp_past = stamps[:lookback][None, ...]
        future = x[lookback:]

        s1_ids, s2_ids = tokenizer.encode(candle, half=True)
        _, _, hidden = model(s1_ids, s2_ids, stamp_past)
        query = np.asarray(hidden[0, -1, :])

        sims, idxs, tickers, futures, series = db.search(query, k)
        if sims[0] < sim_threshold:
            continue

        rets = futures[:, -1, 3] - series[:, -1, 3]
        w = np.exp(sims / tau)
        w /= w.sum()
        pred_move = float((w * rets).sum())

        actual_move = float(future[-1, 3] - candle[0][-1, 3])
        actual_up = actual_move > 0
        actual_ups += int(actual_up)
        evaluated += 1
        if (pred_move > 0) == actual_up:
            positive += 1

        logger.debug("%s: pred move %+.3f over %d bars (k=%d, top sim %.3f), actual %+.3f",
                     ticker, pred_move, future.shape[0], k, sims[0], actual_move)

    if evaluated == 0:
        logger.warning("No query cleared sim_threshold=%.2f over %d queries.", sim_threshold, total)
        return {"total": total, "evaluated": 0}

    hit_rate = positive / evaluated
    up_rate = actual_ups / evaluated
    baseline = max(up_rate, 1.0 - up_rate)

    logger.info("Evaluated %d/%d queries (sim_threshold=%.2f).", evaluated, total, sim_threshold)
    logger.info("Hit rate: %.1f%% | always-bet baseline: %.1f%% | advantage: %+.1f pp",
                hit_rate * 100, baseline * 100, (hit_rate - baseline) * 100)

    return {
        "total": total,
        "evaluated": evaluated,
        "hit_rate": hit_rate,
        "up_rate": up_rate,
        "baseline": baseline,
    }


def rag_single_example(dataset, db_path, lookback, tokenizer, model, k=1, batch_idx=0, sample_idx=0):
    """Fetch one query window, retrieve its k neighbors and plot candles side by side.

    Left column: candlestick OHLC (query + actual future; neighbor + stored
    future). Right column: volume/amount lines. The gray vline marks the end
    of the lookback window. Returns the raw search results.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    db = RagDB.load(db_path)

    batch = None
    for i, b in enumerate(dataset):
        if i == batch_idx:
            batch = b
            break
    if batch is None:
        raise ValueError(f"batch_idx {batch_idx} out of range")

    x = batch["x"][sample_idx]
    stamps = batch["stamps"][sample_idx]
    ticker = batch["symbol"][sample_idx]

    candle = x[:lookback][None, ...]
    stamp_past = stamps[:lookback][None, ...]
    future = x[lookback:]

    s1_ids, s2_ids = tokenizer.encode(candle, half=True)
    _, _, hidden = model(s1_ids, s2_ids, stamp_past)
    query = np.asarray(hidden[0, -1, :])

    sims, idxs, tickers, futures, series = db.search(query, k)

    def draw_candles(ax, past, fut, title, future_edges=None):
        n_past = len(past)

        def draw(bars, offset, alpha, edge):
            for i in range(len(bars)):
                o, h, l, c = bars[i][0], bars[i][1], bars[i][2], bars[i][3]
                color = "tab:green" if c >= o else "tab:red"
                ax.plot([offset + i, offset + i], [l, h], color=color, lw=0.8, alpha=alpha)
                ax.add_patch(Rectangle((offset + i - 0.3, min(o, c)), 0.6,
                                       max(abs(c - o), 1e-6),
                                       facecolor=color, edgecolor=edge, alpha=alpha))

        draw(past, 0, 1.0, None)
        draw(fut, n_past, 0.55, future_edges)
        ax.axvline(n_past - 0.5, color="gray", linestyle="--", lw=1)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.5, n_past + len(fut) - 0.5)

    fig, axes = plt.subplots(k + 1, 2, figsize=(16, 3.5 * (k + 1)),
                             sharex="col", gridspec_kw={"height_ratios": [2] * (k + 1)})
    if k == 1:
        axes = axes[None, :]

    draw_candles(axes[0, 0], candle[0], future,
                 f"query {ticker} — past + actual future")
    for j in range(k):
        draw_candles(axes[j + 1, 0], series[j], futures[j],
                     f"neighbor {tickers[j]} (sim {sims[j]:.2f}) — past + stored future")

    for row, past, fut, name in [
        (0, candle[0], future, "query"),
        *[(j + 1, series[j], futures[j], "neighbor") for j in range(k)],
    ]:
        full = np.concatenate([past, fut])
        axes[row, 1].plot(full[:, 4], label=f"{name} volume", color="tab:purple", lw=1)
        axes[row, 1].plot(full[:, 5], label=f"{name} amount", color="tab:brown", lw=1)
        axes[row, 1].axvline(len(past) - 0.5, color="gray", linestyle="--", lw=1)
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(alpha=0.3)
        axes[row, 1].set_title(f"{name} volume / amount")

    plt.tight_layout()
    plt.show()

    return sims, idxs, tickers, futures, series
