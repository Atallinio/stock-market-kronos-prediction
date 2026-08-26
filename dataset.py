"""Grain dataloaders over the ArrayRecord window files written by fetch_data.py.

Records are pickled dicts {symbol, x: (window_len, 6) normalized candles,
stamps: (window_len, 5) time features}. The val loader uses batch_size=1 so
callers can index batch["x"][0] directly, mirroring kronos_rag.ipynb.

Extracted from kronos_rag.ipynb.
"""

import pickle

import grain
import numpy as np
from array_record.python.array_record_module import ArrayRecordReader

NUM_WORKERS = 0


class ParsePickleRecord(grain.transforms.Map):
    def map(self, record_bytes: bytes):
        return pickle.loads(record_bytes)


def make_dataloaders(train_arrayrecord_path, val_arrayrecord_path, train_batch_size=64, val_batch_size=1, seed=42):
    """Build the train/val Grain dataloaders.

    Train: shuffled, num_epochs=1. Val: ordered (no shuffle), num_epochs=1.
    """

    data_source = grain.sources.ArrayRecordDataSource(train_arrayrecord_path)
    transformations = [
        ParsePickleRecord(),
        grain.transforms.Batch(batch_size=train_batch_size, drop_remainder=True)
    ]
    sampler = grain.samplers.IndexSampler(
        num_records=len(data_source),
        shuffle=True,
        seed=seed,
        num_epochs=1,
    )
    train_dataloader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=NUM_WORKERS,
    )

    data_source = grain.sources.ArrayRecordDataSource(val_arrayrecord_path)
    transformations = [
        ParsePickleRecord(),
        grain.transforms.Batch(batch_size=val_batch_size, drop_remainder=True)
    ]
    sampler = grain.samplers.IndexSampler(
        num_records=len(data_source),
        shuffle=False,
        num_epochs=1,
    )
    val_dataloader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=NUM_WORKERS,
    )

    return train_dataloader, val_dataloader
