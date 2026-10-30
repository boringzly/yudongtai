"""Bounded CPU-only loading and copy-efficient change detection batches."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np


class ThreadedBatchLoader:
    """Ordered loading without child processes or CUDA calls.

    Each thread creates its own dataset (and GDAL handles). Only one batch of
    sample futures is retained ahead; close waits for running reads before closing
    datasets. Use as a context manager so inference/write errors also clean up.
    """

    def __init__(self, factory, length, batch_size, workers, collate):
        if batch_size < 1 or workers < 1:
            raise ValueError('batch_size and workers must be positive')
        self.factory, self.length = factory, length
        self.batch_size, self.workers, self.collate = batch_size, workers, collate
        self._iterator = None

    def __len__(self):
        return (self.length + self.batch_size - 1) // self.batch_size

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self._iterator is not None:
            self._iterator.close()

    def __iter__(self):
        self._iterator = self._batches()
        return self._iterator

    def _batches(self):
        local = threading.local()
        datasets, pending = [], deque()
        lock = threading.Lock()

        def load(index):
            if not hasattr(local, 'dataset'):
                local.dataset = self.factory()
                with lock:
                    datasets.append(local.dataset)
            return local.dataset[index]

        executor = ThreadPoolExecutor(max_workers=self.workers,
                                      thread_name_prefix='change-cpu')
        next_index = 0
        try:
            # Refill consumed futures so only one batch is queued ahead.
            capacity = self.batch_size
            while next_index < min(capacity, self.length):
                pending.append(executor.submit(load, next_index))
                next_index += 1
            while pending:
                samples = []
                for _ in range(min(self.batch_size, len(pending))):
                    samples.append(pending.popleft().result())
                    if next_index < self.length:
                        pending.append(executor.submit(load, next_index))
                        next_index += 1
                yield self.collate(samples)
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True)
            for dataset in datasets:
                close = getattr(dataset, 'close', None)
                if close is not None:
                    close()


def pack_valid_batch(pre, post, mask, indices):
    """Same NCHW pixels/order as filtering both tensors then concatenating.

    CPU tensors expose zero-copy NumPy views. Scan uint8 masks directly instead
    of expanding a full batch to int64 for torch.sum; copy image pixels only to
    the final six-channel input buffer.
    """
    pre, post, mask = np.asarray(pre), np.asarray(post), np.asarray(mask)
    valid = np.any(mask.reshape(mask.shape[0], -1), axis=1)
    positions = np.flatnonzero(valid)
    channels = pre.shape[1]
    images = np.empty((len(positions), channels + post.shape[1], *pre.shape[2:]),
                      dtype=np.result_type(pre.dtype, post.dtype))
    for destination, source in enumerate(positions):
        images[destination, :channels] = pre[source]
        images[destination, channels:] = post[source]
    valid_masks = mask if valid.all() else mask[positions]
    valid_indices = [np.asarray(component)[positions] for component in indices]
    return images, valid_masks, valid_indices
