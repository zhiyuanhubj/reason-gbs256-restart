#!/usr/bin/env python
"""Overlap streaming packing with GPU work without committing read-ahead.

``IterablePackingDataset`` normally advances only when the training thread asks
for the next batch.  This plugin drives its existing iterator on a daemon
producer thread and keeps a small bounded queue of already packed batches.

The important checkpoint invariant is that read-ahead is *speculative*.
Alongside every produced batch, the producer snapshots the exact streaming
state immediately after that batch.  The consumer publishes that snapshot only
when it takes the batch from the ready queue.  Checkpoints therefore contain
the last consumed state, not the producer's live read-ahead cursor.  Any queued
batches are deterministically regenerated after resume instead of being
skipped.

Enable only in diagnostic/opt-in jobs with::

    SWIFT_ASYNC_PACKING_PREFETCH_DEPTH=2
    --external_plugins scripts/swift_async_packing_prefetch.py
"""

from __future__ import annotations

import atexit
import copy
import os
import queue
import threading
import weakref
from functools import wraps
from typing import Any

from swift.dataset.packing import IterablePackingDataset
from swift.megatron.trainers.base import BaseMegatronTrainer
from swift.utils import get_logger


logger = get_logger()

_ORIGINAL_ITER = IterablePackingDataset.__iter__
_ORIGINAL_STATE_DICT = IterablePackingDataset.streaming_state_dict
_ORIGINAL_LOAD_STATE_DICT = IterablePackingDataset.load_streaming_state_dict
_ORIGINAL_INIT = IterablePackingDataset.__init__
_ORIGINAL_TRAIN = BaseMegatronTrainer.train
_LIVE_PACKERS = weakref.WeakSet()


class _PackingStopped(Exception):
    pass


class _InterruptibleOutputQueue:
    """Only the parent reader uses this wrapper; workers retain the raw queue."""

    def __init__(self, raw, stopped):
        self.raw = raw
        self.stopped = stopped

    def get(self):
        while not self.stopped.is_set():
            try:
                return self.raw.get(timeout=0.1)
            except queue.Empty:
                continue
        raise _PackingStopped()


def _packing_init(self, *args, **kwargs):
    # Add parent-only threading objects after workers have been spawned.
    try:
        _ORIGINAL_INIT(self, *args, **kwargs)
    finally:
        self._packing_stopped = threading.Event()
        self._packing_producers = []
        self._packing_closed = False
        _LIVE_PACKERS.add(self)
        # Queue creation lazily imports multiprocessing.util and registers its
        # finalizer. Register afterwards so our cleanup runs first (LIFO).
        atexit.unregister(_close_all_packers)
        atexit.register(_close_all_packers)
    self._out_queue = _InterruptibleOutputQueue(self._out_queue, self._packing_stopped)


def _close_packer(self):
    if self._packing_closed:
        return
    self._packing_closed = True
    self._packing_stopped.set()
    # Only speculative prefetch is discarded here, after train/save has returned.
    # Never join an input feeder whose reader may already have been terminated.
    queues = [getattr(self, name, None) for name in ('_in_queue', '_out_queue')]
    queues = [q.raw if isinstance(q, _InterruptibleOutputQueue) else q for q in queues]
    for q in queues:
        if q is not None:
            q.cancel_join_thread()
    for producer, stopped in self._packing_producers:
        stopped.set()
    for producer, _ in self._packing_producers:
        producer.join(timeout=2)
        if producer.is_alive():
            logger.warning('Packing producer did not stop within 2s; terminating its workers.')
    workers = getattr(self, 'workers', [])
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        worker.join(timeout=2)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=2)
        if worker.is_alive():
            logger.warning('Packing worker %s did not exit after kill.', worker.pid)
    for q in queues:
        if q is not None:
            q.close()
    _LIVE_PACKERS.discard(self)


def _close_all_packers():
    packers = list(_LIVE_PACKERS)
    for packer in packers:
        packer._packing_stopped.set()
    for packer in packers:
        try:
            _close_packer(packer)
        except Exception:
            # Cleanup must not hide the original training/save exception.
            logger.exception('Failed to close streaming packer.')


@wraps(_ORIGINAL_TRAIN)
def _train_with_packing_cleanup(self, *args, **kwargs):
    try:
        return _ORIGINAL_TRAIN(self, *args, **kwargs)
    finally:
        _close_all_packers()


def _depth() -> int:
    value = int(os.environ.get('SWIFT_ASYNC_PACKING_PREFETCH_DEPTH', '0'))
    if value < 0:
        raise ValueError('SWIFT_ASYNC_PACKING_PREFETCH_DEPTH must be non-negative')
    return value


def _put_until_stopped(ready: queue.Queue, value: Any, stopped: threading.Event) -> bool:
    while not stopped.is_set():
        try:
            ready.put(value, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _prefetch_iter(self):
    if self._packing_closed:
        raise RuntimeError('Cannot iterate a closed streaming packer')
    depth = _depth()
    if depth == 0:
        yield from _ORIGINAL_ITER(self)
        return

    # Capture the initial durable state before the producer is allowed to move
    # the underlying mixer/packer cursor.
    initial_state = _ORIGINAL_STATE_DICT(self)
    self._async_prefetch_committed_state = copy.deepcopy(initial_state)

    ready: queue.Queue = queue.Queue(maxsize=depth)
    stopped = threading.Event()
    source = _ORIGINAL_ITER(self)

    def produce() -> None:
        try:
            for packed_batch in source:
                # The original iterator is paused at its yield here. Its state
                # is exactly the next durable boundary for this packed batch.
                state_after_batch = copy.deepcopy(_ORIGINAL_STATE_DICT(self))
                if not _put_until_stopped(
                        ready, ('batch', packed_batch, state_after_batch), stopped):
                    return
            _put_until_stopped(ready, ('eof', None, None), stopped)
        except BaseException as exc:  # propagate producer failures to trainer
            if not self._packing_stopped.is_set():
                _put_until_stopped(ready, ('error', exc, None), stopped)
        finally:
            source.close()

    producer = threading.Thread(
        target=produce,
        name=f'swift-packing-prefetch-{id(self):x}',
        daemon=True,
    )
    self._packing_producers = [(p, s) for p, s in self._packing_producers if p.is_alive()]
    self._packing_producers.append((producer, stopped))
    producer.start()
    logger.info(f'Async streaming packing prefetch started (depth={depth}).')

    try:
        while True:
            kind, payload, state_after_batch = ready.get()
            if kind == 'eof':
                return
            if kind == 'error':
                raise RuntimeError('async streaming packing producer failed') from payload
            # Publish only when the training-side iterator consumes the batch.
            # The producer may already be further ahead, but checkpoints never
            # observe that speculative cursor.
            self._async_prefetch_committed_state = state_after_batch
            yield payload
    finally:
        stopped.set()


def _committed_streaming_state_dict(self):
    depth = _depth()
    if depth > 0 and hasattr(self, '_async_prefetch_committed_state'):
        return copy.deepcopy(self._async_prefetch_committed_state)
    return _ORIGINAL_STATE_DICT(self)


def _load_committed_streaming_state_dict(self, state):
    result = _ORIGINAL_LOAD_STATE_DICT(self, state)
    if _depth() > 0:
        self._async_prefetch_committed_state = copy.deepcopy(state)
    return result


IterablePackingDataset.__init__ = _packing_init
IterablePackingDataset.__iter__ = _prefetch_iter
IterablePackingDataset.streaming_state_dict = _committed_streaming_state_dict
IterablePackingDataset.load_streaming_state_dict = _load_committed_streaming_state_dict
BaseMegatronTrainer.train = _train_with_packing_cleanup

logger.info(
    'Installed async streaming packing prefetch with consume-committed checkpoint state.')
