"""Time-series cross-validation with purging and embargo.

Plain k-fold is invalid here, and so is a naive ``TimeSeriesSplit``. Triple-
barrier labels overlap: the label at bar ``t`` depends on bars up to
``t + horizon``. If bar ``t`` sits in train and ``t + 3`` in test, the training
label already encodes the test period's price path, and the resulting accuracy
is fiction.

The fix (Lopez de Prado's purging and embargo) is to delete training bars whose
label window overlaps the test window, and to additionally drop a buffer of bars
immediately after it, since features are serially correlated across the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class PurgedWalkForward:
    """Expanding-window walk-forward splitter.

    Each fold trains on everything before the test window (minus the purge) and
    tests on the block that follows - the same information ordering the live
    system has.
    """

    n_folds: int = 5
    initial_train_frac: float = 0.5
    embargo: int = 36

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        start = int(n_samples * self.initial_train_frac)
        if start <= 0 or start >= n_samples:
            raise ValueError(
                f"initial_train_frac={self.initial_train_frac} leaves no usable "
                f"train/test split for {n_samples} samples"
            )

        remaining = n_samples - start
        fold_size = remaining // self.n_folds
        if fold_size <= 0:
            raise ValueError(
                f"{n_samples} samples cannot be divided into {self.n_folds} folds "
                f"after an initial train block of {start}"
            )

        for k in range(self.n_folds):
            test_start = start + k * fold_size
            test_end = n_samples if k == self.n_folds - 1 else test_start + fold_size
            # Purge the tail of the training block: those labels resolve inside
            # the test window.
            train_end = max(0, test_start - self.embargo)
            if train_end < 50:
                continue
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            if len(test_idx) == 0:
                continue
            yield train_idx, test_idx


@dataclass
class PurgedKFoldInner:
    """Contiguous-block CV used to build out-of-fold base predictions.

    The meta learner must be fitted on predictions the base models made for data
    they had not seen; otherwise it observes near-perfect in-sample base output,
    concludes both base models are infallible, and produces a wildly
    overconfident live probability. Blocks are contiguous (not shuffled) and
    purged on both sides, because a test block here has training data after it
    as well as before.
    """

    n_folds: int = 4
    embargo: int = 36

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        fold_size = n_samples // self.n_folds
        if fold_size <= 0:
            raise ValueError(f"{n_samples} samples is too few for {self.n_folds} folds")
        idx = np.arange(n_samples)
        for k in range(self.n_folds):
            test_start = k * fold_size
            test_end = n_samples if k == self.n_folds - 1 else (k + 1) * fold_size
            test_mask = (idx >= test_start) & (idx < test_end)
            # Two-sided purge around the test block.
            purge_mask = (idx >= test_start - self.embargo) & (idx < test_end + self.embargo)
            train_idx = idx[~purge_mask]
            test_idx = idx[test_mask]
            if len(train_idx) < 50 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx
