# =============================================================================
# data_loader.py
# Shared Data Loader — DL Pipeline
# Stress Detection Project — Lebanese University 2026
#
# Used by all training scripts (LSTM, CNN, ST-GCN).
# Loads .npy windows, assigns labels from filenames, tags each window with
# its participant, and produces Leave-One-Subject-Out (LOSO) splits.
#
# IMPORTANT — this loader does NO normalization.
#   Normalization must happen INSIDE each LOSO fold in the training script,
#   fit on the training participants only. Normalizing here would leak the
#   test participant's statistics into training (the bug fixed in the ML phase).
#
# Shape-agnostic: it slices on axis 0 (window count) and never inspects the
#   rest of the shape, so the same code loads:
#       graph    windows  (140, 15, 3)
#       mtf      windows  (64, 64, 5)
#       valence  windows  (140, 1)
#
# Usage in a training notebook:
#       from data_loader import load_windows, loso_splits
#       X, y, pids = load_windows(DL_PATH, file_type="valence")
#       for Xtr, ytr, Xte, yte, held_out in loso_splits(X, y, pids):
#           ... normalize on Xtr, train, test on Xte ...
# =============================================================================

import os
import logging
import numpy as np

# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================

# Task -> label mapping (locked decision: T1 baseline=0, T2/T3 stressed=1)
LABEL_MAP = {"T1": 0, "T2": 1, "T3": 1}

# Which tasks to feed the model. Confirmed with team: all three included.
# To exclude T2 later, change this one line to ["T1", "T3"].
DEFAULT_TRAIN_TASKS = ["T1", "T2", "T3"]

# =============================================================================
# LOGGING
# =============================================================================

log = logging.getLogger("data_loader")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


# =============================================================================
# LOAD
# =============================================================================

def load_windows(dl_path,
                 file_type,
                 train_tasks=None,
                 label_map=None):
    """
    Load all windows of one file type into aligned arrays.

    Parameters
    ----------
    dl_path     : folder containing p{N} subfolders
    file_type   : "graph" | "mtf" | "valence"
    train_tasks : list of tasks to include (default ["T1","T2","T3"])
    label_map   : task -> label dict (default {T1:0, T2:1, T3:1})

    Returns
    -------
    X    : np.array, shape (total_windows, <window shape>)
    y    : np.array, shape (total_windows,)  — labels
    pids : np.array, shape (total_windows,)  — participant id per window
    """
    if train_tasks is None:
        train_tasks = DEFAULT_TRAIN_TASKS
    if label_map is None:
        label_map = LABEL_MAP

    windows, labels, participant_ids = [], [], []

    # ── Discover participant folders, in a fixed sorted order ──────────────
    folders = [f for f in os.listdir(dl_path)
               if f.startswith("p") and f[1:].isdigit()
               and os.path.isdir(os.path.join(dl_path, f))]
    # sort numerically by participant number, not lexically (p2 before p10)
    folders = sorted(folders, key=lambda f: int(f[1:]))

    log.info("=" * 64)
    log.info(f"LOADING file_type='{file_type}'  tasks={train_tasks}")
    log.info(f"Source: {dl_path}")
    log.info(f"Found {len(folders)} participant folders")
    log.info("-" * 64)

    files_loaded  = 0
    files_missing = 0

    for folder in folders:
        pid = int(folder[1:])

        for task in train_tasks:
            fname = f"{folder}_{task}_{file_type}.npy"
            fpath = os.path.join(dl_path, folder, fname)

            if not os.path.exists(fpath):
                files_missing += 1
                log.info(f"  MISSING  {fname}")
                continue

            arr = np.load(fpath)            # (num_windows, <window shape>)
            n   = arr.shape[0]
            lbl = label_map[task]

            # slice on axis 0 only — keep the rest of the shape untouched
            for w in range(n):
                windows.append(arr[w])
                labels.append(lbl)
                participant_ids.append(pid)

            files_loaded += 1

    # ── Convert to arrays ──────────────────────────────────────────────────
    X    = np.array(windows)
    y    = np.array(labels, dtype=np.int64)
    pids = np.array(participant_ids, dtype=np.int64)

    # ── Alignment guard (cheap insurance against lockstep desync) ──────────
    assert len(X) == len(y) == len(pids), \
        f"ALIGNMENT ERROR: X={len(X)} y={len(y)} pids={len(pids)}"

    # ── Summary report ──────────────────────────────────────────────────────
    log.info("-" * 64)
    log.info(f"Files loaded   : {files_loaded}")
    log.info(f"Files missing  : {files_missing}")
    log.info(f"Total windows  : {len(X)}")
    if len(X) > 0:
        log.info(f"Window shape   : {X.shape[1:]}")
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        ratio = (n_pos / n_neg) if n_neg > 0 else float('inf')
        log.info(f"Label 0 (calm) : {n_neg}")
        log.info(f"Label 1 (stress): {n_pos}")
        log.info(f"Class balance  : {ratio:.2f} : 1  (stress : calm)")
        log.info(f"Participants   : {sorted(set(pids.tolist()))}")
    log.info("=" * 64)

    return X, y, pids


# =============================================================================
# LOSO SPLITS
# =============================================================================

def loso_splits(X, y, pids, verbose=True):
    """
    Generate Leave-One-Subject-Out splits.

    Yields, for each participant in sorted order:
        X_train, y_train, X_test, y_test, held_out_pid

    Folds whose test set is empty or single-class are skipped with a warning
    (these produce meaningless accuracy and can crash some metrics).

    Normalization is intentionally NOT applied here.
    """
    unique_pids = sorted(set(pids.tolist()))

    if verbose:
        log.info(f"\nLOSO: {len(unique_pids)} rounds planned")

    for held_out in unique_pids:
        test_mask  = (pids == held_out)
        train_mask = ~test_mask

        X_test,  y_test  = X[test_mask],  y[test_mask]
        X_train, y_train = X[train_mask], y[train_mask]

        # ── Guard: empty or single-class test fold ──────────────────────────
        if len(y_test) == 0:
            log.info(f"  Round p{held_out}: SKIPPED — empty test set")
            continue
        if len(set(y_test.tolist())) < 2:
            only = y_test[0]
            log.info(f"  Round p{held_out}: SKIPPED — "
                     f"test set is single-class (all label {only})")
            continue

        if verbose:
            log.info(f"  Round p{held_out}: "
                     f"train={len(y_train)}  test={len(y_test)}")

        yield X_train, y_train, X_test, y_test, held_out


# =============================================================================
# QUICK SELF-TEST  (run this file directly to verify it works)
# =============================================================================

if __name__ == "__main__":
    import sys

    DL_PATH = r"C:\ML\StressDetection\DL"
    test_type = sys.argv[1] if len(sys.argv) > 1 else "valence"

    X, y, pids = load_windows(DL_PATH, file_type=test_type)

    if len(X) > 0:
        # Show the first 2 LOSO rounds as a dry run
        log.info("\nDry run of first 2 LOSO rounds:")
        count = 0
        for Xtr, ytr, Xte, yte, ho in loso_splits(X, y, pids, verbose=True):
            count += 1
            if count >= 2:
                break
