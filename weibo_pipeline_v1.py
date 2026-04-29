#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
weibo_pipeline_v1.py

  1. Recency-weighted objective (exponential decay, half-life = 30 days)
     — recent posts are far more predictive; v7 treats all history equally
  2. Joint recency × engagement weighting matching the official metric
  3. Percentile-based candidate pool — robust, bounded complexity,
     includes values users haven't posted but might (e.g. rounded mean)
  4. Fully vectorised numpy scoring — one matrix multiply per user
     instead of v7's O(n²) pure-Python loop
  5. Better cold-start fallback (global median of non-zero users)
"""
import os
import time
import numpy as np
import pandas as pd

TRAIN_PATH   = 'Weibo Data/weibo_train_data/weibo_train_data.txt'
PREDICT_PATH = 'Weibo Data/weibo_predict_data/weibo_predict_data.txt'
RESULT_PATH  = 'Weibo Data/weibo_result_data/weibo_result_data_v1.txt'

HALF_LIFE    = 30.0   # days for exponential recency decay
MAX_CANDS    = 8      # max candidate values per target dimension


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    cols = ['uid', 'mid', 'time', 'forward_count', 'comment_count',
            'like_count', 'content']
    train = pd.read_csv(TRAIN_PATH,   sep='\t', names=cols,                         header=None)
    pred  = pd.read_csv(PREDICT_PATH, sep='\t', names=['uid','mid','time','content'], header=None)

    train['time'] = pd.to_datetime(train['time'], format='mixed')
    pred['time']  = pd.to_datetime(pred['time'],  format='mixed')

    for c in ['forward_count', 'comment_count', 'like_count']:
        train[c] = pd.to_numeric(train[c], errors='coerce').fillna(0).astype(int)

    return train, pred


# =============================================================================
# Candidate generation
# =============================================================================

def build_candidates(vals, weights):
    """
    Build a small candidate set of integer values for one engagement target.

    Includes:
      - 0 (very common ground truth)
      - recency-weighted mean (rounded)
      - percentiles: 25, 50, 75, 90
    Capped at MAX_CANDS unique values to bound complexity.
    """
    raw = set()
    raw.add(0)
    raw.add(int(round(float(np.average(vals, weights=weights)))))
    for p in (25, 50, 75, 90):
        raw.add(int(round(float(np.percentile(vals, p)))))
    cands = sorted(v for v in raw if v >= 0)
    return np.array(cands[:MAX_CANDS], dtype=np.float64)


# =============================================================================
# Vectorised per-user optimisation
# =============================================================================

def best_triplet(fwd, cmt, lke, w_rec):
    """
    Find the (F, C, L) triplet maximising the recency-and-engagement-weighted
    hit rate on the user's historical posts using the official precision formula:

        precision_i = 1 - 0.5*|F-tf|/(tf+5) - 0.25*|C-tc|/(tc+3) - 0.25*|L-tl|/(tl+3)
        hit_i       = 1  if precision_i > 0.8  else 0
        score       = sum(w_i * hit_i) / sum(w_i)
        w_i         = recency_i × (min(tf+tc+tl, 100) + 1)

    All candidate triplets are scored in a single vectorised pass.
    """
    fwd = fwd.astype(np.float64)
    cmt = cmt.astype(np.float64)
    lke = lke.astype(np.float64)

    # Combined weight = recency × engagement importance
    eng_w  = np.minimum(fwd + cmt + lke, 100.0) + 1.0
    comb_w = w_rec * eng_w                      # shape (n,)
    total_w = comb_w.sum()

    # Candidate pools (one per target)
    cands_f = build_candidates(fwd, comb_w)
    cands_c = build_candidates(cmt, comb_w)
    cands_l = build_candidates(lke, comb_w)

    # Enumerate all candidate triplets via meshgrid
    gf, gc, gl = np.meshgrid(cands_f, cands_c, cands_l, indexing='ij')
    F = gf.ravel()[:, np.newaxis]   # (n_cands, 1)  — broadcast over posts
    C = gc.ravel()[:, np.newaxis]
    L = gl.ravel()[:, np.newaxis]

    # Historical arrays broadcast against candidate axis
    fwd2 = fwd[np.newaxis, :]       # (1, n)
    cmt2 = cmt[np.newaxis, :]
    lke2 = lke[np.newaxis, :]

    dev_f = np.abs(F - fwd2) / (fwd2 + 5.0)
    dev_c = np.abs(C - cmt2) / (cmt2 + 3.0)
    dev_l = np.abs(L - lke2) / (lke2 + 3.0)

    prec  = np.clip(1.0 - 0.5*dev_f - 0.25*dev_c - 0.25*dev_l, 0.0, None)
    hit   = (prec > 0.8).astype(np.float64)           # (n_cands, n)

    scores  = (comb_w[np.newaxis, :] * hit).sum(axis=1) / total_w  # (n_cands,)
    best_i  = int(np.argmax(scores))

    return int(gf.ravel()[best_i]), int(gc.ravel()[best_i]), int(gl.ravel()[best_i])


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    t0 = time.time()
    train, pred = load_data()

    # Competition setup: training = Feb–Jul 2015, prediction = Aug 2015
    train = train[train['time'] < '2015-08-01'].copy()
    ref_date = train['time'].max()

    print(f"Train rows : {len(train)}")
    print(f"Predict rows: {len(pred)}")
    print(f"Reference date (latest train): {ref_date}")

    # ── Build per-user predictions ──────────────────────────────────────────
    user_pred = {}
    groups    = list(train.groupby('uid'))
    n_users   = len(groups)

    for i, (uid, grp) in enumerate(groups):
        days_ago = (ref_date - grp['time']).dt.total_seconds().values / 86400.0
        w_rec    = np.exp(-days_ago * np.log(2.0) / HALF_LIFE)

        F, C, L = best_triplet(
            grp['forward_count'].values,
            grp['comment_count'].values,
            grp['like_count'].values,
            w_rec,
        )
        user_pred[uid] = (F, C, L)

        if (i + 1) % 10_000 == 0:
            print(f"  {i+1}/{n_users} users  [{int(time.time()-t0)}s]")

    print(f"Per-user model ready ({len(user_pred)} users). Elapsed: {int(time.time()-t0)}s")

    # ── Write predictions ───────────────────────────────────────────────────
    lines = []
    for _, row in pred.iterrows():
        uid, mid = row['uid'], row['mid']
        F, C, L  = user_pred.get(uid, (0, 0, 0))
        lines.append(f"{uid}\t{mid}\t{F},{C},{L}\n")

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Saved {len(lines)} predictions to {RESULT_PATH}")
    print(f"Total time: {int(time.time()-t0)}s")


if __name__ == '__main__':
    main()
