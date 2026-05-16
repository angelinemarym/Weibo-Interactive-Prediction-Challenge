#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cv_sweep.py

CV hyperparameter sweep for (HALF_LIFE, RECENCY_WINDOW).

Split: train on pre-July 2015 data, evaluate on July 2015 posts.
Metric: unweighted hit rate over all July posts.
  precision_i = 1 - 0.5*|F-tf|/(tf+5) - 0.25*|C-tc|/(tc+3) - 0.25*|L-tl|/(tl+3)
  hit_i = precision_i > 0.8
  score = mean(hit_i)

Uses the exact v3 algorithm (grid search + single +/-1 refinement).
"""
import time
import numpy as np
import pandas as pd
from itertools import product

TRAIN_PATH = 'Weibo Data/weibo_train_data/weibo_train_data.txt'
MAX_CANDS  = 8

HALF_LIVES       = [30, 45, 60, 75, 90, 120, 150, 180]
RECENCY_WINDOWS  = [7, 14, 21, 30, 45, 60]


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    cols = ['uid', 'mid', 'time', 'forward_count', 'comment_count',
            'like_count', 'content']
    train = pd.read_csv(TRAIN_PATH, sep='\t', names=cols, header=None)
    train['time'] = pd.to_datetime(train['time'], format='mixed')
    for c in ['forward_count', 'comment_count', 'like_count']:
        train[c] = pd.to_numeric(train[c], errors='coerce').fillna(0).astype(int)
    return train


# =============================================================================
# Candidate generation (same as v3)
# =============================================================================

def build_candidates(vals, weights):
    raw = {0}
    raw.add(int(round(float(np.average(vals, weights=weights)))))
    for p in (25, 50, 75, 90):
        raw.add(int(round(float(np.percentile(vals, p)))))
    return np.array(sorted(v for v in raw if v >= 0)[:MAX_CANDS], dtype=np.float64)


# =============================================================================
# Scoring helper (same as v3)
# =============================================================================

def score_triplets(F_col, C_col, L_col, fwd2, cmt2, lke2, comb_w, total_w):
    dev_f = np.abs(F_col - fwd2) / (fwd2 + 5.0)
    dev_c = np.abs(C_col - cmt2) / (cmt2 + 3.0)
    dev_l = np.abs(L_col - lke2) / (lke2 + 3.0)
    prec  = np.clip(1.0 - 0.5*dev_f - 0.25*dev_c - 0.25*dev_l, 0.0, None)
    hit   = (prec > 0.8).astype(np.float64)
    return (comb_w[np.newaxis, :] * hit).sum(axis=1) / total_w


# =============================================================================
# v3 algorithm: grid search + single +/-1 refinement
# =============================================================================

def best_triplet(fwd, cmt, lke, days_ago, hl, rw):
    fwd = fwd.astype(np.float64)
    cmt = cmt.astype(np.float64)
    lke = lke.astype(np.float64)

    w_rec    = np.exp(-days_ago * np.log(2.0) / hl)
    full_eng = np.minimum(fwd + cmt + lke, 100.0) + 1.0
    eng_w    = np.where(days_ago <= rw, full_eng, 1.0)
    comb_w   = w_rec * eng_w
    total_w  = comb_w.sum()

    fwd2 = fwd[np.newaxis, :]
    cmt2 = cmt[np.newaxis, :]
    lke2 = lke[np.newaxis, :]

    cands_f = build_candidates(fwd, comb_w)
    cands_c = build_candidates(cmt, comb_w)
    cands_l = build_candidates(lke, comb_w)

    gf, gc, gl = np.meshgrid(cands_f, cands_c, cands_l, indexing='ij')
    F = gf.ravel()[:, np.newaxis]
    C = gc.ravel()[:, np.newaxis]
    L = gl.ravel()[:, np.newaxis]

    grid_scores = score_triplets(F, C, L, fwd2, cmt2, lke2, comb_w, total_w)
    best_i = int(np.argmax(grid_scores))
    F0 = int(gf.ravel()[best_i])
    C0 = int(gc.ravel()[best_i])
    L0 = int(gl.ravel()[best_i])
    best_score = grid_scores[best_i]

    deltas = (-1, 0, 1)
    nb_F = np.array([max(0, F0+dF) for dF in deltas for dC in deltas for dL in deltas],
                    dtype=np.float64)[:, np.newaxis]
    nb_C = np.array([max(0, C0+dC) for dF in deltas for dC in deltas for dL in deltas],
                    dtype=np.float64)[:, np.newaxis]
    nb_L = np.array([max(0, L0+dL) for dF in deltas for dC in deltas for dL in deltas],
                    dtype=np.float64)[:, np.newaxis]

    nb_scores = score_triplets(nb_F, nb_C, nb_L, fwd2, cmt2, lke2, comb_w, total_w)
    best_nb = int(np.argmax(nb_scores))

    if nb_scores[best_nb] > best_score:
        return int(nb_F[best_nb, 0]), int(nb_C[best_nb, 0]), int(nb_L[best_nb, 0])
    return F0, C0, L0


# =============================================================================
# CV evaluation for one (hl, rw) pair
# =============================================================================

def cv_evaluate(train_data, val_data, hl, rw):
    ref_date = train_data['time'].max()
    groups   = train_data.groupby('uid')

    user_pred = {}
    for uid, grp in groups:
        days_ago = (ref_date - grp['time']).dt.total_seconds().values / 86400.0
        F, C, L = best_triplet(
            grp['forward_count'].values,
            grp['comment_count'].values,
            grp['like_count'].values,
            days_ago, hl, rw,
        )
        user_pred[uid] = (F, C, L)

    # Evaluate on validation posts
    hits = []
    for _, row in val_data.iterrows():
        uid = row['uid']
        F, C, L = user_pred.get(uid, (0, 0, 0))
        tf = float(row['forward_count'])
        tc = float(row['comment_count'])
        tl = float(row['like_count'])
        prec = (1.0
                - 0.5 * abs(F - tf) / (tf + 5.0)
                - 0.25 * abs(C - tc) / (tc + 3.0)
                - 0.25 * abs(L - tl) / (tl + 3.0))
        hits.append(prec > 0.8)

    return float(np.mean(hits))


# =============================================================================
# Main sweep
# =============================================================================

def main():
    t0 = time.time()
    print("Loading data...")
    train = load_data()

    # CV split: train on pre-July, validate on July
    train_data = train[train['time'] < '2015-07-01'].copy()
    val_data   = train[(train['time'] >= '2015-07-01') &
                       (train['time'] <  '2015-08-01')].copy()

    print(f"Train size: {len(train_data)}  Val size: {len(val_data)}")
    print(f"Val users with train data: {val_data['uid'].isin(train_data['uid']).sum()}")
    print()

    results = []
    combos  = list(product(HALF_LIVES, RECENCY_WINDOWS))
    n_combos = len(combos)

    for idx, (hl, rw) in enumerate(combos):
        t1 = time.time()
        score = cv_evaluate(train_data, val_data, hl, rw)
        elapsed = time.time() - t1
        results.append((score, hl, rw))
        print(f"[{idx+1:2d}/{n_combos}] hl={hl:3d}  rw={rw:2d}  CV={score:.4f}  ({elapsed:.0f}s)")

    print()
    print("=== Top 10 configurations ===")
    results.sort(reverse=True)
    for score, hl, rw in results[:10]:
        print(f"  hl={hl:3d}  rw={rw:2d}  CV={score:.4f}")

    best_score, best_hl, best_rw = results[0]
    print(f"\nBest: HALF_LIFE={best_hl}  RECENCY_WINDOW={best_rw}  CV={best_score:.4f}")
    print(f"Total time: {int(time.time()-t0)}s")


if __name__ == '__main__':
    main()
