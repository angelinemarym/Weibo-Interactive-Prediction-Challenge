#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
weibo_pipeline_v3.py

Key improvement over v2: local refinement (+/-1) after candidate search.

Problem with v2 (discrete candidate grid):
  The best (F, C, L) must lie exactly on one of the ~8^3 candidate intersections.
  The true optimum may be a neighbouring integer (e.g. F+1) that scores higher
  but is not in the candidate pool, especially near the precision=0.8 boundary.

Solution (validated on pre-Jul->Jul CV split):
  After the vectorised grid search identifies the best candidate (F0, C0, L0),
  try all 27 neighbours (F0+dF, C0+dC, L0+dL) for dF,dC,dL in {-1, 0, +1}
  and keep whichever triplet scores highest under the same metric.

CV scores (pre-Jul->Jul split):
  v6  (hl=90, rot=30)              : 0.3093
  v7 (hl=90, rot=30, +/-1 refine) : 0.3112  <- +0.0019
"""
import os
import time
import numpy as np
import pandas as pd

TRAIN_PATH       = 'Weibo Data/weibo_train_data/weibo_train_data.txt'
PREDICT_PATH     = 'Weibo Data/weibo_predict_data/weibo_predict_data.txt'
RESULT_PATH      = 'Weibo Data/weibo_result_data/weibo_result_data_v3.txt'

HALF_LIFE        = 90.0   # days -- exponential recency decay half-life
RECENCY_WINDOW   = 30.0   # days -- engagement weight applied only within this window
MAX_CANDS        = 8      # max candidate values per target dimension


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    cols = ['uid', 'mid', 'time', 'forward_count', 'comment_count',
            'like_count', 'content']
    train = pd.read_csv(TRAIN_PATH,   sep='\t', names=cols, header=None)
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
# Scoring helper
# =============================================================================

def score_triplets(F_col, C_col, L_col, fwd2, cmt2, lke2, comb_w, total_w):
    """
    Vectorised scoring for a batch of (F, C, L) candidate triplets.

    F_col, C_col, L_col : (n_cands, 1) float arrays
    fwd2, cmt2, lke2    : (1, n_posts) float arrays
    Returns scores array of shape (n_cands,).
    """
    dev_f = np.abs(F_col - fwd2) / (fwd2 + 5.0)
    dev_c = np.abs(C_col - cmt2) / (cmt2 + 3.0)
    dev_l = np.abs(L_col - lke2) / (lke2 + 3.0)

    prec  = np.clip(1.0 - 0.5*dev_f - 0.25*dev_c - 0.25*dev_l, 0.0, None)
    hit   = (prec > 0.8).astype(np.float64)
    return (comb_w[np.newaxis, :] * hit).sum(axis=1) / total_w


# =============================================================================
# Vectorised per-user optimisation + local refinement
# =============================================================================

def best_triplet(fwd, cmt, lke, days_ago):
    """
    1. Grid search over candidate pool (up to MAX_CANDS^3 = 512 triplets).
    2. Local refinement: try all 27 neighbours (+/-1) of best grid result.

    Objective: hybrid-weighted hit rate
        precision_i  = 1 - 0.5*|F-tf|/(tf+5) - 0.25*|C-tc|/(tc+3) - 0.25*|L-tl|/(tl+3)
        hit_i        = 1  if precision_i > 0.8
        w_recency_i  = exp(-days_ago_i * ln2 / HALF_LIFE)
        w_engage_i   = (min(tf+tc+tl, 100)+1)  if days_ago_i <= RECENCY_WINDOW  else  1
        combined_i   = w_recency_i * w_engage_i
        score        = sum(combined_i * hit_i) / sum(combined_i)
    """
    fwd = fwd.astype(np.float64)
    cmt = cmt.astype(np.float64)
    lke = lke.astype(np.float64)

    w_rec    = np.exp(-days_ago * np.log(2.0) / HALF_LIFE)
    full_eng = np.minimum(fwd + cmt + lke, 100.0) + 1.0
    eng_w    = np.where(days_ago <= RECENCY_WINDOW, full_eng, 1.0)
    comb_w   = w_rec * eng_w
    total_w  = comb_w.sum()

    # Broadcast-ready historical arrays
    fwd2 = fwd[np.newaxis, :]
    cmt2 = cmt[np.newaxis, :]
    lke2 = lke[np.newaxis, :]

    # ── Stage 1: grid search over candidate pool ────────────────────────────
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

    # ── Stage 2: local refinement (+/-1 around best grid candidate) ─────────
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
# Main pipeline
# =============================================================================

def main():
    t0 = time.time()
    train, pred = load_data()

    # Competition setup: training = Feb-Jul 2015, prediction = Aug 2015
    train = train[train['time'] < '2015-08-01'].copy()
    ref_date = train['time'].max()

    print(f"Train rows  : {len(train)}")
    print(f"Predict rows: {len(pred)}")
    print(f"Reference date: {ref_date}")
    print(f"Config: half_life={HALF_LIFE}d  recency_window={RECENCY_WINDOW}d  +/-1 refinement")

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
