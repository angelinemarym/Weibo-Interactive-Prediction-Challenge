#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
weibo_pipeline_v10.py

Improvement over v9: TF-IDF weighted bigram similarity (per-user IDF).

Problem with v9 (plain Jaccard):
  Weibo posts are short (~100 chars → ~98 bigrams each).  Two posts on
  completely different topics still share many common bigrams
  (punctuation, numbers, common words like "的/了/我").  Plain Jaccard
  similarity is dominated by these stop-bigrams and carries little
  topical signal → weight ≈ 1.0 for nearly all post pairs → zero effect.

Solution: TF-IDF weighted cosine similarity.
  For each user, compute IDF over their OWN training posts:
      idf(bg) = log(N / df(bg))   where df(bg) = # training posts containing bg.
  A bigram shared by ALL training posts (common filler) gets idf≈0.
  A bigram shared by only 1-2 training posts (topical) gets high idf.

  Similarity between prediction post p and training post t:
      sim(p, t) = dot(tfidf(p), tfidf(t)) / (||tfidf(p)|| * ||tfidf(t)||)

  Content weight: w_content(p, t) = 1 + ALPHA * sim(p, t)

Everything else is identical to v3:
  hl=90d, rw=30d, unweighted-percentile candidates {0,wm,p25,p50,p75,p90},
  single ±1 refinement.
"""
import os
import time
import math
import numpy as np
import pandas as pd
from collections import defaultdict

TRAIN_PATH     = 'Weibo Data/weibo_train_data/weibo_train_data.txt'
PREDICT_PATH   = 'Weibo Data/weibo_predict_data/weibo_predict_data.txt'
RESULT_PATH    = 'Weibo Data/weibo_result_data/weibo_result_data_v10.txt'

HALF_LIFE      = 90.0
RECENCY_WINDOW = 30.0
MAX_CANDS      = 8
ALPHA          = 3.0   # content similarity multiplier


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
# Per-user TF-IDF bigram similarity
# =============================================================================

def to_bigrams(text):
    s = str(text)
    return [s[i:i+2] for i in range(len(s) - 1)]


def build_tfidf_vecs(contents):
    """
    Build TF-IDF vectors for a list of post contents within ONE user.

    Returns:
      vecs : list of dict {bigram: tfidf_weight}
      norms: list of L2 norms (for cosine normalisation)
    """
    n = len(contents)
    if n == 0:
        return [], []

    # Term frequencies per post
    tf_list = []
    df = defaultdict(int)
    for text in contents:
        bgs = to_bigrams(text)
        tf = defaultdict(int)
        for bg in bgs:
            tf[bg] += 1
        tf_list.append(tf)
        for bg in tf:
            df[bg] += 1

    # IDF: log(N / df); bigrams appearing in all posts get idf=0
    idf = {bg: math.log(n / cnt) for bg, cnt in df.items() if cnt < n}

    # TF-IDF vectors and their L2 norms
    vecs, norms = [], []
    for tf in tf_list:
        vec = {}
        for bg, freq in tf.items():
            if bg in idf:
                vec[bg] = freq * idf[bg]
        norm = math.sqrt(sum(v * v for v in vec.values())) if vec else 0.0
        vecs.append(vec)
        norms.append(norm)

    return vecs, norms, idf


def cosine_sim(pred_vec, pred_norm, train_vec, train_norm):
    """Cosine similarity between two TF-IDF vectors (as dicts)."""
    if pred_norm == 0 or train_norm == 0:
        return 0.0
    dot = sum(pred_vec.get(bg, 0.0) * w for bg, w in train_vec.items())
    return dot / (pred_norm * train_norm)


def pred_tfidf_vec(text, idf):
    """Build TF-IDF vector for a prediction post using per-user IDF."""
    bgs = to_bigrams(text)
    tf = defaultdict(int)
    for bg in bgs:
        tf[bg] += 1
    vec = {bg: freq * idf[bg] for bg, freq in tf.items() if bg in idf}
    norm = math.sqrt(sum(v * v for v in vec.values())) if vec else 0.0
    return vec, norm


# =============================================================================
# Candidate generation + scoring (same as v3)
# =============================================================================

def build_candidates(vals, weights):
    raw = {0}
    raw.add(int(round(float(np.average(vals, weights=weights)))))
    for p in (25, 50, 75, 90):
        raw.add(int(round(float(np.percentile(vals, p)))))
    return np.array(sorted(v for v in raw if v >= 0)[:MAX_CANDS], dtype=np.float64)


def score_triplets(F_col, C_col, L_col, fwd2, cmt2, lke2, comb_w, total_w):
    dev_f = np.abs(F_col - fwd2) / (fwd2 + 5.0)
    dev_c = np.abs(C_col - cmt2) / (cmt2 + 3.0)
    dev_l = np.abs(L_col - lke2) / (lke2 + 3.0)
    prec  = np.clip(1.0 - 0.5*dev_f - 0.25*dev_c - 0.25*dev_l, 0.0, None)
    hit   = (prec > 0.8).astype(np.float64)
    return (comb_w[np.newaxis, :] * hit).sum(axis=1) / total_w


def best_triplet(fwd, cmt, lke, days_ago, w_content):
    fwd = fwd.astype(np.float64)
    cmt = cmt.astype(np.float64)
    lke = lke.astype(np.float64)

    w_rec    = np.exp(-days_ago * np.log(2.0) / HALF_LIFE)
    full_eng = np.minimum(fwd + cmt + lke, 100.0) + 1.0
    eng_w    = np.where(days_ago <= RECENCY_WINDOW, full_eng, 1.0)
    comb_w   = w_rec * eng_w * w_content
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
    F0, C0, L0 = int(gf.ravel()[best_i]), int(gc.ravel()[best_i]), int(gl.ravel()[best_i])
    best_score = grid_scores[best_i]

    deltas = (-1, 0, 1)
    nb_F = np.array([max(0, F0+dF) for dF in deltas for dC in deltas for dL in deltas],
                    dtype=np.float64)[:, np.newaxis]
    nb_C = np.array([max(0, C0+dC) for dF in deltas for dC in deltas for dL in deltas],
                    dtype=np.float64)[:, np.newaxis]
    nb_L = np.array([max(0, L0+dL) for dF in deltas for dC in deltas for dL in deltas],
                    dtype=np.float64)[:, np.newaxis]

    nb_scores = score_triplets(nb_F, nb_C, nb_L, fwd2, cmt2, lke2, comb_w, total_w)
    best_nb   = int(np.argmax(nb_scores))
    if nb_scores[best_nb] > best_score:
        return int(nb_F[best_nb, 0]), int(nb_C[best_nb, 0]), int(nb_L[best_nb, 0])
    return F0, C0, L0


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    t0 = time.time()
    train, pred = load_data()

    train    = train[train['time'] < '2015-08-01'].copy()
    ref_date = train['time'].max()

    print(f"Train rows  : {len(train)}")
    print(f"Predict rows: {len(pred)}")
    print(f"Config: hl={HALF_LIFE}d  rw={RECENCY_WINDOW}d  ALPHA={ALPHA}  tfidf-bigram")

    # Group predictions by uid
    pred_by_uid = {}
    for row in pred.itertuples(index=False):
        pred_by_uid.setdefault(row.uid, []).append(row)

    predictions = {}   # (uid, mid) -> (F, C, L)
    groups  = list(train.groupby('uid'))
    n_users = len(groups)

    for i, (uid, grp) in enumerate(groups):
        if uid not in pred_by_uid:
            continue

        days_ago = (ref_date - grp['time']).dt.total_seconds().values / 86400.0
        fwd      = grp['forward_count'].values
        cmt      = grp['comment_count'].values
        lke      = grp['like_count'].values
        contents = grp['content'].tolist()

        # Build per-user TF-IDF vectors for training posts
        train_vecs, train_norms, idf = build_tfidf_vecs(contents)

        for pred_row in pred_by_uid[uid]:
            # TF-IDF vector for this specific prediction post
            pv, pn = pred_tfidf_vec(pred_row.content, idf)

            # Content similarity weight for each training post
            w_cont = np.array(
                [1.0 + ALPHA * cosine_sim(pv, pn, tv, tn)
                 for tv, tn in zip(train_vecs, train_norms)],
                dtype=np.float64,
            )

            F, C, L = best_triplet(fwd, cmt, lke, days_ago, w_cont)
            predictions[(uid, pred_row.mid)] = (F, C, L)

        if (i + 1) % 5_000 == 0:
            print(f"  {i+1}/{n_users} users  [{int(time.time()-t0)}s]")

    print(f"Predictions: {len(predictions)}. Elapsed: {int(time.time()-t0)}s")

    lines = []
    for row in pred.itertuples(index=False):
        uid, mid = row.uid, row.mid
        F, C, L  = predictions.get((uid, mid), (0, 0, 0))
        lines.append(f"{uid}\t{mid}\t{F},{C},{L}\n")

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Saved {len(lines)} predictions to {RESULT_PATH}")
    print(f"Total time: {int(time.time()-t0)}s")


if __name__ == '__main__':
    main()
