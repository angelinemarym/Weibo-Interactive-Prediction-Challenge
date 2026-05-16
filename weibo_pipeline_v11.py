#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
weibo_pipeline_v11.py

Changes from v9: ALPHA 12 -> 30, RECENCY_WINDOW 30 -> 45.

Sweep 5 ablation val scores (Jul-2015, 3k users):
  v9 baseline (A=12, rw=30): 0.3689
  ALPHA=30:                  +0.0048
  rw=45d:                    +0.0032
"""
import os
import re
import time
import math
import numpy as np
import pandas as pd
from collections import defaultdict

_RE_MENTION = re.compile(r'@\S+')

TRAIN_PATH     = 'Weibo Data/weibo_train_data/weibo_train_data.txt'
PREDICT_PATH   = 'Weibo Data/weibo_predict_data/weibo_predict_data.txt'
RESULT_PATH    = 'Weibo Data/weibo_result_data/weibo_result_data_v11.txt'

RECENCY_WINDOW = 45.0
MAX_CANDS      = 8
ALPHA          = 30.0


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

def strip_mentions(text):
    return _RE_MENTION.sub('', str(text))


def to_bigrams(text):
    s = strip_mentions(text)
    return [s[i:i+2] for i in range(len(s) - 1)]


def build_tfidf_vecs(contents):
    n = len(contents)
    if n == 0:
        return [], [], {}

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

    idf = {bg: math.log(n / cnt) for bg, cnt in df.items() if cnt < n}

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
    if pred_norm == 0 or train_norm == 0:
        return 0.0
    dot = sum(pred_vec.get(bg, 0.0) * w for bg, w in train_vec.items())
    return dot / (pred_norm * train_norm)


def pred_tfidf_vec(text, idf):
    bgs = to_bigrams(text)
    tf = defaultdict(int)
    for bg in bgs:
        tf[bg] += 1
    vec = {bg: freq * idf[bg] for bg, freq in tf.items() if bg in idf}
    norm = math.sqrt(sum(v * v for v in vec.values())) if vec else 0.0
    return vec, norm


# =============================================================================
# Candidate generation + scoring
# =============================================================================

def build_candidates(vals, weights):
    raw = {0}
    raw.add(int(round(float(np.average(vals, weights=weights)))))
    for p in (5, 10, 25, 50, 75, 90):   # p5 added vs v12
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

    full_eng = np.minimum(fwd + cmt + lke, 100.0) + 1.0
    eng_w    = np.where(days_ago <= RECENCY_WINDOW, full_eng, 1.0)
    comb_w   = eng_w * w_content  # no recency decay
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

    deltas = (-2, -1, 0, 1, 2)
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
    print(f"Config: no-recency-decay  rw={RECENCY_WINDOW}d  ALPHA={ALPHA}  tfidf-bigram  p5+p10  +-2-refinement  strip-mentions")

    pred_by_uid = {}
    for row in pred.itertuples(index=False):
        pred_by_uid.setdefault(row.uid, []).append(row)

    predictions = {}
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

        train_vecs, train_norms, idf = build_tfidf_vecs(contents)

        for pred_row in pred_by_uid[uid]:
            pv, pn = pred_tfidf_vec(pred_row.content, idf)

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
