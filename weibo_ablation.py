#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
weibo_ablation.py

Ablation study — sweep 5.  Baseline: v9 (strip @mentions, ALPHA=12, no decay,
rw=30, ±2 refinement, test score 0.3157).

Validation split now mirrors the actual submission scenario:
  Train: Feb–Jun 2015  (time < 2015-07-01)
  Val  : Jul 2015      (2015-07-01 <= time < 2015-08-01)
  (Predict file is entirely Aug 2015, so Jul is one month prior — same lag
   as the old Jun split was to the old Jun baseline.)

Metric (competition):
  precision_i = 1 - 0.5*|F-tf|/(tf+5) - 0.25*|C-tc|/(tc+3) - 0.25*|L-tl|/(tl+3)
  hit_i       = precision_i > 0.8
  weight_i    = min(tf+tc+tl, 100) + 1
  score       = sum(weight_i * hit_i) / sum(weight_i)

Optimisation: TF-IDF and w_content arrays are computed once per user and reused
across all variants, so the study runs in roughly the same time as one pipeline run.
"""

import math, time, sys, re
import numpy as np
import pandas as pd
from collections import defaultdict

TRAIN_PATH    = 'Weibo Data/weibo_train_data/weibo_train_data.txt'
MAX_VAL_USERS        = 3000  # sample for speed; set None for full eval
MAX_TRAIN_POSTS_USER = 300   # cap training history per user (keep most recent)
MAX_VAL_POSTS_USER   = 20    # cap val posts per user (random sample)
MAX_CANDS     = 8
HALF_LIFE     = 90.0
RECENCY_WINDOW = 30.0


# =============================================================================
# Data loading
# =============================================================================

def load_and_split(seed=42):
    cols = ['uid', 'mid', 'time', 'forward_count', 'comment_count',
            'like_count', 'content']
    df = pd.read_csv(TRAIN_PATH, sep='\t', names=cols, header=None)
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    for c in ['forward_count', 'comment_count', 'like_count']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    train = df[df['time'] < '2015-07-01'].copy()
    val   = df[(df['time'] >= '2015-07-01') & (df['time'] < '2015-08-01')].copy()
    known = set(train['uid'].unique())
    val   = val[val['uid'].isin(known)].copy()

    if MAX_VAL_USERS is not None:
        rng     = np.random.default_rng(seed)
        sampled = rng.choice(val['uid'].unique(),
                             size=min(MAX_VAL_USERS, val['uid'].nunique()),
                             replace=False)
        val = val[val['uid'].isin(sampled)].copy()

    return train, val


# =============================================================================
# Content preprocessing
# =============================================================================

_RE_URL      = re.compile(r'http\S+')
_RE_MENTION  = re.compile(r'@\S+')
_RE_HASH_SEP = re.compile(r'#')          # strip # delimiters, keep hashtag text
_RE_HASH_ALL = re.compile(r'#[^#]*#')   # strip entire #topic# tokens

def preprocess(text, cfg):
    s = str(text)
    mode = cfg.get('preprocess', 'none')
    if mode == 'none':
        return s
    if 'url' in mode:
        s = _RE_URL.sub('', s)
    if 'mention' in mode:
        s = _RE_MENTION.sub('', s)
    if 'hash_sep' in mode:        # keep hashtag words, remove # chars
        s = _RE_HASH_SEP.sub('', s)
    elif 'hash_all' in mode:      # remove entire #topic# blocks
        s = _RE_HASH_ALL.sub('', s)
    return s


# =============================================================================
# TF-IDF bigrams
# =============================================================================

def to_bigrams(text):
    s = str(text)
    return [s[i:i+2] for i in range(len(s) - 1)]


def build_tfidf_vecs(contents, cfg):
    n = len(contents)
    if n == 0:
        return [], [], {}
    tf_list, df = [], defaultdict(int)
    for text in contents:
        bgs = to_bigrams(preprocess(text, cfg))
        tf = defaultdict(int)
        for bg in bgs:
            tf[bg] += 1
        tf_list.append(tf)
        for bg in tf:
            df[bg] += 1
    idf = {bg: math.log(n / cnt) for bg, cnt in df.items() if cnt < n}
    vecs, norms = [], []
    for tf in tf_list:
        vec  = {bg: freq * idf[bg] for bg, freq in tf.items() if bg in idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) if vec else 0.0
        vecs.append(vec)
        norms.append(norm)
    return vecs, norms, idf


def pred_tfidf_vec(text, idf, cfg):
    bgs = to_bigrams(preprocess(text, cfg))
    tf = defaultdict(int)
    for bg in bgs:
        tf[bg] += 1
    vec  = {bg: freq * idf[bg] for bg, freq in tf.items() if bg in idf}
    norm = math.sqrt(sum(v * v for v in vec.values())) if vec else 0.0
    return vec, norm


def cosine_sim(pv, pn, tv, tn):
    if pn == 0 or tn == 0:
        return 0.0
    return sum(pv.get(bg, 0.0) * w for bg, w in tv.items()) / (pn * tn)


# =============================================================================
# Candidate generation + scoring
# =============================================================================

def build_candidates(vals, weights, percentiles, include_zero, max_cands, weighted_percentiles):
    raw = set()
    if include_zero:
        raw.add(0)
    raw.add(int(round(float(np.average(vals, weights=weights)))))
    for p in percentiles:
        if weighted_percentiles:
            # weighted percentile via sorted CDF
            order  = np.argsort(vals)
            sv     = vals[order]
            sw     = weights[order]
            cdf    = np.cumsum(sw) / sw.sum()
            idx    = np.searchsorted(cdf, p / 100.0)
            idx    = min(idx, len(sv) - 1)
            raw.add(int(round(float(sv[idx]))))
        else:
            raw.add(int(round(float(np.percentile(vals, p)))))
    return np.array(sorted(v for v in raw if v >= 0)[:max_cands], dtype=np.float64)


def score_triplets(F_col, C_col, L_col, fwd2, cmt2, lke2, comb_w, total_w):
    dev_f = np.abs(F_col - fwd2) / (fwd2 + 5.0)
    dev_c = np.abs(C_col - cmt2) / (cmt2 + 3.0)
    dev_l = np.abs(L_col - lke2) / (lke2 + 3.0)
    prec  = np.clip(1.0 - 0.5*dev_f - 0.25*dev_c - 0.25*dev_l, 0.0, None)
    hit   = (prec > 0.8).astype(np.float64)
    return (comb_w[np.newaxis, :] * hit).sum(axis=1) / total_w


def predict_one(fwd, cmt, lke, days_ago, w_content, cfg):
    """Return (F, C, L) for one prediction post under config cfg."""
    w_rec = np.exp(-days_ago * math.log(2.0) / cfg['half_life'])

    eng_cap = cfg.get('eng_cap', 100)
    if cfg['use_recency_window']:
        if cfg.get('engagement_boost', True):
            full_eng = np.minimum(fwd + cmt + lke, float(eng_cap)) + 1.0
        else:
            full_eng = np.ones(len(fwd))   # no engagement boost: flat weight within window
        eng_w = np.where(days_ago <= cfg['recency_window'], full_eng, 1.0)
    else:
        eng_w = np.ones(len(fwd))

    comb_w = (w_rec if cfg['use_recency_decay'] else np.ones(len(fwd))) * eng_w * w_content

    # top-k similarity filtering: keep only the k most similar posts
    topk = cfg.get('topk_sim')
    if topk and len(fwd) > topk:
        top_idx = np.argpartition(w_content, -topk)[-topk:]
        fwd      = fwd[top_idx]
        cmt      = cmt[top_idx]
        lke      = lke[top_idx]
        days_ago = days_ago[top_idx]
        comb_w   = comb_w[top_idx]

    total_w = comb_w.sum()

    fwd2 = fwd[np.newaxis, :]
    cmt2 = cmt[np.newaxis, :]
    lke2 = lke[np.newaxis, :]

    mc  = cfg.get('max_cands', MAX_CANDS)
    wp  = cfg.get('weighted_percentiles', False)
    cands_f = build_candidates(fwd, comb_w, cfg['percentiles'], cfg['include_zero'], mc, wp)
    cands_c = build_candidates(cmt, comb_w, cfg['percentiles'], cfg['include_zero'], mc, wp)
    cands_l = build_candidates(lke, comb_w, cfg['percentiles'], cfg['include_zero'], mc, wp)

    gf, gc, gl = np.meshgrid(cands_f, cands_c, cands_l, indexing='ij')
    F  = gf.ravel()[:, np.newaxis]
    C  = gc.ravel()[:, np.newaxis]
    L  = gl.ravel()[:, np.newaxis]

    gs      = score_triplets(F, C, L, fwd2, cmt2, lke2, comb_w, total_w)
    best_i  = int(np.argmax(gs))
    F0, C0, L0 = int(gf.ravel()[best_i]), int(gc.ravel()[best_i]), int(gl.ravel()[best_i])
    best_s  = gs[best_i]

    refine = cfg['use_local_refinement']
    if refine:
        r = refine if isinstance(refine, int) else 1
        d = list(range(-r, r + 1))
        nb_F = np.array([max(0, F0+df) for df in d for dc in d for dl in d], dtype=np.float64)[:, np.newaxis]
        nb_C = np.array([max(0, C0+dc) for df in d for dc in d for dl in d], dtype=np.float64)[:, np.newaxis]
        nb_L = np.array([max(0, L0+dl) for df in d for dc in d for dl in d], dtype=np.float64)[:, np.newaxis]
        nb_s  = score_triplets(nb_F, nb_C, nb_L, fwd2, cmt2, lke2, comb_w, total_w)
        nb_b  = int(np.argmax(nb_s))
        if nb_s[nb_b] > best_s:
            return int(nb_F[nb_b, 0]), int(nb_C[nb_b, 0]), int(nb_L[nb_b, 0])

    return F0, C0, L0


def hit_score(F, C, L, tf, tc, tl):
    prec = (1.0
            - 0.5  * abs(F - tf) / (tf + 5.0)
            - 0.25 * abs(C - tc) / (tc + 3.0)
            - 0.25 * abs(L - tl) / (tl + 3.0))
    hit    = 1.0 if prec > 0.8 else 0.0
    weight = float(min(tf + tc + tl, 100) + 1)
    return hit, weight


# =============================================================================
# Ablation configurations
# =============================================================================

# v12 is current best: strip @mentions, ALPHA=50, no decay, rw=35, ±2 (test 0.3192)
BASE = dict(
    preprocess           = 'mention',
    alpha                = 50.0,
    half_life            = HALF_LIFE,
    recency_window       = 35.0,
    use_recency_decay    = False,
    use_recency_window   = True,
    use_local_refinement = 2,
    percentiles          = [5, 10, 25, 50, 75, 90],
    include_zero         = True,
    max_cands            = 8,
    topk_sim             = None,
    weighted_percentiles = False,
)

VARIANTS = [
    # ── v12 baseline ───────────────────────────────────────────────────────────
    ("v12 baseline (A=50, rw=35, ±2)",     {**BASE}),
    ("±1 local refinement",                {**BASE, "use_local_refinement": 1}),
    ("±3 local refinement",                {**BASE, "use_local_refinement": 3}),
    ("no local refinement",                {**BASE, "use_local_refinement": False}),
]

# All unique alpha values we need — compute w_content for each once
ALPHAS = sorted({cfg['alpha'] for _, cfg in VARIANTS})


# =============================================================================
# Single-pass ablation (precompute TF-IDF + w_content once per user)
# =============================================================================

def run_ablation(train, val):
    ref_date   = train['time'].max()
    groups     = {uid: g for uid, g in train.groupby('uid')}
    val_by_uid = {}
    for row in val.itertuples(index=False):
        val_by_uid.setdefault(row.uid, []).append(row)

    n_variants = len(VARIANTS)
    totw  = np.zeros(n_variants)
    toth  = np.zeros(n_variants)

    # Group variant indices by preprocessing mode so TF-IDF is built once per mode
    PREPROCESS_MODES = sorted({cfg.get('preprocess', 'none') for _, cfg in VARIANTS})
    # For each (preprocess_mode, alpha) pair, collect variant indices
    from collections import defaultdict as _dd
    mode_alpha_vidx = _dd(list)
    for v_idx, (_, cfg) in enumerate(VARIANTS):
        key = (cfg.get('preprocess', 'none'), cfg['alpha'])
        mode_alpha_vidx[key].append(v_idx)

    t0 = time.time()
    for u_idx, (uid, pred_rows) in enumerate(val_by_uid.items()):
        if uid not in groups:
            continue

        grp = groups[uid]
        if MAX_TRAIN_POSTS_USER and len(grp) > MAX_TRAIN_POSTS_USER:
            grp = grp.nlargest(MAX_TRAIN_POSTS_USER, 'time')

        days_ago = (ref_date - grp['time']).dt.total_seconds().values / 86400.0
        fwd      = grp['forward_count'].values.astype(np.float64)
        cmt      = grp['comment_count'].values.astype(np.float64)
        lke      = grp['like_count'].values.astype(np.float64)
        raw_contents = grp['content'].tolist()

        # Build TF-IDF once per unique preprocessing mode
        tfidf_by_mode = {}
        for mode in PREPROCESS_MODES:
            dummy_cfg = {'preprocess': mode}
            tfidf_by_mode[mode] = build_tfidf_vecs(raw_contents, dummy_cfg)

        # cap val posts per user
        if MAX_VAL_POSTS_USER and len(pred_rows) > MAX_VAL_POSTS_USER:
            rng_u = np.random.default_rng(u_idx)
            idx   = rng_u.choice(len(pred_rows), size=MAX_VAL_POSTS_USER, replace=False)
            pred_rows = [pred_rows[i] for i in idx]

        for row in pred_rows:
            tf, tc, tl = row.forward_count, row.comment_count, row.like_count

            # Compute similarities once per preprocessing mode × alpha combination
            sims_by_mode = {}
            for mode in PREPROCESS_MODES:
                train_vecs, train_norms, idf = tfidf_by_mode[mode]
                dummy_cfg = {'preprocess': mode}
                pv, pn = pred_tfidf_vec(row.content, idf, dummy_cfg)
                sims_by_mode[mode] = np.array(
                    [cosine_sim(pv, pn, tv, tn) for tv, tn in zip(train_vecs, train_norms)],
                    dtype=np.float64)

            # w_content keyed by (mode, alpha)
            w_cont = {(mode, a): 1.0 + a * sims_by_mode[mode]
                      for mode in PREPROCESS_MODES for a in ALPHAS}

            for v_idx, (_, cfg) in enumerate(VARIANTS):
                key       = (cfg.get('preprocess', 'none'), cfg['alpha'])
                w_content = w_cont[key]
                F, C, L   = predict_one(fwd, cmt, lke, days_ago, w_content, cfg)
                hit, wt   = hit_score(F, C, L, tf, tc, tl)
                totw[v_idx] += wt
                toth[v_idx] += wt * hit

        if (u_idx + 1) % 500 == 0:
            elapsed = int(time.time() - t0)
            print(f"  {u_idx+1}/{len(val_by_uid)} users  [{elapsed}s]", flush=True)

    scores = np.where(totw > 0, toth / totw, 0.0)
    return scores


# =============================================================================
# Main
# =============================================================================

RESULT_FILE = 'ablation_results_sweep9.txt'


def main():
    print("Loading data …", flush=True)
    train, val = load_and_split()
    print(f"Train: {len(train):,} rows  |  Val: {len(val):,} rows  "
          f"({val['uid'].nunique():,} users)\n", flush=True)

    t0     = time.time()
    scores = run_ablation(train, val)
    elapsed = int(time.time() - t0)
    print(f"\nDone in {elapsed}s\n", flush=True)

    base  = scores[0]
    width = max(len(n) for n, _ in VARIANTS)
    lines = []
    lines.append(f"Ablation sweep 9 — local refinement radius (v12 baseline, Jul 2015 val)  ({MAX_VAL_USERS} sampled users)")
    lines.append(f"Total time: {elapsed}s\n")
    lines.append("=" * (width + 24))
    lines.append(f"{'Variant':<{width}}  {'Val score':>10}  {'delta':>10}")
    lines.append("-" * (width + 24))
    for i, (name, _) in enumerate(VARIANTS):
        delta  = scores[i] - base
        marker = "  <- baseline" if i == 0 else ""
        lines.append(f"{name:<{width}}  {scores[i]:>10.4f}  {delta:>+10.4f}{marker}")
    lines.append("=" * (width + 24))

    output = "\n".join(lines)
    # write to file (UTF-8)
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        f.write(output + "\n")
    # print safely (replace any non-ASCII)
    print(output.encode('ascii', errors='replace').decode('ascii'), flush=True)
    print(f"\nResults saved to {RESULT_FILE}", flush=True)


if __name__ == '__main__':
    main()
