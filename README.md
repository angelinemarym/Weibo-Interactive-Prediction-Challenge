# Weibo Engagement Prediction — Pipeline v13

## Overview

This project predicts Weibo post engagement (forwards, comments, likes) for the
**Aliyun Tianchi Challenge: Weibo User Post Engagement Prediction**.

**Competition link:** https://tianchi.aliyun.com/competition/entrance/231574/information

---

## Competition Setup

| Split | Period | Purpose |
|---|---|---|
| Training | Feb – Jul 2015 | Learn per-user engagement patterns |
| Prediction | Aug 2015 | Produce one (F, C, L) triplet per test post |

**Evaluation metric — weighted hit rate:**

```
precision_i = 1 - 0.5 * |F - tf| / (tf + 5)
                - 0.25 * |C - tc| / (tc + 3)
                - 0.25 * |L - tl| / (tl + 3)

hit_i   = 1  if  precision_i > 0.8
weight_i = min(tf + tc + tl, 100) + 1

score = sum(weight_i * hit_i) / sum(weight_i)
```

A prediction is a **hit** when its (F, C, L) triplet is within ~20% of the true
values on each dimension (with small-count smoothing). High-engagement posts carry
more weight (capped at 101).

---

## Algorithm

![Pipeline Framework](image/framework.jpg)

v13 is a **per-user, per-post** model. For each prediction post it finds the
(F, C, L) integer triplet that maximises the weighted hit rate across all of
that user's historical posts, using content similarity to the prediction post
as an additional weight signal.

### Step 1 — Compute per-post content weights (TF-IDF bigram cosine)

For each user, a per-user IDF is built over all training post contents:

```
idf(bg) = log(N / df(bg))    for bigrams appearing in fewer than N posts
         (bigrams in ALL posts get idf=0, filtering out common filler)
```

For each prediction post `p`, cosine similarity against each training post `t`:

```
sim(p, t) = dot(tfidf(p), tfidf(t)) / (||tfidf(p)|| * ||tfidf(t)||)

w_content(p, t) = 1 + ALPHA * sim(p, t)     ALPHA = 8.0
```

This up-weights training posts that are topically similar to the prediction post,
so the model predicts engagement patterns from the most relevant past posts.

### Step 2 — Compute combined post weights

Each historical post `i` receives a combined weight:

```
w_recency_i = exp(-days_ago_i * ln2 / HALF_LIFE)          # exponential decay
w_engage_i  = min(tf+tc+tl, 100) + 1  if days_ago_i <= RECENCY_WINDOW
            = 1                         otherwise

combined_i  = w_recency_i * w_engage_i * w_content_i
```

**Why hybrid weighting?**
Multiplying recency and engagement gives old high-engagement spikes enormous weight.
Restricting the engagement boost to the last `RECENCY_WINDOW` days prevents this.

Parameters:
- `HALF_LIFE = 90.0` days — recency decay half-life
- `RECENCY_WINDOW = 30.0` days — window for full engagement weighting
- `ALPHA = 8.0` — TF-IDF content similarity multiplier

### Step 3 — Build candidate pools

For each of the three dimensions (F, C, L), a small set of up to 8 integer
candidates is built from the user's history:

```
candidates = {0} ∪ {weighted_mean} ∪ {p5, p10, p25, p50, p75, p90}
```

- `0` is included because the majority of posts have zero engagement.
- The weighted mean uses `combined_i` as weights.
- Percentiles are unweighted (capture the shape of the raw distribution).
- `p5` and `p10` provide low-end anchors for users with mostly low-engagement posts.

### Step 4 — Vectorised grid search

All combinations of the three candidate pools are scored in a single NumPy
matrix operation (up to 8³ = 512 triplets):

```python
# (n_cands, 1) broadcast against (1, n_posts)
dev_f = |F - fwd| / (fwd + 5)
dev_c = |C - cmt| / (cmt + 3)
dev_l = |L - lke| / (lke + 3)

prec  = clip(1 - 0.5*dev_f - 0.25*dev_c - 0.25*dev_l, 0, ∞)
hit   = (prec > 0.8)
score = sum(combined_w * hit, axis=posts) / sum(combined_w)
```

The triplet with the highest score is selected as `(F0, C0, L0)`.

### Step 5 — Local refinement (±1)

After the grid search, all 27 neighbours are evaluated:

```
for dF in {-1, 0, +1}:
  for dC in {-1, 0, +1}:
    for dL in {-1, 0, +1}:
      score (max(0, F0+dF), max(0, C0+dC), max(0, L0+dL))
```

The best-scoring triplet (grid or neighbour) is kept.

---

## Competition Score History

| Version | Competition score | Key change |
|---|---|---|
| v3 | 0.3127 | Baseline: recency×engagement weights, grid search, ±1 refinement |
| v4 | 0.3090 | Weighted percentiles, ±2 refinement, more candidates (hurt) |
| v5 | 0.3115 | Iterative hill climbing (overfits, hurt) |
| v6 | 0.3098 | Exact sweep-line coordinate descent (overfits, hurt) |
| v7 | 0.3080 | hl=180, rw=7 from CV sweep (CV misleading, worst) |
| v8 | 0.3093 | Added p95, p99 candidates (hurt) |
| v9 | 0.3126 | Per-post bigram Jaccard similarity (no signal, neutral) |
| v10 | 0.3139 | TF-IDF bigram cosine similarity, ALPHA=3 (first improvement) |
| v11 | 0.3102 | + temporal features, pred-relative days_ago (hurt) |
| v12 | 0.3144 | ALPHA=5, added p10 candidate |
| **v13** | **0.3147** | **ALPHA=8, added p5 candidate (best)** |
| v14 | 0.3144 | Bigram+trigram (trigrams too sparse for short posts, hurt) |
| v15 | 0.3144 | ALPHA=12 (over-concentrates weight, peaked at 8) |
| v16 | 0.3146 | Sublinear TF log(1+tf) (marginal, bigrams short enough) |

---

## Data Format

### Training data — tab-separated, 7 columns
```
uid  mid  time  forward_count  comment_count  like_count  content
```

### Prediction data — tab-separated, 4 columns
```
uid  mid  time  content
```

### Output — tab-separated
```
uid  mid  forward_count,comment_count,like_count
```

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run
```bash
python weibo_pipeline_v13.py
```

Output is written to `Weibo Data/weibo_result_data/weibo_result_data_v13.txt`.

### Run on HPC (SLURM)
```bash
chmod +x run_pipeline.sh
sbatch run_pipeline.sh
```

---

## Key Parameters

| Parameter | Value | Effect |
|---|---|---|
| `HALF_LIFE` | 90 days | Recency decay — posts from 90 days ago get half the weight |
| `RECENCY_WINDOW` | 30 days | Posts older than this contribute recency weight only |
| `MAX_CANDS` | 8 | Maximum candidates per dimension in grid search |
| `ALPHA` | 8.0 | TF-IDF content similarity multiplier |

---

## File Structure

```
weibo-baseline/
├── weibo_pipeline_v6.py         # Current production pipeline (score 0.3147)
├── weibo_pipeline_v3.py          # Baseline pipeline (score 0.3127)
├── run_pipeline.sh               # SLURM batch script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── Weibo Data/
    ├── weibo_train_data/
    │   └── weibo_train_data.txt  # Training dataset (~1.2M rows)
    ├── weibo_predict_data/
    │   └── weibo_predict_data.txt # Test dataset (~177K rows)
    └── weibo_result_data/
        └── weibo_result_data_v13.txt  # v13 predictions (best)
```

---

## Requirements

- Python 3.7+
- numpy >= 1.21.0
- pandas >= 1.3.0

---

## Author

Developed for the Aliyun Tianchi Weibo Challenge.

## License

Competition project — refer to competition terms and conditions.
