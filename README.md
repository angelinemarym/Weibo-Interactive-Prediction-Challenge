# Weibo Engagement Prediction — Pipeline v3

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

v3 is a **per-user, single-prediction** model. For each user it finds one
(F, C, L) integer triplet that maximises the weighted hit rate across all of
that user's historical posts, then applies that same triplet to every test post
belonging to that user.

### Step 1 — Compute post weights

Each historical post `i` receives a combined weight:

```
w_recency_i = exp(-days_ago_i * ln2 / HALF_LIFE)          # exponential decay
w_engage_i  = min(tf+tc+tl, 100) + 1  if days_ago_i <= RECENCY_WINDOW
            = 1                         otherwise

combined_i  = w_recency_i * w_engage_i
```

**Why hybrid weighting?**
Multiplying recency and engagement together gives old high-engagement spikes
enormous weight (e.g. a 90-day-old post with 100 engagements:
`exp(-90*ln2/90) * 101 = 0.5 * 101 = 50.5`), causing the model to predict
inflated counts for users who have since gone quiet. Restricting the engagement
boost to the last `RECENCY_WINDOW` days prevents this: old posts contribute via
recency decay only (weight = 1), while recent posts benefit from both signals.

Parameters:
- `HALF_LIFE = 90.0` days — recency decay half-life
- `RECENCY_WINDOW = 30.0` days — window for full engagement weighting

### Step 2 — Build candidate pools

For each of the three dimensions (F, C, L), a small set of up to 8 integer
candidates is built from the user's history:

```
candidates = {0} ∪ {weighted_mean} ∪ {p25, p50, p75, p90}
```

- `0` is included because the majority of posts have zero engagement.
- The weighted mean uses `combined_i` as weights.
- Percentiles are unweighted (capture the shape of the raw distribution).

### Step 3 — Vectorised grid search

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

### Step 4 — Local refinement (±1)

The grid search is limited to the discrete candidate pool. The true optimum may
be an adjacent integer not in the pool — especially near the precision = 0.8
boundary where a ±1 shift can flip a miss to a hit.

After the grid search, all 27 neighbours are evaluated:

```
for dF in {-1, 0, +1}:
  for dC in {-1, 0, +1}:
    for dL in {-1, 0, +1}:
      score (max(0, F0+dF), max(0, C0+dC), max(0, L0+dL))
```

The best-scoring triplet (grid or neighbour) is kept.

---

## CV Results

| Version | Change | CV score (pre-Jul→Jul) |
|---|---|---|
| v1 | Simple per-user heuristic, no recency weighting | 0.3004 |
| v2 | Recency decay (hl=30), percentile candidates, vectorised scoring | 0.3065 |
| v2 → v3 base | Hybrid engagement weighting (hl=90, rot=30) | 0.3093 |
| **v3** | + local refinement (±1) | **0.3112** |

Oracle ceiling (true July median per user): **0.3490**

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
python weibo_pipeline_v3.py
```

Output is written to `Weibo Data/weibo_result_data/weibo_result_data_v3.txt`.

### Run on HPC (SLURM)
```bash
chmod +x run_pipeline.sh
sbatch run_pipeline.sh
```

---

## Key Parameters

| Parameter | Value | Effect |
|---|---|---|
| `HALF_LIFE` | 90 days | Recency decay — posts from 90 days ago get half the weight of today's posts |
| `RECENCY_WINDOW` | 30 days | Posts older than this contribute recency weight only (engagement weight = 1) |
| `MAX_CANDS` | 8 | Maximum candidates per dimension in grid search |

---

## File Structure

```
weibo-baseline/
├── weibo_pipeline_v3.py          # Current production pipeline
├── run_pipeline.sh               # SLURM batch script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── Weibo Data/
    ├── weibo_train_data/
    │   └── weibo_train_data.txt  # Training dataset (~1.2M rows)
    ├── weibo_predict_data/
    │   └── weibo_predict_data.txt # Test dataset (~177K rows)
    └── weibo_result_data/
        └── weibo_result_data_v3.txt  # v3 predictions
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
