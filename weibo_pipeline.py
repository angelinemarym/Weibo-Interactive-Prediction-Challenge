"""
Weibo Engagement Prediction Pipeline
=====================================
Predicts forward_count, comment_count, like_count for each weibo.

Data format assumptions:
  - weibo_train_data.txt : tab-separated
      uid | mid | time | forward_count | comment_count | like_count | content
  - weibo_predict_data.txt : tab-separated
      uid | mid | time | content
  - Output: weibo_result_data.txt
      uid mid forward_count,comment_count,like_count
"""

import ctypes
import gc
import os
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Limit OpenMP threads before LightGBM is imported
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ── Try importing LightGBM, fall back to RandomForest ──────────────────────
try:
    import lightgbm as lgb
    USE_LGBM = True
    print("[✓] LightGBM available — using gradient boosting")
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    USE_LGBM = False
    print("[!] LightGBM not found — falling back to RandomForest")

# ═══════════════════════════════════════════════════════════════════
# 0. CONFIG
# ═══════════════════════════════════════════════════════════════════
TRAIN_FILE   = "Weibo Data/weibo_train_data/weibo_train_data.txt"
PREDICT_FILE = "Weibo Data/weibo_predict_data/weibo_predict_data.txt"
OUTPUT_FILE  = "Weibo Data/weibo_result_data/weibo_result_data.txt"

TARGETS = ["forward_count", "comment_count", "like_count"]

# Smoothing denominators matching the competition's deviation formula
SMOOTHING = {"forward_count": 5, "comment_count": 3, "like_count": 3}

# Keep all rows with any engagement > 0; subsample zero-rows to fill the rest.
# Lower = less RAM; raise if you have headroom and want better accuracy.
MAX_TRAIN_ROWS = 200_000

LGBM_PARAMS = {
    "objective":         "regression_l1",
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 50,
    "max_bin":           63,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "n_estimators":      100,
    "n_jobs":            1,
    "verbose":           -1,
}

# ═══════════════════════════════════════════════════════════════════
# MEMORY HELPERS
# ═══════════════════════════════════════════════════════════════════
def _mem() -> str:
    rss = virt = -1
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) // 1024
                elif line.startswith("VmSize:"):
                    virt = int(line.split()[1]) // 1024
    except Exception:
        pass
    return f"RSS={rss} MB  Virt={virt} MB"

def _free(label: str = ""):
    gc.collect()
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    gc.collect()
    if label:
        print(f"    [mem] {label}: {_mem()}")


# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════
def load_data(train_path: str, predict_path: str):
    train_cols   = ["uid", "mid", "time", "forward_count",
                    "comment_count", "like_count", "content"]
    predict_cols = ["uid", "mid", "time", "content"]

    train = pd.read_csv(
        train_path, sep="\t", header=None, names=train_cols,
        dtype={"uid": str, "mid": str, "time": str, "content": str},
        on_bad_lines="skip",
    )
    predict = pd.read_csv(
        predict_path, sep="\t", header=None, names=predict_cols,
        dtype={"uid": str, "mid": str, "time": str, "content": str},
        on_bad_lines="skip",
    )

    for df in (train, predict):
        t = pd.to_datetime(df["time"], format="%Y%m%d", errors="coerce")
        df["month"]      = t.dt.month.fillna(0).astype("int8")
        df["day"]        = t.dt.day.fillna(0).astype("int8")
        df["dayofweek"]  = t.dt.dayofweek.fillna(0).astype("int8")
        df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
        df.drop(columns=["time"], inplace=True)

    for col in TARGETS:
        train[col] = (pd.to_numeric(train[col], errors="coerce")
                      .fillna(0).clip(lower=0).astype("float32"))

    print(f"[✓] Train rows  : {len(train):,}")
    print(f"[✓] Predict rows: {len(predict):,}")
    return train, predict


# ═══════════════════════════════════════════════════════════════════
# 2. CONTENT FEATURES  (extracted immediately, content then dropped)
# ═══════════════════════════════════════════════════════════════════
URL_RE     = re.compile(r"http[s]?://\S+")
MENTION_RE = re.compile(r"@[\w\u4e00-\u9fff]+")
TOPIC_RE   = re.compile(r"#[^#]+#")

def extract_and_drop_content(df: pd.DataFrame) -> None:
    text = df["content"].fillna("")
    df["content_len"]       = text.str.len().astype("int32")
    df["url_count"]         = text.str.count(URL_RE.pattern).astype("int8")
    df["mention_count"]     = text.str.count(MENTION_RE.pattern).astype("int8")
    df["topic_count"]       = text.str.count(TOPIC_RE.pattern).astype("int8")
    df["has_url"]           = (df["url_count"] > 0).astype("int8")
    df["has_mention"]       = (df["mention_count"] > 0).astype("int8")
    df["has_topic"]         = (df["topic_count"] > 0).astype("int8")
    df["has_retweet_mark"]  = text.str.contains("转发", na=False).astype("int8")
    df["exclamation_count"] = text.str.count("！|!").astype("int8")
    df["question_count"]    = text.str.count("？|\\?").astype("int8")
    df["emoji_count"]       = text.apply(
        lambda x: sum(1 for c in x if ord(c) > 0x1F300)
    ).astype("int16")
    df.drop(columns=["content"], inplace=True)


# ═══════════════════════════════════════════════════════════════════
# 3. USER-LEVEL FEATURES
# ═══════════════════════════════════════════════════════════════════
def build_user_features(train_slim: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for target in TARGETS:
        prefix = target.replace("_count", "")
        part = (
            train_slim.groupby("uid", sort=False)[target]
            .agg(**{
                f"user_{prefix}_mean":  "mean",
                f"user_{prefix}_std":   "std",
                f"user_{prefix}_max":   "max",
                f"user_{prefix}_min":   "min",
                f"user_{prefix}_count": "count",
            })
            .reset_index()
        )
        dfs.append(part)

    user_df = dfs[0]
    for part in dfs[1:]:
        user_df = user_df.merge(part, on="uid", how="outer")
    del dfs; gc.collect()

    post_counts = (train_slim.groupby("uid", sort=False)["mid"]
                   .count().rename("user_post_count").reset_index())
    user_df = user_df.merge(post_counts, on="uid", how="left")

    user_df["user_fwd_comment_ratio"] = (
        user_df["user_forward_mean"] / (user_df["user_comment_mean"] + 1))
    user_df["user_engagement_total"] = (
        user_df["user_forward_mean"] +
        user_df["user_comment_mean"] +
        user_df["user_like_mean"])
    user_df["user_std_mean_ratio_fwd"] = (
        user_df["user_forward_std"] / (user_df["user_forward_mean"] + 1))
    return user_df


def build_user_time_features(train_slim: pd.DataFrame) -> pd.DataFrame:
    feats = []
    for target in TARGETS:
        prefix = target.replace("_count", "")
        grp = (train_slim.groupby(["uid", "month"], sort=False)[target]
               .mean().reset_index()
               .rename(columns={target: f"user_{prefix}_month_mean"}))
        feats.append(grp)
    out = feats[0]
    for f in feats[1:]:
        out = out.merge(f, on=["uid", "month"], how="outer")
    return out


# ═══════════════════════════════════════════════════════════════════
# 4. FULL FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame,
                   user_feats: pd.DataFrame,
                   user_time_feats: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(user_feats, on="uid", how="left")
    df = df.merge(user_time_feats, on=["uid", "month"], how="left")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    drop = {"uid", "mid", "forward_count", "comment_count", "like_count"}
    return [c for c in df.columns if c not in drop and df[c].dtype != object]


def subsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    engaged = df[(df[TARGETS] > 0).any(axis=1)]
    zeros   = df[(df[TARGETS] == 0).all(axis=1)]
    n_keep  = max(0, max_rows - len(engaged))
    if n_keep < len(zeros):
        zeros = zeros.sample(n=n_keep, random_state=42)
    sub = pd.concat([engaged, zeros], ignore_index=True)
    print(f"    Subsampled: {len(sub):,} rows "
          f"({len(engaged):,} engaged + {len(zeros):,} zero-rows)")
    return sub


# ═══════════════════════════════════════════════════════════════════
# 5. TRAIN ONE MODEL AT A TIME, PREDICT IMMEDIATELY, FREE IT
#    → only 1 model in memory at once
# ═══════════════════════════════════════════════════════════════════
def train_and_predict(train_X: np.ndarray,
                      train_ys: dict,
                      predict_X: np.ndarray) -> dict:
    predictions = {}
    importances = {}

    for target in TARGETS:
        y = train_ys[target]
        print(f"\n  ── [{target}] ──")
        print(f"     train n={len(y):,}  mean={y.mean():.2f}"
              f"  median={np.median(y):.1f}  max={y.max():.0f}")

        if USE_LGBM:
            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(train_X, y)
            try:
                model.booster_.free_dataset()   # release C++ training bins
            except Exception:
                pass
            if USE_LGBM:
                importances[target] = model.feature_importances_.copy()
        else:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10,
                min_samples_leaf=10, n_jobs=1, random_state=42)
            model.fit(train_X, y)

        _free(f"fit {target}")

        pred = np.round(model.predict(predict_X).clip(0)).astype(np.int32)
        predictions[target] = pred

        del model
        _free(f"predict+free {target}")

    return predictions, importances


# ═══════════════════════════════════════════════════════════════════
# 6. EVALUATION METRIC
# ═══════════════════════════════════════════════════════════════════
def summarise_targets(train_ys: dict) -> None:
    """Print target distribution stats (models are already freed)."""
    for target, arr in train_ys.items():
        print(f"    {target}: mean={arr.mean():.2f}  median={np.median(arr):.1f}"
              f"  max={arr.max():.0f}")


# ═══════════════════════════════════════════════════════════════════
# 7. OUTPUT
# ═══════════════════════════════════════════════════════════════════
def write_predictions(uid_mid: pd.DataFrame,
                      predictions: dict,
                      output_path: str):
    results = uid_mid.reset_index(drop=True)
    for target in TARGETS:
        results[target] = predictions[target]

    lines = (results["uid"] + " " + results["mid"] + " "
             + results["forward_count"].astype(str) + ","
             + results["comment_count"].astype(str) + ","
             + results["like_count"].astype(str))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    lines.to_csv(output_path, index=False, header=False)

    print(f"\n[✓] Predictions saved → {output_path}")
    print(f"    Total rows: {len(results):,}")
    print("\n    Sample output:")
    for line in lines.head(3):
        print(f"    {line}")


# ═══════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  WEIBO ENGAGEMENT PREDICTION PIPELINE")
    print("=" * 55)
    print(f"  [start] {_mem()}")

    # ── 1. Load ──────────────────────────────────────────────────
    print("\n[1] Loading data …")
    if not Path(TRAIN_FILE).exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    train, predict = load_data(TRAIN_FILE, PREDICT_FILE)

    predict_uid_mid = predict[["uid", "mid"]].copy()

    print("    Extracting content features …")
    extract_and_drop_content(train)
    extract_and_drop_content(predict)
    _free("content drop")

    # ── 2. User features ──────────────────────────────────────────
    print(f"\n[2] Building user-level features …")
    slim_cols  = ["uid", "mid", "month"] + TARGETS
    train_slim = train[slim_cols]
    user_feats      = build_user_features(train_slim)
    user_time_feats = build_user_time_features(train_slim)
    del train_slim
    _free("user feats")

    # ── 3. Feature matrices ───────────────────────────────────────
    print(f"\n[3] Assembling feature matrices …")
    train_feats = build_features(train, user_feats, user_time_feats)
    del train
    _free("train drop")
    print(f"    train_feats shape={train_feats.shape}  {_mem()}")

    predict_feats = build_features(predict, user_feats, user_time_feats)
    del predict, user_feats, user_time_feats
    _free("predict drop")
    print(f"    predict_feats shape={predict_feats.shape}  {_mem()}")

    feat_cols = get_feature_columns(train_feats)
    print(f"    Feature count: {len(feat_cols)}")

    # ── 4. Extract numpy arrays, free DataFrames ──────────────────
    print(f"\n[4] Extracting arrays …")
    if MAX_TRAIN_ROWS and len(train_feats) > MAX_TRAIN_ROWS:
        print(f"    Subsampling to {MAX_TRAIN_ROWS:,} rows …")
        train_feats = subsample(train_feats, MAX_TRAIN_ROWS)
        _free("subsample")

    train_X  = train_feats[feat_cols].values.astype("float32")
    train_ys = {t: train_feats[t].values.astype("float32") for t in TARGETS}
    del train_feats
    _free("train_feats → arrays")
    print(f"    train_X: {train_X.shape}  {_mem()}")

    predict_X = predict_feats[feat_cols].values.astype("float32")
    del predict_feats
    _free("predict_feats → array")
    print(f"    predict_X: {predict_X.shape}  {_mem()}")

    # ── 5. Train + predict (one model at a time) ──────────────────
    print(f"\n[5] Training and predicting (1 model at a time) …")
    predictions, importances = train_and_predict(train_X, train_ys, predict_X)
    del train_X, predict_X
    _free("arrays freed post-train")

    # ── 6. Feature importance ─────────────────────────────────────
    if USE_LGBM and importances:
        print("\n[6] Top-10 features (forward_count model) …")
        imp = pd.Series(
            importances["forward_count"], index=feat_cols
        ).sort_values(ascending=False)
        for name, val in imp.head(10).items():
            print(f"    {name:<40s}  {val:.0f}")

    # ── 7. Write predictions ──────────────────────────────────────
    print(f"\n[7] Writing predictions …  {_mem()}")
    write_predictions(predict_uid_mid, predictions, OUTPUT_FILE)

    print("\n[✓] Pipeline complete.")


if __name__ == "__main__":
    main()
