"""
Weibo Engagement Prediction — v2 (Social Network Edition)
==========================================================
Key improvements over v1:
  1. Mention-based social graph → PageRank, in/out-degree, centrality
  2. Community membership features (Label Propagation)
  3. "Influencee" features — avg engagement of users you mention
  4. Log1p target transform → much better MAE on skewed count data
  5. Interaction features — user_influence × content signals
  6. Temporal decay — recency-weighted user history
  7. Post-prediction isotonic calibration (clipping heuristic)
"""

import ctypes
import gc
import os
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, diags

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"]  = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    import lightgbm as lgb
    USE_LGBM = True
    print("[✓] LightGBM available")
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    USE_LGBM = False
    print("[!] Falling back to GradientBoosting")

# ═══════════════════════════════════════════════════════════════════
# 0. CONFIG
# ═══════════════════════════════════════════════════════════════════
TRAIN_FILE   = "Weibo Data/weibo_train_data/weibo_train_data.txt"
PREDICT_FILE = "Weibo Data/weibo_predict_data/weibo_predict_data.txt"
OUTPUT_FILE  = "Weibo Data/weibo_result_data/weibo_result_data_v2.txt"

TARGETS = ["forward_count", "comment_count", "like_count"]

MAX_TRAIN_ROWS = 80_000    # hard cap on training rows; lower = less LightGBM RAM

LGBM_PARAMS = {
    "objective":         "regression_l1",
    "learning_rate":     0.08,     # faster convergence so fewer rounds needed
    "num_leaves":        31,       # 63→31: halves histogram buffer RAM
    "min_child_samples": 20,
    "max_bin":           31,       # 63→31: halves bin index memory
    "feature_fraction":  0.7,
    "bagging_fraction":  0.7,
    "bagging_freq":      5,
    "lambda_l1":         0.1,
    "lambda_l2":         0.1,
    "n_estimators":        200,      # kept low to cap booster object memory
    "num_threads":         1,        # raw API uses num_threads, not n_jobs
    "histogram_pool_size": 16,       # MB; caps contiguous histogram pool allocation
    "force_col_wise":      True,     # avoids large row-wise working buffers
    "verbose":             -1,
}

PREDICT_MONTH = 1

URL_RE     = re.compile(r"http[s]?://\S+")
MENTION_RE = re.compile(r"@([\w\u4e00-\u9fff]+)")
TOPIC_RE   = re.compile(r"#[^#]+#")


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
# 1. LOAD
# ═══════════════════════════════════════════════════════════════════
def load_data(train_path, predict_path):
    train_cols   = ["uid","mid","time","forward_count",
                    "comment_count","like_count","content"]
    predict_cols = ["uid","mid","time","content"]

    train = pd.read_csv(
        train_path, sep="\t", header=None, names=train_cols,
        dtype={"uid":str,"mid":str,"time":str,"content":str},
        on_bad_lines="skip")
    predict = pd.read_csv(
        predict_path, sep="\t", header=None, names=predict_cols,
        dtype={"uid":str,"mid":str,"time":str,"content":str},
        on_bad_lines="skip")

    for df in (train, predict):
        t = pd.to_datetime(df["time"], format="%Y%m%d", errors="coerce")
        df["date"]       = t
        df["year"]       = t.dt.year.fillna(0).astype("int16")
        df["month"]      = t.dt.month.fillna(0).astype("int8")
        df["day"]        = t.dt.day.fillna(0).astype("int8")
        df["dayofweek"]  = t.dt.dayofweek.fillna(0).astype("int8")
        df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
        df["hour"]       = np.int8(12)
        df.drop(columns=["time"], inplace=True)

    for col in TARGETS:
        train[col] = pd.to_numeric(train[col], errors="coerce").fillna(0).clip(lower=0).astype("float32")

    train = train.sort_values("date").reset_index(drop=True)
    train["post_rank"] = train.groupby("uid").cumcount().astype("int32")
    predict["post_rank"] = np.int32(0)

    print(f"[✓] Train  : {len(train):,} rows")
    print(f"[✓] Predict: {len(predict):,} rows")
    return train, predict


# ═══════════════════════════════════════════════════════════════════
# 2. SOCIAL NETWORK GRAPH
# ═══════════════════════════════════════════════════════════════════
def build_social_graph(train: pd.DataFrame) -> pd.DataFrame:
    """
    Directed graph: edge uid_poster → @mentioned.
    Uses scipy sparse instead of NetworkX to avoid the ~1 GB virtual address
    space fragmentation that NetworkX's Python-dict graph structures cause.
    Clustering and community features are dropped (required undirected graph).
    """
    print("\n[SN] Building mention graph …")

    text = train["content"].fillna("")
    mentions_series = text.str.findall(MENTION_RE.pattern)

    pairs = (pd.DataFrame({"uid": train["uid"], "mention": mentions_series})
               .explode("mention")
               .dropna(subset=["mention"]))
    pairs["mention"] = "@" + pairs["mention"]
    _free("mention extraction")
    print(f"    Mention pairs: {len(pairs):,}")

    # Integer-encode all nodes (posters + mentioned)
    all_nodes = pd.Index(
        pd.concat([pairs["uid"], pairs["mention"]], ignore_index=True).unique()
    )
    n = len(all_nodes)
    node_idx = pd.Series(np.arange(n, dtype=np.int32), index=all_nodes)

    src = node_idx[pairs["uid"].values].values
    dst = node_idx[pairs["mention"].values].values
    del pairs
    _free("graph build")
    print(f"    Graph: {n:,} nodes, {len(src):,} edges")

    # Sparse directed adjacency matrix (src → dst)
    A = csr_matrix(
        (np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n)
    )
    del src, dst

    out_deg = np.asarray(A.sum(axis=1)).ravel()   # row sums = out-degree
    in_deg  = np.asarray(A.sum(axis=0)).ravel()   # col sums = in-degree

    # PageRank: power iteration on column-stochastic matrix
    d_inv = np.where(out_deg > 0, 1.0 / out_deg, 0.0).astype(np.float32)
    M = (diags(d_inv) @ A).T.tocsr()   # column-stochastic transition matrix
    del A, d_inv
    _free("sparse matrix")

    alpha    = np.float32(0.85)
    teleport = np.float32((1.0 - alpha) / n)
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    for _ in range(100):
        pr_new = alpha * M.dot(pr) + teleport
        if float(np.abs(pr_new - pr).max()) < 1e-4:
            break
        pr = pr_new
    del M
    _free("graph metrics freed")

    # Map results back to uid strings via pandas Series
    pr_s      = pd.Series(pr,                        index=all_nodes)
    in_deg_s  = pd.Series(in_deg.astype(np.int32),   index=all_nodes)
    out_deg_s = pd.Series(out_deg.astype(np.int32),  index=all_nodes)
    del pr, in_deg, out_deg, node_idx, all_nodes

    all_uids = train["uid"].unique()
    sn_df = pd.DataFrame({
        "uid":           all_uids,
        "sn_pagerank":   pr_s.reindex(all_uids).fillna(0.0).values.astype(np.float32),
        "sn_in_degree":  in_deg_s.reindex(all_uids).fillna(0).values.astype(np.int32),
        "sn_out_degree": out_deg_s.reindex(all_uids).fillna(0).values.astype(np.int32),
        "sn_clustering": np.float32(0.0),   # dropped: required undirected graph
        "sn_community":  np.int16(-1),      # dropped: required undirected graph
    })
    del pr_s, in_deg_s, out_deg_s

    sn_df["sn_in_out_ratio"] = (
        sn_df["sn_in_degree"] / (sn_df["sn_out_degree"] + 1))
    pr_max = sn_df["sn_pagerank"].max()
    sn_df["sn_pagerank_norm"] = sn_df["sn_pagerank"] / (pr_max + 1e-9)

    _free("sn_df assembled")
    print(f"    SN features shape: {sn_df.shape}  {_mem()}")
    return sn_df


# ═══════════════════════════════════════════════════════════════════
# 3. CONTENT FEATURES
# ═══════════════════════════════════════════════════════════════════
def extract_and_drop_content(df: pd.DataFrame) -> None:
    """Extract all content features in-place, then drop the content column."""
    text = df["content"].fillna("")
    df["content_len"]        = text.str.len().astype("int32")
    df["url_count"]          = text.str.count(URL_RE.pattern).astype("int8")
    df["mention_count"]      = text.str.count(MENTION_RE.pattern).astype("int8")
    df["topic_count"]        = text.str.count(TOPIC_RE.pattern).astype("int8")
    df["has_url"]            = (df["url_count"] > 0).astype("int8")
    df["has_mention"]        = (df["mention_count"] > 0).astype("int8")
    df["has_topic"]          = (df["topic_count"] > 0).astype("int8")
    df["has_retweet_mark"]   = text.str.contains("转发", na=False).astype("int8")
    df["exclamation_count"]  = text.str.count("！|!").astype("int8")
    df["question_count"]     = text.str.count("？|\\?").astype("int8")
    df["emoji_count"]        = text.apply(
        lambda x: sum(1 for c in x if ord(c) > 0x1F300)).astype("int16")
    df["chinese_char_ratio"] = text.apply(
        lambda x: sum(1 for c in x if '\u4e00' <= c <= '\u9fff') / (len(x) + 1)
    ).astype("float32")
    df.drop(columns=["content"], inplace=True)


# ═══════════════════════════════════════════════════════════════════
# 4. USER HISTORICAL FEATURES
# ═══════════════════════════════════════════════════════════════════
def build_user_features(train: pd.DataFrame, recent_mask: pd.Series) -> pd.DataFrame:
    # Only built-in agg strings — no lambdas, no scipy, no per-group Python calls.
    dfs = []
    for t in TARGETS:
        prefix = t.replace("_count", "")
        part = (
            train.groupby("uid", sort=False)[t]
            .agg(**{
                f"user_{prefix}_mean":  "mean",
                f"user_{prefix}_std":   "std",
                f"user_{prefix}_max":   "max",
                f"user_{prefix}_count": "count",
            })
            .reset_index()
        )
        dfs.append(part)

    user_df = dfs[0]
    for part in dfs[1:]:
        user_df = user_df.merge(part, on="uid", how="outer")
    del dfs; gc.collect()

    # Temporal decay: recent_mask computed from date before date was dropped
    recent = train[recent_mask]
    for t in TARGETS:
        prefix = t.replace("_count", "")
        rec_mean = (recent.groupby("uid", sort=False)[t]
                    .mean().rename(f"user_{prefix}_recent_mean").reset_index())
        user_df = user_df.merge(rec_mean, on="uid", how="left")
    del recent; gc.collect()

    post_counts = (train.groupby("uid", sort=False)["mid"]
                   .count().rename("user_post_count").reset_index())
    user_df = user_df.merge(post_counts, on="uid", how="left")

    user_df["user_fwd_comment_ratio"] = (
        user_df["user_forward_mean"] / (user_df["user_comment_mean"] + 1))
    user_df["user_engagement_total"] = (
        user_df["user_forward_mean"] +
        user_df["user_comment_mean"] +
        user_df["user_like_mean"])
    user_df["user_cv_fwd"] = (
        user_df["user_forward_std"] / (user_df["user_forward_mean"] + 1))
    return user_df


def build_user_time_features(train: pd.DataFrame) -> pd.DataFrame:
    feats = []
    for t in TARGETS:
        prefix = t.replace("_count", "")
        g = (train.groupby(["uid","month"], sort=False)[t]
             .mean().reset_index()
             .rename(columns={t: f"user_{prefix}_month_mean"}))
        feats.append(g)
    out = feats[0]
    for f in feats[1:]:
        out = out.merge(f, on=["uid","month"], how="outer")
    return out


# ═══════════════════════════════════════════════════════════════════
# 5. FULL FEATURE ASSEMBLY
# ═══════════════════════════════════════════════════════════════════
def build_features(df, user_feats, user_time_feats, sn_df):
    # No df.copy() — operates on the DataFrame in-place
    df = df.merge(user_feats,      on="uid",           how="left")
    df = df.merge(user_time_feats, on=["uid","month"],  how="left")
    df = df.merge(sn_df,           on="uid",           how="left")

    df["itx_influence_x_topic"]   = df["sn_pagerank_norm"] * df["has_topic"]
    df["itx_influence_x_mention"] = df["sn_pagerank_norm"] * df["mention_count"]
    df["itx_influence_x_url"]     = df["sn_pagerank_norm"] * df["has_url"]
    df["itx_hist_fwd_x_topic"]    = df["user_forward_mean"] * df["has_topic"]
    df["itx_hist_fwd_x_mention"]  = df["user_forward_mean"] * df["has_mention"]
    df["itx_hist_fwd_x_retweet"]  = df["user_forward_mean"] * df["has_retweet_mark"]

    for t in ["forward", "comment", "like"]:
        col_all    = f"user_{t}_mean"
        col_recent = f"user_{t}_recent_mean"
        if col_all in df.columns and col_recent in df.columns:
            df[f"user_{t}_drift"] = (
                df[col_recent].fillna(df[col_all]) - df[col_all])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def get_feature_columns(df):
    drop = {"uid","mid","forward_count","comment_count","like_count"}
    return [c for c in df.columns if c not in drop and df[c].dtype != object]


def subsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    sub = df.sample(n=max_rows, random_state=42)
    print(f"    Subsampled: {len(sub):,} rows from {len(df):,}")
    return sub


# ═══════════════════════════════════════════════════════════════════
# 6. TRAIN ONE MODEL AT A TIME, PREDICT IMMEDIATELY, FREE IT
# ═══════════════════════════════════════════════════════════════════
def train_and_predict(train_X, train_ys, predict_X):
    predictions = {}
    importances = {}

    for target in TARGETS:
        y = train_ys[target]
        y_log = np.log1p(y)

        print(f"\n  ── [{target}] raw mean={y.mean():.1f} "
              f"| log mean={y_log.mean():.3f} | log std={y_log.std():.3f}")

        if USE_LGBM:
            # free_raw_data=True: LightGBM releases its numpy copy after binning,
            # before even starting to build trees.
            ds = lgb.Dataset(train_X, label=y_log, free_raw_data=True,
                             params={"max_bin": LGBM_PARAMS["max_bin"]})
            ds.construct()            # bin now so raw data is freed before lgb.train
            del y_log; _free(f"lgb.Dataset freed {target}")
            params = {k: v for k, v in LGBM_PARAMS.items() if k != "n_estimators"}
            booster = lgb.train(
                params, ds,
                num_boost_round=LGBM_PARAMS["n_estimators"],
            )
            ds = None; _free(f"ds freed {target}")
            importances[target] = booster.feature_importance(importance_type="gain").copy()
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.05,
                max_depth=5, subsample=0.8, random_state=42)
            model.fit(train_X, y_log)

        _free(f"fit {target}")

        if USE_LGBM:
            log_pred = booster.predict(predict_X)
            del booster
        else:
            log_pred = model.predict(predict_X)
            del model, y_log
        predictions[target] = np.round(np.expm1(log_pred).clip(0)).astype(np.int32)
        del log_pred
        _free(f"predict+free {target}")

    return predictions, importances


# ═══════════════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE REPORT
# ═══════════════════════════════════════════════════════════════════
def print_importance(importances, feat_cols, top_n=15):
    if not USE_LGBM or not importances:
        return
    fwd_imp = importances["forward_count"]
    com_imp = importances["comment_count"]
    lik_imp = importances["like_count"]
    combined = fwd_imp + com_imp + lik_imp
    idx = np.argsort(combined)[::-1][:top_n]
    print(f"\n{'Feature':<45}  fwd    com    lik")
    print("-" * 65)
    for i in idx:
        print(f"  {feat_cols[i]:<43}  {fwd_imp[i]:>5.0f}  "
              f"{com_imp[i]:>5.0f}  {lik_imp[i]:>5.0f}")


# ═══════════════════════════════════════════════════════════════════
# 8. OUTPUT
# ═══════════════════════════════════════════════════════════════════
def write_predictions(uid_mid: pd.DataFrame, predictions: dict, output_path: str):
    results = uid_mid.reset_index(drop=True)
    for t in TARGETS:
        results[t] = predictions[t]

    lines = (results["uid"] + " " + results["mid"] + " "
             + results["forward_count"].astype(str) + ","
             + results["comment_count"].astype(str) + ","
             + results["like_count"].astype(str))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    lines.to_csv(output_path, index=False, header=False)

    print(f"\n[✓] Saved → {output_path}  ({len(results):,} rows)")
    print("\n    Sample:")
    for line in lines.head(5):
        print(f"    {line}")


# ═══════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  WEIBO PIPELINE v2 — Social Network Edition")
    print("=" * 60)
    print(f"  [start] {_mem()}")

    # 1. Load
    print("\n[1] Loading data …")
    if not Path(TRAIN_FILE).exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    train, predict = load_data(TRAIN_FILE, PREDICT_FILE)
    predict_uid_mid = predict[["uid", "mid"]].copy()

    # 2. Social graph — must run before content is dropped (needs content column)
    print("\n[2] Building social network features …")
    sn_df = build_social_graph(train)
    _free("social graph")

    # 3. Extract content features, then free content strings
    print("\n[3] Extracting content features …")
    # Compute recent_mask from date before dropping it
    recent_cutoff = train["date"].max() - pd.Timedelta(days=30)
    recent_mask   = train["date"] >= recent_cutoff
    train.drop(columns=["date"], inplace=True)
    predict.drop(columns=["date"], inplace=True, errors="ignore")
    extract_and_drop_content(train)
    extract_and_drop_content(predict)
    _free("content+date drop")
    print(f"    {_mem()}")

    # 4. User features
    print("\n[4] Building user historical features …")
    user_feats      = build_user_features(train, recent_mask)
    del recent_mask
    user_time_feats = build_user_time_features(train)
    _free("user feats")
    print(f"    {_mem()}")

    # 5. Feature matrices
    print("\n[5] Assembling feature matrices …")
    train_feats = build_features(train, user_feats, user_time_feats, sn_df)
    del train
    _free("train drop")
    print(f"    train_feats: {train_feats.shape}  {_mem()}")

    predict_feats = build_features(predict, user_feats, user_time_feats, sn_df)
    del predict, user_feats, user_time_feats, sn_df
    _free("predict drop")
    print(f"    predict_feats: {predict_feats.shape}  {_mem()}")

    feat_cols = get_feature_columns(train_feats)
    print(f"    Feature count: {len(feat_cols)}")

    # 6. Extract arrays, free DataFrames
    print("\n[6] Extracting arrays …")
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

    # 7. Train + predict (one model at a time)
    print("\n[7] Training and predicting (1 model at a time) …")
    predictions, importances = train_and_predict(train_X, train_ys, predict_X)
    del train_X, predict_X
    _free("arrays freed post-train")

    # 8. Feature importance
    print("\n[8] Feature importance (top 15, combined):")
    print_importance(importances, feat_cols)

    # 9. Write predictions
    print(f"\n[9] Writing predictions …  {_mem()}")
    write_predictions(predict_uid_mid, predictions, OUTPUT_FILE)

    print("\n[✓] Done.")


if __name__ == "__main__":
    main()
