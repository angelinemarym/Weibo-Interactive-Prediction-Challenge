import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
import os
import sys
import logging
import time

# Try to import psutil for memory monitoring (optional)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════
def setup_logging(log_file='debug.log'):
    """Configure logging to both console and file"""
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

def get_memory_info():
    """Get current memory usage (or empty string if psutil not available)"""
    if not HAS_PSUTIL:
        return "N/A"
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        rss_mb = mem.rss / 1024 / 1024
        return f"{rss_mb:.1f} MB"
    except Exception:
        return "N/A"

def log_step(step_name, status='START', duration=None, extra_info=''):
    """Log a step with timestamp and optional duration"""
    mem_info = f"Memory: {get_memory_info()}"
    if status == 'START':
        logger.info(f"[{step_name}] STARTING ... ({mem_info})")
    elif status == 'DONE':
        time_info = f" Time: {duration:.2f}s" if duration else ""
        extra = f" {extra_info}" if extra_info else ""
        logger.info(f"[{step_name}] ✓ DONE{time_info} ({mem_info}){extra}")
    elif status == 'ERROR':
        logger.error(f"[{step_name}] ✗ ERROR: {extra_info}")

class Timer:
    """Context manager for timing operations"""
    def __init__(self, name):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        log_step(self.name, 'START')
        return self

    def __exit__(self, *args):
        duration = time.time() - self.start_time
        log_step(self.name, 'DONE', duration=duration)

class SocialNetworkEngagementPredictor:
    def __init__(self):
        self.user_stats = None
        self.content_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scalers = {}
        self.models = {
            'forward': None,
            'comment': None,
            'like': None
        }
        self.feature_cols = None
        self.evaluation_metrics = {}
        
    def extract_user_network_features(self, df):
        """
        Extract user-level social network features using vectorized groupby
        instead of iterating over each user (massive speedup)
        """
        with Timer("extract_user_network_features"):
            logger.info(f"  Processing {len(df):,} rows across {df['uid'].nunique():,} unique users")

            # Vectorized groupby aggregation - replaces O(n*m) loop
            user_features = df.groupby('uid', observed=True).agg({
                'forward_count': ['sum', 'mean', 'std', 'count'],
                'comment_count': ['sum', 'mean', 'std'] if 'comment_count' in df.columns else {},
                'like_count': ['sum', 'mean', 'std'] if 'like_count' in df.columns else {},
            }).fillna(0)

            logger.info(f"  Created {len(user_features):,} user feature rows")

            # Flatten multi-level columns
            user_features.columns = [f'user_{col[0]}_{col[1]}' for col in user_features.columns]
            user_features = user_features.rename(columns={
                'user_forward_count_count': 'user_total_posts',
                'user_forward_count_sum': 'user_total_forwards',
                'user_forward_count_mean': 'user_avg_forwards',
                'user_forward_count_std': 'user_std_forwards',
            })

            # Total engagement - compute only once
            user_features['user_total_engagement'] = (
                user_features.get('user_forward_count_sum', 0) +
                user_features.get('user_comment_count_sum', 0) +
                user_features.get('user_like_count_sum', 0)
            )

            logger.info(f"  User features shape: {user_features.shape}")
            return user_features
    
    def extract_temporal_features(self, df):
        """
        Extract temporal patterns which significantly impact engagement [[34, 37]]
        """
        with Timer("extract_temporal_features"):
            if 'time' not in df.columns:
                logger.warning("  No 'time' column found, returning empty temporal features")
                return pd.DataFrame(index=df.index)

            temporal_features = pd.DataFrame(index=df.index)

            # Day of week (0=Monday, 6=Sunday)
            temporal_features['day_of_week'] = df['time'].dt.dayofweek
            temporal_features['is_weekend'] = (temporal_features['day_of_week'] >= 5).astype(int)

            # Hour of day (if available)
            temporal_features['hour'] = df['time'].dt.hour
            temporal_features['is_peak_hour'] = (
                (temporal_features['hour'] >= 9) & (temporal_features['hour'] <= 11) |
                (temporal_features['hour'] >= 19) & (temporal_features['hour'] <= 22)
            ).astype(int)

            # Day of month
            temporal_features['day_of_month'] = df['time'].dt.day
            temporal_features['is_month_start'] = df['time'].dt.is_month_start.astype(int)
            temporal_features['is_month_end'] = df['time'].dt.is_month_end.astype(int)

            # Month
            temporal_features['month'] = df['time'].dt.month

            logger.info(f"  Temporal features shape: {temporal_features.shape}")
            return temporal_features
        return temporal_features
    
    def extract_content_features(self, df, fit_vectorizer=True):
        """
        Extract content-based features using vectorized string methods
        instead of .apply() for massive speedup on large datasets

        Args:
            fit_vectorizer: If True, fit TF-IDF on this data. If False, use existing fit.
        """
        with Timer("extract_content_features"):
            content_features = pd.DataFrame(index=df.index)

            if 'content' not in df.columns:
                logger.warning("  No 'content' column found, returning empty content features")
                return content_features

            # Vectorized string operations (100x faster than .apply())
            text = df['content'].fillna('')
            content_features['content_length'] = text.str.len()
            content_features['word_count'] = text.str.split().str.len()
            content_features['avg_word_length'] = content_features['content_length'] / (content_features['word_count'] + 1)

            # Regex counts on string series (still vectorized)
            content_features['hashtag_count'] = text.str.count(r'#\w+')
            content_features['mention_count'] = text.str.count(r'@\w+')
            content_features['url_count'] = text.str.count(r'http[s]?://\S+')
            content_features['question_count'] = text.str.count(r'\?')
            content_features['exclamation_count'] = text.str.count(r'!')

            # TF-IDF features: fit on training, transform on predict
            if fit_vectorizer:
                logger.info(f"  Fitting TF-IDF vectorizer on {len(text):,} texts...")
                tfidf_matrix = self.content_vectorizer.fit_transform(text)
            else:
                logger.info(f"  Transforming {len(text):,} texts with pre-fit vectorizer...")
                tfidf_matrix = self.content_vectorizer.transform(text)

            logger.info(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                index=df.index,
                columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
            )

            result = pd.concat([content_features, tfidf_df], axis=1)
            logger.info(f"  Content features shape: {result.shape}")
            return result

    
    def extract_engagement_ratio_features(self, df):
        """
        Calculate engagement ratios - important for understanding content quality
        """
        ratio_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['forward_count', 'comment_count', 'like_count']):
            total_engagement = df['forward_count'] + df['comment_count'] + df['like_count'] + 1
            
            ratio_features['forward_ratio'] = df['forward_count'] / total_engagement
            ratio_features['comment_ratio'] = df['comment_count'] / total_engagement
            ratio_features['like_ratio'] = df['like_count'] / total_engagement
            
            # Engagement diversity
            ratio_features['engagement_diversity'] = (
                -1 * (
                    ratio_features['forward_ratio'] * np.log(ratio_features['forward_ratio'] + 1e-10) +
                    ratio_features['comment_ratio'] * np.log(ratio_features['comment_ratio'] + 1e-10) +
                    ratio_features['like_ratio'] * np.log(ratio_features['like_ratio'] + 1e-10)
                )
            )
        
        return ratio_features
    
    def prepare_features(self, train_df, predict_df=None):
        """
        Combine all feature extraction methods
        """
        with Timer("prepare_features"):
            try:
                # Validate required columns
                required_cols = ['uid', 'mid']
                for col in required_cols:
                    if col not in train_df.columns:
                        raise ValueError(f"Missing required column: {col}")

                logger.info(f"Processing training data: {len(train_df):,} rows")
                user_features = self.extract_user_network_features(train_df)

                train_temporal = self.extract_temporal_features(train_df)

                train_content = self.extract_content_features(train_df, fit_vectorizer=True)

                train_ratios = self.extract_engagement_ratio_features(train_df)

                # Merge all features for training data
                logger.info("Merging training features...")
                train_features = train_df.merge(
                    user_features,
                    left_on='uid',
                    right_index=True,
                    how='left'
                )
                train_features = pd.concat([train_features.reset_index(drop=True),
                                           train_temporal.reset_index(drop=True),
                                           train_content.reset_index(drop=True),
                                           train_ratios.reset_index(drop=True)], axis=1)
                logger.info(f"Training features shape: {train_features.shape}")

                if predict_df is not None:
                    logger.info(f"Processing prediction data: {len(predict_df):,} rows")
                    pred_temporal = self.extract_temporal_features(predict_df)
                    pred_content = self.extract_content_features(predict_df, fit_vectorizer=False)
                    pred_ratios = self.extract_engagement_ratio_features(predict_df)

                    pred_features = predict_df.merge(
                        user_features,
                        left_on='uid',
                        right_index=True,
                        how='left'
                    )
                    pred_features = pd.concat([pred_features.reset_index(drop=True),
                                              pred_temporal.reset_index(drop=True),
                                              pred_content.reset_index(drop=True),
                                              pred_ratios.reset_index(drop=True)], axis=1)
                    logger.info(f"Prediction features shape: {pred_features.shape}")

                    return train_features, pred_features

                return train_features
            except Exception as e:
                logger.error(f"Error in prepare_features: {str(e)}", exc_info=True)
                raise
    
    def train_models(self, train_features):
        """
        Train separate models for each engagement metric
        Using LightGBM native API for memory efficiency (matching v2 fixes)
        """
        with Timer("train_models"):
            try:
                # Define feature columns (exclude non-feature columns)
                exclude_cols = ['uid', 'mid', 'time', 'content', 'forward_count', 'comment_count', 'like_count']
                feature_cols = [col for col in train_features.columns if col not in exclude_cols]

                # Remove any NaN column names
                feature_cols = [col for col in feature_cols if pd.notna(col)]

                X = train_features[feature_cols].fillna(0).values.astype('float32')

                logger.info(f"Training on {len(feature_cols)} features with {X.shape[0]:,} samples")

                # Memory-optimized LightGBM parameters (matching v2 fixes)
                lgb_params = {
                    'objective': 'regression_l1',
                    'learning_rate': 0.08,
                    'num_leaves': 31,
                    'min_child_samples': 20,
                    'max_bin': 31,
                    'feature_fraction': 0.7,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 5,
                    'lambda_l1': 0.1,
                    'lambda_l2': 0.1,
                    'num_threads': 1,  # critical: single-threaded to avoid per-thread buffers
                    'histogram_pool_size': 16,  # MB; caps contiguous allocation
                    'force_col_wise': True,  # avoids large row-wise working buffers
                    'verbose': -1,
                }

                # Train models for each target
                for idx, target in enumerate(['forward_count', 'comment_count', 'like_count'], 1):
                    if target not in train_features.columns:
                        logger.warning(f"Target column {target} not found, skipping...")
                        continue

                    with Timer(f"train_model_{idx}/3_{target}"):
                        logger.info(f"Training model {idx}/3 for {target}")
                        y = train_features[target].values.astype('float32')

                        # Log target statistics
                        logger.info(f"  Target stats: mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.2f}, max={y.max():.2f}")

                        # Use log1p transform for skewed count data
                        y_log = np.log1p(y)

                        # Use native lgb.train API with free_raw_data=True for memory efficiency
                        logger.info(f"  Creating LightGBM dataset with free_raw_data=True...")
                        ds = lgb.Dataset(X, label=y_log, free_raw_data=True,
                                         params={'max_bin': lgb_params['max_bin']})
                        ds.construct()  # Force binning before training

                        logger.info(f"  Training {200} boosting rounds...")
                        booster = lgb.train(
                            {k: v for k, v in lgb_params.items() if k != 'n_estimators'},
                            ds,
                            num_boost_round=200,  # conservative: 500→200 to save memory/time
                        )
                        del ds
                        logger.info(f"  Booster created with {booster.num_trees()} trees")

                        # Predict and inverse log1p
                        logger.info(f"  Making predictions on {X.shape[0]:,} samples...")
                        y_pred_log = booster.predict(X)
                        y_pred = np.expm1(y_pred_log).clip(0)

                        # Calculate metrics
                        metrics = {
                            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                            'mae': mean_absolute_error(y, y_pred),
                            'r2': r2_score(y, y_pred),
                        }

                        target_key = target.split('_')[0]
                        self.models[target_key] = booster
                        self.evaluation_metrics[target_key] = metrics

                        logger.info(f"  ✓ Model metrics - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")

                self.feature_cols = feature_cols
                logger.info("✓ All models trained successfully!")

            except Exception as e:
                logger.error(f"Error in train_models: {str(e)}", exc_info=True)
                raise
    
    def predict(self, pred_features):
        """
        Predict engagement metrics for new posts
        """
        with Timer("predict"):
            try:
                if self.feature_cols is None:
                    raise ValueError("Model not trained. Call train_models first.")

                logger.info(f"Making predictions for {len(pred_features):,} samples")

                # Ensure all feature columns exist
                missing_cols = [col for col in self.feature_cols if col not in pred_features.columns]
                if missing_cols:
                    logger.warning(f"Missing features {missing_cols}, filling with 0")
                    for col in missing_cols:
                        pred_features[col] = 0

                X_pred = pred_features[self.feature_cols].fillna(0).values.astype('float32')

                # Inverse log1p transform for predictions
                logger.info("Predicting engagement metrics...")
                predictions = pd.DataFrame({
                    'uid': pred_features['uid'].values,
                    'mid': pred_features['mid'].values,
                    'forward_count': np.expm1(self.models['forward'].predict(X_pred)).clip(0),
                    'comment_count': np.expm1(self.models['comment'].predict(X_pred)).clip(0),
                    'like_count': np.expm1(self.models['like'].predict(X_pred)).clip(0)
                })

                # Ensure non-negative integer predictions
                for col in ['forward_count', 'comment_count', 'like_count']:
                    predictions[col] = predictions[col].round().astype(int)

                logger.info(f"✓ Predictions completed. Shape: {predictions.shape}")
                logger.info(f"  forward_count: mean={predictions['forward_count'].mean():.2f}, max={predictions['forward_count'].max()}")
                logger.info(f"  comment_count: mean={predictions['comment_count'].mean():.2f}, max={predictions['comment_count'].max()}")
                logger.info(f"  like_count: mean={predictions['like_count'].mean():.2f}, max={predictions['like_count'].max()}")

                return predictions

            except Exception as e:
                logger.error(f"Error in predict: {str(e)}", exc_info=True)
                raise
    
    def format_submission(self, predictions):
        """
        Format predictions according to competition requirements
        """
        submission = predictions.copy()
        
        # Format: uid, mid, forward_count, comment_count, like_count
        # Separated by tabs and commas as specified
        submission_str = []
        for _, row in submission.iterrows():
            line = f"{row['uid']}\t{row['mid']}\t{row['forward_count']},{row['comment_count']},{row['like_count']}"
            submission_str.append(line)
        
        return '\n'.join(submission_str)
    
    def calculate_competition_precision(self, predictions_df, actual_df):
        """
        Calculate Weibo competition precision metric
        
        Based on the official competition metric:
        - Deviation for each metric: deviation_i = |count_p_i - count_t_i| / (count_t_i + 3)
        - Individual precision: precision_i = 1 - 0.5*dev_f - 0.25*dev_c - 0.25*dev_l
        - Overall precision: sum of weighted precision where precision_i > 0.9
        
        Args:
            predictions_df: DataFrame with predicted counts [forward_count, comment_count, like_count]
            actual_df: DataFrame with actual counts [forward_count, comment_count, like_count]
        
        Returns:
            dict: Contains individual precisions and overall precision
        """
        try:
            # Calculate deviations for each metric
            eps = 3  # Smoothing constant
            
            # Forward deviation
            deviation_f = np.abs(predictions_df['forward_count'] - actual_df['forward_count']) / (actual_df['forward_count'] + eps)
            
            # Comment deviation
            deviation_c = np.abs(predictions_df['comment_count'] - actual_df['comment_count']) / (actual_df['comment_count'] + eps)
            
            # Like deviation
            deviation_l = np.abs(predictions_df['like_count'] - actual_df['like_count']) / (actual_df['like_count'] + eps)
            
            # Individual precision for each sample
            precision_scores = 1 - (0.5 * deviation_f + 0.25 * deviation_c + 0.25 * deviation_l)
            
            # Calculate overall precision with 0.9 threshold
            total_count = actual_df['forward_count'] + actual_df['comment_count'] + actual_df['like_count']
            weights = total_count + 1
            
            # Sign function: 1 if precision > 0.9, else 0
            precision_mask = (precision_scores > 0.9).astype(int)
            
            # Overall precision
            overall_precision = (weights * precision_mask).sum() / weights.sum()
            
            return {
                'deviation_forward': deviation_f,
                'deviation_comment': deviation_c,
                'deviation_like': deviation_l,
                'individual_precision': precision_scores,
                'overall_precision': overall_precision,
                'precision_above_threshold': precision_mask.sum(),
                'total_samples': len(predictions_df),
                'mean_individual_precision': precision_scores.mean(),
                'median_individual_precision': precision_scores.median()
            }
            
        except Exception as e:
            print(f"Error calculating competition precision: {str(e)}")
            return None
    
    def save_evaluation_results(self, predictions, output_path='weibo_result_data_v3_metrics.txt', actual_values=None):
        """
        Save comprehensive evaluation metrics and results to file
        
        Args:
            predictions: DataFrame with predicted engagement counts
            output_path: Path to save results
            actual_values: Optional DataFrame with actual engagement counts for competition precision calculation
        """
        try:
            # Calculate competition precision if actual values provided
            competition_precision = None
            if actual_values is not None:
                competition_precision = self.calculate_competition_precision(predictions, actual_values)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("WEIBO ENGAGEMENT PREDICTION - PIPELINE v3 RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                # Header information
                f.write(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Number of features: {len(self.feature_cols)}\n")
                f.write(f"Number of predictions: {len(predictions)}\n\n")
                
                # Competition Precision Metric (if available)
                if competition_precision is not None:
                    f.write("-" * 80 + "\n")
                    f.write("COMPETITION PRECISION METRICS (Official Metric)\n")
                    f.write("-" * 80 + "\n\n")
                    f.write(f"Overall Precision: {competition_precision['overall_precision']:.6f}\n")
                    f.write(f"Mean Individual Precision: {competition_precision['mean_individual_precision']:.6f}\n")
                    f.write(f"Median Individual Precision: {competition_precision['median_individual_precision']:.6f}\n")
                    f.write(f"Samples with Precision > 0.9: {competition_precision['precision_above_threshold']}/{competition_precision['total_samples']}\n\n")
                    f.write("Deviations (Mean):\n")
                    f.write(f"  Forward Deviation:  {competition_precision['deviation_forward'].mean():.6f}\n")
                    f.write(f"  Comment Deviation:  {competition_precision['deviation_comment'].mean():.6f}\n")
                    f.write(f"  Like Deviation:     {competition_precision['deviation_like'].mean():.6f}\n\n")
                
                # Model evaluation metrics
                f.write("-" * 80 + "\n")
                f.write("MODEL EVALUATION METRICS\n")
                f.write("-" * 80 + "\n\n")
                
                for target_key, metrics in self.evaluation_metrics.items():
                    f.write(f"{target_key.upper()} MODEL METRICS:\n")
                    f.write(f"  Best Iteration: {metrics.get('best_iteration', 'N/A')}\n")
                    f.write(f"  Training RMSE:  {metrics['rmse_train']:.6f}\n")
                    f.write(f"  Validation RMSE: {metrics['rmse_val']:.6f}\n")
                    f.write(f"  Training MAE:   {metrics['mae_train']:.6f}\n")
                    f.write(f"  Validation MAE:  {metrics['mae_val']:.6f}\n")
                    f.write(f"  Training R²:    {metrics['r2_train']:.6f}\n")
                    f.write(f"  Validation R²:   {metrics['r2_val']:.6f}\n\n")
                
                # Prediction statistics
                f.write("-" * 80 + "\n")
                f.write("PREDICTION STATISTICS\n")
                f.write("-" * 80 + "\n\n")
                
                for col in ['forward_count', 'comment_count', 'like_count']:
                    col_short = col.split('_')[0]
                    f.write(f"{col_short.upper()} COUNT STATISTICS:\n")
                    f.write(f"  Mean:   {predictions[col].mean():.2f}\n")
                    f.write(f"  Median: {predictions[col].median():.2f}\n")
                    f.write(f"  Std:    {predictions[col].std():.2f}\n")
                    f.write(f"  Min:    {predictions[col].min()}\n")
                    f.write(f"  Max:    {predictions[col].max()}\n")
                    f.write(f"  25%:    {predictions[col].quantile(0.25):.0f}\n")
                    f.write(f"  75%:    {predictions[col].quantile(0.75):.0f}\n\n")
                
                # Overall statistics
                total_engagement = predictions['forward_count'].sum() + \
                                   predictions['comment_count'].sum() + \
                                   predictions['like_count'].sum()
                f.write(f"TOTAL PREDICTED ENGAGEMENT: {int(total_engagement)}\n")
                f.write(f"  Forwards:  {int(predictions['forward_count'].sum())}\n")
                f.write(f"  Comments:  {int(predictions['comment_count'].sum())}\n")
                f.write(f"  Likes:     {int(predictions['like_count'].sum())}\n\n")
                
                # Feature information
                f.write("-" * 80 + "\n")
                f.write("FEATURE INFORMATION\n")
                f.write("-" * 80 + "\n\n")
                f.write(f"Total features used: {len(self.feature_cols)}\n\n")
                f.write("Feature list:\n")
                for i, feat in enumerate(self.feature_cols, 1):
                    f.write(f"  {i}. {feat}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
            
            print(f"✓ Evaluation results saved to {output_path}")
            
        except Exception as e:
            print(f"Error saving evaluation results: {str(e)}")
            raise

class AdvancedNetworkFeatures:
    """
    Advanced social network analysis features
    Based on influence measurement literature [[21-24, 27]]
    """
    
    def __init__(self):
        self.user_interaction_graph = None
        
    def build_user_interaction_graph(self, df):
        """
        Build a user interaction graph based on content similarities
        This creates implicit network connections
        """
        G = nx.Graph()
        
        # Add users as nodes
        users = df['uid'].unique()
        G.add_nodes_from(users)
        
        # Create edges based on content similarity (users posting similar content)
        # This is a proxy for potential social connections
        user_content = df.groupby('uid')['content'].apply(lambda x: ' '.join(x.astype(str)))
        
        # Simple similarity: users who post at similar times
        user_times = df.groupby('uid')['time'].apply(lambda x: x.dt.hour.mean())
        
        for i, uid1 in enumerate(users):
            for uid2 in users[i+1:]:
                # Time-based similarity
                time_diff = abs(user_times.get(uid1, 12) - user_times.get(uid2, 12))
                if time_diff < 3:  # Similar posting times
                    weight = 1 / (time_diff + 1)
                    G.add_edge(uid1, uid2, weight=weight)
        
        self.user_interaction_graph = G
        return G
    
    def extract_centrality_features(self, df):
        """
        Extract network centrality measures for each user
        Key metrics from SNA: degree, betweenness, closeness centrality [[23, 27]]
        """
        if self.user_interaction_graph is None:
            self.build_user_interaction_graph(df)
        
        G = self.user_interaction_graph
        
        centrality_features = {}
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        for uid in df['uid'].unique():
            centrality_features[uid] = {
                'degree_centrality': degree_centrality.get(uid, 0),
                'betweenness_centrality': betweenness_centrality.get(uid, 0),
                'closeness_centrality': closeness_centrality.get(uid, 0),
                'network_size': len(G.neighbors(uid)) if G.has_node(uid) else 0
            }
        
        return pd.DataFrame.from_dict(centrality_features, orient='index')
    
    def extract_user_influence_score(self, df):
        """
        Calculate composite influence score based on multiple dimensions [[21, 22, 25]]
        """
        if 'forward_count' not in df.columns:
            return pd.DataFrame(index=df['uid'].unique(), columns=['influence_score'], data=0)
        
        user_influence = df.groupby('uid').agg({
            'forward_count': ['sum', 'mean', 'std'],
            'comment_count': ['sum', 'mean'],
            'like_count': ['sum', 'mean'],
            'mid': 'count'
        }).reset_index()
        
        user_influence.columns = ['uid', 
                                  'total_forwards', 'avg_forwards', 'std_forwards',
                                  'total_comments', 'avg_comments',
                                  'total_likes', 'avg_likes',
                                  'post_count']
        
        # Composite influence score
        user_influence['influence_score'] = (
            0.5 * (user_influence['total_forwards'] / (user_influence['total_forwards'].max() + 1)) +
            0.25 * (user_influence['total_comments'] / (user_influence['total_comments'].max() + 1)) +
            0.25 * (user_influence['total_likes'] / (user_influence['total_likes'].max() + 1))
        )
        
        return user_influence[['uid', 'influence_score']]

def main(train_path=None, predict_path=None, output_path=None, actual_values_path=None):
    """
    Main pipeline execution

    Args:
        train_path: Path to training data (default: Weibo Data/weibo_train_data/weibo_train_data.txt)
        predict_path: Path to prediction data (default: Weibo Data/weibo_predict_data/weibo_predict_data.txt)
        output_path: Output file path (default: weibo_result_data.txt)
        actual_values_path: Path to actual engagement values for competition precision calculation (optional)
    """
    logger.info("=" * 80)
    logger.info("WEIBO ENGAGEMENT PREDICTION PIPELINE v3")
    logger.info("=" * 80)

    try:
        # Set default paths
        if train_path is None:
            train_path = 'Weibo Data/weibo_train_data/weibo_train_data.txt'
        if predict_path is None:
            predict_path = 'Weibo Data/weibo_predict_data/weibo_predict_data.txt'
        if output_path is None:
            output_path = 'Weibo Data/weibo_result_data/weibo_result_data_v3.txt'

        # Validate paths exist
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not os.path.exists(predict_path):
            raise FileNotFoundError(f"Prediction data not found: {predict_path}")

        # Load data
        with Timer("load_data"):
            logger.info(f"Loading training data from {train_path}")
            train_columns = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content']
            pred_columns = ['uid', 'mid', 'time', 'content']

            train_df = pd.read_csv(train_path, sep='\t', header=None, names=train_columns)
            predict_df = pd.read_csv(predict_path, sep='\t', header=None, names=pred_columns)

            logger.info(f"  Training data: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
            logger.info(f"  Prediction data: {predict_df.shape[0]:,} rows × {predict_df.shape[1]} columns")

            # Validate columns in training data
            required_train_cols = ['uid', 'mid']
            for col in required_train_cols:
                if col not in train_df.columns:
                    raise ValueError(f"Missing required column in training data: {col}")

            # Validate columns in prediction data (may have different columns)
            required_pred_cols = ['uid', 'mid']
            for col in required_pred_cols:
                if col not in predict_df.columns:
                    raise ValueError(f"Missing required column in prediction data: {col}")

            # Convert time to datetime
            if 'time' in train_df.columns:
                train_df['time'] = pd.to_datetime(train_df['time'], errors='coerce')
            if 'time' in predict_df.columns:
                predict_df['time'] = pd.to_datetime(predict_df['time'], errors='coerce')

            # Convert engagement metrics to numeric
            for col in ['forward_count', 'comment_count', 'like_count']:
                if col in train_df.columns:
                    train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)

        # Initialize predictor
        logger.info("Initializing SocialNetworkEngagementPredictor...")
        predictor = SocialNetworkEngagementPredictor()

        # Prepare features
        with Timer("prepare_features"):
            train_features, pred_features = predictor.prepare_features(train_df, predict_df)
            logger.info(f"  Train features: {train_features.shape}")
            logger.info(f"  Pred features: {pred_features.shape}")

        # Add advanced network features
        logger.info("\n" + "=" * 80)
        logger.info("EXTRACTING ADVANCED NETWORK FEATURES")
        logger.info("=" * 80)
        with Timer("network_features"):
            network_extractor = AdvancedNetworkFeatures()

            train_centrality = network_extractor.extract_centrality_features(train_df)
            train_influence = network_extractor.extract_user_influence_score(train_df)

            # Merge network features
            train_features = train_features.merge(
                train_centrality, left_on='uid', right_index=True, how='left'
            )
            train_features = train_features.merge(
                train_influence, on='uid', how='left'
            )

            pred_features = pred_features.merge(
                train_centrality, left_on='uid', right_index=True, how='left'
            )
            pred_features = pred_features.merge(
                train_influence, on='uid', how='left'
            )

            # Fill NaN values
            train_features = train_features.fillna(0)
            pred_features = pred_features.fillna(0)

            logger.info(f"  Final train features: {train_features.shape}")
            logger.info(f"  Final pred features: {pred_features.shape}")

        # Train models
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING MODELS")
        logger.info("=" * 80)
        predictor.train_models(train_features)

        # Make predictions
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING PREDICTIONS")
        logger.info("=" * 80)
        predictions = predictor.predict(pred_features)

        logger.info(f"✓ Generated {len(predictions):,} predictions")

        # Format and save submission
        logger.info("\n" + "=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)
        with Timer("save_results"):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            submission = predictor.format_submission(predictions)

            with open(output_path, 'w') as f:
                f.write(submission)

            logger.info(f"✓ Results saved to {output_path}")

        # Save evaluation metrics
        metrics_output_path = output_path.replace('.txt', '_metrics.txt')

        # Load actual values if path provided
        actual_values = None
        if actual_values_path is not None and os.path.exists(actual_values_path):
            try:
                logger.info(f"Loading actual engagement values from {actual_values_path}...")
                actual_columns = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content']
                actual_values = pd.read_csv(actual_values_path, sep='\t', header=None, names=actual_columns)
                actual_values = actual_values[actual_values['mid'].isin(predictions['mid'])]
                actual_values = actual_values[['forward_count', 'comment_count', 'like_count']]
                logger.info(f"✓ Loaded actual values for {len(actual_values):,} posts")
            except Exception as e:
                logger.warning(f"Could not load actual values: {str(e)}")
                actual_values = None

        predictor.save_evaluation_results(predictions, metrics_output_path, actual_values=actual_values)

        logger.info("\n" + "=" * 80)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return predictions

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    predictions = main()