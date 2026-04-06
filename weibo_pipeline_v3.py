import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
import os
import sys

warnings.filterwarnings('ignore')

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
        
    def extract_user_network_features(self, df):
        """
        Extract user-level social network features
        Based on influence metrics from SNA literature [[23, 27]]
        """
        user_features = defaultdict(dict)
        
        # Group by user and calculate historical statistics
        for uid in df['uid'].unique():
            user_data = df[df['uid'] == uid]
            
            # Degree centrality proxy: total historical engagement
            total_forwards = user_data['forward_count'].sum() if 'forward_count' in user_data.columns else 0
            total_comments = user_data['comment_count'].sum() if 'comment_count' in user_data.columns else 0
            total_likes = user_data['like_count'].sum() if 'like_count' in user_data.columns else 0
            
            # Influence metrics
            user_features[uid] = {
                'user_total_posts': len(user_data),
                'user_avg_forwards': user_data['forward_count'].mean() if 'forward_count' in user_data.columns else 0,
                'user_avg_comments': user_data['comment_count'].mean() if 'comment_count' in user_data.columns else 0,
                'user_avg_likes': user_data['like_count'].mean() if 'like_count' in user_data.columns else 0,
                'user_total_engagement': total_forwards + total_comments + total_likes,
                'user_std_forwards': user_data['forward_count'].std() if 'forward_count' in user_data.columns else 0,
                'user_std_comments': user_data['comment_count'].std() if 'comment_count' in user_data.columns else 0,
                'user_std_likes': user_data['like_count'].std() if 'like_count' in user_data.columns else 0,
                'user_engagement_variance': (
                    user_data['forward_count'].var() + 
                    user_data['comment_count'].var() + 
                    user_data['like_count'].var()
                ) / 3 if 'forward_count' in user_data.columns else 0,
            }
            
            # Temporal patterns: posting frequency [[32, 35]]
            if 'time' in user_data.columns:
                user_features[uid]['user_posting_frequency'] = len(user_data) / (
                    (user_data['time'].max() - user_data['time'].min()).days + 1
                )
            
        return pd.DataFrame.from_dict(user_features, orient='index')
    
    def extract_temporal_features(self, df):
        """
        Extract temporal patterns which significantly impact engagement [[34, 37]]
        """
        if 'time' not in df.columns:
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
        
        return temporal_features
    
    def extract_content_features(self, df):
        """
        Extract content-based features including text analysis and sentiment
        Content characteristics affect virality [[1, 6, 11]]
        """
        content_features = pd.DataFrame(index=df.index)
        
        if 'content' not in df.columns:
            return content_features
            
        # Basic text statistics
        content_features['content_length'] = df['content'].apply(lambda x: len(str(x)))
        content_features['word_count'] = df['content'].apply(lambda x: len(str(x).split()))
        content_features['avg_word_length'] = content_features['content_length'] / (content_features['word_count'] + 1)
        
        # Hashtag and mention features
        content_features['hashtag_count'] = df['content'].apply(lambda x: len(re.findall(r'#\w+', str(x))))
        content_features['mention_count'] = df['content'].apply(lambda x: len(re.findall(r'@\w+', str(x))))
        content_features['url_count'] = df['content'].apply(lambda x: len(re.findall(r'http[s]?://\S+', str(x))))
        
        # Question and exclamation (engagement triggers)
        content_features['question_count'] = df['content'].apply(lambda x: str(x).count('?'))
        content_features['exclamation_count'] = df['content'].apply(lambda x: str(x).count('!'))
        
        # Emoji count (if applicable)
        content_features['emoji_count'] = df['content'].apply(
            lambda x: len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', str(x)))
        )
        
        # TF-IDF features for content topics
        tfidf_matrix = self.content_vectorizer.fit_transform(df['content'].fillna(''))
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=df.index,
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        
        return pd.concat([content_features, tfidf_df], axis=1)
    
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
        try:
            # Validate required columns
            required_cols = ['uid', 'mid']
            for col in required_cols:
                if col not in train_df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            print("Extracting user network features...")
            user_features = self.extract_user_network_features(train_df)
            
            print("Extracting temporal features...")
            train_temporal = self.extract_temporal_features(train_df)
            
            print("Extracting content features...")
            train_content = self.extract_content_features(train_df)
            
            print("Extracting engagement ratio features...")
            train_ratios = self.extract_engagement_ratio_features(train_df)
            
            # Merge all features for training data
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
            
            if predict_df is not None:
                print("Preparing prediction data features...")
                pred_temporal = self.extract_temporal_features(predict_df)
                pred_content = self.extract_content_features(predict_df)
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
                
                return train_features, pred_features
            
            return train_features
        except Exception as e:
            print(f"Error in prepare_features: {str(e)}")
            raise
    
    def train_models(self, train_features):
        """
        Train separate models for each engagement metric
        Using LightGBM for efficient gradient boosting
        """
        try:
            # Define feature columns (exclude non-feature columns)
            exclude_cols = ['uid', 'mid', 'time', 'content', 'forward_count', 'comment_count', 'like_count']
            feature_cols = [col for col in train_features.columns if col not in exclude_cols]
            
            # Remove any NaN column names
            feature_cols = [col for col in feature_cols if pd.notna(col)]
            
            X = train_features[feature_cols].fillna(0)
            
            print(f"Training on {len(feature_cols)} features")
            
            # Initialize LightGBM parameters
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 500,
                'random_state': 42
            }
            
            # Train models for each target
            for target in ['forward_count', 'comment_count', 'like_count']:
                if target not in train_features.columns:
                    print(f"Warning: Target column {target} not found, skipping...")
                    continue
                    
                print(f"\nTraining model for {target}...")
                y = train_features[target]
                
                # Train-validation split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train model
                model = lgb.LGBMRegressor(**lgb_params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                target_key = target.split('_')[0]
                self.models[target_key] = model
                print(f"Model trained. Validation RMSE: {model.best_score_['valid_0']['rmse']:.2f}")
            
            self.feature_cols = feature_cols
            print("\nAll models trained successfully!")
            
        except Exception as e:
            print(f"Error in train_models: {str(e)}")
            raise
    
    def predict(self, pred_features):
        """
        Predict engagement metrics for new posts
        """
        try:
            if self.feature_cols is None:
                raise ValueError("Model not trained. Call train_models first.")
            
            # Ensure all feature columns exist
            missing_cols = [col for col in self.feature_cols if col not in pred_features.columns]
            if missing_cols:
                print(f"Warning: Missing features {missing_cols}, filling with 0")
                for col in missing_cols:
                    pred_features[col] = 0
            
            X_pred = pred_features[self.feature_cols].fillna(0)
            
            predictions = pd.DataFrame({
                'uid': pred_features['uid'],
                'mid': pred_features['mid'],
                'forward_count': self.models['forward'].predict(X_pred),
                'comment_count': self.models['comment'].predict(X_pred),
                'like_count': self.models['like'].predict(X_pred)
            })
            
            # Ensure non-negative integer predictions
            for col in ['forward_count', 'comment_count', 'like_count']:
                predictions[col] = predictions[col].clip(lower=0).round().astype(int)
            
            return predictions
            
        except Exception as e:
            print(f"Error in predict: {str(e)}")
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
            0.4 * (user_influence['total_forwards'] / (user_influence['total_forwards'].max() + 1)) +
            0.3 * (user_influence['total_comments'] / (user_influence['total_comments'].max() + 1)) +
            0.3 * (user_influence['total_likes'] / (user_influence['total_likes'].max() + 1))
        )
        
        return user_influence[['uid', 'influence_score']]

def main(train_path=None, predict_path=None, output_path=None):
    """
    Main pipeline execution
    
    Args:
        train_path: Path to training data (default: Weibo Data/weibo_train_data/weibo_train_data.txt)
        predict_path: Path to prediction data (default: Weibo Data/weibo_predict_data/weibo_predict_data.txt)
        output_path: Output file path (default: weibo_result_data.txt)
    """
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
        print("Loading data...")
        # Define column names for the data files (they don't have headers)
        train_columns = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content']
        pred_columns = ['uid', 'mid', 'time', 'content']
        
        train_df = pd.read_csv(train_path, sep='\t', header=None, names=train_columns)
        predict_df = pd.read_csv(predict_path, sep='\t', header=None, names=pred_columns)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Training columns: {list(train_df.columns)}")
        print(f"Prediction data shape: {predict_df.shape}")
        print(f"Prediction columns: {list(predict_df.columns)}")
        
        # Validate columns in training data
        required_train_cols = ['uid', 'mid']
        for col in required_train_cols:
            if col not in train_df.columns:
                raise ValueError(f"Missing required column in training data: {col}")
        
        # Validate columns in prediction data (may have different columns)
        required_pred_cols = ['uid', 'mid']
        for col in required_pred_cols:
            if col not in predict_df.columns:
                raise ValueError(f"Missing required column in prediction data: {col}. Available columns: {list(predict_df.columns)}")
        
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
        print("\nInitializing predictor...")
        predictor = SocialNetworkEngagementPredictor()
        
        # Prepare features
        print("\n--- FEATURE EXTRACTION ---")
        train_features, pred_features = predictor.prepare_features(train_df, predict_df)
        
        print(f"Training features shape: {train_features.shape}")
        print(f"Prediction features shape: {pred_features.shape}")
        
        # Add advanced network features
        print("\n--- NETWORK ANALYSIS ---")
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
        
        print(f"Final training features shape: {train_features.shape}")
        print(f"Final prediction features shape: {pred_features.shape}")
        
        # Train models
        print("\n--- MODEL TRAINING ---")
        predictor.train_models(train_features)
        
        # Make predictions
        print("\n--- GENERATING PREDICTIONS ---")
        predictions = predictor.predict(pred_features)
        
        print(f"Generated {len(predictions)} predictions")
        print(f"Prediction statistics:\n{predictions[['forward_count', 'comment_count', 'like_count']].describe()}")
        
        # Format and save submission
        print(f"\nSaving results to {output_path}...")
        submission = predictor.format_submission(predictions)
        
        with open(output_path, 'w') as f:
            f.write(submission)
        
        print(f"✓ Predictions saved successfully!")
        
        return predictions
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    predictions = main()