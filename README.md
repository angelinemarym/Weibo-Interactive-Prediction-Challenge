# Weibo Engagement Prediction - Pipeline v3

## Overview

This project implements a machine learning pipeline to predict social media engagement metrics for Weibo posts. It was developed as part of the **Aliyun Tianchi Challenge: Weibo User Post Engagement Prediction**.

**Competition Link:** https://tianchi.aliyun.com/competition/entrance/231574/information

## Challenge Objective

Predict the number of **forwards, comments, and likes** for Weibo posts based on user profile features, temporal patterns, and content characteristics. The competition focuses on building accurate engagement prediction models using real Weibo social network data.

## Pipeline Architecture

### Overview
The v3 pipeline is a comprehensive social network analysis-based approach that combines:
- User network features (historical statistics, engagement patterns)
- Temporal features (time of day, day of week, seasonal patterns)
- Content features (text analysis, TF-IDF, linguistic patterns)
- Network centrality measures (degree, betweenness, closeness)
- User influence scoring

### Key Components

#### 1. **SocialNetworkEngagementPredictor** Class
Main prediction pipeline with the following methods:

- **`extract_user_network_features()`**: Extracts user-level statistics
  - Total posts, average engagement metrics
  - Engagement variance and diversity
  - Historical posting frequency
  
- **`extract_temporal_features()`**: Time-based features
  - Day of week and weekend indicator
  - Hour of day and peak hour indicator
  - Day/month temporal patterns
  
- **`extract_content_features()`**: Text and content analysis
  - Content length and word count
  - Hashtag, mention, and URL counts
  - Question and exclamation marks
  - Emoji presence
  - TF-IDF vectorization (1000 features)
  
- **`extract_engagement_ratio_features()`**: Engagement proportions
  - Forward/comment/like ratios
  - Engagement diversity (entropy-based)
  
- **`train_models()`**: Trains separate LightGBM models for each metric
  - One model per engagement type (forward, comment, like)
  - 80/20 train-validation split
  - Early stopping with 50-round patience
  
- **`predict()`**: Generates predictions for new posts

#### 2. **AdvancedNetworkFeatures** Class
Social network analysis features:

- **`build_user_interaction_graph()`**: Creates implicit network graph
  - Nodes: users
  - Edges: weighted by posting time similarity
  
- **`extract_centrality_features()`**: Network centrality measures
  - Degree centrality
  - Betweenness centrality
  - Closeness centrality
  - Network neighborhood size
  
- **`extract_user_influence_score()`**: Composite influence metric
  - Weighted combination of engagement metrics
  - Normalized by maximum values

### Model Details

**Algorithm:** LightGBM (Light Gradient Boosting Machine)

**Hyperparameters:**
```python
{
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'n_estimators': 500,
    'random_state': 42
}
```

**Output:** Non-negative integers (predictions clipped to 0 and rounded)

## Data Format

### Training Data (7 columns)
```
uid | mid | time | forward_count | comment_count | like_count | content
```

### Prediction Data (4 columns)
```
uid | mid | time | content
```

### Output Format
```
uid\tmid\tforward_count,comment_count,like_count
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Locally
```bash
python weibo_pipeline_v3.py
```

### 3. Run on HPC (SLURM)
```bash
chmod +x run_pipeline.sh
sbatch run_pipeline.sh
```

Check status:
```bash
squeue -u $USER
tail -f logs/pipeline_*.log
```

## File Structure

```
weibo-baseline/
├── weibo_pipeline_v3.py          # Main pipeline (production-ready)
├── weibo_pipeline_v2.py          # Previous version
├── weibo_pipeline.py             # Initial version
├── run_pipeline.sh               # SLURM batch script (CPU)
├── requirements.txt              # Python dependencies
├── check_columns.py              # Data inspection utility
├── HPC_README.md                 # HPC submission guide
├── README.md                     # This file
└── Weibo Data/
    ├── weibo_train_data/
    │   └── weibo_train_data.txt  # Training dataset (1.2M rows)
    ├── weibo_predict_data/
    │   └── weibo_predict_data.txt # Test dataset (177K rows)
    └── weibo_result_data/
        ├── weibo_result_data_v3.txt    # v3 predictions
        ├── weibo_result_data_v2.txt    # v2 predictions
        └── weibo_result_data.txt       # v1 predictions
```

## Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- lightgbm >= 3.0.0
- networkx >= 2.6

All requirements are in `requirements.txt`

## Performance

### Feature Count
- User network features: ~10
- Temporal features: ~8
- Content features: ~1020 (including TF-IDF)
- Engagement ratio features: ~4
- Network centrality features: ~4
- **Total: ~1046 features**

### Training Time
Typical runtime on 8 CPUs with 32GB RAM: ~15-30 minutes

## Customization

### Modify File Paths
```python
from weibo_pipeline_v3 import main

predictions = main(
    train_path='custom/path/train.txt',
    predict_path='custom/path/predict.txt',
    output_path='custom/output.txt'
)
```

### Adjust LightGBM Parameters
Edit in `train_models()` method:
```python
lgb_params = {
    'learning_rate': 0.05,  # Reduce for better generalization
    'num_leaves': 63,       # Increase for more complexity
    'n_estimators': 1000,   # More iterations
    # ... other parameters
}
```

### Enable/Disable Features
Modify feature extraction calls in `prepare_features()`:
```python
# Comment out to disable specific features
train_temporal = self.extract_temporal_features(train_df)
train_content = self.extract_content_features(train_df)
# ... etc
```

## Error Handling

The pipeline includes comprehensive error handling:
- ✓ File existence validation
- ✓ Column validation with helpful error messages
- ✓ Type conversion with error coercion
- ✓ NaN/missing value handling
- ✓ Feature dimension mismatch detection
- ✓ Graceful degradation for missing features

## Debugging

Enable verbose output by checking `logs/debug.log` after HPC submission:

```bash
cat logs/debug.log
tail -f logs/pipeline_*.log
```

## Future Improvements

- [ ] Cross-validation for better model evaluation
- [ ] Hyperparameter tuning (grid/random search)
- [ ] Feature importance analysis
- [ ] Ensemble methods (stacking multiple models)
- [ ] Deep learning approaches (LSTM for temporal patterns)
- [ ] Advanced NLP (BERT embeddings for content)
- [ ] Sentiment analysis components
- [ ] User follower graph integration

## References

The pipeline architecture is based on Social Network Analysis (SNA) literature:
- Centrality measures for influence prediction [23, 27]
- Temporal patterns in social media [32, 34, 37]
- Content virality factors [1, 6, 11]
- Engagement ratio analysis [varies]

## Author

Developed for Aliyun Tianchi Weibo Challenge

## License

Competition project - refer to competition terms and conditions
