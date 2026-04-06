# Competition Precision Metric

## Overview

The Weibo Engagement Prediction Challenge uses a custom precision metric that measures prediction accuracy. This pipeline now implements the official competition metric for evaluation.

## Formula

The competition precision is calculated as follows:

### 1. Per-Metric Deviations

For each engagement type (forward, comment, like):

$$deviation_f = \frac{|count_p^f - count_t^f|}{count_t^f + 3}$$

$$deviation_c = \frac{|count_p^c - count_t^c|}{count_t^c + 3}$$

$$deviation_l = \frac{|count_p^l - count_t^l|}{count_t^l + 3}$$

Where:
- $count_p^f$ = predicted forward count
- $count_t^f$ = true/actual forward count (similar for comment and like)
- The constant `3` is a smoothing term to avoid division by zero

### 2. Individual Sample Precision

For each post, calculate weighted precision:

$$precision_i = 1 - 0.5 \times deviation_f - 0.25 \times deviation_c - 0.25 \times deviation_l$$

This gives:
- 50% weight to forward accuracy
- 25% weight to comment accuracy
- 25% weight to like accuracy

### 3. Overall Precision

The final metric is calculated as:

$$precision = \frac{\sum_{precision_i > 0.9} (count_t + 1)}{\sum (count_t + 1)}$$

Where:
- Only samples with $precision_i > 0.9$ are counted (quality threshold)
- Each sample is weighted by its total actual engagement count
- The term `+1` prevents zero weights for posts with no engagement

## Implementation

### Basic Usage

```python
from weibo_pipeline_v3 import SocialNetworkEngagementPredictor

predictor = SocialNetworkEngagementPredictor()

# After training and making predictions
predictions = pd.DataFrame({
    'forward_count': [10, 5, 15],
    'comment_count': [2, 1, 3],
    'like_count': [50, 25, 40]
})

actual = pd.DataFrame({
    'forward_count': [12, 4, 14],
    'comment_count': [1, 2, 2],
    'like_count': [55, 20, 45]
})

result = predictor.calculate_competition_precision(predictions, actual)
print(f"Overall Precision: {result['overall_precision']:.4f}")
```

### With Actual Values File

Run the pipeline with actual engagement values:

```bash
python weibo_pipeline_v3.py
```

If you have a file with actual engagement values, you can pass it:

```python
from weibo_pipeline_v3 import main

predictions = main(
    actual_values_path='path/to/actual_values.txt'
)
```

## Output Metrics

When actual values are available, the metrics file includes:

```
COMPETITION PRECISION METRICS (Official Metric)
────────────────────────────────────────────────────────────────────────────────

Overall Precision: 0.8234
Mean Individual Precision: 0.7956
Median Individual Precision: 0.8120
Samples with Precision > 0.9: 45230/177922

Deviations (Mean):
  Forward Deviation:  0.1234
  Comment Deviation:  0.2345
  Like Deviation:     0.1567
```

### Interpretation

- **Overall Precision**: Final competition score (0-1)
- **Mean Individual Precision**: Average per-sample precision across all posts
- **Median Individual Precision**: Median per-sample precision
- **Samples with Precision > 0.9**: Number of well-predicted posts
- **Deviations**: How far predictions deviate from actual values on average

## Performance Insights

### Good Precision Score (>0.85)
- Model accurately predicts engagement across all metrics
- Well-calibrated predictions relative to actual engagement

### Moderate Precision Score (0.70-0.85)
- Some predictions are significantly off
- May need feature engineering improvements
- Consider adjusting model hyperparameters

### Low Precision Score (<0.70)
- Model needs substantial improvement
- Consider:
  - Adding more relevant features
  - Collecting more training data
  - Trying different algorithms
  - Addressing data quality issues

## Notes

1. **Data Format**: Ensure actual values file has the same format as training data
2. **Metric Comparison**: This metric is stricter than simple MSE/MAE due to the 0.9 threshold
3. **Weighting**: Posts with more engagement are weighted more heavily
4. **Robustness**: The +3 smoothing helps handle edge cases with very few engagements

## References

This metric is defined in the official Aliyun Tianchi Weibo Challenge documentation.
See: https://tianchi.aliyun.com/competition/entrance/231574/information
