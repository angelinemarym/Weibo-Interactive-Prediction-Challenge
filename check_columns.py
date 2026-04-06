import pandas as pd

# Check training data
try:
    train_df = pd.read_csv('Weibo Data/weibo_train_data/weibo_train_data.txt', sep='\t', nrows=1)
    print("Training data columns:", list(train_df.columns))
    print("Training data head:")
    print(train_df.head())
except Exception as e:
    print(f"Error reading training data: {e}")

print("\n" + "="*50 + "\n")

# Check prediction data
try:
    pred_df = pd.read_csv('Weibo Data/weibo_predict_data/weibo_predict_data.txt', sep='\t', nrows=1)
    print("Prediction data columns:", list(pred_df.columns))
    print("Prediction data head:")
    print(pred_df.head())
except Exception as e:
    print(f"Error reading prediction data: {e}")
