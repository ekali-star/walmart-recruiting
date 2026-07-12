import pandas as pd
import numpy as np

def time_based_split(df, date_col='Date', val_weeks=10):
    df = df.sort_values(date_col)
    cutoff = df[date_col].max() - pd.Timedelta(weeks=val_weeks)
    train = df[df[date_col] <= cutoff]
    val = df[df[date_col] > cutoff]
    return train, val

def wmae(y_true, y_pred, is_holiday):
    weights = np.where(is_holiday, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)