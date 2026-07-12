import pandas as pd
import numpy as np

def build_features(train_df, features_df, stores_df):
    df = train_df.merge(features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
    df = df.merge(stores_df, on='Store', how='left')
    df['Date'] = pd.to_datetime(df['Date'])

    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    df[markdown_cols] = df[markdown_cols].fillna(0)
    df['Total_MarkDown'] = df[markdown_cols].sum(axis=1)

    df = df.sort_values(['Store', 'Date'])
    df['CPI'] = df.groupby('Store')['CPI'].ffill()
    df['Unemployment'] = df.groupby('Store')['Unemployment'].ffill()

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)

    df = pd.get_dummies(df, columns=['Type'], prefix='Type')

    return df