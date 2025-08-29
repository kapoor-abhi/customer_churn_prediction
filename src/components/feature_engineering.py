import os
import argparse
import yaml
import pandas as pd
import numpy as np
import dask.dataframe as dd
import logging



print("helllo i am from feature engineering how are you")
def feature_engineering(config_path):
    # ----------------- Step 1: Load Configuration and Data -----------------
    print("Loading configuration and processed data...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    root_dir = config['data_ingestion']['root_dir']
    processed_data_dir = os.path.join(root_dir, 'processed')
    train_df = pd.read_parquet(os.path.join(processed_data_dir, 'train.parquet'))
    members_df = pd.read_parquet(os.path.join(processed_data_dir, 'members.parquet'))
    transactions_df = pd.read_parquet(os.path.join(processed_data_dir, 'transactions.parquet'))
    user_logs_dd = dd.read_parquet(os.path.join(processed_data_dir, 'user_logs.parquet'))
    print("Data loaded.\n")

    # ----------------- Step 2: Transaction Features -----------------
    print("Engineering transaction features...")
    epsilon = 1e-6
    snapshot_date = transactions_df['transaction_date'].max() + pd.Timedelta(days=1)
    promo_transactions = transactions_df[transactions_df['actual_amount_paid'] == 0]
    promo_counts = promo_transactions.groupby('msno').size().reset_index(name='promo_transaction_count')
    transactions_features = transactions_df.groupby('msno').agg(
        total_transactions=('msno', 'count'),
        total_payment=('actual_amount_paid', 'sum'),
        avg_payment_value=('actual_amount_paid', 'mean'),
        avg_plan_days=('payment_plan_days', 'mean'),
        total_cancel_count=('is_cancel', 'sum'),
        last_transaction_date=('transaction_date', 'max')
    ).reset_index()
    transactions_features = pd.merge(transactions_features, promo_counts, on='msno', how='left')
    transactions_features['promo_transaction_count'].fillna(0, inplace=True)
    transactions_features['days_since_last_transaction'] = (
        snapshot_date - transactions_features['last_transaction_date']).dt.days
    transactions_features['cancel_rate'] = (transactions_features['total_cancel_count'] /
                                            (transactions_features['total_transactions'] + epsilon))
    transactions_features['promo_ratio'] = (transactions_features['promo_transaction_count'] /
                                            (transactions_features['total_transactions'] + epsilon))
    transactions_features.drop(columns=['last_transaction_date'], inplace=True)
    print("Transaction features engineered.\n")

    # ----------------- Step 3: User Logs Feature Engineering with Dask -----------------
    print("Engineering user log features (Dask: may take time)...")
    user_logs_dd['date'] = dd.to_datetime(user_logs_dd['date'])
    max_seconds_in_a_day = 86400
    user_logs_dd['total_secs'] = user_logs_dd['total_secs'].clip(upper=max_seconds_in_a_day)
    user_logs_dd['total_songs_daily'] = (
        user_logs_dd['num_25'] + user_logs_dd['num_50'] +
        user_logs_dd['num_75'] + user_logs_dd['num_985'] + user_logs_dd['num_100']
    )
    numeric_logs = ['total_secs', 'num_unq', 'num_100', 'total_songs_daily', 'num_25','num_50','num_75','num_985']
    logs_agg_dd = user_logs_dd[['msno', 'date'] + numeric_logs]
    # Only aggregate appropriate columns!
    user_logs_features_dd = logs_agg_dd.groupby('msno').agg(
        total_secs_played=('total_secs', 'sum'),
        avg_secs_played_daily=('total_secs', 'mean'),
        total_unique_songs=('num_unq', 'sum'),
        avg_unique_songs_daily=('num_unq', 'mean'),
        total_songs_played=('total_songs_daily', 'sum'),
        total_songs_100_percent=('num_100', 'sum'),
        active_days=('date', 'count'),
    ).reset_index()
    user_logs_features_df = user_logs_features_dd.compute()
    user_logs_features_df['completion_rate'] = (
        user_logs_features_df['total_songs_100_percent'] /
        (user_logs_features_df['total_songs_played'] + epsilon)
    )
    user_logs_features_df['uniqueness_rate'] = (
        user_logs_features_df['total_unique_songs'] /
        (user_logs_features_df['total_songs_played'] + epsilon)
    )
    print("User logs features engineered.\n")

    # ----------------- Step 4: Advanced Feature Engineering - Trends & Recency -----------------
    print("Starting advanced feature engineering...\nThis will take several minutes as it involves multiple Dask computations.\n")
    end_of_month = pd.to_datetime('2017-03-31')
    mid_month = pd.to_datetime('2017-03-15')
    # First half
    print("Computing features for the first half of the month...")
    first_half_logs = user_logs_dd[user_logs_dd['date'] < mid_month]
    first_half_features = first_half_logs.groupby('msno').agg(
        active_days_first_half=('date', 'count'),
        total_secs_first_half=('total_secs', 'sum')
    ).compute()
    # Second half
    print("Computing features for the second half of the month...")
    second_half_logs = user_logs_dd[user_logs_dd['date'] >= mid_month]
    second_half_features = second_half_logs.groupby('msno').agg(
        active_days_second_half=('date', 'count'),
        total_secs_second_half=('total_secs', 'sum')
    ).compute()
    # Recency
    print("Computing engagement recency features...")
    recency_features = user_logs_dd.groupby('msno').agg(
        last_listen_date=('date', 'max')
    ).compute()
    recency_features['days_since_last_listen'] = (
        end_of_month - recency_features['last_listen_date']
    ).dt.days
    recency_features.drop(columns=['last_listen_date'], inplace=True)
    # Combine
    print("Merging all new features together...")
    trend_features_df = recency_features.copy()
    trend_features_df = trend_features_df.merge(first_half_features, on='msno', how='outer')
    trend_features_df = trend_features_df.merge(second_half_features, on='msno', how='outer')
    trend_features_df.fillna(0, inplace=True)
    # Final ratios
    print("Calculating final trend ratios...")
    trend_features_df['activity_trend_abs'] = (
        trend_features_df['active_days_second_half'] -
        trend_features_df['active_days_first_half']
    )
    trend_features_df['secs_trend_ratio'] = (
        trend_features_df['total_secs_second_half'] /
        (trend_features_df['total_secs_first_half'] + epsilon)
    )
    trend_features_df.replace([np.inf, -np.inf], 0, inplace=True)
    trend_features_df['secs_trend_ratio'] = trend_features_df['secs_trend_ratio'].clip(upper=10)
    print("\nAdvanced feature engineering complete.")
    print("------ New Trend & Recency Features ------")
    print(trend_features_df.head())
    print("\nDescription of the new features:")
    print(trend_features_df[['days_since_last_listen', 'activity_trend_abs', 'secs_trend_ratio']].describe())

    # ----------------- Step 5: Master Merge & Final Imputation -----------------
    print("\nMerging all features into the master DataFrame and final imputation...")
    df_master = pd.merge(train_df, members_df, on='msno', how='left')
    df_master = pd.merge(df_master, transactions_features, on='msno', how='left')
    df_master = pd.merge(df_master, user_logs_features_df, on='msno', how='left')
    df_master = pd.merge(df_master, trend_features_df, on='msno', how='left')
    df_master['gender'].fillna('Unknown', inplace=True)
    df_master['city'].fillna(0, inplace=True)
    df_master['registered_via'].fillna(0, inplace=True)
    age_bins = [0, 18, 25, 35, 50, 80]
    age_labels = ['0-18', '19-25', '26-35', '36-50', '51-80']
    df_master['age_group'] = pd.cut(df_master['bd'], bins=age_bins, labels=age_labels, right=False)
    df_master['age_group'] = df_master['age_group'].cat.add_categories('Unknown').fillna('Unknown')
    # Fill all numeric features with 0
    num_cols = df_master.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df_master[col].fillna(0, inplace=True)
    print("Final feature engineering and imputation complete.")

    # ----------------- Step 6: Save Final Output -----------------
    featured_data_dir = os.path.join(root_dir, 'featured')
    os.makedirs(featured_data_dir, exist_ok=True)
    output_path = os.path.join(featured_data_dir, 'master_features.parquet')
    df_master.to_parquet(output_path)
    print(f"\nAll features saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering for KKBox Churn Prediction")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    feature_engineering(args.config)
