import os
import argparse
import yaml
import pandas as pd
import numpy as np
import dask.dataframe as dd
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def data_ingestion(config_path):
    """
    This component loads raw data, performs initial cleaning, and saves
    the cleaned dataframes to the processed data directory.
    """
    try:
        logging.info("Starting data ingestion component.")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        root_dir = config['data_ingestion']['root_dir']
        raw_data_dir = config['data_ingestion']['raw_data_dir']
        processed_data_dir = os.path.join(root_dir, 'processed')
        os.makedirs(processed_data_dir, exist_ok=True)
        logging.info(f"Processed data will be saved to: {processed_data_dir}")

        # Define file paths
        train_path = os.path.join(raw_data_dir, config['data_ingestion']['train_data_file'])
        members_path = os.path.join(raw_data_dir, config['data_ingestion']['members_data_file'])
        transactions_path = os.path.join(raw_data_dir, config['data_ingestion']['transactions_data_file'])
        user_logs_path = os.path.join(raw_data_dir, config['data_ingestion']['user_logs_data_file'])

        # --- Load and Clean Members Data ---
        logging.info("Loading and cleaning members data.")
        members_df = pd.read_csv(members_path)
        members_df.loc[(members_df['bd'] < 10) | (members_df['bd'] > 80), 'bd'] = np.nan
        members_df['gender'].fillna('unknown', inplace=True)
        members_df['registration_init_time'] = pd.to_datetime(members_df['registration_init_time'], format='%Y%m%d')
        
        # --- Load and Clean Transactions Data ---
        logging.info("Loading and cleaning transactions data.")
        transactions_df = pd.read_csv(transactions_path)
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'], format='%Y%m%d')
        transactions_df['membership_expire_date'] = pd.to_datetime(transactions_df['membership_expire_date'], format='%Y%m%d')

        # --- Load Train Data ---
        logging.info("Loading train data.")
        train_df = pd.read_csv(train_path)

        # --- Load User Logs Data (using Dask for memory efficiency) ---
        logging.info("Loading user logs data with Dask.")
        user_logs_dd = dd.read_csv(user_logs_path)
        user_logs_dd['date'] = dd.to_datetime(user_logs_dd['date'], format='%Y%m%d')

        # --- Save Processed Data ---
        # Using Parquet format is more efficient for saving dataframes than CSV
        logging.info("Saving processed dataframes to Parquet format.")
        train_df.to_parquet(os.path.join(processed_data_dir, 'train.parquet'))
        members_df.to_parquet(os.path.join(processed_data_dir, 'members.parquet'))
        transactions_df.to_parquet(os.path.join(processed_data_dir, 'transactions.parquet'))
        # Dask saves its dataframe in a directory
        user_logs_dd.to_parquet(os.path.join(processed_data_dir, 'user_logs.parquet'))

        logging.info("Data ingestion component finished successfully.")

    except Exception as e:
        logging.error(f"An error occurred in the data ingestion component: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Ingestion for KKBox Churn Prediction")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    data_ingestion(args.config)