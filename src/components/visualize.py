import os
import argparse
import yaml
import pandas as pd
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_data(config_path):
    """
    This component loads the final featured data and provides a
    quick summary and visualization.
    """
    try:
        logging.info("Starting data visualization component.")
        
        # Load configuration to find the data path
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        root_dir = config['data_ingestion']['root_dir']
        featured_data_dir = os.path.join(root_dir, 'featured')
        master_features_path = os.path.join(featured_data_dir, 'master_features.parquet')

        # --- Load the Final Data ---
        logging.info(f"Loading master feature set from: {master_features_path}")
        df = pd.read_parquet(master_features_path)

        # --- Display Data Summary ---
        print("\n" + "="*50)
        print("      MASTER FEATURE DATAFRAME SUMMARY")
        print("="*50)
        
        print(f"\n[INFO] Shape of the dataframe: {df.shape}")
        
        print("\n[INFO] Data types and non-null counts:")
        df.info()
        
        print("\n[INFO] First 5 rows of the dataframe:")
        print(df.head())
        
        print("\n[INFO] Churn distribution:")
        print(df['is_churn'].value_counts(normalize=True))
        
        print("\n" + "="*50)
        
        logging.info("Data visualization component finished successfully.")

    except FileNotFoundError:
        logging.error(f"Error: The file {master_features_path} was not found. Please ensure the feature_engineering stage has been run.")
        raise
    except Exception as e:
        logging.error(f"An error occurred in the visualization component: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize final feature data for KKBox Churn Prediction")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    visualize_data(args.config)