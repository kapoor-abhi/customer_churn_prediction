import os
import argparse
import yaml
import pandas as pd
import logging
import sys

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_data_schema(config_path, schema_path):
    """
    Validates the schema of raw data files against a predefined schema.
    """
    try:
        logging.info("Starting data validation component.")
        
        # Load configs
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        with open(schema_path, 'r') as f:
            schema_config = yaml.safe_load(f)
            
        raw_data_dir = config['data_ingestion']['raw_data_dir']
        
        validation_status = True
        
        # Iterate over each file defined in the schema config
        for filename, expected_schema in schema_config.items():
            file_path = os.path.join(raw_data_dir, filename)
            logging.info(f"--- Validating schema for: {filename} ---")
            
            try:
                df = pd.read_csv(file_path, nrows=5) # Read only a few rows for efficiency
                actual_schema = df.dtypes.astype(str).to_dict()
                
                if actual_schema == expected_schema:
                    logging.info(f"Schema VALID for {filename}")
                else:
                    logging.error(f"Schema INVALID for {filename}")
                    logging.error(f"Expected: {expected_schema}")
                    logging.error(f"Found:    {actual_schema}")
                    validation_status = False
                    
            except FileNotFoundError:
                logging.error(f"File not found: {file_path}")
                validation_status = False
                
        if not validation_status:
            logging.error("Data schema validation failed. Halting pipeline.")
            sys.exit(1) # Exit with a non-zero status code to stop DVC
            
        logging.info("Data validation component finished successfully. All schemas are valid.")

    except Exception as e:
        logging.error(f"An error occurred during data validation: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Schema Validation for KKBox Churn Prediction")
    parser.add_argument('--config', type=str, required=True, help='Path to the main configuration file.')
    parser.add_argument('--schema', type=str, required=True, help='Path to the schema definition file.')
    args = parser.parse_args()
    validate_data_schema(args.config, args.schema)