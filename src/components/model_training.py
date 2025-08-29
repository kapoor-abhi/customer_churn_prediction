import os
import argparse
import yaml
import pandas as pd
import numpy as np
import logging
import joblib
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import mlflow
import mlflow.lightgbm

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def model_training(config_path, params_path):
    """
    This component trains the model using cross-validation, and tracks the
    experiment with MLflow.
    """
    try:
        logging.info("Starting model training component.")
        
        # Load configs and params
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)

        # --- Load Data ---
        root_dir = config['data_ingestion']['root_dir']
        featured_data_dir = os.path.join(root_dir, 'featured')
        model_dir = os.path.join(root_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        logging.info("Loading master feature set.")
        df = pd.read_parquet(os.path.join(featured_data_dir, 'master_features.parquet'))

        # --- Prepare Data for Modeling ---
        logging.info("Preparing data for modeling.")
        features = [col for col in df.columns if col not in ['msno', 'is_churn', 'bd', 'registration_init_time']]
        X = df[features]
        y = df['is_churn']

        # One-hot encode categorical features
        X = pd.get_dummies(X, columns=['gender', 'city', 'registered_via', 'age_group'], dummy_na=False)
        
        # Sanitize column names for LightGBM
        X.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X.columns]

        # --- Start MLflow Experiment Tracking ---
        mlflow.set_experiment("KKBox Churn Prediction")
        
        with mlflow.start_run():
            logging.info("MLflow run started.")
            
            # Log all hyperparameters from params.yaml
            mlflow.log_params(params['lgbm_params'])
            mlflow.log_param("n_splits", params['n_splits'])

            # --- Stratified K-Fold Cross-Validation ---
            logging.info(f"Starting {params['n_splits']}-Fold Cross-Validation.")
            skf = StratifiedKFold(n_splits=params['n_splits'], shuffle=True, random_state=params['random_state'])

            fold_auc_scores = []
            trained_models = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                logging.info(f"===== FOLD {fold+1} =====")
                
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                # Identify numeric columns for scaling (excluding OHE columns)
                numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
                
                # --- FIX: Prevent data leakage by fitting scaler ONLY on train data ---
                scaler = StandardScaler()
                X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
                X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

                # Calculate scale_pos_weight for imbalanced data
                scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
                
                lgbm = lgb.LGBMClassifier(**params['lgbm_params'], 
                                          scale_pos_weight=scale_pos_weight,
                                          random_state=params['random_state'])

                lgbm.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)], 
                         eval_metric='auc',
                         callbacks=[lgb.early_stopping(50, verbose=False)])

                val_preds_proba = lgbm.predict_proba(X_val)[:, 1]
                fold_auc = roc_auc_score(y_val, val_preds_proba)
                fold_auc_scores.append(fold_auc)
                logging.info(f"AUC for Fold {fold+1}: {fold_auc:.5f}")
                
                trained_models.append(lgbm)
                
                # Save the scaler for this fold
                joblib.dump(scaler, os.path.join(model_dir, f'scaler_fold_{fold+1}.pkl'))
                
                del X_train, y_train, X_val, y_val
                gc.collect()

            # --- Log Metrics and Model to MLflow ---
            # --- FIX: Correctly calculate and log mean_auc ---
            mean_auc = np.mean(fold_auc_scores)
            std_auc = np.std(fold_auc_scores)
            
            logging.info(f"\nMean ROC AUC Score: {mean_auc:.5f}")
            logging.info(f"Standard Deviation of AUC Scores: {std_auc:.5f}")
            
            mlflow.log_metric("mean_roc_auc", mean_auc)
            mlflow.log_metric("std_roc_auc", std_auc)
            
            # Log the first model as the primary artifact
            mlflow.lightgbm.log_model(trained_models[0], "model", registered_model_name="LGBMChurnModel")
            logging.info("Model and metrics logged to MLflow.")

            # Save the first model as a pipeline output artifact
            joblib.dump(trained_models[0], os.path.join(model_dir, 'lgbm_model.pkl'))
            logging.info(f"Model artifact saved to {model_dir}")

    except Exception as e:
        logging.error(f"An error occurred in the model training component: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training for KKBox Churn Prediction")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--params', type=str, required=True, help='Path to the parameters file.')
    args = parser.parse_args()
    model_training(args.config, args.params)