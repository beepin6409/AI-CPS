"""
preprocess_pipeline.py
----------------------
Purpose:
    Preprocess customer churn dataset for AI/ML project.
    - Load raw data from GitHub-scraped file
    - Clean and normalize data
    - Encode categorical variables
    - Remove leakage columns
    - Split into training, test, and activation datasets
    - Save cleaned data and train/test/activation CSVs
    - Save encoders and scaler for reproducibility

Author: Bipin Ghimire
"""

import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

# -------------------------------
# Configurable Parameters
# -------------------------------
RAW_FILE_NAME = "customer_churn_dataset.csv"
RAW_SUBFOLDER = "raw"
PROCESSED_SUBFOLDER = "processed"
TRAIN_RATIO = 0.8
RANDOM_STATE = 42
TARGET_COLUMN = "churn"
LEAKAGE_COLUMNS = ["customer_id"]

# -------------------------------
# Paths
# -------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
RAW_FILE_PATH = os.path.join(REPO_ROOT, "data", "customer-churn-dataset", RAW_SUBFOLDER, RAW_FILE_NAME)
PROCESSED_PATH = os.path.join(REPO_ROOT, "data", "customer-churn-dataset", PROCESSED_SUBFOLDER)
os.makedirs(PROCESSED_PATH, exist_ok=True)

ENCODERS_FILE = os.path.join(PROCESSED_PATH, "encoders.pkl")
SCALER_FILE = os.path.join(PROCESSED_PATH, "scaler.pkl")

# -------------------------------
# Function Definitions
# -------------------------------
def load_raw_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded raw data with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load raw CSV: {e}")
        raise

def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in LEAKAGE_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
            logging.info(f"Removed leakage column: {col}")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=['float64','int64']).columns

    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    logging.info("Missing values handled.")
    return df

def encode_categorical(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include='object').columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        logging.info(f"Encoded column: {col}")
    return df, encoders

def normalize_numeric(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=['float64','int64']).columns
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    logging.info(f"Normalized numeric columns: {list(num_cols)}")
    return df, scaler

def split_train_test_activation(df: pd.DataFrame):
    train_df, test_df = train_test_split(df, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
    activation_df = test_df.sample(n=1, random_state=RANDOM_STATE)
    logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}, Activation shape: {activation_df.shape}")
    return train_df, test_df, activation_df

def save_csv(df: pd.DataFrame, file_name: str):
    path = os.path.join(PROCESSED_PATH, file_name)
    df.to_csv(path, index=False)
    logging.info(f"Saved {file_name} at {path}")

def save_artifact(obj, file_path: str):
    joblib.dump(obj, file_path)
    logging.info(f"Saved artifact at {file_path}")

# -------------------------------
# Main Pipeline
# -------------------------------
def main():
    # Step 1: Load raw data
    df = load_raw_data(RAW_FILE_PATH)

    # Step 2: Preprocessing
    df = remove_leakage_columns(df)
    df = handle_missing_values(df)
    df, encoders = encode_categorical(df)
    df, scaler = normalize_numeric(df)

    # Step 3: Save cleaned joint dataset
    save_csv(df, "joint_data_collection.csv")

    # Step 4: Split train/test/activation
    train_df, test_df, activation_df = split_train_test_activation(df)
    save_csv(train_df, "training_data.csv")
    save_csv(test_df, "test_data.csv")
    save_csv(activation_df, "activation_data.csv")

    # Step 5: Save encoders and scaler for reproducibility
    save_artifact(encoders, ENCODERS_FILE)
    save_artifact(scaler, SCALER_FILE)

    logging.info("Preprocessing pipeline completed successfully.")

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    main()
