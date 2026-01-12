"""
data_ingestion.py
-----------------
Purpose:
    Scrape the customer churn dataset from a public GitHub repository and save
    it in the raw folder of the customer-churn-dataset.
    This ensures reproducibility and proper folder structure for preprocessing.

Folder Structure:
    data/
        customer-churn-dataset/
            raw/           <- downloaded CSV stored here
            processed/     <- will be used for preprocessing outputs

Outputs:
    customer_churn_dataset.csv   <- stored in raw folder

Usage:
    python data_ingestion.py

Author: Bipin Ghimire
"""

import os
import requests

# -------------------------------
# Configurations
# -------------------------------
GITHUB_RAW_URL = "https://raw.githubusercontent.com/beepin6409/Churn_data/main/customer_churn_dataset.csv"
RAW_FILE_NAME = "customer_churn_dataset.csv"

# -------------------------------
# Determine portable raw folder path
# -------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))           # directory of this script
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))   # go 2 levels up to repo root
RAW_DATA_PATH = os.path.join(REPO_ROOT, "data", "customer-churn-dataset", "raw")
RAW_FILE_PATH = os.path.join(RAW_DATA_PATH, RAW_FILE_NAME)

# -------------------------------
# Function to scrape dataset
# -------------------------------
def scrape_raw_data(github_url: str, raw_folder: str, file_name: str) -> str:
    """
    Downloads CSV from GitHub and stores it in raw folder.

    Args:
        github_url (str): Direct URL to CSV on GitHub
        raw_folder (str): Local folder path to store raw CSV
        file_name (str): Name of the CSV file to save locally

    Returns:
        str: Full local path of downloaded CSV, or None if failed
    """
    try:
        # Ensure raw folder exists
        os.makedirs(raw_folder, exist_ok=True)

        # Download CSV
        response = requests.get(github_url)
        response.raise_for_status()  # Raise HTTPError if status not 200

        # Save file
        file_path = os.path.join(raw_folder, file_name)
        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"[SUCCESS] Dataset downloaded and saved at: {file_path}")
        return file_path

    except requests.exceptions.HTTPError as http_err:
        print(f"[ERROR] HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"[ERROR] Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"[ERROR] Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"[ERROR] Unknown requests error occurred: {req_err}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

    return None

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    local_file = scrape_raw_data(GITHUB_RAW_URL, RAW_DATA_PATH, RAW_FILE_NAME)
    if local_file is None:
        print("[FAILURE] Dataset could not be downloaded. Check GitHub URL or network.")
    else:
        print("[INFO] Dataset ingestion complete. Ready for preprocessing.")
