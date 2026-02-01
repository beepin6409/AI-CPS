"""
Data Scraper for Customer Churn Dataset
Author: Bipin Ghimire
Date: January 2026
Course: M. Grum: Advanced AI-based Application Systems

This script scrapes the Customer Churn dataset from a GitHub-hosted webpage
endpoint (CSV rendered as plain text) and stores it in the raw data folder.
"""

import os
import requests

# -------------------------------------------------
# Configuration
# -------------------------------------------------
GITHUB_BLOB_URL = "https://github.com/beepin6409/Churn_data/blob/main/customer_churn_dataset.csv"
RAW_FILE_NAME = "customer_churn_dataset.csv"

# -------------------------------------------------
# Resolve raw data path (PIPELINE-SAFE)
# -------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
RAW_DATA_DIR = os.path.join(REPO_ROOT, "data", "customer-churn-dataset", "raw")
RAW_FILE_PATH = os.path.join(RAW_DATA_DIR, RAW_FILE_NAME)

# -------------------------------------------------
# Function 1: Convert blob URL to raw text URL
# -------------------------------------------------
def convert_blob_to_raw(blob_url: str) -> str:
    """
    Convert a GitHub blob URL into a raw content URL.

    Args:
        blob_url (str): GitHub blob URL

    Returns:
        str: Raw content URL
    """
    return blob_url.replace("/blob/", "/raw/")

# -------------------------------------------------
# Function 2: Scrape CSV content from GitHub webpage
# -------------------------------------------------
def scrape_csv_text(raw_url: str) -> str:
    """
    Scrape CSV text content from a GitHub raw webpage endpoint.

    Args:
        raw_url (str): Raw GitHub URL pointing to CSV content

    Returns:
        str: CSV content as plain text
    """
    print(" Starting data scraping process...")
    print(f" Raw URL: {raw_url}")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(raw_url, headers=headers)
    response.raise_for_status()

    print(f" Successfully scraped data (Status Code: {response.status_code})")
    return response.text

# -------------------------------------------------
# Function 3: Save scraped CSV into raw folder
# -------------------------------------------------
def save_raw_csv(csv_text: str, output_path: str) -> str:
    """
    Save scraped CSV text into the raw data directory.

    Args:
        csv_text (str): CSV content as text
        output_path (str): Full path to save CSV

    Returns:
        str: Path of saved file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(csv_text)

    print(f" Raw data saved successfully at: {output_path}")
    return output_path

# -------------------------------------------------
# Main execution
# -------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("🚀 CUSTOMER CHURN DATA SCRAPER")
    print("=" * 60 + "\n")

    try:
        # Step 1: Convert blob URL to raw URL
        raw_url = convert_blob_to_raw(GITHUB_BLOB_URL)

        # Step 2: Scrape CSV text content
        csv_text = scrape_csv_text(raw_url)

        # Step 3: Save CSV to raw folder
        save_raw_csv(csv_text, RAW_FILE_PATH)

        print("\n Scraping completed successfully!")

    except Exception as e:
        print(f"\n Scraping failed: {e}")

    print("\n" + "=" * 60)

# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    main()
