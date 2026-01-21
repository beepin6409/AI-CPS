"""
ols_model.py
------------
Purpose:
    Train an Ordinary Least Squares (OLS) regression model using Statsmodels to
    perform the same customer churn prediction task as the AI model. The script
    evaluates performance on both training and testing datasets, stores the model
    summary and performance indicators, and generates required visualizations
    including scatter plots and regression diagnostic plots.

Folder Structure:
    data/
        customer-churn-dataset/
            processed/                 <- input CSV files are loaded from here
            output/                    <- generated summaries + plots stored here

Inputs (from processed/):
    training_data.csv                  <- training material for OLS fitting
    test_data.csv                      <- testing material for evaluation
    activation_data.csv                <- activation material for inference (optional)

Outputs (to output/):
    ols_model_summary.txt              <- statsmodels OLS summary (coefficients, p-values, etc.)
    ols_performance.txt                <- training/testing performance indicators (MSE, MAE, R²)
    ols_scatter_plots.pdf              <- scatter plots (train/test + OLS prediction line)
    UE_04_App2_DiagnosticPlots.pdf     <- statsmodels regression diagnostic plots
    ols_vif_table.csv                  <- variance inflation factors (VIF) table from diagnostics

Usage:
    python ols_model.py

Author: Paul Bakos
"""

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from diagnostic_plots import LinearRegDiagnostic
matplotlib.use("Agg")



PROCESSED_SUBFOLDER = "processed"
OUTPUT_SUBFOLDER = "output"

# -------------------------------
# Paths
# -------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

PROCESSED_PATH = os.path.join(REPO_ROOT, "data", "customer-churn-dataset", PROCESSED_SUBFOLDER)
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "customer-churn-dataset", OUTPUT_SUBFOLDER)

# inputs
TEST_DATA_PATH = os.path.join(PROCESSED_PATH, "test_data.csv")
TRAIN_DATA_PATH = os.path.join(PROCESSED_PATH, "training_data.csv")
ACTIVATION_DATA_PATH = os.path.join(PROCESSED_PATH, "activation_data.csv")

# outputs
SUMMARY_PATH = os.path.join(OUTPUT_PATH, "ols_model_summary.txt")
OLS_PERFORMANCE = os.path.join(OUTPUT_PATH, "ols_performance.txt")
AI_TRAINLOG_CSV = os.path.join(OUTPUT_SUBFOLDER, "AI_training_log.csv")
AI_TRAIN_SUMMARY_TXT = os.path.join(OUTPUT_SUBFOLDER, "AI_training_summary.txt")
SCATTER_PATH = os.path.join(OUTPUT_PATH, "ols_scatter_plots.pdf")

DIAG_PDF_PATH = os.path.join(OUTPUT_PATH, "ols_diagnostic_plots.pdf")
VIF_CSV_PATH  = os.path.join(OUTPUT_PATH, "ols_vif_table.csv")


def main():
    # -------------------------------
    # Import data
    # -------------------------------
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df  = pd.read_csv(TEST_DATA_PATH)
    act_df   = pd.read_csv(ACTIVATION_DATA_PATH)


    # -------------------------------
    # OLS Model
    # -------------------------------

    TARGET = "churn"

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]

    X_act = act_df.copy()
    if TARGET in X_act.columns:
        X_act = X_act.drop(columns=[TARGET])

    X_test = X_test.reindex(columns=X_train.columns)
    X_act  = X_act.reindex(columns=X_train.columns)

    X_train_sm = sm.add_constant(X_train, has_constant="add")
    X_test_sm  = sm.add_constant(X_test,  has_constant="add")


    # -------------------------------
    # Compile AI Model and save summary
    # -------------------------------
    ols_res = sm.OLS(y_train, X_train_sm).fit()
    print(ols_res.summary())

    with open(SUMMARY_PATH, "w") as f:
        f.write(str(ols_res.summary()))

    print("saved:", SUMMARY_PATH)

    # -------------------------------
    # Store training/validation performance (learningBase)
    # -------------------------------

    # Predictions
    y_train_pred = ols_res.predict(X_train_sm)
    y_test_pred  = ols_res.predict(X_test_sm)

    # Indicators
    metrics = {
        "MSE": [mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)],
        "MAE": [mean_absolute_error(y_train, y_train_pred), mean_absolute_error(y_test, y_test_pred)],
        "R2":  [r2_score(y_train, y_train_pred),          r2_score(y_test, y_test_pred)]
    }

    df_metrics = pd.DataFrame(metrics, index=["TRAIN", "TEST"])
    print(df_metrics)

    # Speichern als TXT (optional, aber praktisch)
    with open(OLS_PERFORMANCE, "w") as f:
        f.write("=== OLS Performance Indicators ===\n")
        f.write(df_metrics.to_string())
        f.write("\n")

    print("✓ saved:", OLS_PERFORMANCE)


    # -------------------------------
    # Scatter Plot (Train orange, Test blue, ANN red line) + save as PDF
    # -------------------------------

    features = list(X_train.columns)
    n = len(features)

    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4*nrows))
    axes = np.array(axes).reshape(-1)

    base = X_train.mean().copy()

    for i, feat in enumerate(features):
        ax = axes[i]

        # Train/Test Scatter
        ax.scatter(X_train[feat], y_train, alpha=0.35, color="orange", s=12, label="train" if i == 0 else None)
        ax.scatter(X_test[feat],  y_test,  alpha=0.35, color="blue",   s=12, label="test"  if i == 0 else None)

        # x-grid für rote Linie
        x_min = min(X_train[feat].min(), X_test[feat].min())
        x_max = max(X_train[feat].max(), X_test[feat].max())
        x_grid = np.linspace(x_min, x_max, 200)

        # Grid bauen: alle Features = mean, nur feat variiert
        grid_df = pd.DataFrame([base.values] * len(x_grid), columns=features)
        grid_df[feat] = x_grid

        # OLS prediction (rote Linie)
        grid_sm = sm.add_constant(grid_df, has_constant="add")
        grid_sm = grid_sm.reindex(columns=ols_res.model.exog_names)

        y_line = ols_res.predict(grid_sm)
        ax.plot(x_grid, y_line, color="red", linewidth=2, label="OLS model" if i == 0 else None)

        ax.set_title(f"{feat} vs churn")
        ax.set_xlabel(feat)
        ax.set_ylabel("churn / OLS prediction")
        ax.grid(True)

    # Leere Subplots löschen
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.legend(loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(SCATTER_PATH, bbox_inches="tight")
    plt.close(fig)

    print("saved:", SCATTER_PATH)


    # -------------------------------
    # Diagnostic Plots 
    # -------------------------------

    diag = LinearRegDiagnostic(ols_res)

    vif_table, fig, ax = diag(plot_context="seaborn-v0_8-paper")

    fig.savefig(DIAG_PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    vif_table.to_csv(VIF_CSV_PATH, index=False)

    print("saved:", DIAG_PDF_PATH)
    print("saved:", VIF_CSV_PATH)

if __name__ == "__main__":
    main()