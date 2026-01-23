"""
ols_model.py
------------
Purpose:
    Train an Ordinary Least Squares (OLS) regression model using Statsmodels
    for customer churn prediction.
"""

import os
import pickle   # <<< ADDED
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from diagnostic_plots import LinearRegDiagnostic
matplotlib.use("Agg")


PROCESSED_SUBFOLDER = "processed"

# -------------------------------
# Paths
# -------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# input data
PROCESSED_PATH = os.path.join(REPO_ROOT, "data", "customer-churn-dataset", PROCESSED_SUBFOLDER)

TRAIN_DATA_PATH = os.path.join(PROCESSED_PATH, "training_data.csv")
TEST_DATA_PATH  = os.path.join(PROCESSED_PATH, "test_data.csv")
ACTIVATION_DATA_PATH = os.path.join(PROCESSED_PATH, "activation_data.csv")

# output: models
MODELS_BASE_PATH = os.path.join(REPO_ROOT, "models", "customer_churn", "ols")
os.makedirs(MODELS_BASE_PATH, exist_ok=True)

OLS_MODEL_PATH = os.path.join(MODELS_BASE_PATH, "currentOlsSolution.pkl")  # <<< ADDED

# output: documentation
DOCS_BASE_PATH = os.path.join(REPO_ROOT, "documentation", "customer_churn", "ols")
os.makedirs(DOCS_BASE_PATH, exist_ok=True)

SUMMARY_PATH = os.path.join(DOCS_BASE_PATH, "ols_model_summary.txt")
OLS_PERFORMANCE = os.path.join(DOCS_BASE_PATH, "ols_performance.txt")
SCATTER_PATH = os.path.join(DOCS_BASE_PATH, "ols_scatter_plots.pdf")
DIAG_PDF_PATH = os.path.join(DOCS_BASE_PATH, "ols_diagnostic_plots.pdf")
VIF_CSV_PATH  = os.path.join(DOCS_BASE_PATH, "ols_vif_table.csv")


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
    # Fit OLS Model
    # -------------------------------
    ols_res = sm.OLS(y_train, X_train_sm).fit()
    print(ols_res.summary())

    # save summary
    with open(SUMMARY_PATH, "w") as f:
        f.write(str(ols_res.summary()))

    # -------------------------------
    # SAVE OLS MODEL (NEW, REQUIRED)
    # -------------------------------
    with open(OLS_MODEL_PATH, "wb") as f:
        pickle.dump(ols_res, f)

    print("saved:", OLS_MODEL_PATH)

    # -------------------------------
    # Store performance
    # -------------------------------
    y_train_pred = ols_res.predict(X_train_sm)
    y_test_pred  = ols_res.predict(X_test_sm)

    metrics = {
        "MSE": [mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)],
        "MAE": [mean_absolute_error(y_train, y_train_pred), mean_absolute_error(y_test, y_test_pred)],
        "R2":  [r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)]
    }

    df_metrics = pd.DataFrame(metrics, index=["TRAIN", "TEST"])

    with open(OLS_PERFORMANCE, "w") as f:
        f.write("=== OLS Performance Indicators ===\n")
        f.write(df_metrics.to_string())
        f.write("\n")

    # -------------------------------
    # Scatter Plot
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

        ax.scatter(X_train[feat], y_train, alpha=0.35, color="orange", s=12)
        ax.scatter(X_test[feat],  y_test,  alpha=0.35, color="blue",   s=12)

        x_min = min(X_train[feat].min(), X_test[feat].min())
        x_max = max(X_train[feat].max(), X_test[feat].max())
        x_grid = np.linspace(x_min, x_max, 200)

        grid_df = pd.DataFrame([base.values] * len(x_grid), columns=features)
        grid_df[feat] = x_grid

        grid_sm = sm.add_constant(grid_df, has_constant="add")
        grid_sm = grid_sm.reindex(columns=ols_res.model.exog_names)

        y_line = ols_res.predict(grid_sm)
        ax.plot(x_grid, y_line, color="red", linewidth=2)

        ax.set_title(f"{feat} vs churn")
        ax.set_xlabel(feat)
        ax.set_ylabel("churn / OLS prediction")
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(SCATTER_PATH, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------
    # Diagnostic Plots
    # -------------------------------
    diag = LinearRegDiagnostic(ols_res)
    vif_table, fig, ax = diag(plot_context="seaborn-v0_8-paper")

    fig.savefig(DIAG_PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    vif_table.to_csv(VIF_CSV_PATH, index=False)


if __name__ == "__main__":
    main()
