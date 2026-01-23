import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np


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
MODELS_BASE_PATH = os.path.join(REPO_ROOT, "models", "customer_churn", "ann")
os.makedirs(MODELS_BASE_PATH, exist_ok=True)

AI_MODEL_PATH = os.path.join(MODELS_BASE_PATH, "currentAiSolution.keras")

# output: documentation
DOCS_BASE_PATH = os.path.join(REPO_ROOT, "documentation", "customer_churn", "ann")
os.makedirs(DOCS_BASE_PATH, exist_ok=True)

SUMMARY_PATH = os.path.join(DOCS_BASE_PATH, "ai_model_summary.txt")
AI_TRAINLOG_CSV = os.path.join(DOCS_BASE_PATH, "AI_training_log.txt")
AI_TRAIN_SUMMARY_TXT = os.path.join(DOCS_BASE_PATH, "AI_training_summary.txt")
TRAINING_TESTING_CURVE_PATH = os.path.join(DOCS_BASE_PATH, "ai_model_training_testing_curve.pdf")
SCATTER_PATH = os.path.join(DOCS_BASE_PATH, "AI_model_scatter_plot.pdf")


def main():

    # -------------------------------
    # Import data
    # -------------------------------
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df  = pd.read_csv(TEST_DATA_PATH)
    act_df   = pd.read_csv(ACTIVATION_DATA_PATH)

    # -------------------------------
    # AI Model
    # -------------------------------
    TARGET = "churn"

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]

    X_act = act_df.copy()
    if TARGET in X_act.columns:
        X_act = X_act.drop(columns=[TARGET])

    # -------------------------------
    # Compile AI Model
    # -------------------------------
    model = Sequential([
        Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    model.save(AI_MODEL_PATH)

    # -------------------------------
    # Store training/validation performance
    # -------------------------------
    hist_df = pd.DataFrame(history.history)
    hist_df["epoch"] = range(1, len(hist_df) + 1)
    hist_df.to_csv(AI_TRAINLOG_CSV, index=False)

    final_epoch = int(hist_df["epoch"].iloc[-1])
    final_loss = float(hist_df["loss"].iloc[-1])
    final_acc  = float(hist_df["accuracy"].iloc[-1])
    final_val_loss = float(hist_df["val_loss"].iloc[-1])
    final_val_acc  = float(hist_df["val_accuracy"].iloc[-1])

    with open(AI_TRAIN_SUMMARY_TXT, "w") as f:
        f.write("=== AI training summary ===\n")
        f.write(f"epochs_run: {final_epoch}\n")
        f.write(f"final_loss: {final_loss}\n")
        f.write(f"final_accuracy: {final_acc}\n")
        f.write(f"final_val_loss: {final_val_loss}\n")
        f.write(f"final_val_accuracy: {final_val_acc}\n")

    # -------------------------------
    # Training and testing curves
    # -------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(TRAINING_TESTING_CURVE_PATH, bbox_inches="tight")
    plt.close()

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

        y_line = model.predict(grid_df, verbose=0).flatten()
        ax.plot(x_grid, y_line, color="red", linewidth=2)

        ax.set_title(f"{feat} vs churn")
        ax.set_xlabel(feat)
        ax.set_ylabel("churn / predicted prob.")
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(SCATTER_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
