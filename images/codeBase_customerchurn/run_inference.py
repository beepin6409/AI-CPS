import os
import pandas as pd
import pickle
import tensorflow as tf

KB_PATH = "/tmp/knowledgeBase"
ACT_PATH = "/tmp/activationBase/activation_data.csv"

def main():
    # load activation data
    X = pd.read_csv(ACT_PATH)

    # drop target if present
    if "churn" in X.columns:
        X = X.drop(columns=["churn"])

    files = os.listdir(KB_PATH)

    if any(f.endswith(".keras") for f in files):
        model_path = os.path.join(KB_PATH, "currentSolution.keras")
        model = tf.keras.models.load_model(model_path)
        preds = model.predict(X).flatten()
        print("ANN predictions:", preds)

    elif any(f.endswith(".pkl") for f in files):
        model_path = os.path.join(KB_PATH, "currentSolution.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        X_sm = X.copy()
        X_sm.insert(0, "const", 1.0)
        preds = model.predict(X_sm)
        print("OLS predictions:", preds.values)

    else:
        raise RuntimeError("No supported model found in knowledgeBase")

if __name__ == "__main__":
    main()
