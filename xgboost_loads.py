import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load training and validation sets
train_df = pd.read_csv("dataset.csv")
test_df = pd.read_csv("test.csv")

# All load levels to train separate models for
load_levels = ["no-load", "avg-load", "full-load"]

for load in load_levels:
    print(f"\n=== Training for load level: {load} ===")

    # Filter for current load level
    train_filtered = train_df[train_df["load_level"] == load]
    test_filtered = test_df[test_df["load_level"] == load]

    # Drop metadata
    drop_cols = ["id", "attack_type", "load_level"]
    X_train = train_filtered.drop(columns=drop_cols + ["label"])
    y_train = train_filtered["label"]
    X_test = test_filtered.drop(columns=drop_cols + ["label"])
    y_test = test_filtered["label"]

    # Train binary classifier
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
