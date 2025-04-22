import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Load Data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Settings
drop_cols = ["id", "attack_type", "load_level"]
load_levels = ["no-load", "avg-load", "full-load"]

# Model Params
model_params = {
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 20,
    "max_depth": 5,
    "learning_rate": 0.2,
    "random_state": 42
}

# Train & Evaluate per Load Level
for load in load_levels:
    print(f"\nLoad Level: {load}")

    # Subset by load level
    train_subset = train_df[train_df["load_level"] == load]
    test_subset = test_df[test_df["load_level"] == load]

    # Drop metadata
    X_train = train_subset.drop(columns=drop_cols + ["label"])
    y_train = train_subset["label"]
    X_test = test_subset.drop(columns=drop_cols + ["label"])
    y_test = test_subset["label"]

    # Train model
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1: {f1_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Feature importance
    importance = model.get_booster().get_score(importance_type='gain')
    importance_sorted = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    
    print("\nFeature Importances (Gain):")
    for feature, score in importance_sorted.items():
        print(f"{feature}: {score:.4f}")

    # Plot all features
    xgb.plot_importance(model, max_num_features=len(importance), title=f"All Feature Importances ({load})", importance_type='gain')
    plt.tight_layout()
    plt.show()
