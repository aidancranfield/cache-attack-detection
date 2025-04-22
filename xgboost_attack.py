import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Drop metadata
drop_cols = ["id", "load_level", "attack_type"]
FEATURE_COLS = train_df.drop(columns=drop_cols + ["label"]).columns.tolist()

# Define attack types to isolate
attack_types = ["fr", "pp", "ff", "er"]

# Add class column for consistent filtering
train_df["class"] = train_df.apply(lambda row: row["attack_type"] if row["label"] == 1 else "benign", axis=1)
test_df["class"] = test_df.apply(lambda row: row["attack_type"] if row["label"] == 1 else "benign", axis=1)


print(train_df["attack_type"].value_counts())

# Train and evaluate each model
for attack in attack_types:
    print(f"\n\nTraining model for: {attack} vs benign (balanced)")

    # Balance training set
    attack_train = train_df[train_df["class"] == attack].copy()
    benign_train = train_df[train_df["class"] == "benign"].sample(n=len(attack_train), random_state=42)
    train_bal = pd.concat([attack_train, benign_train])

    # Balance test set
    attack_test = test_df[test_df["class"] == attack].copy()
    benign_test = test_df[test_df["class"] == "benign"].sample(n=len(attack_test), random_state=42)
    test_bal = pd.concat([attack_test, benign_test])

    # Features and binary labels
    X_train = train_bal[FEATURE_COLS]
    y_train = (train_bal["class"] == attack).astype(int)

    X_test = test_bal[FEATURE_COLS]
    y_test = (test_bal["class"] == attack).astype(int)

    # Train model
    model = xgb.XGBClassifier(
        tree_method="hist", 
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=20,
        max_depth=5,
        learning_rate=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["benign", attack], digits=4) )
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Feature importance
    importance = model.get_booster().get_score(importance_type='gain')
    importance_sorted = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    
    print("\nFeature Importances (Gain):")
    for feature, score in importance_sorted.items():
        print(f"{feature}: {score:.4f}")

    MODEL_PATH = "model.json"
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Plot all features
    xgb.plot_importance(model, max_num_features=len(importance), title=f"All Feature Importances ({attack})", importance_type='gain')
    plt.tight_layout()
    plt.show()


