import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load Data
train_df = pd.read_csv("dataset.csv")
test_df = pd.read_csv("test.csv")

# Drop metadata
drop_cols = ["id", "attack_type", "load_level"]
X_train = train_df.drop(columns=drop_cols + ["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=drop_cols + ["label"])
y_test = test_df["label"]

# Train XGBoost
model = xgb.XGBClassifier(
    tree_method="hist", 
    predictor="cuda",
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_vtest, y_pred))

# Feature Importance Plot
xgb.plot_importance(model, max_num_features=10, title="Top 10 Feature Importances")
plt.tight_layout()
plt.show()
