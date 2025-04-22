import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv("train.csv")
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
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importance = model.get_booster().get_score(importance_type='gain')
importance_sorted = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    
print("\nFeature Importances (Gain):")
for feature, score in importance_sorted.items():
    print(f"{feature}: {score:.4f}")

# Plot all features
xgb.plot_importance(model, max_num_features=len(importance), title=f"All Feature Importances", importance_type='gain')
plt.tight_layout()
plt.show()
