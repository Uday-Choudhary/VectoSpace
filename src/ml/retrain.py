import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(SCRIPT_DIR, "..", "..", "datasets", "train_cleaned.csv")
TEST_PATH = os.path.join(SCRIPT_DIR, "..", "..", "datasets", "test_cleaned.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

#Load data
print(f"Loading training data from {TRAIN_PATH}")
train_df = pd.read_csv(TRAIN_PATH)
print(f"Loading test data from {TEST_PATH}")
test_df = pd.read_csv(TEST_PATH)

target_col = "final_grade"
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

#Train Random Forest
print("Training Random Forest Classifier (no n_jobs)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
labels = sorted(y_train.unique())
target_names = [f"Grade {i}" for i in labels]

print(f"\nRandom Forest Accuracy: {acc*100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
print(f"Saving model to {rf_path}")
with open(rf_path, "wb") as f:
    pickle.dump(rf_model, f)


print(f"\nmodel.feature_names_in_ = {list(rf_model.feature_names_in_)}")
print("\n✅ Done! Updated random_forest.pkl saved (no n_jobs).")
