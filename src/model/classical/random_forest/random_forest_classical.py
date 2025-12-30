import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from data.preprocessing import (
    compute_class_weights,
    load_nsl_kdd,
    preprocess_nsl_kdd,
)

TRAIN_PATH = "src/data/KDDTrain+.txt"
TEST_PATH = "src/data/KDDTest+.txt"
RANDOM_STATE = 42
N_ESTIMATORS = 200

# Load datasets
df_train = load_nsl_kdd(TRAIN_PATH)
df_test = load_nsl_kdd(TEST_PATH)

# Preprocess data
X_train_proc, X_test_proc, y_train, y_test, ohe = preprocess_nsl_kdd(df_train, df_test)

print("Train label counts:\n", y_train.value_counts(normalize=False))
print("Test label counts:\n", y_test.value_counts(normalize=False))
print("Processed shapes: X_train:", X_train_proc.shape, "X_test:", X_test_proc.shape)

# Compute class weights
class_weight_dict = compute_class_weights(y_train)
print("Class weights:", class_weight_dict)

start = time.time()
# ---- train Random Forest on TRAIN SET ----
clf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight=class_weight_dict
)
clf.fit(X_train_proc, y_train)
end = time.time()
print(f"Training time: {end - start:.2f} seconds")

start = time.time()
# ---- evaluate on TEST SET ----
y_pred = clf.predict(X_test_proc)
y_proba = clf.predict_proba(X_test_proc)[:, 1] if hasattr(clf, "predict_proba") else None
end = time.time()
print(f"Evaluation time: {end - start:.2f} seconds")

print("\nTest accuracy:    ", accuracy_score(y_test, y_pred))
if y_proba is not None and len(np.unique(y_test)) == 2:
    try:
        auc = roc_auc_score(y_test, y_proba)
        print("Test ROC AUC:     ", round(auc, 4))
    except Exception:
        pass

print("\nClassification report (test):\n", classification_report(y_test, y_pred, digits=4))
print("Confusion matrix (test):\n", confusion_matrix(y_test, y_pred))

# ---- save model & encoder ----
joblib.dump(clf, "rf_nsl_kdd_train_model.pkl")
joblib.dump(ohe, "ohe_nsl_kdd.pkl")
print("Saved: rf_nsl_kdd_train_model.pkl and ohe_nsl_kdd.pkl")
