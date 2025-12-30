import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data.preprocessing import (
    compute_class_weights,
    load_cic_ids,
    preprocess_cic_ids,
    show_distribution,
    CIC_IDS_LABEL_COL,
)

DATA_PATH = "src/data/CIC-IDS2017-merged.csv"
RANDOM_STATE = 42

# Load dataset
df_data = load_cic_ids(DATA_PATH)
print(f"Data shape: {df_data.shape}")

show_distribution(df_data, CIC_IDS_LABEL_COL, "Data Distribution")

# Split into train and test
df_train, df_test = train_test_split(
    df_data,
    train_size=100000,
    test_size=10000,
    random_state=RANDOM_STATE
)

show_distribution(df_train, CIC_IDS_LABEL_COL, "Train Distribution")
show_distribution(df_test, CIC_IDS_LABEL_COL, "Test Distribution")

# Preprocess data
X_train_proc, X_test_proc, y_train, y_test, encoder = preprocess_cic_ids(df_train, df_test)

print("Train label counts:\n", y_train.value_counts(normalize=False))
print("Test label counts:\n", y_test.value_counts(normalize=False))
print("Processed shapes: X_train:", X_train_proc.shape, "X_test:", X_test_proc.shape)

# Compute class weights
class_weight_dict = compute_class_weights(y_train)
print("Class weights:", class_weight_dict)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
clf.fit(X_train_proc, y_train)

# Evaluate on test set
y_pred = clf.predict(X_test_proc)
y_proba = clf.predict_proba(X_test_proc)[:, 1] if hasattr(clf, "predict_proba") else None

print("\nTest accuracy:", accuracy_score(y_test, y_pred))
if y_proba is not None and len(np.unique(y_test)) == 2:
    try:
        auc = roc_auc_score(y_test, y_proba)
        print("Test ROC AUC:", round(auc, 4))
    except Exception:
        pass

print("\nClassification report (test):\n", classification_report(y_test, y_pred, digits=4))
print("Confusion matrix (test):\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(clf, "dt_cic_ids_model.pkl")
if encoder is not None:
    joblib.dump(encoder, "encoder_cic_ids.pkl")
    print("Saved: dt_cic_ids_model.pkl and encoder_cic_ids.pkl")
else:
    print("Saved: dt_cic_ids_model.pkl")
print("Confusion matrix (test):\n", confusion_matrix(y_test, y_pred))

# ---- save model & encoder ----
joblib.dump(clf, "rf_nsl_kdd_train_model.pkl")
print("Saved: rf_nsl_kdd_train_model.pkl")
