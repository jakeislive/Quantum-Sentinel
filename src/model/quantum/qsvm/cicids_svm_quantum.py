import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time

import joblib
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data.preprocessing import (
    CIC_IDS_LABEL_COL,
    load_cic_ids,
    preprocess_cic_ids,
    scale_features,
    show_distribution,
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
    train_size=1000,
    test_size=100,
    random_state=RANDOM_STATE
)

show_distribution(df_train, CIC_IDS_LABEL_COL, "Train Distribution")
show_distribution(df_test, CIC_IDS_LABEL_COL, "Test Distribution")

# Preprocess data
X_train_proc, X_test_proc, y_train, y_test, encoder = preprocess_cic_ids(df_train, df_test)

print("Train label counts:\n", y_train.value_counts(normalize=False))
print("Test label counts:\n", y_test.value_counts(normalize=False))
print("Processed shapes: X_train:", X_train_proc.shape, "X_test:", X_test_proc.shape)

# Scale features
X_train_proc_scaled, X_test_proc_scaled, scaler = scale_features(X_train_proc, X_test_proc)

# Reduce dimension to number of qubits for the quantum kernel
# Choose the number of qubits you can afford (8 is a reasonable starting point).
n_qubits = 2
pca = PCA(n_components=n_qubits, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_proc_scaled)
X_test_pca  = pca.transform(X_test_proc_scaled)

joblib.dump(pca, "pca_nsl_kdd.pkl")
print(f"PCA fitted -> reduced to {n_qubits} dims and saved (pca_nsl_kdd.pkl).")

adhoc_feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement="linear")

sampler = Sampler()

fidelity = ComputeUncompute(sampler=sampler)

adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)

start=time.time()

qsvc = QSVC(quantum_kernel=adhoc_kernel)
qsvc.fit(X_train_pca, y_train)

end=time.time()
print(f"Training time: {end - start:.2f} seconds")

start = time.time()

qsvc_score = qsvc.score(X_test_pca, y_test)
y_pred = qsvc.predict(X_test_pca)
y_proba = qsvc.predict_proba(X_test_pca)[:, 1] if hasattr(qsvc, "predict_proba") else None

end = time.time()
print(f"Evaluation time: {end - start:.2f} seconds")

# print your test score and the time
print(f"QSVC classification test score: {qsvc_score}")

print("\nTest accuracy:    ", accuracy_score(y_test, y_pred))
if y_proba is not None and len(np.unique(y_test)) == 2:
    try:
        auc = roc_auc_score(y_test, y_proba)
        print("Test ROC AUC:     ", round(auc, 4))
    except Exception:
        pass

print("\nClassification report (test):\n", classification_report(y_test, y_pred, digits=4))
print("Confusion matrix (test):\n", confusion_matrix(y_test, y_pred))
