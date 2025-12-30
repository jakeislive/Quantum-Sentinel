import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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
    train_size=500,
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

def build_vqc_circuit(n_qubits, feature_reps=1, ansatz_reps=2, entanglement="linear"):
    """
    Build a VQC: feature map (ZZFeatureMap) composed with a variational ansatz (RealAmplitudes).
    Returns (circuit, input_params_list, weight_params_list).
    """
    # Feature map: encodes classical features into the circuit
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=feature_reps, entanglement='linear')
    # Variational ansatz: trainable parameters
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=ansatz_reps, entanglement=entanglement)
    # Compose: feature_map followed by ansatz
    full_circ = feature_map.compose(ansatz)  # feature_map -> ansatz

    # Extract parameters: qiskit.Parameters are returned in nondeterministic ordering depending on circuit,
    # but EstimatorQNN expects explicit lists for input and weight parameters
    input_params = list(feature_map.parameters)   # parameters coming from feature map (named like x0,x1,...)
    weight_params = list(ansatz.parameters)       # trainable parameters from ansatz (theta...)

    return full_circ, input_params, weight_params

vqc_circuit, vqc_input_params, vqc_weight_params = build_vqc_circuit(
    n_qubits=n_qubits,
    feature_reps=1,
    ansatz_reps=3,
    entanglement="linear"
)

# Create the QNN: EstimatorQNN wraps the circuit and exposes forward/backprop for TorchConnector
vqc_qnn = EstimatorQNN(
    circuit=vqc_circuit,
    input_params=vqc_input_params,
    weight_params=vqc_weight_params,
    input_gradients=True
)

# Wrap as a PyTorch layer
quantum_layer = TorchConnector(vqc_qnn)

class VQCClassifier(nn.Module):
    def __init__(self, quantum_layer):
        super().__init__()
        self.quantum = quantum_layer
        self.fc = nn.Linear(1, 1)   # tiny classical head

    def forward(self, x):
        # x should be shape (batch, n_qubits) already (float32)
        out = self.quantum(x)      # returns shape (batch, 1) by default
        out = self.fc(out)         # (batch,1)
        return out.squeeze(1)      # -> (batch,)

# Instantiate model and move to device
device = torch.device("cpu")
model = VQCClassifier(quantum_layer)
model.to(device)

# ---- prepare data as torch tensors ----
# X_train_pca, X_test_pca are numpy arrays from your PCA stage. Ensure dtype float32.
X_train_q = torch.tensor(np.array(X_train_pca, dtype=np.float32), dtype=torch.float32, device=device)
y_train_q = torch.tensor(np.array(y_train, dtype=np.float32), dtype=torch.float32, device=device)

X_test_q  = torch.tensor(np.array(X_test_pca, dtype=np.float32), dtype=torch.float32, device=device)
y_test_q  = torch.tensor(np.array(y_test, dtype=np.float32), dtype=torch.float32, device=device)

batch_size = 32
train_dataset = TensorDataset(X_train_q, y_train_q)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("Torch tensor shapes: X_train:", X_train_q.shape, "X_test:", X_test_q.shape)

loss_fn = nn.BCEWithLogitsLoss()   # logits -> sigmoid internally
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

start=time.time()
epochs = 100
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)              # shape (batch,)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} — Loss: {epoch_loss:.6f}")

end=time.time()
print(f"Training time: {end - start:.2f} seconds")

# ---- evaluation on test set ----

start = time.time()
model.eval()

with torch.no_grad():
    # 1. Raw model outputs: logits
    logits_test = model(X_test_q)
    print("Logits:", logits_test)

    # 2. Convert logits → probabilities
    test_proba = torch.sigmoid(logits_test)
    print("Probabilities:", test_proba)

    # 3. Convert probabilities → class predictions (0 or 1)
    y_pred = (test_proba >= 0.5).float()
    print("Predictions:", y_pred)

correct = (y_pred == y_test_q).float().sum().item()
accuracy = correct / len(y_test_q)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# print("\nClassification report (test):\n", classification_report(y_test, y_pred, digits=4))
# print("Confusion matrix (test):\n", confusion_matrix(y_test, y_pred))

end = time.time()
print(f"Evaluation time: {end - start:.2f} seconds")

# ---- optionally save the PyTorch model weights ----
torch.save(model.state_dict(), "vqc_nsl_kdd_torch.pth")
print("Saved QVC classifier state to vqc_nsl_kdd_torch.pth")
