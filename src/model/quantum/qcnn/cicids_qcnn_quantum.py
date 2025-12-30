import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
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
n_qubits = 4
pca = PCA(n_components=n_qubits, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_proc_scaled)
X_test_pca  = pca.transform(X_test_proc_scaled)

print("PCA shapes: X_train:", X_train_pca.shape, "X_test:", X_test_pca.shape)

def build_qcnn_circuit(n_qubits, reps_feature=1, reps_variational=2):
    """
    Robust QCNN-like circuit builder.
    - input_params: ParameterVector of length n_qubits (one param per qubit feature)
    - weight_params: ParameterVector of length (n_qubits * 3 * reps_variational)
      (3 param-sets per qubit per variational layer: [ry, rz] before entangling + rz after)
    """
    # Feature parameters
    input_params = ParameterVector("x", length=n_qubits)

    # Total trainable parameters: 3 * n_qubits per variational layer
    params_per_layer = 3 * n_qubits
    total_weights = params_per_layer * reps_variational
    weight_params = ParameterVector("w", length=total_weights)

    qc = QuantumCircuit(n_qubits)

    # Feature map: RX rotations encoding classical features
    for i in range(n_qubits):
        for _ in range(reps_feature):
            qc.rx(input_params[i], i)

    # Use slices of weight_params for each layer to avoid index mistakes
    for layer in range(reps_variational):
        base = layer * params_per_layer
        # slice indices for this layer:
        # - first block: 2 params per qubit (ry, rz)  => length 2*n_qubits
        # - second block: 1 param per qubit (rz)      => length n_qubits
        # layout: [ry_0, rz_0, ry_1, rz_1, ..., ry_{q-1}, rz_{q-1}, rz_0', rz_1', ..., rz_{q-1}']
        # Build the first block (ry, rz per qubit)
        for q in range(n_qubits):
            i_ry = base + (2 * q)
            i_rz = base + (2 * q + 1)
            qc.ry(weight_params[i_ry], q)
            qc.rz(weight_params[i_rz], q)

        # entangling layer (nearest-neighbor)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

        # final rz per qubit for this layer (pooling/completion)
        rz_block_start = base + 2 * n_qubits
        for q in range(n_qubits):
            qc.rz(weight_params[rz_block_start + q], q)

        if n_qubits >= 2 and (layer % 2 == 0):
            pool_base = base + 2 * n_qubits
            for q in range(0, n_qubits - 1, 2):
                idx = pool_base + (q % n_qubits)
                qc.crx(weight_params[idx], q, q + 1)

    return qc, list(input_params), list(weight_params)

qcnn_circuit, qcnn_input_params, qcnn_weight_params = build_qcnn_circuit(n_qubits=n_qubits, reps_feature=1, reps_variational=3)

qnn = EstimatorQNN(
    circuit=qcnn_circuit,
    input_params=qcnn_input_params,
    weight_params=qcnn_weight_params,
    input_gradients=True,     # allow backprop
)

# ---- wrap qnn as a PyTorch module ----
quantum_layer = TorchConnector(qnn)

class QCNNClassifier(nn.Module):
    def __init__(self, quantum_layer):
        super().__init__()
        self.quantum = quantum_layer
        self.fc = nn.Linear(1, 1)   # tiny classical head

    def forward(self, x):
        out = self.quantum(x)          # shape: (batch, 1)
        out = self.fc(out)             # shape: (batch, 1)
        return out.squeeze(1)          # -> (batch,)

model = QCNNClassifier(quantum_layer)

# ---- training setup ----
device = torch.device("cpu")  # change to "cuda" only if you have a GPU and the primitives support it
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
epochs = 40
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
torch.save(model.state_dict(), "qcnn_nsl_kdd_torch.pth")
print("Saved QCNN classifier state to qcnn_nsl_kdd_torch.pth")
