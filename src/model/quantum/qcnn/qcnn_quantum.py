#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
import numpy as np
import time
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit.circuit import Parameter, ParameterVector
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


# In[87]:


TRAIN_PATH = "src/data/KDDTrain+.txt"
TEST_PATH  = "src/data/KDDTest+.txt"
RANDOM_STATE = 42


# In[88]:


columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]


# In[89]:


# ---- utility to load file ----
def load_nsl_kdd(path):
    # NSL-KDD files are comma-separated (no header). skipinitialspace helps if spaces after commas.
    return pd.read_csv(path, header=None, names=columns, sep=",", skipinitialspace=True)

def show_distribution(data, title):
    print(f"\n=== {title} ===")
    dist = data["label"].value_counts().reset_index()
    dist.columns = ['class_label', 'samples']   
    dist['pct'] = (dist['samples'] / len(data) * 100).round(3)
    print(dist)


# In[90]:


# ---- load train & test ----
df_train = load_nsl_kdd(TRAIN_PATH)
df_test  = load_nsl_kdd(TEST_PATH)

show_distribution(df_train, "Train Distribution")
show_distribution(df_test, "Test Distribution")


# In[91]:


df_train, _ = train_test_split(
    df_train,
    train_size=500,
    stratify=df_train["label"],
    random_state=42
)
df_test = df_test.sample(n=100, random_state=42)


show_distribution(df_train, "Train Distribution")
show_distribution(df_test, "Test Distribution")


# In[92]:


# ---- label mapping: binary ----
df_train["label_binary"] = df_train["label"].apply(lambda s: 0 if s.strip() == "normal" else 1)
df_test["label_binary"]  = df_test["label"].apply(lambda s: 0 if s.strip() == "normal" else 1)

print("Train label counts:\n", df_train["label_binary"].value_counts(normalize=False))
print("Test label counts:\n", df_test["label_binary"].value_counts(normalize=False))


# In[93]:


# ---- feature/target split ----
y_train = df_train["label_binary"]
y_test  = df_test["label_binary"]

X_train = df_train.drop(columns=["label", "difficulty", "label_binary"])
X_test  = df_test.drop(columns=["label", "difficulty", "label_binary"])


# In[94]:


# ---- categorical columns ----
cat_cols = ["protocol_type", "service", "flag"]

# ---- fit OneHotEncoder on train and transform both (handle unknown categories in test) ----
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe.fit(X_train[cat_cols])

X_train_ohe = pd.DataFrame(
    ohe.transform(X_train[cat_cols]),
    columns=ohe.get_feature_names_out(cat_cols),
    index=X_train.index
)
X_test_ohe = pd.DataFrame(
    ohe.transform(X_test[cat_cols]),
    columns=ohe.get_feature_names_out(cat_cols),
    index=X_test.index
)


# In[95]:


# ---- drop original categorical columns and concat encoded columns ----
X_train_num = X_train.drop(columns=cat_cols).reset_index(drop=True)
X_test_num  = X_test.drop(columns=cat_cols).reset_index(drop=True)

X_train_proc = pd.concat([X_train_num, X_train_ohe.reset_index(drop=True)], axis=1)
X_test_proc  = pd.concat([X_test_num,  X_test_ohe.reset_index(drop=True)], axis=1)

print("Processed shapes: X_train:", X_train_proc.shape, "X_test:", X_test_proc.shape)


# In[96]:


# ---- ensure ordering / dtypes consistent ----
# (After OneHotEncoder with get_feature_names_out, columns match between train/test)
# If any differences remain (rare), reindex test to train columns:
X_test_proc = X_test_proc.reindex(columns=X_train_proc.columns, fill_value=0)

# ---- optional: compute class weights for binary problem ----
classes = np.unique(y_train)
class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)


# In[97]:


scaler = StandardScaler()
# Fit on train, transform both
X_train_proc_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_proc),
    columns=X_train_proc.columns,
    index=X_train_proc.index
)
X_test_proc_scaled = pd.DataFrame(
    scaler.transform(X_test_proc),
    columns=X_test_proc.columns,
    index=X_test_proc.index
)


# In[98]:


# Reduce dimension to number of qubits for the quantum kernel
n_qubits = 4
pca = PCA(n_components=n_qubits, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_proc_scaled)
X_test_pca  = pca.transform(X_test_proc_scaled)

print("PCA shapes: X_train:", X_train_pca.shape, "X_test:", X_test_pca.shape)

# In[99]:


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


# In[100]:


qcnn_circuit, qcnn_input_params, qcnn_weight_params = build_qcnn_circuit(n_qubits=n_qubits, reps_feature=1, reps_variational=3)


# In[101]:


qnn = EstimatorQNN(
    circuit=qcnn_circuit,
    input_params=qcnn_input_params,
    weight_params=qcnn_weight_params,
    input_gradients=True,     # allow backprop
)


# ---- wrap qnn as a PyTorch module ----
quantum_layer = TorchConnector(qnn)


# In[102]:


class QCNNClassifier(nn.Module):
    def __init__(self, quantum_layer):
        super().__init__()
        self.quantum = quantum_layer
        self.fc = nn.Linear(1, 1)   # tiny classical head

    def forward(self, x):
        out = self.quantum(x)          # shape: (batch, 1)
        out = self.fc(out)             # shape: (batch, 1)
        return out.squeeze(1)          # -> (batch,)


# In[103]:


model = QCNNClassifier(quantum_layer)

# In[105]:


# ---- training setup ----
device = torch.device("cpu")  # change to "cuda" only if you have a GPU and the primitives support it
model.to(device)


# In[104]:


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





# In[ ]:


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


# In[ ]:


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
