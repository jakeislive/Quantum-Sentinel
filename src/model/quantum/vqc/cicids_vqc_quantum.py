#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils import compute_class_weight

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


DATA_PATH = "src/data/CIC-IDS2017-merged.csv"   
RANDOM_STATE = 42


# In[3]:


label = "Label"
columns = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"]


# In[4]:


# ---- utility to load file ----
def load_nsl_kdd(path):
    # NSL-KDD files are comma-separated (no header). skipinitialspace helps if spaces after commas.
    return pd.read_csv(path, header=None, names=columns, sep=",", skipinitialspace=True)

def show_distribution(data, title):
    print(f"\n=== {title} ===")
    dist = data[label].value_counts().reset_index()
    dist.columns = ['class_label', 'samples']   
    dist['pct'] = (dist['samples'] / len(data) * 100).round(3)
    print(dist)

# ---- load train & test ----
df_data = load_nsl_kdd(DATA_PATH)
df_data.shape


# In[5]:


show_distribution(df_data, "Data Distribution")

df_train, df_test = train_test_split(
    df_data,
    train_size=500,
    test_size=100,
    random_state=RANDOM_STATE
)

show_distribution(df_train, "Train Distribution")
show_distribution(df_test, "Test Distribution")


# In[6]:


# ---- label mapping: binary ----
df_train["label_binary"] = df_train[label].apply(lambda s: 0 if s.strip() == "BENIGN" else 1)
df_test["label_binary"]  = df_test[label].apply(lambda s: 0 if s.strip() == "BENIGN" else 1)

print("Train label counts:\n", df_train["label_binary"].value_counts(normalize=False))
print("Test label counts:\n", df_test["label_binary"].value_counts(normalize=False))


# In[7]:


# ---- feature/target split ----
y_train = df_train["label_binary"]
y_test  = df_test["label_binary"]

X_train = df_train.drop(columns=[label, "label_binary", "Destination Port"])
X_test  = df_test.drop(columns=[label, "label_binary", "Destination Port"])


# In[8]:


X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[9]:


numeric_cols = X_train.select_dtypes(include=["number"]).columns

X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].mean())

categorical_cols = X_train.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
    X_test[col] = X_test[col].fillna(X_test[col].mode()[0])


# In[10]:


X_train_proc = X_train
X_test_proc = X_test


# In[11]:


classes = np.unique(y_train)
class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)


# In[12]:
try:
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
except Exception as e:
    print("Scaling failed:", e)
    exit


# In[13]:


# Reduce dimension to number of qubits for the quantum kernel
# Choose the number of qubits you can afford (8 is a reasonable starting point).
n_qubits = 2
pca = PCA(n_components=n_qubits, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_proc_scaled)
X_test_pca  = pca.transform(X_test_proc_scaled)

joblib.dump(pca, "pca_nsl_kdd.pkl")
print(f"PCA fitted -> reduced to {n_qubits} dims and saved (pca_nsl_kdd.pkl).")


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


# Instantiate model and move to device
device = torch.device("cpu")
model = VQCClassifier(quantum_layer)
model.to(device)


# In[18]:


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


# In[19]:


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


# In[21]:


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


# In[ ]:




