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


# In[2]:


TRAIN_PATH = "src/data/KDDTrain+.txt"
TEST_PATH  = "src/data/KDDTest+.txt"
RANDOM_STATE = 42


# In[3]:


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


# In[4]:


# ---- utility to load file ----
def load_nsl_kdd(path):
    # NSL-KDD files are comma-separated (no header). skipinitialspace helps if spaces after commas.
    return pd.read_csv(path, header=None, names=columns, sep=",", skipinitialspace=True)

# ---- load train & test ----
df_train = load_nsl_kdd(TRAIN_PATH)
df_test  = load_nsl_kdd(TEST_PATH)


# In[5]:


def show_distribution(data, title):
    print(f"\n=== {title} ===")
    dist = data["label"].value_counts().reset_index()
    dist.columns = ['class_label', 'samples']   
    dist['pct'] = (dist['samples'] / len(data) * 100).round(3)
    print(dist)


# In[6]:


show_distribution(df_train, "Train Distribution")
show_distribution(df_test, "Test Distribution")

df_train, _ = train_test_split(
    df_train,
    train_size=1000,
    stratify=df_train["label"],
    random_state=42
)
df_test = df_test.sample(n=100, random_state=42)


show_distribution(df_train, "Train Distribution")
show_distribution(df_test, "Test Distribution")


# In[7]:


# ---- label mapping: binary ----
df_train["label_binary"] = df_train["label"].apply(lambda s: 0 if s.strip() == "normal" else 1)
df_test["label_binary"]  = df_test["label"].apply(lambda s: 0 if s.strip() == "normal" else 1)

print("Train label counts:\n", df_train["label_binary"].value_counts(normalize=False))
print("Test label counts:\n", df_test["label_binary"].value_counts(normalize=False))


# In[8]:


# ---- feature/target split ----
y_train = df_train["label_binary"]
y_test  = df_test["label_binary"]

X_train = df_train.drop(columns=["label", "difficulty", "label_binary"])
X_test  = df_test.drop(columns=["label", "difficulty", "label_binary"])


# In[9]:


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


# In[10]:


# ---- drop original categorical columns and concat encoded columns ----
X_train_num = X_train.drop(columns=cat_cols).reset_index(drop=True)
X_test_num  = X_test.drop(columns=cat_cols).reset_index(drop=True)

X_train_proc = pd.concat([X_train_num, X_train_ohe.reset_index(drop=True)], axis=1)
X_test_proc  = pd.concat([X_test_num,  X_test_ohe.reset_index(drop=True)], axis=1)

print("Processed shapes: X_train:", X_train_proc.shape, "X_test:", X_test_proc.shape)


# In[11]:


# ---- ensure ordering / dtypes consistent ----
# (After OneHotEncoder with get_feature_names_out, columns match between train/test)
# If any differences remain (rare), reindex test to train columns:
X_test_proc = X_test_proc.reindex(columns=X_train_proc.columns, fill_value=0)

# ---- optional: compute class weights for binary problem ----
classes = np.unique(y_train)
class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)


# In[12]:


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
epochs = 200
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




