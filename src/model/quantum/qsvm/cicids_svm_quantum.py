#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import time
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils import compute_class_weight

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


# In[46]:


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
    train_size=1000,
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


# In[55]:


# Reduce dimension to number of qubits for the quantum kernel
# Choose the number of qubits you can afford (8 is a reasonable starting point).
n_qubits = 2
pca = PCA(n_components=n_qubits, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_proc_scaled)
X_test_pca  = pca.transform(X_test_proc_scaled)

joblib.dump(pca, "pca_nsl_kdd.pkl")
print(f"PCA fitted -> reduced to {n_qubits} dims and saved (pca_nsl_kdd.pkl).")


# In[56]:


adhoc_feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement="linear")

sampler = Sampler()

fidelity = ComputeUncompute(sampler=sampler)

adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)


# In[ ]:


start=time.time()

qsvc = QSVC(quantum_kernel=adhoc_kernel)
qsvc.fit(X_train_pca, y_train)

end=time.time()
print(f"Training time: {end - start:.2f} seconds")


# In[ ]:


start = time.time()

qsvc_score = qsvc.score(X_test_pca, y_test)
y_pred = qsvc.predict(X_test_pca)
y_proba = qsvc.predict_proba(X_test_pca)[:, 1] if hasattr(qsvc, "predict_proba") else None

end = time.time()
print(f"Evaluation time: {end - start:.2f} seconds")


# In[ ]:


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


# In[ ]:




