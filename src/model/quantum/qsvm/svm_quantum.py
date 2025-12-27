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


TRAIN_PATH = "src/data/KDDTrain+.txt"
TEST_PATH  = "src/data/KDDTest+.txt"
RANDOM_STATE = 42


# In[47]:


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


# In[48]:


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

# ---- load train & test ----
df_train = load_nsl_kdd(TRAIN_PATH)
df_test  = load_nsl_kdd(TEST_PATH)

show_distribution(df_train, "Train Distribution")
show_distribution(df_test, "Test Distribution")


# In[91]:


df_train, _ = train_test_split(
    df_train,
    train_size=1500,
    stratify=df_train["label"],
    random_state=42
)
df_test = df_test.sample(n=500, random_state=42)


show_distribution(df_train, "Train Distribution")
show_distribution(df_test, "Test Distribution")


# In[49]:


# ---- label mapping: binary ----
df_train["label_binary"] = df_train["label"].apply(lambda s: 0 if s.strip() == "normal" else 1)
df_test["label_binary"]  = df_test["label"].apply(lambda s: 0 if s.strip() == "normal" else 1)

print("Train label counts:\n", df_train["label_binary"].value_counts(normalize=False))
print("Test label counts:\n", df_test["label_binary"].value_counts(normalize=False))


# In[50]:


# ---- feature/target split ----
y_train = df_train["label_binary"]
y_test  = df_test["label_binary"]

X_train = df_train.drop(columns=["label", "difficulty", "label_binary"])
X_test  = df_test.drop(columns=["label", "difficulty", "label_binary"])


# In[51]:


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


# In[52]:


# ---- drop original categorical columns and concat encoded columns ----
X_train_num = X_train.drop(columns=cat_cols).reset_index(drop=True)
X_test_num  = X_test.drop(columns=cat_cols).reset_index(drop=True)

X_train_proc = pd.concat([X_train_num, X_train_ohe.reset_index(drop=True)], axis=1)
X_test_proc  = pd.concat([X_test_num,  X_test_ohe.reset_index(drop=True)], axis=1)

print("Processed shapes: X_train:", X_train_proc.shape, "X_test:", X_test_proc.shape)


# In[53]:


# ---- ensure ordering / dtypes consistent ----
X_test_proc = X_test_proc.reindex(columns=X_train_proc.columns, fill_value=0)

classes = np.unique(y_train)
class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)


# In[54]:


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


# In[55]:


# Reduce dimension to number of qubits for the quantum kernel
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




