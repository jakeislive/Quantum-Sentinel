#!/usr/bin/env python
# coding: utf-8

# In[48]:


# Classical K-means Anomaly Detection - NSL-KDD
# ========================

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[49]:


TRAIN_PATH = "../../../data/KDDTrain+.txt"   
TEST_PATH  = "../../../data/KDDTest+.txt"    
RANDOM_STATE = 42
N_ESTIMATORS = 200


# In[51]:


columns = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
        "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
        "label", "difficulty"
    ]


# In[52]:


df_train = pd.read_csv(TRAIN_PATH, header=None, names=colnames,sep=",", skipinitialspace=True)
df_test = pd.read_csv(TEST_PATH, header=None, names=colnames,sep=",", skipinitialspace=True)
df = pd.concat([df_train, df_test], ignore_index=True)

print(df_train.shape, df_test.shape)
df_train.head()


# In[53]:


# Convert labels to normal/anomaly
df["binary_label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

print("Dataset Loaded:", df.shape)


# In[54]:


# ----------------------------------------
# Preprocess Data
# ----------------------------------------

cat_cols = ["protocol_type", "service", "flag"]
num_cols = [c for c in df.columns if c not in cat_cols + ["label", "binary_label"]]


# In[57]:


# Encode categorical
enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = enc.fit_transform(df[cat_cols])


# In[58]:


# Numeric processing
X_num = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values


# In[59]:


# Scale numeric
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Combine features
X = np.hstack([X_num_scaled, X_cat])
y = df["binary_label"].values

print("Feature matrix:", X.shape)


# In[60]:


# ----------------------------------------
# Split for tuning threshold (unsupervised training)
# ----------------------------------------

# Only "normal" used for training K-means!
X_normal = X[y == 0]

# Split off labeled validation set for threshold selection
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# In[61]:


# ----------------------------------------
# Train K-means on normal-only data
# ----------------------------------------

k = 1  # single centroid for anomaly scoring
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_normal)

centroid = kmeans.cluster_centers_[0]


# In[62]:


# ----------------------------------------
# Compute anomaly scores (distance from centroid)
# ----------------------------------------

def anomaly_score(X):
    return np.linalg.norm(X - centroid, axis=1)

scores_val = anomaly_score(X_val)


# In[63]:


# ----------------------------------------
# Choose threshold using validation set
# ----------------------------------------

# Try percentiles of scores
percentiles = np.linspace(80, 99.5, 40)
best_f1 = -1
best_threshold = None

for p in percentiles:
    thresh = np.percentile(scores_val, p)
    preds = (scores_val > thresh).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print("Best threshold:", best_threshold)
print("Best validation F1:", best_f1)


# In[66]:


# ----------------------------------------
# Final evaluation on full test set
# ----------------------------------------

scores = anomaly_score(X)
preds = (scores > best_threshold).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary")
auc = roc_auc_score(y, scores)
accuracy = (preds == y).mean()  # NEW

print("\n=== Final Evaluation ===")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", auc)
print("Accuracy:", accuracy)  # NEW


# In[67]:


# ----------------------------------------
# Plot ROC Curve
# ----------------------------------------

fpr, tpr, _ = roc_curve(y, scores)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("K-means Anomaly Detection - ROC Curve")
plt.grid()
plt.show()


# In[ ]:




