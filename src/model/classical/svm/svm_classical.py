#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils import compute_class_weight
import joblib
import numpy as np
import time

# In[2]:


TRAIN_PATH = "src/data/KDDTrain+.txt"   
TEST_PATH  = "src/data/KDDTest+.txt"    
RANDOM_STATE = 42
N_ESTIMATORS = 200



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


# ---- label mapping: binary ----
df_train["label_binary"] = df_train["label"].apply(lambda s: 0 if s.strip() == "normal" else 1)
df_test["label_binary"]  = df_test["label"].apply(lambda s: 0 if s.strip() == "normal" else 1)

print("Train label counts:\n", df_train["label_binary"].value_counts(normalize=False))
print("Test label counts:\n", df_test["label_binary"].value_counts(normalize=False))


# In[6]:


# ---- feature/target split ----
y_train = df_train["label_binary"]
y_test  = df_test["label_binary"]

X_train = df_train.drop(columns=["label", "difficulty", "label_binary"])
X_test  = df_test.drop(columns=["label", "difficulty", "label_binary"])


# In[7]:


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



# In[8]:


# ---- drop original categorical columns and concat encoded columns ----
X_train_num = X_train.drop(columns=cat_cols).reset_index(drop=True)
X_test_num  = X_test.drop(columns=cat_cols).reset_index(drop=True)

X_train_proc = pd.concat([X_train_num, X_train_ohe.reset_index(drop=True)], axis=1)
X_test_proc  = pd.concat([X_test_num,  X_test_ohe.reset_index(drop=True)], axis=1)

print("Processed shapes: X_train:", X_train_proc.shape, "X_test:", X_test_proc.shape)


# ---- ensure ordering / dtypes consistent ----
X_test_proc = X_test_proc.reindex(columns=X_train_proc.columns, fill_value=0)

classes = np.unique(y_train)
class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)


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

joblib.dump(scaler, "standard_scaler_nsl_kdd.pkl")
print("Saved scaler: standard_scaler_nsl_kdd.pkl")
# =================================


# In[ ]:


start = time.time()
clf = SVC(
    kernel='rbf',              # or 'linear', 'poly', 'sigmoid' depending on your case
    C=1.0,                     # regularization strength
    class_weight=class_weight_dict,  # handle class imbalance if needed
    probability=False,           # enable probability estimates (optional, slower)
    random_state=RANDOM_STATE
)
# NOTE: using the scaled features for training
clf.fit(X_train_proc_scaled, y_train)
end = time.time()
print(f"Training time: {end - start:.2f} seconds")


# In[ ]:

start = time.time()
# ---- evaluate on TEST SET ----
y_pred = clf.predict(X_test_proc_scaled)
y_proba = clf.predict_proba(X_test_proc_scaled)[:, 1] if hasattr(clf, "predict_proba") else None
end = time.time()
print(f"Evaluation time: {end - start:.2f} seconds")

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


# ---- save model & encoder ----
joblib.dump(clf, "svm_nsl_kdd_train_model.pkl")
joblib.dump(ohe, "ohe_nsl_kdd.pkl")
print("Saved: svm_nsl_kdd_train_model.pkl, ohe_nsl_kdd.pkl, and standard_scaler_nsl_kdd.pkl")
