#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, classification_report,
    roc_curve, auc, precision_recall_curve, silhouette_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings


# In[2]:


from google.colab import files
uploaded = files.upload()  # file selector


# In[3]:


RANDOM_STATE = 42
N_ESTIMATORS = 200


# In[15]:


colnames = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
        "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
        "label", "difficulty" # Added 'difficulty' to account for the extra column in the dataset
    ]


# In[16]:


TRAIN_PATH = "./KDDTrain+.txt"
TEST_PATH  = "./KDDTest+.txt"

df_train = pd.read_csv(TRAIN_PATH, names=colnames, sep=",", skipinitialspace=True, index_col=False)
df_test  = pd.read_csv(TEST_PATH, names=colnames, sep=",", skipinitialspace=True, index_col=False)

print(df_train.shape, df_test.shape)
df_train.head()


# In[17]:


df = pd.concat([df_train, df_test], ignore_index=True)


# In[44]:


y = (df["label"] != "normal").astype(int).values  # 0 normal, 1 attack

cat_cols = ["protocol_type", "service", "flag"]
num_cols = [c for c in df.columns if c not in cat_cols + ["label"]]

X_num = df[num_cols].astype(float).values
enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = enc.fit_transform(df[cat_cols])

X = np.hstack([X_num, X_cat])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[46]:


# Autoencoder helper
# -----------------------
def prepare_presumed_normal(X, y=None, contamination_frac=0.3, random_state=42):
    """
    If y provided: return X[y==0] (all labelled normal).
    If y is None: use IsolationForest to estimate anomaly scores, take lowest `contamination_frac` fraction as presumed-normal.
    """
    if y is not None:
        normal_mask = (y == 0)
        X_normal = X[normal_mask]
        return X_normal
    else:
        iso = IsolationForest(contamination=contamination_frac, random_state=random_state)
        iso.fit(X)
        scores = iso.decision_function(X)  # higher => more normal
        # select top (1 - contamination_frac) fraction as normal
        cutoff = np.percentile(scores, 100.0 * contamination_frac)
        X_normal = X[scores >= cutoff]
        return X_normal


# In[47]:


def build_and_train_autoencoder(X_train_normal, input_dim, latent_dim=8, epochs=50, batch_size=128, patience=5, verbose=0):
    try:
        import tensorflow as tf
        from tensorflow.keras import models, layers, callbacks
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False

    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow (Keras) not available. Install tensorflow to use autoencoder.")

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(max(latent_dim*4, 32), activation="relu")(inputs)
    x = layers.Dense(max(latent_dim*2, 16), activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="relu")(x)
    x = layers.Dense(max(latent_dim*2, 16), activation="relu")(latent)
    x = layers.Dense(max(latent_dim*4, 32), activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    es = callbacks.EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True, verbose=0)
    model.fit(X_train_normal, X_train_normal, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=verbose)
    return model


# In[48]:


# Detector wrappers (give anomaly scores and binary preds)
# -----------------------
def iso_forest_detector(X_train, X_eval, contamination=0.1, random_state=42):
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X_train)
    # anomaly score: negative of decision_function so higher = more anomalous
    scores = -iso.decision_function(X_eval)  # larger => more anomalous
    # binary predictions by thresholding at quantile contamination
    cutoff = np.quantile(scores, 1.0 - contamination)
    preds = (scores >= cutoff).astype(int)
    return scores, preds, iso


# In[49]:


def oneclass_svm_detector(X_train, X_eval, nu=0.05, kernel='rbf', gamma='scale'):
    oc = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    oc.fit(X_train)
    # decision_function: positive = inlier (higher => more normal)
    scores_raw = -oc.decision_function(X_eval)  # invert so higher = more anomalous
    # predicted labels
    preds = (scores_raw >= np.quantile(scores_raw, 1.0 - nu)).astype(int)
    return scores_raw, preds, oc


# In[50]:


def autoencoder_detector(model, X_eval, contamination=0.1):
    recon = model.predict(X_eval)
    errors = np.mean(np.square(X_eval - recon), axis=1)
    cutoff = np.quantile(errors, 1.0 - contamination)
    preds = (errors >= cutoff).astype(int)
    return errors, preds


# In[51]:


# Grid-search tuning
# -----------------------
def grid_search_detectors(X_train, y_train, X_val, y_val=None, supervised_tune=True, random_state=42):
    """
    Grid-search over detectors.
    If supervised_tune == True and y_val is provided: pick hyperparams maximizing F1 on val set.
    If supervised_tune == False: use unsupervised heuristics:
      - KMeans: silhouette_score on combined data
      - Others: pick params maximizing a separation metric on anomaly scores (difference between top and bottom quantiles)
    Returns best configurations and optionally trained models.
    """
    results = {}

    # parameter grids
    iso_grid = {"contamination": [0.01, 0.03, 0.05, 0.1, 0.15]}
    ocsvm_grid = {"nu": [0.01, 0.03, 0.05, 0.1], "gamma": ["scale", "auto"], "kernel": ["rbf", "poly"]}
    lof_grid = {"n_neighbors": [10, 20, 35], "contamination": [0.01, 0.03, 0.05, 0.1]}
    kmeans_grid = {"n_clusters": [2, 3, 4], "contamination": [0.01, 0.03, 0.05, 0.1]}

    # Helper scoring for unsupervised mode: separation metric of anomaly scores
    def score_separation(scores):
        # higher if top quantile mean >> bottom quantile mean
        top = np.mean(np.quantile(scores, [0.75, 0.9, 0.99]))
        bottom = np.mean(np.quantile(scores, [0.01, 0.1, 0.25]))
        return top - bottom

    # --- IsolationForest grid ---
    best_iso = None
    best_iso_score = -np.inf
    for c in iso_grid["contamination"]:
        scores, preds, model = iso_forest_detector(X_train, X_val, contamination=c, random_state=random_state)
        if supervised_tune and (y_val is not None):
            # evaluate via F1
            _, _, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
            metric = f1
        else:
            metric = score_separation(scores)
        if metric > best_iso_score:
            best_iso_score = metric
            best_iso = {"contamination": c, "scores": scores, "preds": preds, "model": model}
    results["isolation_forest"] = best_iso

    # --- OneClassSVM grid ---
    best_oc = None
    best_oc_score = -np.inf
    for nu in ocsvm_grid["nu"]:
        for gamma in ocsvm_grid["gamma"]:
            for kernel in ocsvm_grid["kernel"]:
                try:
                    scores, preds, model = oneclass_svm_detector(X_train, X_val, nu=nu, kernel=kernel, gamma=gamma)
                except Exception:
                    continue
                if supervised_tune and (y_val is not None):
                    _, _, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
                    metric = f1
                else:
                    metric = score_separation(scores)
                if metric > best_oc_score:
                    best_oc_score = metric
                    best_oc = {"nu": nu, "gamma": gamma, "kernel": kernel, "scores": scores, "preds": preds, "model": model}
    results["oneclass_svm"] = best_oc

    # --- LOF grid ---
    best_lof = None
    best_lof_score = -np.inf
    for nnb in lof_grid["n_neighbors"]:
        for c in lof_grid["contamination"]:
            try:
                scores, preds, model = lof_detector(X_train, X_val, n_neighbors=nnb, contamination=c)
            except Exception:
                continue
            if supervised_tune and (y_val is not None):
                _, _, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
                metric = f1
            else:
                metric = score_separation(scores)
            if metric > best_lof_score:
                best_lof_score = metric
                best_lof = {"n_neighbors": nnb, "contamination": c, "scores": scores, "preds": preds, "model": model}
    results["lof"] = best_lof

    # --- KMeans grid (use silhouette if unsupervised) ---
    best_km = None
    best_km_score = -np.inf
    # silhouette requires >1 cluster and at least 2 samples per cluster; we'll try on X_val or X_train
    silhouette_target = X_val if X_val is not None else X_train
    for ncl in kmeans_grid["n_clusters"]:
        for c in kmeans_grid["contamination"]:
            try:
                scores, preds, model = kmeans_distance_detector(X_train, silhouette_target, n_clusters=ncl, contamination=c)
            except Exception:
                continue
            if supervised_tune and (y_val is not None):
                # when val labels exist, evaluate directly on val
                # but here we computed preds on silhouette_target; recompute preds on X_val if needed
                _, preds_val, _ = kmeans_distance_detector(X_train, X_val, n_clusters=ncl, contamination=c)
                _, _, f1, _ = precision_recall_fscore_support(y_val, preds_val, average="binary", zero_division=0)
                metric = f1
            else:
                # unsupervised: silhouette score (higher is better)
                try:
                    sil = silhouette_score(silhouette_target, model.predict(silhouette_target))
                    metric = sil
                except Exception:
                    metric = -np.inf
            if metric > best_km_score:
                best_km_score = metric
                best_km = {"n_clusters": ncl, "contamination": c, "scores": scores, "preds": preds, "model": model}
    results["kmeans"] = best_km

    return results


# In[52]:


#  Evaluation & plotting
# -----------------------
def evaluate_and_plot(name, y_true, scores, preds, ax_pr=None, ax_roc=None):
    """
    Print evaluation and add to PR/ROC axes if provided.
    scores: continuous anomaly score (higher => more anomalous)
    preds: binary predictions (1 anomaly, 0 normal)
    """
    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(classification_report(y_true, preds, digits=4))

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        if ax_roc is not None:
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    except Exception:
        pass

    # Precision-Recall or bar summary
    if ax_pr is not None:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, scores)
        # approximate average precision by area under PR curve via trapezoid
        # but here we just plot a small bar with F1 for comparison
        ax_pr.bar(name, f1)



# In[53]:


# Runner that wires everything together
# ----------------------- Helpers -----------------------
def load_preprocess_nslkdd(sample_size=None, pca_components=None, random_state=42):
    # Reusing existing loaded data and preprocessing logic
    global df_train, df_test, colnames, cat_cols, num_cols, y, X_scaled, X, df

    # If PCA is requested, apply it. Otherwise, use X_scaled directly
    if pca_components is not None and 0 < pca_components < X_scaled.shape[1]:
        pca = PCA(n_components=pca_components, random_state=random_state)
        X_reduced = pca.fit_transform(X_scaled)
        X_final = X_reduced
    else:
        X_final = X_scaled

    # If sample_size is requested, apply sampling
    if sample_size is not None and sample_size < len(y):
        # Use train_test_split to get a stratified sample
        X_sample, _, y_sample, _ = train_test_split(
            X_final, y, train_size=sample_size, stratify=y, random_state=random_state
        )
        return X_sample, y_sample, df # df here refers to the original combined dataframe
    else:
        return X_final, y, df


def lof_detector(X_train, X_eval, n_neighbors=20, contamination=0.1, novelty=True):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=novelty)
    if novelty:
        lof.fit(X_train)
        # anomaly score: negative of decision_function so higher = more anomalous
        scores = -lof.decision_function(X_eval)
    else: # if novelty=False, fit and predict on the same data. Usually used for outlier detection on a single dataset.
        lof.fit(X_eval) # For outlier detection, fit on the data being evaluated
        scores = -lof.negative_outlier_factor_ # LOF score is negative, so negate for consistency (higher=more anomalous)

    # binary predictions by thresholding at quantile contamination
    cutoff = np.quantile(scores, 1.0 - contamination)
    preds = (scores >= cutoff).astype(int)
    return scores, preds, lof

def kmeans_distance_detector(X_train, X_eval, n_clusters=2, contamination=0.1, random_state=42):
    # Fit KMeans on the training data
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans.fit(X_train)

    # For each point in X_eval, calculate its distance to the closest centroid
    distances = kmeans.transform(X_eval)
    # The anomaly score is the minimum distance to any centroid (higher = more anomalous)
    scores = np.min(distances, axis=1)

    # Determine anomaly threshold based on contamination
    cutoff = np.quantile(scores, 1.0 - contamination)
    preds = (scores >= cutoff).astype(int)

    return scores, preds, kmeans

# ----------------------- Main Experiment Runner -----------------------
def run_experiment(sample_size=3000, pca_components=30, supervised_tune=True, test_size=0.2, random_state=42):
    print("Loading & preprocessing ...")

    # TF_AVAILABLE should be defined from the autoencoder cell to indicate if TF is installed.
    global TF_AVAILABLE
    try:
        import tensorflow as tf
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False

    # Call the new helper function
    X, y, current_df = load_preprocess_nslkdd(sample_size=sample_size, pca_components=pca_components, random_state=random_state)
    print("Data shape:", X.shape, "Anomaly ratio:", np.mean(y))
    print()

    # Split into train / val / test (we use labels only for tuning when supervised_tune True)
    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_hold, y_hold, test_size=0.5, random_state=random_state, stratify=y_hold)

    # Grid search best hyperparams (supervised or unsupervised mode)
    print("Running grid search for detectors (supervised_tune=%s) ..." % supervised_tune)
    grid_results = grid_search_detectors(X_train, y_train, X_val, y_val if supervised_tune else None, supervised_tune=supervised_tune, random_state=random_state)
    print("Grid search done.")
    print()

    # Prepare plotting axes
    fig_roc, ax_roc = plt.subplots(figsize=(6,6))
    ax_roc.plot([0,1],[0,1],'--', linewidth=0.5, label='random')
    ax_pr_fig, ax_pr = plt.subplots(figsize=(8,4))

    # Plot bar chart of F1 scores for quick comparison; ax_pr is bar-plot axis

    # Evaluate each best detector on the test set
    # IsolationForest
    iso_conf = grid_results["isolation_forest"]
    scores_iso, preds_iso, iso_model = iso_forest_detector(X_train, X_test, contamination=iso_conf["contamination"], random_state=random_state)
    evaluate_and_plot("IsolationForest", y_test, scores_iso, preds_iso, ax_pr=ax_pr, ax_roc=ax_roc)

    # OneClassSVM
    oc_conf = grid_results["oneclass_svm"]
    scores_oc, preds_oc, oc_model = oneclass_svm_detector(X_train, X_test, nu=oc_conf["nu"], kernel=oc_conf["kernel"], gamma=oc_conf["gamma"])
    evaluate_and_plot("OneClassSVM", y_test, scores_oc, preds_oc, ax_pr=ax_pr, ax_roc=ax_roc)

    # LOF
    lof_conf = grid_results["lof"]
    # For LOF we usually fit on evaluation set or use novelty=True for novel samples, but grid used fit on silhouette_target.
    # We'll fit LOF on combined train+val for better coverage
    X_lof_train = np.vstack([X_train, X_val])
    # Ensure novelty=True for consistent scoring across train and eval, especially for test set evaluation
    scores_lof, preds_lof, lof_model = lof_detector(X_lof_train, X_test, n_neighbors=lof_conf["n_neighbors"], contamination=lof_conf["contamination"], novelty=True)
    evaluate_and_plot("LocalOutlierFactor", y_test, scores_lof, preds_lof, ax_pr=ax_pr, ax_roc=ax_roc)

    # KMeans
    km_conf = grid_results["kmeans"]
    scores_km, preds_km, km_model = kmeans_distance_detector(X_train, X_test, n_clusters=km_conf["n_clusters"], contamination=km_conf["contamination"], random_state=random_state)
    evaluate_and_plot("KMeans-distance", y_test, scores_km, preds_km, ax_pr=ax_pr, ax_roc=ax_roc)

    # Autoencoder: train on presumed-normal subset
    if TF_AVAILABLE:
        print("Preparing presumed-normal subset for autoencoder training...")
        if supervised_tune:
            X_train_normal_ae = prepare_presumed_normal(X_train, y=y_train, contamination_frac=0.3, random_state=random_state)
        else:
            # If unsupervised, we use contamination_frac to estimate normal data from X_train
            X_train_normal_ae = prepare_presumed_normal(X_train, y=None, contamination_frac=0.3, random_state=random_state)

        print("Autoencoder training on {} presumed-normal samples (shape {}).".format(X_train_normal_ae.shape[0], X_train_normal_ae.shape))
        ae_model = build_and_train_autoencoder(X_train_normal_ae, input_dim=X.shape[1], latent_dim=min(32, X.shape[1]//2), epochs=100, batch_size=256, patience=10, verbose=0)
        scores_ae, preds_ae = autoencoder_detector(ae_model, X_test, contamination=0.1) # Contamination for AE is typically fixed or tuned separately
        evaluate_and_plot("Autoencoder", y_test, scores_ae, preds_ae, ax_pr=ax_pr, ax_roc=ax_roc)
    else:
        print("TensorFlow not available; skipping autoencoder.")

    # Finalize plots
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()

    ax_pr.set_ylabel("F1 score")
    ax_pr.set_title("F1 Score Comparison (bar)")
    ax_pr.set_xticklabels(ax_pr.get_xticklabels(), rotation=45, ha='right')
    ax_pr.figure.tight_layout()

    plt.show()

    # Return objects for further inspection
    return {
        "grid_results": grid_results,
        "models": {
            "iso": iso_model,
            "ocsvm": oc_model,
            "lof": lof_model if 'lof_model' in locals() else None,
            "kmeans": km_model if 'km_model' in locals() else None,
            "autoencoder": ae_model if TF_AVAILABLE else None
        }
    }

# -----------------------
# If run as script, run default experiment
# -----------------------
if __name__ == "__main__":
    # adjust sample_size up/down depending on your machine; 3000 is moderate
    RES = run_experiment(sample_size=3000, pca_components=30, supervised_tune=True, test_size=0.2, random_state=42)

