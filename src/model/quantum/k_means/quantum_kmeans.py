# Quantum K-means Anomaly Detection - NSL-KDD
# 1) Import classical & qiskit library & Quantum Library

import math

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2, PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

TRAIN_PATH = "../../../data/KDDTrain+.txt"
TEST_PATH  = "../../../data/KDDTest+.txt"
RANDOM_STATE = 42

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

df_train = pd.read_csv(TRAIN_PATH, header=None, names=columns,sep=",", skipinitialspace=True)
df_test = pd.read_csv(TEST_PATH, header=None, names=columns,sep=",", skipinitialspace=True)
df = pd.concat([df_train, df_test], ignore_index=True)

print(df_train.shape, df_test.shape)
df_train.head()

# ---------------------------
# 2) Load & preprocess NSL-KDD
# ---------------------------
def load_nsl_kdd(data_folder="data", train_file="KDDTrain+.txt", test_file="KDDTest+.txt", sample_size=None):
    """
    Returns X (features), y (0=normal,1=anomaly) combining train+test or a subset.
    NSL-KDD original columns: 41 features + label + difficulty. We'll keep the 41 features + label.
    """
    train_path = os.path.join(data_folder, train_file)
    test_path = os.path.join(data_folder, test_file)
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        ok = download_nsl_kdd(data_folder)
        if not ok:
            raise FileNotFoundError("NSL-KDD files not found and auto-download failed. Please put KDDTrain+.txt and KDDTest+.txt into the data folder.")

    # Keep only relevant columns (feature columns 0..40)
    X_raw = df.drop(columns=["label"])
    y_raw = df["label"].copy()
    # Convert label to binary: normal -> 0, anything else (attack) -> 1
    y = (y_raw != "normal").astype(int).values

    # Preprocess:
    # - categorical columns: protocol_type, service, flag => one-hot
    cat_cols = ["protocol_type","service","flag"]
    num_cols = [c for c in X_raw.columns if c not in cat_cols]

    # One-hot encode categorical
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = enc.fit_transform(X_raw[cat_cols])
    X_num = X_raw[num_cols].astype(float).values

    # Scale numeric features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    X = np.hstack([X_num_scaled, X_cat])

    # optional sub-sampling to keep simulation cost reasonable
    if sample_size is not None and sample_size < X.shape[0]:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], sample_size, replace=False)
        X = X[idx]
        y = y[idx]
    return X, y

# ---------------------------
# 3) Reduce dimensions (PCA) to fit into n_qubits
# ---------------------------
def reduce_dimensionality(X, n_qubits=4):
    """
    PCA to reduce X to n_qubits features. Returns X_reduced (shape [n_samples, n_qubits]) and the PCA object.
    Note: we assume features are scaled already.
    """
    pca = PCA(n_components=n_qubits, random_state=42)
    X_reduced = pca.fit_transform(X)
    # normalize to range [0, pi] (common for angle encodings)
    # scale each component to mean 0, std 1 then to [-1,1] then to [0,pi]
    X_std = (X_reduced - X_reduced.mean(axis=0)) / (X_reduced.std(axis=0) + 1e-9)
    X_scaled = (X_std - X_std.min()) / (X_std.max() - X_std.min() + 1e-9)  # [0,1]
    X_angles = X_scaled * math.pi  # map to [0, pi]
    return X_angles, pca

# ---------------------------
# 4) Build quantum feature map & compute kernel matrix (statevector fidelity)
# ---------------------------
def build_feature_map(n_qubits, reps=1, map_type="zz"):
    """
    Build a circuit that encodes n_qubits classical angles into a state.
    map_type: 'zz' (ZZFeatureMap) or 'pauli' (PauliFeatureMap) or 'z' (ZFeatureMap)
    """
    if map_type == "zz":
        # ZZFeatureMap is good for kernel experiments
        return ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement="full")
    elif map_type == "pauli":
        return PauliFeatureMap(feature_dimension=n_qubits, reps=reps, paulis=['Z','ZZ'])
    elif map_type == "z":
        return ZFeatureMap(feature_dimension=n_qubits, reps=reps)
    else:
        raise ValueError("Unknown map_type")

def compute_kernel_matrix(X_angles, feature_map, backend=None):
    """
    Compute kernel matrix K where K[i,j] = |<phi(x_i)|phi(x_j)>|^2 (state fidelity)
    Uses statevector simulation to prepare states and compute inner products.
    X_angles: shape (n_samples, n_features)
    feature_map: qiskit circuit with parameter vector ordering matching X_angles columns
    """
    if backend is None:
        backend = Aer.get_backend('aer_simulator_statevector')
    n = X_angles.shape[0]
    # Pre-compile the circuits once for speed
    param_names = feature_map.parameters  # ParameterVector
    # We'll create circuits per data point, bind parameters, run statevector
    statevectors = []
    for i in tqdm(range(n), desc="Preparing quantum states"):
        qc = feature_map.copy() # Create a copy of the feature map
        # Assign parameters using the new method. inplace=True modifies qc directly.
        qc.assign_parameters({p: float(X_angles[i, idx]) for idx, p in enumerate(param_names)}, inplace=True)
        # Ensure the simulator yields statevector directly:
        sv = Statevector.from_instruction(qc)  # directly simulates the statevector
        statevectors.append(sv.data)  # complex vector

    # compute kernel matrix (fidelity/squared absolute overlap)
    K = np.zeros((n, n), dtype=float)
    for i in tqdm(range(n), desc="Computing kernel matrix"):
        v_i = statevectors[i]
        for j in range(i, n):
            v_j = statevectors[j]
            overlap = np.vdot(v_i, v_j)  # inner product <v_i|v_j>
            Kij = np.abs(overlap)**2
            K[i, j] = Kij
            K[j, i] = Kij
    return K

# ---------------------------
# 5) Kernel KMeans implementation (works with precomputed kernel matrix)
# ---------------------------
def kernel_kmeans(K, n_clusters=2, max_iter=100, tol=1e-6, random_state=42):
    """
    K: precomputed kernel matrix (n_samples x n_samples)
    Returns labels (n_samples)
    """
    n = K.shape[0]
    rng = np.random.RandomState(random_state)
    # initialize labels randomly
    labels = rng.randint(0, n_clusters, size=n)
    diagK = np.diag(K).copy()
    for it in range(max_iter):
        changed = False
        # compute cluster assignments via distances in feature space:
        # distance(x, cluster_j) = K_xx - (2/|C_j|) sum_{i in C_j} K_xi + (1/|C_j|^2) sum_{i,l in C_j} K_il
        cluster_indices = [np.where(labels == j)[0] for j in range(n_clusters)]
        # Precompute cluster constants
        cluster_sumK = []
        cluster_sum_pairK = []
        cluster_size = []
        for idxs in cluster_indices:
            if len(idxs) == 0:
                cluster_sumK.append(None)
                cluster_sum_pairK.append(0.0)
                cluster_size.append(0)
            else:
                cluster_sumK.append(np.sum(K[:, idxs], axis=1))  # sum over rows => shape (n,)
                # sum_{i,l in C} K_il
                cluster_sum_pairK.append(np.sum(K[np.ix_(idxs, idxs)]))
                cluster_size.append(len(idxs))
        # assign labels
        new_labels = np.zeros_like(labels)
        for x in range(n):
            distances = np.zeros(n_clusters)
            for j in range(n_clusters):
                if cluster_size[j] == 0:
                    distances[j] = np.inf
                    continue
                term1 = K[x, x]
                term2 = -2.0 * (cluster_sumK[j][x] / cluster_size[j])
                term3 = (cluster_sum_pairK[j] / (cluster_size[j]**2))
                distances[j] = term1 + term2 + term3
            new_labels[x] = np.argmin(distances)
        if np.all(new_labels == labels):
            # converged
            break
        labels = new_labels
    return labels

# ---------------------------
# 6) Anomaly decision from cluster labels (k=2 default)
# ---------------------------
def cluster_labels_to_anomaly(labels, y_true):
    """
    For binary anomaly detection: map clusters -> normal/anomaly depending on majority of true labels.
    Returns predicted_anomaly (0 normal, 1 anomaly)
    """
    unique_clusters = np.unique(labels)
    mapping = {}
    for c in unique_clusters:
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            mapping[c] = 1  # if empty cluster map to anomaly (rare)
            continue
        # choose majority label within cluster as predicted label for that cluster
        majority = int(round(np.mean(y_true[idxs])))
        mapping[c] = majority
    preds = np.array([mapping[c] for c in labels])
    return preds, mapping

# ---------------------------
# 7) Putting it all together
# ---------------------------
def quantum_kernel_kmeans_pipeline(sample_size=600, n_qubits=4, reps=1, map_type="zz", k_clusters=2):

    print("Loading dataset...")
    X, y = load_nsl_kdd(sample_size=sample_size)
    print("Original X shape:", X.shape)

    print(f"Reducing dimension to {n_qubits} (PCA) ...")
    X_angles, pca = reduce_dimensionality(X, n_qubits=n_qubits)
    print("X_angles shape:", X_angles.shape)

    print(f"Building feature map ({map_type}, reps={reps}) ...")
    feature_map = build_feature_map(n_qubits=n_qubits, reps=reps, map_type=map_type)

    print("Computing quantum kernel matrix")
    K = compute_kernel_matrix(X_angles, feature_map)

    print("Kernel KMeans...")
    labels_q = kernel_kmeans(K, n_clusters=k_clusters)

    preds_q, mapping = cluster_labels_to_anomaly(labels_q, y)
    print("Cluster -> anomaly mapping:", mapping)

    # evaluation
    acc = accuracy_score(y, preds_q)
    prec, rec, f1, _ = precision_recall_fscore_support(y, preds_q, average="binary", zero_division=0)
    print("Quantum Kernel KMeans results:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(classification_report(y, preds_q, digits=4))

    # Classical baseline (KMeans on PCA reduced features in Euclidean space)
    print("Classical KMeans baseline on PCA reduced features (Euclidean)")
    # Use the same PCA features before angle scaling by inverse transform
    # compute PCA on the original X again but with same n_qubits
    # use X_angles but treat it as vector features
    km = KMeans(n_clusters=k_clusters, random_state=42).fit(X_angles)
    labels_c = km.labels_
    preds_c, mapping_c = cluster_labels_to_anomaly(labels_c, y)
    acc_c = accuracy_score(y, preds_c)
    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(y, preds_c, average="binary", zero_division=0)
    print("Classical KMeans baseline results:")
    print(f"Accuracy: {acc_c:.4f}, Precision: {prec_c:.4f}, Recall: {rec_c:.4f}, F1: {f1_c:.4f}")
    print(classification_report(y, preds_c, digits=4))

    return {
        "X": X, "y": y, "X_angles": X_angles, "K": K,
        "labels_q": labels_q, "preds_q": preds_q, "mapping": mapping,
        "labels_c": labels_c, "preds_c": preds_c
    }

# ---------------------------
# 8) Run the pipeline (example)
# ---------------------------
if __name__ == "__main__":
    # tune these to compute/time budget:
    SAMPLE_SIZE = 600
    N_QUBITS = 4         # number of qubits / PCA components (4 qubits -> 16-dim statevector)
    REPS = 1
    MAP_TYPE = "zz"      # 'zz' recommended
    results = quantum_kernel_kmeans_pipeline(sample_size=SAMPLE_SIZE, n_qubits=N_QUBITS, reps=REPS, map_type=MAP_TYPE, k_clusters=2)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

# Extract predictions and true labels
y_true = results['y']
preds_q = results['preds_q']
preds_c = results['preds_c']

# --- 1) ROC Curve ---
# We'll treat cluster label 1 as anomaly "score" for ROC (binary label)
fpr_q, tpr_q, _ = roc_curve(y_true, preds_q)
roc_auc_q = roc_auc_score(y_true, preds_q)

fpr_c, tpr_c, _ = roc_curve(y_true, preds_c)
roc_auc_c = roc_auc_score(y_true, preds_c)

plt.figure(figsize=(8,6))
plt.plot(fpr_q, tpr_q, label=f'Quantum K-Means ROC (AUC={roc_auc_q:.3f})', color='darkorange')
plt.plot(fpr_c, tpr_c, label=f'Classical K-Means ROC (AUC={roc_auc_c:.3f})', color='blue')
plt.plot([0,1],[0,1],'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# --- 2) Precision-Recall Curve ---
precision_q, recall_q, _ = precision_recall_curve(y_true, preds_q)
precision_c, recall_c, _ = precision_recall_curve(y_true, preds_c)

plt.figure(figsize=(8,6))
plt.plot(recall_q, precision_q, label='Quantum K-Means', color='darkorange')
plt.plot(recall_c, precision_c, label='Classical K-Means', color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()

# --- 3) Histogram of Predicted Anomalies ---
plt.figure(figsize=(8,6))
plt.hist(preds_q[y_true==0], bins=2, alpha=0.6, label='Normal', color='green')
plt.hist(preds_q[y_true==1], bins=2, alpha=0.6, label='Anomaly', color='red')
plt.xlabel('Predicted Cluster (1=anomaly)')
plt.ylabel('Count')
plt.title('Quantum K-Means Predicted Anomaly Distribution')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.hist(preds_c[y_true==0], bins=2, alpha=0.6, label='Normal', color='green')
plt.hist(preds_c[y_true==1], bins=2, alpha=0.6, label='Anomaly', color='red')
plt.xlabel('Predicted Cluster (1=anomaly)')
plt.ylabel('Count')
plt.title('Classical K-Means Predicted Anomaly Distribution')
plt.legend()
plt.show()
