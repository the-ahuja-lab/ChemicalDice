# -*- coding: utf-8 -*-
"""
CDI Similarity Search (k-NN) Module
Standardized for integration into the ChemicalDice package.
"""

import os
import gc
import warnings
import argparse
import numpy as np
import pandas as pd
import multiprocessing

from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import normalize
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =========================================================
# HELPER: LOAD EMBEDDINGS
# =========================================================
def load_embeddings(base_dir, descriptor):
    """Scan directory for descriptor CSV or Parquet files."""
    if not os.path.exists(base_dir):
        return None

    for file in os.listdir(base_dir):
        if descriptor in file and (file.endswith(".parquet") or file.endswith(".csv")):   
            path = os.path.join(base_dir, file)
            print(f"   ✅ Found {descriptor}: {file}")

            if file.endswith(".parquet"):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)

            if "id" in df.columns:
                df.rename(columns={"id": "SMILES"}, inplace=True)
            elif "smiles" in df.columns:
                df.rename(columns={"smiles": "SMILES"}, inplace=True)

            return df.drop_duplicates(subset="SMILES").set_index("SMILES")
    return None

# =========================================================
# HELPER: SINGLE LOOCV FOLD
# =========================================================
def run_single_fold(train_idx, test_idx, X, y, smiles_array, k_max=20, k_base=5):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Evaluate up to k_max neighbors
    k_val = min(k_max, len(train_idx))
    knn = KNeighborsClassifier(n_neighbors=k_val, metric="cosine")
    knn.fit(X_train, y_train)

    # Base prediction at k_base (Standard Baseline)
    k_eval = min(k_base, len(train_idx))
    knn_base = KNeighborsClassifier(n_neighbors=k_eval, metric="cosine")
    knn_base.fit(X_train, y_train)
    y_pred_base = knn_base.predict(X_test)[0]
    y_prob_base = knn_base.predict_proba(X_test)[0, 1]

    # Extract distances and neighbor indices for K=k_max
    distances, neighbor_indices_local = knn.kneighbors(X_test)
    global_indices = train_idx[neighbor_indices_local[0]]

    neighbor_smiles = smiles_array[global_indices]
    neighbor_labels = y[global_indices]

    return y_test[0], y_pred_base, y_prob_base, distances[0], neighbor_smiles, neighbor_labels  

# =========================================================
# MAIN PIPELINE
# =========================================================
def run_similarity_search(folder, label_csv, target_cols, descriptor="CDI", sampling="NONE", output_dir="similarity_results", n_cores=None, seed=42):
    if n_cores is None:
        n_cores = min(15, multiprocessing.cpu_count())
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🚀 Starting Similarity Search | Dataset: {label_csv} | Descriptor: {descriptor}")

    label_path = os.path.join(folder, label_csv)
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file {label_path} not found.")

    labels_master = pd.read_csv(label_path)
    if "SMILES" not in labels_master.columns:
        # Try to find a smiles column case-insensitive
        cols = {c.upper(): c for c in labels_master.columns}
        if "SMILES" in cols:
            labels_master.rename(columns={cols["SMILES"]: "SMILES"}, inplace=True)
        else:
            raise ValueError("Label CSV must contain a 'SMILES' column.")

    # Load embeddings
    emb_df = load_embeddings(folder, descriptor)
    if emb_df is None:
        raise FileNotFoundError(f"No embeddings found for {descriptor} in {folder}")

    for target_col in target_cols:
        if target_col not in labels_master.columns:
            print(f"⚠️ Target {target_col} not in {label_csv}. Skipping.")
            continue

        print(f"--- Processing Target: {target_col} ---")
        
        # Guarantee Alignment
        task_labels = labels_master[["SMILES", target_col]].dropna()
        common_smiles = sorted(list(set(task_labels["SMILES"]) & set(emb_df.index)))
        task_labels = task_labels[task_labels["SMILES"].isin(common_smiles)].drop_duplicates(subset="SMILES").set_index("SMILES")
        
        smiles_full = np.array(common_smiles)
        y_full = task_labels.loc[common_smiles, target_col].values

        if sampling == "DOWNSAMPLING":
            print(f"   Applying Downsampling. Common molecules: {len(smiles_full)}")      
            rus = RandomUnderSampler(random_state=seed)
            indices = np.arange(len(smiles_full)).reshape(-1, 1)
            indices_res, y_resampled = rus.fit_resample(indices, y_full)
            indices_res = indices_res.flatten()
            smiles_resampled = smiles_full[indices_res]
        else:
            print(f"   No sampling. Common molecules: {len(smiles_full)}")
            smiles_resampled = smiles_full
            y_resampled = y_full

        # Prepare Features
        X = emb_df.loc[smiles_resampled].values
        X = normalize(X, norm="l2")
        y = y_resampled

        # Run LOOCV
        print(f"--- Running LOOCV (k_max=20) ---")
        loo = LeaveOneOut()
        results_fold = Parallel(n_jobs=n_cores, backend="loky")(
            delayed(run_single_fold)(train_idx, test_idx, X, y, smiles_resampled)     
            for train_idx, test_idx in tqdm(loo.split(X), total=len(X), desc="LOOCV Folds")
        )

        y_true, y_pred, y_prob, all_dists, all_n_smiles, all_n_labels = zip(*results_fold)
        y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob) 

        # --- Process Results & Weighted Metrics ---
        dataset_distances = []
        dataset_metrics = []
        dataset_weighted_k_results = []
        
        dist_matrix = []
        label_matrix = []

        for i in range(len(y_true)):
            dists = list(all_dists[i])
            n_smiles = list(all_n_smiles[i])
            n_labels = list(all_n_labels[i])

            while len(dists) < 20:
                dists.append(np.nan); n_smiles.append(None); n_labels.append(np.nan)  

            dist_matrix.append(dists)
            label_matrix.append(n_labels)

            record = {
                "Assay": target_col, "Descriptor": descriptor,
                "Query_SMILES": smiles_resampled[i],
                "True_Label": y_true[i], "Predicted_Label_k5": y_pred[i],
                "Correct_Prediction": y_true[i] == y_pred[i]
            }

            for n_idx in range(20):
                record[f"Dist_{n_idx+1}"] = dists[n_idx]
                record[f"SMILES_{n_idx+1}"] = n_smiles[n_idx]
                record[f"Label_{n_idx+1}"] = n_labels[n_idx]

            dataset_distances.append(record)

        # Baseline Metrics (k=5)
        dataset_metrics.append({
            "Assay": target_col, "Descriptor": descriptor, "Sampling": sampling,
            "ROC_AUC": roc_auc_score(y_true, y_prob),
            "PR_AUC": average_precision_score(y_true, y_prob),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred)
        })

        # Weighted k-NN for 1 to 20
        dist_np = np.array(dist_matrix)
        label_np = np.array(label_matrix)
        weight_np = 1.0 / (dist_np + 1e-8)

        for k in range(1, 21):
            k_weights = weight_np[:, :k]
            k_labels = label_np[:, :k]
            weighted_sum = np.nansum(k_weights * k_labels, axis=1)
            total_weight = np.nansum(k_weights, axis=1)
            y_prob_weighted = np.divide(weighted_sum, total_weight, out=np.zeros_like(weighted_sum), where=total_weight!=0)
            y_pred_weighted = (y_prob_weighted >= 0.5).astype(int)

            try:
                roc_auc = roc_auc_score(y_true, y_prob_weighted)
                pr_auc = average_precision_score(y_true, y_prob_weighted)
            except ValueError:
                roc_auc, pr_auc = np.nan, np.nan

            dataset_weighted_k_results.append({
                "Assay": target_col, "Descriptor": descriptor,
                "k_neighbors": k, "ROC_AUC": roc_auc, "PR_AUC": pr_auc,
                "Accuracy": accuracy_score(y_true, y_pred_weighted),
                "Balanced_Acc": balanced_accuracy_score(y_true, y_pred_weighted),     
                "F1": f1_score(y_true, y_pred_weighted, zero_division=0)
            })

        # Export Files
        suffix = f"{descriptor}_{target_col}"
        pd.DataFrame(dataset_metrics).to_csv(os.path.join(output_dir, f"metrics_k5_{suffix}.csv"), index=False)
        pd.DataFrame(dataset_distances).to_csv(os.path.join(output_dir, f"distances_20_{suffix}.csv"), index=False)
        pd.DataFrame(dataset_weighted_k_results).to_csv(os.path.join(output_dir, f"weighted_metrics_{suffix}.csv"), index=False)

        print(f"✅ Results for {target_col} saved to {output_dir}")

    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CDI Similarity Search (k-NN LOOCV)")
    parser.add_argument("--data-dir", required=True, help="Directory containing labels and embeddings")
    parser.add_argument("--labels", required=True, help="Filename of the label CSV")
    parser.add_argument("--targets", nargs="+", required=True, help="Target column names")
    parser.add_argument("--descriptor", default="CDI", help="Descriptor name to search for in files")
    parser.add_argument("--sampling", choices=["NONE", "DOWNSAMPLING"], default="NONE")
    parser.add_argument("--output", default="./similarity_out", help="Output directory")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_similarity_search(args.data_dir, args.labels, args.targets, args.descriptor, args.sampling, args.output, args.cores, args.seed)
