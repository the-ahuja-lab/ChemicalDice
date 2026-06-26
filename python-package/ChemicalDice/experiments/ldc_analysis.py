import os
import glob
import time
import psutil
import argparse
import sys
from collections import Counter
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, cohen_kappa_score, balanced_accuracy_score,
                             confusion_matrix)
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier      
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler

def get_ram_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return f"{mem_bytes / (1024 ** 3):.2f} GB" if mem_bytes >= 1024**3 else f"{mem_bytes / (1024 ** 2):.2f} MB"

def handle_class_imbalance(X_train, y_train, resampling_strategy=None, verbose=True):
    counter = Counter(y_train)
    if len(counter) < 2:
        return X_train, y_train, {"Strategy": "Only one class present"}

    minority_class = min(counter, key=counter.get)
    majority_class = max(counter, key=counter.get)

    minority_count = counter[minority_class]
    majority_count = counter[majority_class]
    total = sum(counter.values())

    IR = majority_count / minority_count if minority_count > 0 else float('inf')
    
    X_res, y_res = X_train, y_train
    strategy = "Default: No resampling"

    if resampling_strategy == "DOWNSAMPLING" or IR > 5:
        strategy = "DOWNSAMPLING the majority class with RandomUnderSampler."
        resampler = RandomUnderSampler(random_state=42)
        try:
            X_res, y_res = resampler.fit_resample(X_train, y_train)
        except Exception as e:
            strategy = f"Resampling failed ({e})"

    if verbose:
        print(f"🔎 Class Distribution → Minority: {minority_count}, Majority: {majority_count} | IR = {IR:.2f} | {strategy}")

    return X_res, y_res, {"Strategy": strategy}

def run_ldc_analysis(dataset_name, emb_root, label_root, result_root, train_fractions=[0.10, 0.25, 0.50, 0.75], n_folds=5, seeds=[42, 123, 456]):
    MODELS = {
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(n_jobs=5, random_state=42, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(n_jobs=5, random_state=42, verbose=-1),
        "ExtraTrees": ExtraTreesClassifier(n_jobs=5, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    os.makedirs(result_root, exist_ok=True)
    label_file = os.path.join(label_root, f"{dataset_name}.csv")
    
    if not os.path.exists(label_file):
        print(f"⚠️ Label file not found: {label_file}")
        return

    labels_df = pd.read_csv(label_file)
    
    # Detect identifier column
    label_id_col = "SMILES" if "SMILES" in labels_df.columns else labels_df.columns[0]
    target_cols = [c for c in labels_df.columns if c not in [label_id_col, "id", "index"]]
    
    print(f"🚀 Starting LDC Analysis | Dataset: {dataset_name} | Targets: {len(target_cols)}")

    emb_pattern = os.path.join(emb_root, f"{dataset_name}_*.parquet")
    emb_files = glob.glob(emb_pattern)

    for emb_path in emb_files:
        descriptor = os.path.basename(emb_path).replace(f"{dataset_name}_", "").replace(".parquet", "")
        print(f"\n📂 Descriptor: {descriptor}")
        embeddings = pd.read_parquet(emb_path)

        if "id" in embeddings.columns and "SMILES" not in embeddings.columns:
            embeddings.rename(columns={"id": "SMILES"}, inplace=True)
        if "SMILES" in embeddings.columns:
            embeddings["SMILES"] = embeddings["SMILES"].astype(str).str.strip()
            embeddings = embeddings.set_index("SMILES").sort_index()

        for target_column in target_cols:
            print(f"  ▶️ Target: {target_column}")
            target_labels = labels_df.dropna(subset=[target_column]).copy()
            target_labels[label_id_col] = target_labels[label_id_col].astype(str).str.strip()
            
            valid_ids = list(set(target_labels[label_id_col]) & set(embeddings.index))
            if len(valid_ids) < 50: continue

            target_labels = target_labels[target_labels[label_id_col].isin(valid_ids)]
            X = embeddings.loc[target_labels[label_id_col]].values
            y = target_labels[target_column].values.astype(int)

            if len(np.unique(y)) < 2: continue

            for seed in seeds:
                cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
                    X_train_full, X_test = X[train_idx], X[test_idx]
                    y_train_full, y_test = y[train_idx], y[test_idx]

                    for frac in train_fractions:
                        frac_label = int(frac * 100)
                        try:
                            X_train_sub, _, y_train_sub, _ = train_test_split(
                                X_train_full, y_train_full, train_size=frac, stratify=y_train_full, random_state=seed
                            )
                        except:
                            X_train_sub, _, y_train_sub, _ = train_test_split(
                                X_train_full, y_train_full, train_size=frac, random_state=seed
                            )

                        if len(np.unique(y_train_sub)) < 2: continue

                        X_train_res, y_train_res, _ = handle_class_imbalance(X_train_sub, y_train_sub, verbose=False)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_res)
                        X_test_scaled = scaler.transform(X_test)

                        for model_name, base_model in MODELS.items():
                            model = clone(base_model)
                            model.set_params(random_state=seed)
                            
                            start = time.time()
                            model.fit(X_train_scaled, y_train_res)
                            elapsed = time.time() - start
                            
                            y_pred = model.predict(X_test_scaled)
                            y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
                            
                            res = {
                                "Descriptor": descriptor, "Target": target_column, "Model": model_name,
                                "Seed": seed, "Fold": fold+1, "Fraction": frac_label,
                                "ROC_AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
                                "Balanced_Acc": balanced_accuracy_score(y_test, y_pred),
                                "Time": elapsed
                            }
                            
                            # Save result (append or individual files as per previous pattern)
                            res_file = os.path.join(result_root, f"{dataset_name}_{descriptor}_ldc_results.csv")
                            pd.DataFrame([res]).to_csv(res_file, mode='a', header=not os.path.exists(res_file), index=False)

def main():
    parser = argparse.ArgumentParser(description="Low Data Condition (LDC) Analysis")
    parser.add_argument("--dataset", required=True, help="Dataset name (without .csv extension)")
    parser.add_argument("--emb_dir", default=".", help="Directory containing embedding parquet files")
    parser.add_argument("--label_dir", default=".", help="Directory containing label CSV files")
    parser.add_argument("--output", default="results_ldc", help="Output directory for results")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75], help="Training fractions to evaluate")
    
    args = parser.parse_args()
    run_ldc_analysis(args.dataset, args.emb_dir, args.label_dir, args.output, args.fractions)

if __name__ == "__main__":
    main()
