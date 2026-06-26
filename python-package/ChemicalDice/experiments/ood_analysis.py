# -*- coding: utf-8 -*-
"""Multitask Classification for PCBA with Random and Scaffold Splits"""

import os
import time
import warnings
import gc
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, cohen_kappa_score, balanced_accuracy_score,
                             confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Imblearn imports
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsembleClassifier

# RDKit for Scaffold Splitting
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

pd.set_option("display.max_columns", 25)
warnings.filterwarnings('ignore')

# --- 1. Scaffold Split Logic ---
def generate_scaffold(smiles, include_chirality=False):
    """Compute the Murcko scaffold for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

def scaffold_split_indices(smiles_list, frac_train=0.8, frac_test=0.2):
    """Returns train and test indices based on Murcko scaffolds."""
    scaffolds = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold].append(i)

    # Sort scaffolds by size (largest first)
    scaffold_sets = [
        scaffold_indices for scaffold, scaffold_indices in
        sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)
    ]

    train_indices, test_indices = [], []
    train_cutoff = frac_train * len(smiles_list)

    for scaffold_indices in scaffold_sets:
        if len(train_indices) + len(scaffold_indices) > train_cutoff:
            test_indices.extend(scaffold_indices)
        else:
            train_indices.extend(scaffold_indices)

    return train_indices, test_indices


# --- 2. Class Imbalance Handler ---
def handle_class_imbalance(X_train, y_train, resampling_strategy="DOWNSAMPLING", verbose=True):        
    counter = Counter(y_train)
    if len(counter) < 2:
        return X_train, y_train, {"Strategy": "Only one class present"}

    minority_class = min(counter, key=counter.get)
    majority_class = max(counter, key=counter.get)

    minority_count = counter[minority_class]
    majority_count = counter[majority_class]
    total = sum(counter.values())

    IR = majority_count / minority_count if minority_count > 0 else float('inf')
    abs_prop = (minority_count / total) * 100

    resampler, model = None, None
    X_res, y_res = X_train, y_train

    if resampling_strategy == "DOWNSAMPLING":
        strategy = "DOWNSAMPLING the majority class with RandomUnderSampler."
        resampler = RandomUnderSampler(random_state=42)
    elif IR <= 2 and minority_count > 200:
        strategy = "Balanced / near balanced → No resampling. Use class weights."
    elif 2 < IR <= 10 and minority_count > 200:
        strategy = "Mild imbalance → Apply SMOTE or BorderlineSMOTE."
        resampler = SMOTE(random_state=42) if IR <= 5 else BorderlineSMOTE(random_state=42)
    elif 10 < IR <= 50 and 100 <= minority_count <= 500:
        strategy = "Moderate imbalance → Apply SMOTEENN or class weights."
        resampler = SMOTEENN(random_state=42)
    elif IR > 50 and minority_count < 500 and IR <= 200:
        strategy = "Severe imbalance → Avoid SMOTE. Use EasyEnsembleClassifier."
        model = EasyEnsembleClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    elif IR > 200 and minority_count < 500:
        strategy = "Extreme imbalance → Avoid resampling. Use XGBoost with scale_pos_weight."
        scale_pos_weight = int(round(majority_count / minority_count))
        model = XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric="logloss", random_state=42)
    else:
        strategy = "Default: Use class weights (no resampling)."

    if resampler is not None:
        try:
            X_res, y_res = resampler.fit_resample(X_train, y_train)
        except Exception as e:
            strategy = f"Resampling failed ({e}) -> Defaulting to original."
            X_res, y_res = X_train, y_train

    decision = {
        "IR": round(IR, 3), "Minority Count": minority_count, "Majority Count": majority_count,        
        "Strategy": strategy, "Resampler": type(resampler).__name__ if resampler else None
    }

    if verbose:
        print(f"🔎 Class Distribution → Minority: {minority_count}, Majority: {majority_count} | IR = {IR:.2f}")

    return X_res, y_res, decision


# --- 3. Main Execution ---
import argparse

def main():
    parser = argparse.ArgumentParser(description="Out-of-Distribution (OOD) Analysis with Scaffold vs Random Splits")
    parser.add_argument("--labels", required=True, help="Path to the labels CSV file (e.g., PCBA.csv)")
    parser.add_argument("--embeddings", nargs="+", required=True, help="List of precomputed embedding parquet files")
    parser.add_argument("--output", default="results_ood", help="Directory to save the metrics results")
    parser.add_argument("--strategy", default="DOWNSAMPLING", choices=["DOWNSAMPLING", "AUTO"], 
                        help="Resampling strategy for class imbalance")
    args_cli = parser.parse_args()

    # File configuration
    labels_file = args_cli.labels
    embedding_files = args_cli.embeddings
    results_dir = args_cli.output
    resampling_strategy = args_cli.strategy

    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading global labels from: {labels_file}")
    labels_df_global = pd.read_csv(labels_file)

    # Identify all target columns (e.g., PCBA-1030, PCBA-1379, etc.)
    # Supports PCBA- prefix but can be expanded
    target_columns = [col for col in labels_df_global.columns if col.startswith("PCBA-")]
    if not target_columns:
        # Fallback if no PCBA prefix found, might be a different dataset
        target_columns = [c for c in labels_df_global.columns if c not in ["SMILES", "id", "index"]]
        
    print(f"Found {len(target_columns)} targets to process.")

    # Process one embedding file at a time to save memory
    for emb_file in embedding_files:
        descriptor_name = os.path.basename(emb_file).split(".")[0]
        print(f"\n{'='*80}\n🚀 Loading Embeddings: {emb_file} ({descriptor_name})\n{'='*80}")

        embeddings_full = pd.read_parquet(emb_file)

        # Datav showed the first column is named "id" but contains SMILES strings
        if "id" in embeddings_full.columns and "SMILES" not in embeddings_full.columns:
            embeddings_full.rename(columns={"id": "SMILES"}, inplace=True)

        embeddings_full = embeddings_full.drop_duplicates(subset="SMILES").set_index("SMILES")

        # Iterate through each specific assay task
        for target in target_columns:
            if target in ["PCBA-2100","PCBA-720532","PCBA-1468"]:
                pass
            else:
                continue
            print(f"\n{'#'*60}\n=== Running target: {target} | Descriptor: {descriptor_name} ===\n{'#'*60}")

            # 1. Filter labels for the current target, dropping NaNs (missing labels)
            task_labels = labels_df_global[["SMILES", target]].dropna(subset=[target])
            task_labels = task_labels.drop_duplicates(subset=["SMILES"])

            if len(task_labels) < 50:
                print(f"⚠️ Skipping {target}: Not enough valid samples ({len(task_labels)}).")
                continue

            # 2. Extract matching embeddings safely
            common_smiles = task_labels["SMILES"].isin(embeddings_full.index)
            task_labels = task_labels[common_smiles].reset_index(drop=True)

            # Get X and align order with task_labels
            X_df = embeddings_full.loc[task_labels["SMILES"]].reset_index(drop=True)
            y = task_labels[target].astype(int)
            smiles_array = task_labels["SMILES"].values

            assert len(X_df) == len(y), "Mismatch between X and y length!"
            print(f"Assay Dataset Size: {len(X_df)} samples.")

            # Create specific directory for this target
            target_results_dir = os.path.join(results_dir, target)
            os.makedirs(target_results_dir, exist_ok=True)

            # --- LOOP THROUGH SPLIT TYPES ---
            for split_type in ["Random", "Scaffold"]:
                print(f"\n--- Performing {split_type} Split ---")

                if split_type == "Scaffold":
                    train_idx, test_idx = scaffold_split_indices(smiles_array, frac_train=0.8, frac_test=0.2)
                    X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_df, y, test_size=0.2, random_state=42, stratify=y
                    )

                # Handle Imbalance
                X_train_res, y_train_res, decision = handle_class_imbalance(X_train, y_train, resampling_strategy=resampling_strategy)

                # Avoid running if we only ended up with one class in training
                if len(set(y_train_res)) < 2:
                    print("⚠️ Not enough classes to train. Skipping split.")
                    continue

                # Scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_res)
                X_test_scaled = scaler.transform(X_test)

                result_file = os.path.join(target_results_dir, f"{descriptor_name}_{split_type.lower()}_metrics.csv")

                models = {
                    "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=10),
                    "GaussianNB": GaussianNB(),
                    "RandomForest": RandomForestClassifier(n_jobs=10, random_state=42),
                    "GradientBoosting": GradientBoostingClassifier(random_state=42),
                    "AdaBoost": AdaBoostClassifier(random_state=42),
                    "ExtraTrees": ExtraTreesClassifier(n_jobs=10, random_state=42),
                    "SVM": SVC(probability=True, random_state=42),
                    "XGBoost": XGBClassifier(eval_metric="logloss", n_jobs=10, random_state=42),       
                    "LightGBM": LGBMClassifier(n_jobs=10, importance_type='split', random_state=42, verbose=-1),
                    "CatBoost": CatBoostClassifier(verbose=0, thread_count=10, random_state=42)        
                }

                results = {}
                for name, model in models.items():
                    start_time = time.time()
                    try:
                        model.fit(X_train_scaled, y_train_res)

                        y_pred_test = model.predict(X_test_scaled)
                        y_prob_test = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

                        y_pred_train = model.predict(X_train_scaled)
                        y_prob_train = model.predict_proba(X_train_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

                        elapsed_time = round(time.time() - start_time, 4)

                        tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test, labels=[0,1]).ravel()
                        tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train_res, y_pred_train, labels=[0,1]).ravel()

                        results[name] = {
                            'Time Taken (s)': elapsed_time,
                            'Test ROC AUC': roc_auc_score(y_test, y_prob_test) if y_prob_test is not None else 'N/A',
                            'Test Accuracy': accuracy_score(y_test, y_pred_test),
                            'Test Balanced Accuracy': balanced_accuracy_score(y_test, y_pred_test),    
                            'Test F1 Score': f1_score(y_test, y_pred_test, average='binary', zero_division=0),
                            'Test Precision': precision_score(y_test, y_pred_test, average='binary', zero_division=0),
                            'Test Recall': recall_score(y_test, y_pred_test, average='binary', zero_division=0),
                            'Test Kappa': cohen_kappa_score(y_test, y_pred_test),
                            'Test TP': tp_test, 'Test TN': tn_test, 'Test FP': fp_test, 'Test FN': fn_test,

                            'Train ROC AUC': roc_auc_score(y_train_res, y_prob_train) if y_prob_train is not None else 'N/A',
                            'Train Accuracy': accuracy_score(y_train_res, y_pred_train),
                            'Train Balanced Accuracy': balanced_accuracy_score(y_train_res, y_pred_train),
                            'Train F1 Score': f1_score(y_train_res, y_pred_train, average='binary', zero_division=0),
                            'Train Precision': precision_score(y_train_res, y_pred_train, average='binary', zero_division=0),
                            'Train Recall': recall_score(y_train_res, y_pred_train, average='binary', zero_division=0),
                            'Train Kappa': cohen_kappa_score(y_train_res, y_pred_train),
                            'Train TP': tp_train, 'Train TN': tn_train, 'Train FP': fp_train, 'Train FN': fn_train,
                        }
                    except Exception as e:
                        print(f"❌ Error training {name}: {e}")

                # Save metrics to CSV
                pd.DataFrame.from_dict(results, orient='index').to_csv(result_file)
                print(f"💾 Metrics saved to {result_file}")

            # Clear memory between tasks
            del X_df, X_train, X_test, X_train_scaled, X_test_scaled
            gc.collect()

        # Clear huge embeddings from memory before loading the next descriptor
        del embeddings_full
        gc.collect()

if __name__ == '__main__':
    main()