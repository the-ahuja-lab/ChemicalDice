# -*- coding: utf-8 -*-
"""
CDI Classification Benchmark Pipeline
Standardized for integration into the ChemicalDice package.
"""

import os
import json
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations
from tqdm import tqdm

# Scikit-Learn
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, KNNImputer
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.manifold import Isomap, TSNE, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             roc_auc_score, cohen_kappa_score, balanced_accuracy_score, 
                             confusion_matrix)

# Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Imbalance handling
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.under_sampling import RandomUnderSampler

# =========================================================
# UTILITIES
# =========================================================

def handle_class_imbalance(X_train, y_train, strategy="auto", verbose=True):
    """Detect imbalance and apply the requested or automated strategy."""
    counter = Counter(y_train)
    minority_class = min(counter, key=counter.get)
    majority_class = max(counter, key=counter.get)
    minority_count = counter[minority_class]
    majority_count = counter[majority_class]
    total = sum(counter.values())

    IR = majority_count / minority_count
    abs_prop = (minority_count / total) * 100

    resampler, model = None, None
    X_res, y_res = X_train, y_train

    # Automated Strategy Selection
    if strategy == "auto":
        if IR <= 2 and minority_count > 200:
            strategy_applied = "Balanced / near balanced → No resampling."
        elif 2 < IR <= 10 and minority_count > 200:
            strategy_applied = "Mild imbalance → Apply SMOTE."
            resampler = SMOTE(random_state=42)
        elif 10 < IR <= 50 and 100 <= minority_count <= 500:
            strategy_applied = "Moderate imbalance → Apply SMOTE+ENN."
            resampler = SMOTEENN(random_state=42)
        elif IR > 50 and minority_count < 500:
            strategy_applied = "Severe imbalance → EasyEnsemble."
            model = EasyEnsembleClassifier(n_estimators=10, random_state=42, n_jobs=-1)
        else:
            strategy_applied = "Default: No resampling."
    
    # Manual Strategy Selection
    elif strategy == "smote":
        strategy_applied = "Manual: SMOTE"
        resampler = SMOTE(random_state=42)
    elif strategy == "smoteenn":
        strategy_applied = "Manual: SMOTE+ENN"
        resampler = SMOTEENN(random_state=42)
    elif strategy == "easyensemble":
        strategy_applied = "Manual: EasyEnsemble"
        model = EasyEnsembleClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    elif strategy == "downsampling":
        strategy_applied = "Manual: Downsampling (RandomUnderSampler)"
        resampler = RandomUnderSampler(random_state=42)
    else:
        strategy_applied = "Manual: No resampling"

    if resampler is not None:
        X_res, y_res = resampler.fit_resample(X_train, y_train)

    decision = {
        "IR": round(IR, 3),
        "Strategy": strategy_applied,
        "Resampler": type(resampler).__name__ if resampler else None,
        "Model": type(model).__name__ if model else None
    }

    if verbose:
        print(f"🔎 IR: {IR:.2f} | Strategy: {strategy_applied}")

    return X_res, y_res, decision

def get_reducer(method_name, n_components):
    """Factory for dimensionality reduction models."""
    if method_name == 'pca': return PCA(n_components=n_components, random_state=42)
    if method_name == 'ica': return FastICA(n_components=n_components, random_state=42)
    if method_name == 'kpca': return KernelPCA(n_components=n_components, kernel='linear')
    if method_name == 'tsne': return TSNE(n_components=3, random_state=42)
    if method_name == 'rks': return RBFSampler(n_components=n_components, random_state=42)
    raise ValueError(f"Unknown reduction method: {method_name}")

# =========================================================
# MAIN PIPELINE
# =========================================================

def run_classification_benchmark(label_file, target_col, descriptor_files, output_dir, strategy="auto", seed=42):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    
    print(f"🚀 Starting Classification Benchmark | Target: {target_col}")
    
    # 1. Load Labels
    labels_df = pd.read_csv(label_file)
    labels_df = labels_df.drop_duplicates(subset=['SMILES'], keep='first')
    labels_df.set_index('id', inplace=True)
    
    # 2. Load Descriptors
    datasets = {}
    id_sets = { "label": set(labels_df.index) }
    
    for f in descriptor_files:
        name = os.path.basename(f)
        df = pd.read_csv(f, index_col='id')
        datasets[name] = df
        id_sets[name] = set(df.index)
        
    common_ids = sorted(list(set.intersection(*id_sets.values())))
    print(f"✅ Found {len(common_ids)} common IDs.")
    
    y = labels_df.loc[common_ids, target_col]
    
    # 3. Individual Descriptor Evaluation
    results_summary = []
    
    for d_name, df in datasets.items():
        print(f"📊 Evaluating: {d_name}")
        X = df.loc[common_ids]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        
        # Preprocess
        if "mordred" in d_name.lower():
            imputer = KNNImputer()
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            
        X_train, y_train, _ = handle_class_imbalance(X_train, y_train, strategy=strategy, verbose=False)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train Models
        models = {
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=seed),
            "RandomForest": RandomForestClassifier(random_state=seed),
            "LightGBM": LGBMClassifier(random_state=seed, verbose=-1)
        }
        
        for m_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            results_summary.append({
                "Descriptor": d_name,
                "Model": m_name,
                "ROC-AUC": roc_auc_score(y_test, y_prob),
                "F1": f1_score(y_test, y_pred),
                "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred)
            })
            
    # Save Results
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(os.path.join(output_dir, "classification_results.csv"), index=False)
    print(f"🎉 Benchmark complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CDI Classification Benchmark")
    parser.add_argument("--labels", required=True, help="Path to label CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--descriptors", nargs="+", required=True, help="List of descriptor CSVs")
    parser.add_argument("--output", default="./benchmark_out", help="Output directory")
    parser.add_argument("--strategy", choices=["auto", "smote", "smoteenn", "easyensemble", "downsampling", "none"], default="auto", help="Imbalance handling strategy")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    run_classification_benchmark(args.labels, args.target, args.descriptors, args.output, args.strategy, args.seed)
