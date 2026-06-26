
import os
import glob
import time
import json
import psutil
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import clone # For safely cloning models
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, cohen_kappa_score, balanced_accuracy_score,
                             confusion_matrix)
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier      
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler
import sys
# =========================================================
# CONFIG - UPDATED PATHS

def make_default_config() -> dict:
    return {
        "parquet_dir": "./parquets",
        "label_dir": "./datasets_labelled",
        "results_dir": "./results_cv",
        "label_col": "label",
        "datasets": [],
        "models": {
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "XGBoost": XGBClassifier(device="cuda", random_state=42, eval_metric="logloss"),
            "LightGBM": LGBMClassifier(device="gpu", random_state=42, verbose=-1),
            "ExtraTrees": ExtraTreesClassifier(n_jobs=10, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(random_state=42)
        },
        "seeds": [42, 123, 456]
    }
IDENTIFIER_COLS = ["SMILES","Drug_ID", "ID", "Molecule_ID", "compound_id", "id",  "smiles"]

# =========================================================
# UTILITIES
# =========================================================
def get_ram_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return f"{mem_bytes / (1024 ** 3):.2f} GB" if mem_bytes >= 1024**3 else f"{mem_bytes / (1024 ** 2):.2f} MB"

# def handle_class_imbalance(X_train, y_train, resampling_strategy="No resampling"):
#     if True:
#         return X_train, y_train, {"Strategy": "No resampling"}

def handle_class_imbalance(X_train, y_train, resampling_strategy="Downsampling"):
    counter = Counter(y_train)
    if len(counter) < 2:
        return X_train, y_train, {"Strategy": "No resampling"}

    resampler = RandomUnderSampler(random_state=42) if resampling_strategy == "Downsampling" else None 
    X_res, y_res = resampler.fit_resample(X_train, y_train) if resampler else (X_train, y_train)       

    return X_res, y_res, {
        "IR": round(max(counter.values()) / min(counter.values()), 2) if min(counter.values()) > 0 else 0,
        "Strategy": resampling_strategy,
        "Minority": min(counter.values()),
        "Majority": max(counter.values())
    }


# =========================================================
# MAIN PROCESSING
# =========================================================

def benchmark(cfg: dict):
    EMB_ROOT = cfg.get("parquet_dir", "./parquets")
    LABEL_ROOT = cfg.get("label_dir", "./datasets_labelled")
    RESULT_ROOT = cfg.get("results_dir", "./results_cv")
    TARGET_COLUMNS = [cfg.get("label_col", "label")]
    MODELS = cfg.get("models", make_default_config()["models"])
    SEEDS = cfg.get("seeds", make_default_config()["seeds"])
    datasets = cfg.get("datasets", [])

    if not datasets:
        import glob
        datasets = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(LABEL_ROOT, "*.csv"))]

    for DATASET_NAME in datasets:
        os.makedirs(RESULT_ROOT, exist_ok=True)

        print("🚀 Starting 5-Fold CV × 3 Seeds")
        print(f"Embeddings root: {EMB_ROOT}")
        print(f"Labels root: {LABEL_ROOT}\n")

        # Load the SINGLE master label file
        label_file = os.path.join(LABEL_ROOT, f"{DATASET_NAME}.csv")
        if not os.path.exists(label_file):
            print(f"⚠️  Label file not found: {label_file}")
            exit()

        labels_df = pd.read_csv(label_file)
        print(f"=== Dataset: {DATASET_NAME} | Labels: {labels_df.shape} ===")

        # Auto-detect ID column
        label_id_col = "SMILES"


        # Filter only for explicitly requested target columns
        target_cols = [col for col in TARGET_COLUMNS if col in labels_df.columns]

        if len(target_cols) != len(TARGET_COLUMNS):
            missing = set(TARGET_COLUMNS) - set(target_cols)
            print(f"   ⚠️  Warning: Some specific targets were not found in the file: {missing}")

        if not target_cols:
            print(f"   ❌ None of the explicitly requested targets were found in {label_file}")
            exit()

        print(f"   Found {len(target_cols)} explicitly requested targets: {target_cols}")

        # Find ALL canonical_common parquet files
        emb_pattern = os.path.join(EMB_ROOT, "**/*.csv")
        emb_files = glob.glob(emb_pattern, recursive=True)

        if not emb_files:
            print(f"⚠️  No canonical_common embeddings found in {EMB_ROOT}")
            exit()

        print(f"   Found {len(emb_files)} embedding files.\n")

        for emb_path in emb_files:
            # Extract descriptor name flexibly
            filename = os.path.basename(emb_path)
            for suffix in ["_embeddings.csv"]:
                filename = filename.replace(suffix, "")

            descriptor = filename.split("_")[-1]

            print(f"   📂 Processing embeddings: {os.path.relpath(emb_path, BASE_DIR)} | Descriptor: {descriptor}")

            embeddings = pd.read_csv(emb_path)
            embeddings["SMILES"] = embeddings["SMILES"].astype(str).str.strip()
            embeddings = embeddings.set_index("SMILES").sort_index()

            for target_column in target_cols:
                # Create output directory for this descriptor
                result_dir = os.path.join(RESULT_ROOT, f"{DATASET_NAME}_{descriptor}_cv")
                os.makedirs(result_dir, exist_ok=True)

                print(f"     ▶️ Target: {target_column}")

                # Prepare aligned data
                target_labels = labels_df.dropna(subset=[target_column]).copy()
                target_labels = target_labels.drop_duplicates(subset=[label_id_col], keep="first")

                # Match IDs (string comparison)
                target_labels[label_id_col] = target_labels[label_id_col].astype(str).str.strip()
                valid_ids = list(set(target_labels[label_id_col]) & set(embeddings.index))

                if len(valid_ids) < 20:   # minimum samples threshold
                    print(f"       Skipping: only {len(valid_ids)} matching samples")
                    continue

                target_labels = target_labels[target_labels[label_id_col].isin(valid_ids)]

                X = embeddings.loc[target_labels[label_id_col]].values
                y = target_labels[target_column].values.astype(int)
                print(X,y)
                if len(np.unique(y)) < 2:
                    print(f"       Skipping: single class target")
                    continue

                for seed in SEEDS:
                    print(f"       Seed {seed} → 5-Fold CV")
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        # Handle imbalance on training fold only
                        X_train_res, y_train_res, imb_report = handle_class_imbalance(X_train, y_train)        

                        # Feature scaling
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_res)
                        X_test_scaled = scaler.transform(X_test)

                        # Train & evaluate each model
                        for model_name, base_model in MODELS.items():

                            # Set unique file name per fold/seed
                            result_file = os.path.join(result_dir, f"{descriptor}_{target_column}_{model_name}_fold{fold + 1}_seed{seed}_results.csv")

                            # Logic to skip if file exists
                            if os.path.exists(result_file):
                                continue

                            # Clone creates a truly fresh estimator instance.
                            model = clone(base_model)
                            model.set_params(random_state=seed)

                            start = time.time()
                            model.fit(X_train_scaled, y_train_res)
                            elapsed = round(time.time() - start, 4)

                            y_pred = model.predict(X_test_scaled)
                            y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

                            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                            # Create a flat dictionary for this specific model and fold
                            single_result = {
                                "Dataset": DATASET_NAME,
                                "Descriptor": descriptor,
                                "Target": target_column,
                                "Model": model_name,
                                "Seed": seed,
                                "Fold": fold + 1,
                                "Train_Size": len(X_train_res),
                                "Test_Size": len(X_test),
                                "Time(s)": elapsed,
                                "ROC_AUC": float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None,
                                "Accuracy": float(accuracy_score(y_test, y_pred)),
                                "Balanced_Acc": float(balanced_accuracy_score(y_test, y_pred)),
                                "F1": float(f1_score(y_test, y_pred, zero_division=0)),
                                "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
                                "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
                                "Kappa": float(cohen_kappa_score(y_test, y_pred)),
                                "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
                            }

                            # Save immediately as an independent file
                            df_single = pd.DataFrame([single_result])
                            df_single.to_csv(result_file, index=False)

                print(f"       ✅ Completed all seeds/folds for target: {target_column} | RAM: {get_ram_usage()}")

    print("\n🎉 All experiments completed successfully!")
    print(f"Results are saved directly to separate CSVs in: {RESULT_ROOT}")

if __name__ == "__main__":
    import sys
    cfg = make_default_config()
    if len(sys.argv) > 1:
        BASE_DIR = sys.argv[1]
        cfg["parquet_dir"] = os.path.join(BASE_DIR, "embeddings")
        cfg["label_dir"] = os.path.join(BASE_DIR, "labels")
        cfg["results_dir"] = os.path.join(BASE_DIR, "results_cv")
    if len(sys.argv) > 2:
        cfg["datasets"] = [sys.argv[2]]
    if len(sys.argv) > 3:
        cfg["label_col"] = sys.argv[3]
    benchmark(cfg)
