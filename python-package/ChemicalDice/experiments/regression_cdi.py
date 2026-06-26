# -*- coding: utf-8 -*-
"""
CDI Regression Benchmark Pipeline
Standardized for integration into the ChemicalDice package.
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, KNNImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# =========================================================
# MAIN PIPELINE
# =========================================================

def run_regression_benchmark(label_file, target_col, descriptor_files, output_dir, transform='log', seed=42):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    
    print(f"🚀 Starting Regression Benchmark | Target: {target_col} | Transform: {transform}")
    
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
    
    # 3. Transformation
    pt = None
    y_orig = y.copy()
    
    if transform == 'yeo-johnson':
        pt = PowerTransformer(method='yeo-johnson')
        y = pd.Series(pt.fit_transform(y.values.reshape(-1, 1)).flatten(), index=y.index)
    elif transform == 'log':
        y = np.log1p(y)
    
    # 4. Evaluation
    results_summary = []
    
    for d_name, df in datasets.items():
        print(f"📊 Evaluating: {d_name}")
        X = df.loc[common_ids]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        
        # Preprocess
        if "mordred" in d_name.lower():
            imputer = KNNImputer()
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train Models
        models = {
            "XGBoost": XGBRegressor(random_state=seed),
            "RandomForest": RandomForestRegressor(random_state=seed),
            "ExtraTrees": ExtraTreesRegressor(random_state=seed),
            "Ridge": Ridge()
        }
        
        for m_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Inverse transform for metrics
            if transform == 'yeo-johnson':
                y_test_orig = pt.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
                y_pred_orig = pt.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            elif transform == 'log':
                y_test_orig = np.expm1(y_test)
                y_pred_orig = np.expm1(y_pred)
            else:
                y_test_orig = y_test
                y_pred_orig = y_pred
                
            results_summary.append({
                "Descriptor": d_name,
                "Model": m_name,
                "R2": r2_score(y_test_orig, y_pred_orig),
                "MAE": mean_absolute_error(y_test_orig, y_pred_orig),
                "RMSE": np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
            })
            
    # Save Results
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(os.path.join(output_dir, "regression_results.csv"), index=False)
    print(f"🎉 Benchmark complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CDI Regression Benchmark")
    parser.add_argument("--labels", required=True, help="Path to label CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--descriptors", nargs="+", required=True, help="List of descriptor CSVs")
    parser.add_argument("--output", default="./benchmark_out", help="Output directory")
    parser.add_argument("--transform", choices=['log', 'yeo-johnson', 'none'], default='log')
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    run_regression_benchmark(args.labels, args.target, args.descriptors, args.output, args.transform, args.seed)
