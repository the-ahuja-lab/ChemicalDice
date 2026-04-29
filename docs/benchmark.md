# CDI Benchmark Pipelines

The ChemicalDice ecosystem provides dedicated benchmark pipelines for evaluating the performance of molecular featurizers and aggregators across classification and regression tasks. These pipelines are built around two core benchmarking modules:

### 🧩 The Featurizer Module
Evaluates CDI as a holistic representation against the six state-of-the-art specialized featurizers from which it is derived: **ChemBERTa**, **GROVER**, **ImageMol**, **Signaturizer**, **MOPAC**, and **Mordred**. This module assesses whether multimodal fusion outperforms single-modality "experts."

### ⚙️ The Aggregator Module
Benchmarks the CDI embedding against eight traditional feature-set aggregation and dimensionality reduction methods (e.g., PCA, CCA, ICA, t-SNE, and RKS). It tests whether CDI's learned manifold is more predictive than simple projection-based combinations of heterogeneous features.

---

## 🧪 Classification Benchmark

The classification pipeline handles automated imbalance detection, resampling strategies, and multi-model evaluation (including XGBoost, RandomForest, and LightGBM).

### Execution via Command Line
Use the following command to run the classification benchmark on multiple descriptor sets:

```bash
python -m ChemicalDice.experiments.classification_cdi \
    --labels ./datasets/pgp_broccatelli.csv \
    --target Y \
    --descriptors ./descriptors/CDI.csv ./descriptors/Chemberta.csv ./descriptors/mordred.csv \
    --output ./results/classification \
    --strategy auto \
    --seed 42
```

### Essential Arguments:
*   `--labels`: Path to the CSV file containing labels and molecular IDs.
*   `--target`: The name of the target column in the label file.
*   `--descriptors`: Space-separated list of paths to descriptor CSV files.
*   `--output`: Directory where results and plots will be saved.
*   `--strategy`: Imbalance handling strategy (`auto`, `smote`, `smoteenn`, `easyensemble`, `downsampling`, or `none`).
*   `--seed`: Random seed for reproducibility.

---

## 📈 Regression Benchmark

The regression pipeline includes target transformation (log or Yeo-Johnson), missing value imputation, and evaluation across standard regression metrics ($R^2$, MAE, RMSE).

### Execution via Command Line
Use the following command to run the regression benchmark:

```bash
python -m ChemicalDice.experiments.regression_cdi \
    --labels ./datasets/caco2_wang.csv \
    --target Y \
    --descriptors ./descriptors/CDI.csv ./descriptors/Grover.csv \
    --transform log \
    --output ./results/regression
```

### Essential Arguments:
*   `--labels`: Path to the CSV file containing continuous labels.
*   `--target`: The name of the target column.
*   `--descriptors`: Space-separated list of descriptor CSV paths.
*   `--transform`: Transformation to apply to the target variable (`log`, `yeo-johnson`, or `none`).
*   `--output`: Directory for results.
*   `--seed`: Random seed for reproducibility.

---

## 📊 Output Structure

Both pipelines generate a standardized output structure in the specified `--output` directory:

1.  **`results_summary.csv`**: A consolidated file containing metrics (ROC-AUC, F1, R2, etc.) for every Descriptor × Model combination.
2.  **`results/`**: Individual model performance metrics and confusion matrices.
3.  **`imbalance_report.json`**: (Classification only) Detailed diagnostics of dataset imbalance and the chosen resampling strategy.
