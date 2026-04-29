# Out-of-Distribution (OOD) Analysis

The **OOD Analysis** module evaluates the generalization capabilities of molecular embeddings by comparing performance on **Random Splits** (interpolation) versus **Scaffold Splits** (extrapolation).

### 1. Splitting Strategies
To measure how well a model handles structurally novel compounds, we use two distinct splitting methods:
* **Random Split:** A standard stratified split where molecules are assigned to train/test sets randomly. This measures "in-distribution" performance.
* **Scaffold Split (OOD):** Uses **Murcko Scaffolds** to group molecules by their core structural frameworks. Entire scaffold groups are assigned to either the training or test set, ensuring that the test set contains chemical space never seen during training.

### 2. Intelligent Imbalance Handling
The module includes a sophisticated `handle_class_imbalance` utility that automatically selects the best resampling strategy based on the **Imbalance Ratio (IR)** and minority class count:
* **Downsampling:** Used for mild to moderate imbalance.
* **SMOTE / BorderlineSMOTE:** Synthetic oversampling for datasets with sufficient minority samples.
* **EasyEnsemble:** Boosting-based ensemble for severe imbalance.
* **Scale Pos Weight:** XGBoost-specific weighting for extreme cases (IR > 200).

### Directory Structure Requirements
The OOD script expects label CSVs and precomputed embedding Parquet files to be present in the working directory:

```
WORKSPACE_DIR/
├── PCBA.csv                # Labels file with 'SMILES' and 'PCBA-XXXX' columns
└── PCBA_CDI.parquet        # Precomputed CDI embeddings
```

### Execution
Run the OOD analysis module using the `cdi` utility. You must provide the labels file and the list of embeddings to evaluate.

```bash
cdi ood --labels <LABELS_CSV> --embeddings <EMBEDDING_PARQUETS> --output <OUT_DIR>
```

**Example:** To evaluate CDI embeddings on the PCBA dataset:
```bash
cdi ood --labels PCBA.csv --embeddings PCBA_CDI.parquet --output ./results_ood
```

### Key Parameters
* `--labels`: Path to the CSV containing SMILES and task labels.
* `--embeddings`: One or more paths to precomputed embedding parquet files.
* `--output`: Directory to save the final metrics.
* `--strategy`: Resampling strategy (`DOWNSAMPLING` or `AUTO`).

### Expected Outputs
The module automatically iterates through all available tasks in the labels file (e.g., PCBA targets). It performs both Random and Scaffold splits for every task, generating a rigorous comparative performance report.

Once finished, the following files are saved in your results folder:
* **`results_ood/<TASK_NAME>/<DESCRIPTOR>_random_metrics.csv`**: In-distribution performance baseline.
* **`results_ood/<TASK_NAME>/<DESCRIPTOR>_scaffold_metrics.csv`**: Out-of-distribution (OOD) generalization performance.

**Reported Metrics:**
Each CSV file contains rigorous evaluation metrics including **ROC-AUC**, **Balanced Accuracy**, **F1**, **Precision**, **Recall**, and **Cohen's Kappa**.

