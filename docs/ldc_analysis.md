# Low Data Condition (LDC) Analysis

The **LDC Analysis** module evaluates the robustness and learning efficiency of molecular embeddings when training data is scarce. This is critical for drug discovery tasks where experimental labels (e.g., bioassay results) are often extremely limited.

### 1. Methodology
The module performs a rigorous benchmarking of model performance across varying training set sizes. It systematically subsets the training data while keeping the test set constant to measure how quickly a model reaches its performance plateau.

Typical training fractions evaluated:
* **10%**: Severe data scarcity (Extremne Low-Data)
* **25%**: Moderate data scarcity
* **50%**: Half-dataset evaluation
* **75%**: Near-full dataset evaluation

### 2. Experimental Setup
For each fraction, the module:
1.  **Stratified Subsetting**: Samples a fraction of the training data while preserving the original class distribution.
2.  **Cross-Validation**: Executes 5-fold cross-validation across 3 different random seeds to ensure statistical significance.
3.  **Automated Imbalance Handling**: Applies the `handle_class_imbalance` logic (Downsampling/SMOTE/etc.) specifically to the subsetted data.
4.  **Ensemble Benchmarking**: Evaluates performance across 5 SOTA classifiers (AdaBoost, XGBoost, LightGBM, ExtraTrees, GradientBoosting).

### Execution
Run the LDC analysis using the `cdi` utility:

```bash
cdi ldc --dataset <DATASET_NAME> --emb_dir <EMB_DIR> --label_dir <LABEL_DIR>
```

**Example:**
```bash
cdi ldc --dataset herg_karim --emb_dir ./embeddings --label_dir ./labels --fractions 0.1 0.25 0.5 0.75
```

### Expected Outputs
The module generates a comprehensive CSV file capturing the performance of every model at every data fraction:
* **`results_ldc/<DATASET>_<DESCRIPTOR>_ldc_results.csv`**

Key metrics recorded:
* **ROC-AUC**: Overall classification quality.
* **Balanced Accuracy**: Performance accounting for class imbalance.
* **Time (s)**: Training efficiency at different scales.
