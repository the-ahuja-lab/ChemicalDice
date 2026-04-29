# Benchmarking Pipeline Tutorial

This tutorial guides you through executing the native evaluation pipeline using the automated **`ChemicalDice.sota_pipeline`** suite.

## 🚀 Automated Production Execution

The fastest way to evaluate your models is using the unified `cdi` command-line interface.

### Step 1: Extraction
Generate embeddings for all molecules in your dataset across multiple SOTA models.

```bash
python -m ChemicalDice.sota_pipeline.extract_embeddings \
    --input_csv ./my_dataset.csv \
    --output_dir ./parquets
```

### Step 2: Evaluation
Run the cross-validation benchmark to generate statistical performance metrics.

```bash
cdi benchmark ./base_dir my_dataset label_column
```

---

## 📓 Interactive Exploration (Notebooks)

If you prefer to manipulate the Scikit-Learn pipelines directly or visualize decision boundaries, use the Jupyter Notebooks in the `CDI_Benchmarking/` directory.

### 1. Classification Task (`pgp_broccatelli`)
Open `CDI_Benchmarking/Classification.ipynb`. This notebook targets discrete inhibition mapping applying supervised classifiers measuring **ROC-AUC, F1-Scores, and Balanced Accuracy**.

### 2. Regression Task (`ppbr_az`)
Open `CDI_Benchmarking/Regression.ipynb`. This interpolates continuous plasma protein binding states evaluated across **$R^2$ Score, RMSE, and MAE**.

## 📊 Results Analysis
Once execution completes, navigate to your `results_cv/` directory to find comprehensive CSV reports for every model combination.
