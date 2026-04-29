# Evaluation & Metrics

The ChemicalDice ecosystem leverages rigorous statistical frameworks to validate the fidelity of molecular embeddings across classification and regression tasks.

## 📊 Classification Metrics
For discrete property prediction (e.g., toxicity, activity), CDI performance is measured using seven key performance indicators as highlighted in the bioRxiv study:

1. **AUC-ROC**: Area Under the Receiver Operating Characteristic curve, measuring the ability to distinguish between classes.
2. **Accuracy**: The ratio of correctly predicted observations to the total observations.
3. **Balanced Accuracy**: Arithmetic mean of sensitivity and specificity, critical for imbalanced datasets.
4. **F1 Score**: Harmonic mean of Precision and Recall, providing a balance between the two.
5. **Precision**: Ratio of correctly predicted positive observations to the total predicted positives.
6. **Recall (Sensitivity)**: Ratio of correctly predicted positive observations to all observations in actual class.
7. **Cohen’s Kappa**: Statistical measure of inter-rater agreement for categorical items, accounting for chance.

---

## 📈 Regression Metrics
For continuous property prediction (e.g., solubility, binding affinity), the following metrics are utilized across 10 regression benchmarks:

1. **RMSE**: Root Mean Square Error, measuring the average magnitude of the error.
2. **MAE**: Mean Absolute Error, measuring the average absolute difference between predicted and actual values.
3. **R² (Coefficient of Determination)**: Measuring how well the regression predictions approximate the real data points.


---

## 🧪 Benchmarking Protocols

The Chemical Dice Integrator (CDI) is rigorously evaluated across diverse cheminformatics tasks using two distinct benchmarking modules described in the CDI framework:

### 🧩 The Featurizer Module
The **Featurizer module** benchmarks CDI as a holistic molecular representation against the six state-of-the-art specialized "expert" featurizers from which it is derived:
*   **ChemBERTa**: Language-based semantics.
*   **GROVER**: Graph-based topology.
*   **ImageMol**: Structural image-based features.
*   **Signaturizer**: Biological bioactivity profiles.
*   **MOPAC**: Quantum mechanical properties.
*   **Mordred**: Physicochemical descriptors.

This module evaluates if CDI's multimodal fusion can match or exceed the predictive power of these single-modality specialists.

### ⚙️ The Aggregator Module
The **Aggregator module** compares the CDI embedding against traditional feature-set aggregation and dimensionality reduction methods. In this context, **aggregation** refers to the process of projecting heterogeneous feature spaces into a common latent manifold. CDI is benchmarked against eight established techniques:
*   **Linear**: PCA, CCA, ICA.
*   **Non-Linear/Manifold**: Kernel PCA (kPCA), Isomap, Locally Linear Embedding (LLE), t-SNE.
*   **Projection-based**: Random Kitchen Sinks (RKS).

---
### The Production Pipeline
The formal evaluation process is orchestrated via the modular **`ChemicalDice.sota_pipeline`** sub-package, which handles high-dimensional feature extraction and multi-model statistical evaluation.

For interactive discovery, legacy Jupyter Notebooks are available:
* [**Classification.ipynb**](notebooks/Classification.ipynb): Evaluates independent classifiers against the `pgp_broccatelli` labels.
* [**Regression.ipynb**](notebooks/Regression.ipynb): Trains predictive regression models over the `ppbr_az` labels.

---
## 💻 Practical Evaluation Example

Users can extract 8192-D CDI features and evaluate them against ground-truth labels using the following workflows.

### 1. Fetching CDI Features
Extract the high-dimensional latent representation for your molecules.

**Via CLI:**
```bash
cdi fetch --input my_data.csv --output embeddings.csv --canonicalize
```

**Via Python:**
```python
from ChemicalDice.core.api_client import collect_features_from_csv

# Fetch 8192-D embeddings via API stream
df = collect_features_from_csv("my_data.csv", convert_to_canonical=True)
df.to_csv("embeddings.csv", index=False)
```

### 2. Evaluating with Metrics
Once features are extracted, you can evaluate model performance using standard SOTA classifiers.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score, 
                             accuracy_score, precision_score, recall_score, cohen_kappa_score)

# Load embeddings and labels
df_emb = pd.read_csv("embeddings.csv")
df_labels = pd.read_csv("labels.csv") # Assumes 'label' column exists

# Align and split
X = df_emb.drop(columns=["SMILES"]).values
y = df_labels["label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Classifier
model = XGBClassifier()
model.fit(X_train, y_train)

# Predict and Evaluate
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Calculate Metrics
metrics = {
    "ROC-AUC": roc_auc_score(y_test, y_prob),
    "Accuracy": accuracy_score(y_test, y_pred),
    "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "Cohen's Kappa": cohen_kappa_score(y_test, y_pred)
}

# Print Results
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# Save to CSV
results_df = pd.DataFrame([metrics])
results_df.to_csv("evaluation_results.csv", index=False)
```

