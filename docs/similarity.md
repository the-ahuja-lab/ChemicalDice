# Similarity Search (k-NN)

The ChemicalDice ecosystem includes a dedicated module for performing **Leave-One-Out Cross-Validation (LOOCV)** using distance-weighted k-Nearest Neighbors (k-NN). This tool is essential for evaluating the local chemical intuition of embeddings and identifying neighboring molecules in the latent space.

## ⚙️ How it Works
The module performs the following steps:

1.  **Alignment**: Ensures that labels and embeddings are perfectly aligned by SMILES.
2.  **Normalization**: Applies L2 normalization to features for accurate cosine similarity calculation.
3.  **LOOCV Execution**: For every molecule in the dataset, it identifies the top 20 nearest neighbors from the remaining data.
4.  **Weighted Metrics**: Calculates predictive metrics (ROC-AUC, F1, etc.) for every $k$ from 1 to 20 using inverse-distance weighting.

---

## 🛠 Pre-requisite: Generate CDI Embeddings
Before running similarity search, you must have the CDI embeddings for your dataset. You can generate these using the **CDI API**:

```bash
# Generate embeddings from your label file
cdi fetch --input ./datasets/herg/herg_karim.csv --output ./datasets/herg/CDI_embeddings.csv --canonicalize
```

---

## 🚀 Execution via Command Line

You can run the similarity search pipeline using the following syntax:

```bash
python -m ChemicalDice.experiments.similarity_search \
    --data-dir ./datasets/herg \
    --labels herg_karim.csv \
    --targets Y \
    --descriptor CDI \
    --sampling NONE \
    --output ./results/similarity
```

### Essential Arguments:
*   `--data-dir`: Directory containing your label CSV and the corresponding descriptor (CSV/Parquet).
*   `--labels`: The filename of your label file (e.g., `herg.csv`).
*   `--targets`: Space-separated list of target columns to evaluate.
*   `--descriptor`: The string to match in the descriptor filename (e.g., `CDI` or `mordred`).
*   `--sampling`: Use `DOWNSAMPLING` for highly imbalanced datasets or `NONE` (default).
*   `--output`: Directory where the three result CSVs will be saved.

---

## 📊 Output Files

The pipeline generates three distinct files in the output directory:

1.  **`metrics_k5_...csv`**: Standard baseline metrics calculated at $k=5$.
2.  **`distances_20_...csv`**: Detailed neighbor report containing the SMILES, distance, and labels for the top 20 neighbors of every query molecule.
3.  **`weighted_metrics_...csv`**: A comparative table showing how predictive performance scales as you increase $k$ from 1 to 20.

---

## 🧪 Statistical Insights
By analyzing the `weighted_metrics` output, researchers can determine the "Locality Range" of an embedding—the optimal number of neighbors required to achieve peak predictive accuracy, which serves as a proxy for the quality of the learned chemical manifold.
