# Representative ChEMBL

The **Representative ChEMBL** module provides advanced density-aware sampling techniques to create high-fidelity, manageable subsets of massive chemical databases. This is used to train CDI on a representative 10% fraction of chemical space while maintaining statistical robustness.

### 1. Density-Aware Sampling Method
The core of this module implements a four-stage pipeline to select the most informative molecules:
- **Topological Mapping:** Projects ECFP6 fingerprints into a 30-dimensional UMAP space using the Jaccard metric.
- **Landscape Clustering:** HDBSCAN identifies dense core clusters (common motifs) and sparse "noise" points (rare scaffolds).
- **Log-Weighted Quotas:** Cluster budgets are assigned based on the logarithm of their size, preventing common motifs from overwhelming unique but small clusters.
- **MaxMin Diversity:** A GPU-accelerated diversity picker ensures that molecules within each cluster are as dissimilar as possible.

### Directory Structure Requirements
The clustering module operates most efficiently when provided with a dedicated workspace for large memory-mapped fingerprints and temporary caches:

```
WORKSPACE_DIR/
├── chembl_35_ecfp6.csv     # Input dataset with 'ECFP6' column
└── results/                # (Auto-generated) Output directory for subsets and reports
```

### Execution
Run the clustering module from the command line using the `cdi` utility. You must specify the input CSV and the target fraction (e.g., 0.10 for 10%).

```bash
cdi cluster --input <INPUT_CSV> --target <FRACTION> --out_dir <OUT_DIR>
```

**Example:** To generate a 10% representative subset from a ChEMBL 35 export:
```bash
cdi cluster --input chembl_35_ecfp6.csv --target 0.10 --out_dir ./results
```

### Expected Outputs
The script automatically handles the transition from high-dimensional fingerprints to low-dimensional manifolds. It leverages GPU acceleration for Tanimoto diversity picking to ensure optimal coverage.

Once finished, the following files are saved in your results folder:
- **representative_10pct_hdbscan.csv**: The final, high-fidelity subset ready for CDI training.
- **hdbscan_labels.csv**: A mapping of every molecule to its assigned cluster and membership probability.
- **cluster_report_hdbscan.csv**: A detailed statistical report showing cluster persistence, sample rates, and chemical landscape coverage.
