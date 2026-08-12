# Chemical Dice Integrator (CDI) — Complete Technical Documentation

> **Framework**: Chemical Dice Integrator (CDI)  
> **Repository**: [github.com/the-ahuja-lab/ChemicalDice](https://github.com/the-ahuja-lab/ChemicalDice)  
> **Documentation**: [the-ahuja-lab.github.io/ChemicalDice/](https://the-ahuja-lab.github.io/ChemicalDice/)  
> **License**: MIT  
> **Total Parameter Count**: ~1.09 Billion  

---

## Table of Contents

1. [Overview](#1-overview)
2. [Six Orthogonal Molecular Modalities](#2-six-orthogonal-molecular-modalities)
3. [Architecture: CDI-Basic (Multimodal Fusion Engine)](#3-architecture-cdi-basic-multimodal-fusion-engine)
4. [Architecture: CDI-Generalised (Sequence-to-Embedding Mapping)](#4-architecture-cdi-generalised-sequence-to-embedding-mapping)
5. [Mathematical Formulation & Loss Functions](#5-mathematical-formulation--loss-functions)
6. [Code Architecture: File-by-File Breakdown](#6-code-architecture-file-by-file-breakdown)
7. [Descriptor Calculation Pipeline](#7-descriptor-calculation-pipeline)
8. [Data Preprocessing Methods](#8-data-preprocessing-methods)
9. [Fusion Methods & Dimensionality Reduction](#9-fusion-methods--dimensionality-reduction)
10. [Machine Learning Evaluation Framework](#10-machine-learning-evaluation-framework)
11. [Model Training (Grover-based)](#11-model-training-grover-based)
12. [Cross-Validation & Scaffold Splitting](#12-cross-validation--scaffold-splitting)
13. [Metrics & Evaluation](#13-metrics--evaluation)
14. [Analysis Utilities](#14-analysis-utilities)
15. [Fine-Tuning & Downstream Tasks](#15-fine-tuning--downstream-tasks)
16. [R and Python Package Interfaces](#16-r-and-python-package-interfaces)
17. [Summary of Technical Specifications](#17-summary-of-technical-specifications)

---

## 1. Overview

CDI is a **hierarchical, multimodal deep learning framework** that performs unsupervised integration of five to six complementary molecular embeddings into a single, high-information latent space (8192-dimensional by default). The framework is designed for large-scale cheminformatics, bioinformatics, and AI-driven molecular discovery.

**Core Pipeline:**
1. Input SMILES → Canonicalization → Six parallel descriptor generators → Preprocessing → Feature Fusion → Downstream ML

**Two operational modes:**
- **CDI-Basic**: Full multimodal fusion using a two-tiered autoencoder (requires all six descriptors)
- **CDI-Generalised**: Direct SMILES-to-embedding via a Mamba State-Space Model (SSM) bypassing per-descriptor computation

---

## 2. Six Orthogonal Molecular Modalities

CDI integrates the following six molecular representations:

| # | Modality | Method | Dimensionality | Source File | Description |
|---|----------|--------|---------------|-------------|-------------|
| 1 | **Quantum-Mechanical** | MOPAC | ~2,966 features | `quantum.py` | Electronic properties, orbital landscapes, heat of formation, HOMO/LUMO, dipole moment, etc. |
| 2 | **Topological / Graph-Based** | GROVER | ~3,072 features | `Grover.py`, `models.py` | A multi-view graph neural network (GTransEncoder) with self-attention message passing producing atom and bond embeddings |
| 3 | **Linguistic / Language-Based** | ChemBERTa | ~768 features | `chemberta.py` | Roberta-based transformer (DeepChem/ChemBERTa-77M-MLM) extracting SMILES syntax semantics |
| 4 | **Biological / Bioactivity** | Signaturizer | ~10,000 features | `bioactivity.py` | Pre-trained bioactivity signatures from the Signaturizer 'GLOBAL' model covering ~25,000 bioactivity profiles |
| 5 | **Visual / Image-Based** | ImageMol | ~512 features | `ImageMol.py` | ResNet18-based CNN encoding 2D molecular images into topological representations |
| 6 | **Physicochemical** | Mordred | ~1,613 features | `chemical.py` | Deterministic mathematical descriptors (1D, 2D, 3D) covering geometry, topology, and chemistry |

The five core modalities (excluding one, depending on configuration) are fused by CDI-Basic. A sixth is used when all are available.

### 2.1 Quantum Descriptors (MOPAC)

**File**: `quantum.py`

Uses the MOPAC semi-empirical quantum chemistry package to compute:
- **Heat of formation** (ΔHf)
- **HOMO / LUMO energies** and **HOMO-LUMO gap**
- **Ionization potential**
- **Total energy**, **electronic energy**, **core-core repulsion**
- **Dipole moment** (x, y, z components and magnitude)
- **Cosmic area** and **volume** (3D molecular properties)
- **Polarizability**
- Fully populated MOPAC output parsed via `quantum_need.py`

*Prerequisite*: MOPAC v22.1.1 and 3DMorse compiled separately.

### 2.2 Graph Embeddings (GROVER)

**File**: `Grover.py`, `models.py`, `layers.py`

GROVER (Graph Representation frOm self-superVised mEssage passing tRansformer) provides:
- **GTransEncoder** with multi-headed self-attention message passing
- **Dual encoding**: atom-wise and bond-wise views
- **Three self-supervised pre-training tasks**:
  - **Atom Vocabulary Prediction (AV)**: Predict contextual atom types from masked atom embeddings
  - **Bond Vocabulary Prediction (BV)**: Predict contextual bond types from masked bond embeddings
  - **Functional Group Prediction (FG)**: Predict 85 functional group labels from graph-level readouts
- Pre-trained on **~40M unlabeled molecules** (ZINC15 subset)
- Produces 3,072-dimensional fingerprint from concatenated atom-from-atom, atom-from-bond, bond-from-atom, bond-from-bond readouts

### 2.3 Language Model Embeddings (ChemBERTa)

**File**: `chemberta.py`

Uses `DeepChem/ChemBERTa-77M-MLM` — a **RobertaModel** pre-trained on ~77M SMILES:
- Tokenization: SMILES strings tokenized with RoBERTa tokenizer
- Embedding: Mean pooling of the last hidden state across all tokens
- Output dimension: **768** (hidden_size of ChemBERTa-77M-MLM)
- Column prefix: `ChB77MLM_`

### 2.4 Bioactivity Signatures (Signaturizer)

**File**: `bioactivity.py`

Uses the **Signaturizer** package's `'GLOBAL'` model:
- Predicts bioactivity signatures across **25,000+ bioactivity profiles**
- Based on the chemical signature approach (extended-connectivity fingerprints + neural network)
- Output: 10,000-dimensional signature vector per molecule
- Column prefix: `Sign_`

### 2.5 Image Embeddings (ImageMol)

**File**: `ImageMol.py`, `cnn_model_utils.py`

ImageMol encodes 2D molecular drawings via a **pretrained ResNet18**:
1. SMILES → RDKit Mol → 2D depiction (224×224 PNG)
2. ResNet18 pretrained on ImageNet → 512-dimensional feature vector
3. Publicly available pretrained checkpoint: `ImageMol.pth.tar`

### 2.6 Physicochemical Descriptors (Mordred)

**File**: `chemical.py`

Uses the **Mordred** molecular descriptor calculator:
- **1,613 descriptors** covering:
  - Constitutional descriptors (atom counts, bond counts, molecular weight)
  - Topological indices (Zagreb, BalabanJ, Wiener, etc.)
  - Connectivity indices (Chi, Kappa)
  - Electronic descriptors (E-state, VSA)
  - Geometrical descriptors (3D shape, diameter, radius)
  - Hybrid descriptors
- 3D descriptors require SDF files (generated from SMILES) from `smiles_preprocess.py`

---

## 3. Architecture: CDI-Basic (Multimodal Fusion Engine)

### 3.1 Tier 1: Semantic Commonality Autoencoders (SCA)

**Files**: `arch.py`, `architecture.py`, `getEmbeddings.py`

For each modality *j* (j = 1..6), one autoencoder is constructed:

```
Input:  concat(all other 5 modalities' features)
Target: reconstruct the j-th modality (Leave-One-Out)
```

**Autoencoder structure** (class `Autoencoder`):
- Encoder: Sequence of `Linear → ReLU` layers progressively reducing dimension
- Decoder: Mirroring encoder, `Linear → ReLU` layers expanding back to original dimension
- Latent bottleneck at the smallest dimension in the `dims` list
- Weight initialization: Xavier uniform for weights, constant 0 for biases

**Dimension reduction** (`getAEDimensions`):
```
Given inp_dim (sum of other 5 modalities) → latent_space_dim (j-th modality dimension)
Reduce by factor k (default k=7 or k=3 depending on file) each step:
  this_dim = ceil(this_dim / k)
  Repeat until this_dim ≤ latent_space_dim
```

**Learnable modality weighting**:
```python
# choice=1: per-dimension scaling weights
self.weights = nn.ParameterList([
    nn.Parameter(torch.ones(1, latent_space_dims[i])) 
    for i in range(len(latent_space_dims))
])
```

### 3.2 Tier 2: Super-Embedding Autoencoder (SEA)

After all six SCAs produce their latent encodings `enc[i]`:

```
concat_key = concat(enc[0], enc[1], ..., enc[5])   # concatenated 6 latent spaces
sea_output = Autoencoder(concat_key).encode(concat_key)  # final 8192-D super-embedding
sea_reconstruction = Autoencoder(concat_key)(concat_key) # reconstruction for loss
```

### 3.3 Complete Forward Pass

**(from `arch.py` and `architecture.py`)**

```python
def forward(self, x):
    # 1. Apply learned weights to each modality
    for i in range(len(x)):
        x[i] = x[i] * self.weights[i]
    
    # 2. For each modality j, concatenate all other 5 modalities
    inp = [concat(remove_element_at_index(x, j)) for j in range(6)]
    
    # 3. Encode each LOO input and reconstruct
    for j in range(6):
        enc[j] = encoders[j].encode(inp[j])
        op[j]  = encoders[j](inp[j])  # full AE (encode then decode)
    
    # 4. Tier 2: Super-embedding
    concat_key = concat(enc[0..5])
    concat_key_enc = encoders[6].encode(concat_key)  # 8192-D final embedding
    concat_key_op  = encoders[6](concat_key)          # reconstruction
    
    return enc[0..5], op[0..5], concat_key_enc, concat_key_op
```

---

## 4. Architecture: CDI-Generalised (Sequence-to-Embedding Mapping)

CDI-Generalised replaces the descriptor-computation bottleneck with a **Mamba State-Space Model (SSM)** that maps raw SMILES strings directly to the 8192-D CDI latent space.

**Key properties:**
- **Architecture**: Mamba SSM (SMI-SSED framework)
- **Training**: Supervised regression against CDI-Basic "gold-standard" embeddings
- **Loss**: MSE + angular alignment (cosine similarity) between predicted and target embeddings
- **Inference**: Single model forward pass from SMILES → 8192-D embedding — no MOPAC, Mordred, or other descriptor computation needed
- **Scalability**: Enables high-throughput virtual screening

**SSM objective** (from docs/architecture.md):

$$\mathcal{L}_{SSM} = \frac{1}{N \times D} \sum_{i=1}^{N} \sum_{j=1}^{D} (E_{target, ij} - E_{pred, ij})^2$$

Where:
- $N$ = batch size
- $D$ = 8192 (embedding dimension)
- $E_{target}$ = CDI-Basic gold-standard embeddings
- $E_{pred}$ = Mamba-predicted embeddings

---

## 5. Mathematical Formulation & Loss Functions

### 5.1 CDI-Basic Training Loss

From `getEmbeddings.py`:

```python
# Three components weighted by alpha, beta, gamma (all = 0.33)
total_loss = (α × total_encoding_loss / 6) 
           + (β × total_reconstruction_loss / 6) 
           + (γ × reconstruction_loss_concat)
```

**Encoding loss** (Tier 1 — semantic alignment):
$$\mathcal{L}_{enc} = \frac{1}{6} \sum_{j=1}^{6} \text{MSE}(\text{enc}_j, \mathbf{a}_j)$$

where $\text{enc}_j$ is the latent encoding from the j-th SCA (trained to reconstruct modality *j*), and $\mathbf{a}_j$ is the original j-th modality input.

**Reconstruction loss** (Tier 1 — cross-modal reconstruction):
$$\mathcal{L}_{recon} = \frac{1}{6} \sum_{j=1}^{6} \text{MSE}(\text{AE}_j(\text{concat}_{\neq j}), \text{concat}_{\neq j})$$

where $\text{AE}_j$ is the j-th autoencoder that reconstructs the concatenation of all modalities except *j*.

**Super-embedding loss** (Tier 2):
$$\mathcal{L}_{SEA} = \text{MSE}(\text{concat\_key}, \text{concat\_reconstruction})$$

**Total CDI-Basic loss** (from docs/architecture.md):

$$\mathcal{L}_{total} = \frac{1}{6} \sum_{j=1}^{6} \left( \mathcal{L}_{RE}(\mathbf{a}_j^i, \tilde{\mathbf{a}}_j^i) + \mathcal{L}_{MSE}(\mathbf{p}_j^i, \mathbf{d}_j^i) \right) + \mathcal{L}_{SEA}$$

### 5.2 Optimizer and Scheduler

- **Optimizer**: SGD with learning rate = 0.5, weight_decay = 0
- **Scheduler**: `ReduceLROnPlateau` (patience = 5)
- **Loss function**: MSELoss (nn.MSELoss)

### 5.3 Fine-Tuning Loss

For downstream task adaptation (`FineTuneChemicalDiceIntegrator`):
```python
# Freeze CDI, train additional autoencoder on top
self.finetuner = Autoencoder([8000, ..., user_embed_dim, ..., 8000])
```

---

## 6. Code Architecture: File-by-File Breakdown

### Core Architecture Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `arch.py` | CDI-Basic model architecture | `Autoencoder`, `ChemicalDiceIntegrator`, `FineTuneChemicalDiceIntegrator`, `Classifier` |
| `architecture.py` | Alternative CDI-Basic (different default k) | Same class names, k=7 default vs k=3 |
| `models.py` | GROVER model definitions | `GROVEREmbedding`, `GroverTask`, `GroverFpGeneration`, `GroverFinetuneTask`, `AtomVocabPrediction`, `BondVocabPrediction`, `FunctionalGroupPrediction` |
| `layers.py` | Neural network building blocks | `MPNEncoder`, `GTransEncoder`, `MTBlock`, `MultiHeadedAttention`, `Head`, `Readout`, `SelfAttention`, `PositionwiseFeedForward` |

### Data Handling Files

| File | Purpose |
|------|---------|
| `baseData.py` | `baseData` class — data loading, preprocessing, fusion for non-label data |
| `fusionData.py` | `fusionData` class — complete data fusion pipeline with labels, evaluation, cross-validation |
| `predData.py` | `fusionData` for prediction (no labels) |
| `preprocess_data.py` | `clear_and_process_data()`, `normalize_to_constant_sum()`, `show_dataframe_info()`, missing value checks |
| `saving_data.py` | Train/test split saving for n-fold and scaffold splitting |
| `split_data.py` | Data splitting utilities |
| `splitting.py` / `splitter.py` | Scaffold splitting implementations |

### Descriptor Generation Files

| File | Modality | Method |
|------|----------|--------|
| `quantum.py` | Quantum | MOPAC semi-empirical calculations |
| `chemical.py` | Physicochemical | Mordred descriptor calculator |
| `chemberta.py` | Language | ChemBERTa-77M-MLM transformer |
| `Grover.py` | Graph | GROVER GNN fingerprint |
| `ImageMol.py` | Image | ResNet18 CNN on 2D depictions |
| `bioactivity.py` | Bioactivity | Signaturizer global model |
| `fingerprint.py` | Fingerprint generation | GROVER fingerprint extraction pipeline |
| `smiles_preprocess.py` | SMILES preprocessing | Mol2/SDF creation, canonicalization |

### Training and Evaluation Files

| File | Purpose |
|------|---------|
| `train.py` | `run_training()`, `train()` — GROVER finetuning |
| `pretrain.py` | GROVER self-supervised pretraining |
| `getEmbeddings.py` | CDI autoencoder training (`trainAE_8192`), embedding extraction |
| `evaluate.py` | Model evaluation |
| `predict.py` | Prediction on new data |
| `cross_validate.py` | Cross-validation setup |
| `run_evaluation.py` | Evaluation pipeline |
| `metrics.py` | `get_metric_func()`, `accuracy`, `rmse`, `sensitivity`, etc. |

### Utility Files

| File | Purpose |
|------|---------|
| `utils.py` | General utilities: data loading, scaffold splitting, checkpointing, model building |
| `nn_utils.py` | `initialize_weights()`, `get_activation_function()`, `param_count()`, `select_neighbor_and_aggregate()` |
| `parsing.py` | Argument parsing for GROVER |
| `torchvocab.py` | Vocabulary construction for GROVER pretraining |
| `build_vocab.py` | Vocabulary builder |

### Visualization and Analysis

| File | Purpose |
|------|---------|
| `plot_data.py` | Bar plots, box plots, model comparison plots |
| `analyse_data.py` | `DynamicAutoencoder`, `ccafuse()`, linear/nonlinear analysis functions, CP tensor decomposition |
| `myImports.py` | Centralized imports |

### Configuration

| File | Purpose |
|------|---------|
| `myTrainParams.py` | Training parameter configuration |
| `mytrainparams_new.py` | Updated training parameters |
| `__main__.py` | CLI entry point: `calculate`, `train`, `predict` modes |

---

## 7. Descriptor Calculation Pipeline

The full pipeline is orchestrated in `__main__.py` and uses files in the `training/Chemical_Dice_Integrator_scripts/ChemicalDice/` directory.

### 7.1 SMILES Preprocessing (`smiles_preprocess.py`)

1. **Canonicalization** (`add_canonical_smiles`): Convert SMILES to RDKit canonical form
2. **Mol2 file generation** (`create_mol2_files`): SMILES → 3D conformer → Tripos Mol2 (for MOPAC)
3. **SDF file generation** (`create_sdf_files`): SMILES → 3D conformer → SDF (for Mordred)

### 7.2 Descriptor Computation Order (for `calculate` mode)

```
1. MOPAC quantum descriptors (quantum.py)
2. Mordred physicochemical descriptors (chemical.py)
3. ChemBERTa language embeddings (chemberta.py)
4. ImageMol image embeddings (ImageMol.py)
5. GROVER graph embeddings (Grover.py)
6. Signaturizer bioactivity signatures (bioactivity.py)
```

Each step checks for existing output files before recomputing.

---

## 8. Data Preprocessing Methods

All methods are available in `fusionData.py` and `baseData.py`.

### 8.1 Data Loading (`clear_and_process_data`)

- Reads CSV files with ID column as index
- Automatically identifies separator (tab for ImageMol, comma for others)
- Drops SMILES column from feature DataFrames (stored separately)

### 8.2 Common Sample Alignment (`keep_common_samples`)

Intersects row indices across all data modalities to ensure consistent sample sets:
```python
common_indices = ∩(index_of_df_0, index_of_df_1, ..., index_of_df_n)
```

### 8.3 Missing Value Handling

**`ShowMissingValues()`**: Prints per-dataframe missing value counts.

**`remove_empty_features(threshold=100)`**: Removes columns with > threshold% missing values.
For Mordred specifically, a hard-coded list of ~700 valid descriptor columns is used.

**`ImputeData(method)`**: Five methods available:
| Method | Implementation | Notes |
|--------|---------------|-------|
| `knn` | `KNNImputer(n_neighbors=5)` | Default; also `class_specific` mode with per-class imputation |
| `mean` | `SimpleImputer(strategy='mean')` | Replaces NaN with column mean |
| `median` | `SimpleImputer(strategy='median')` | Replaces NaN with column median |
| `most_frequent` | `SimpleImputer(strategy='most_frequent')` | Mode imputation |
| `interpolate` | `df.interpolate(method='linear')` | Linear interpolation |

### 8.4 Scaling Methods (`scale_data`)

| Type | Formula | Implementation |
|------|---------|---------------|
| `standardize` | $x' = \frac{x - \mu}{\sigma}$ | `StandardScaler` |
| `minmax` | $x' = \frac{x - \min(x)}{\max(x) - \min(x)}$ | `MinMaxScaler` |
| `robust` | $x' = \frac{x - \text{median}(x)}{\text{IQR}(x)}$ | `RobustScaler` |
| `pareto` | $x' = \frac{x}{\sqrt{\sigma}}$ | Manual |

### 8.5 Normalization Methods (`normalize_data`)

| Type | Formula | Notes |
|------|---------|-------|
| `constant_sum` | $x_i' = \frac{x_i}{\sum x_i} \times C$ | Default C=1, axis=1 (rows) |
| `L1` | $x_i' = \frac{x_i}{\sum |x_i|}$ | scikit-learn `normalize(norm='l1')` |
| `L2` | $x_i' = \frac{x_i}{\sqrt{\sum x_i^2}}$ | scikit-learn `normalize(norm='l2')` |
| `max` | $x_i' = \frac{x_i}{\max(|x|)}$ | scikit-learn `normalize(norm='max')` |

### 8.6 Transformation Methods (`transform_data`)

| Type | Operation | Notes |
|------|-----------|-------|
| `cubicroot` | $\sqrt[3]{x}$ | `np.cbrt` |
| `log10` | $\log_{10}(x)$ | `np.log10` |
| `log` | $\ln(x)$ | `np.log` |
| `log2` | $\log_2(x)$ | `np.log2` |
| `sqrt` | $\sqrt{x}$ | `np.sqrt` |
| `powertransformer` | Yeo-Johnson or Box-Cox | `PowerTransformer(method='yeo-johnson' or 'box-cox')` |
| `quantiletransformer` | Map to uniform or normal | `QuantileTransformer(output_distribution='uniform' or 'normal')` |

---

## 9. Fusion Methods & Dimensionality Reduction

All fusion methods are implemented in `fusionData.fuseFeatures()` and executed by `analyse_data.py`.

### 9.1 Linear Methods

#### PCA (Principal Component Analysis)
- **Implementation**: `sklearn.decomposition.PCA`
- **Equation**: $\mathbf{X} = \mathbf{T}\mathbf{P}^\top + \mathbf{E}$, where $\mathbf{T}$ are scores and $\mathbf{P}$ loadings
- **Goal**: Maximize variance in projected space

#### ICA (Independent Component Analysis)
- **Implementation**: `sklearn.decomposition.FastICA`
- **Goal**: Find statistically independent sources

#### IPCA (Incremental PCA)
- **Implementation**: `sklearn.decomposition.IncrementalPCA`
- **Use case**: Out-of-core / large dataset PCA

#### CCA (Canonical Correlation Analysis)
- **Implementation**: `analyse_data.ccafuse()`
- **Two-step**: PCA(50) → CCA → concatenation or summation mode
- **Equation**: Find $\mathbf{w}_x, \mathbf{w}_y$ maximizing $\rho = \text{corr}(\mathbf{X}\mathbf{w}_x, \mathbf{Y}\mathbf{w}_y)$
- Applied pairwise for all combinations of modalities → concatenated

#### PLS-DA (Partial Least Squares Discriminant Analysis)
- **Implementation**: `sklearn.cross_decomposition.PLSRegression`
- **Supervised**: Uses label data to maximize covariance between features and labels
- $\mathbf{X} = \mathbf{T}\mathbf{P}^\top + \mathbf{E}$, $\mathbf{Y} = \mathbf{U}\mathbf{Q}^\top + \mathbf{F}$, with $\mathbf{T}$ and $\mathbf{U}$ maximally covarying

### 9.2 Nonlinear Methods

#### t-SNE (t-distributed Stochastic Neighbor Embedding)
- **Implementation**: `sklearn.manifold.TSNE`
- **Default**: n_components=3
- **Use**: Visualization / exploratory analysis

#### Kernel PCA
- **Implementation**: `sklearn.decomposition.KernelPCA(kernel='linear')`
- Maps data to higher-dimensional space via kernel trick

#### Isomap
- **Implementation**: `sklearn.manifold.Isomap(n_neighbors=5)`
- Preserves geodesic distances

#### LLE (Locally Linear Embedding)
- **Implementation**: `sklearn.manifold.LocallyLinearEmbedding`
- Preserves local linear reconstruction weights

#### Spectral Embedding (SEM)
- **Implementation**: `sklearn.manifold.SpectralEmbedding`
- Uses graph Laplacian eigenvalues

#### Random Kitchen Sinks (RKS)
- **Implementation**: `sklearn.kernel_approximation.RBFSampler`
- Random Fourier features for kernel approximation

#### Autoencoder (AE)
- **Implementation**: `analyse_data.DynamicAutoencoder` + `apply_analysis_nonlinear3`
- Architecture: `input → [128, 64, 36, 18] → latent → symmetric reconstruction`
- Optimizer: Adam, lr=0.001, epochs=20

#### CDI (Chemical Dice Integrator)
- **Implementation**: `getEmbeddings.AutoencoderReconstructor_training_8192` / `_other` / `_single`
- Full two-tiered autoencoder as described in Section 3
- **Default**: 8192-D output (or custom dimension list)
- Training: SGD with lr=0.5, ReduceLROnPlateau, configurable epochs and k-factors

#### Tensor Decomposition (CP/PARAFAC)
- **Implementation**: `tensorly.decomposition.parafac`
- **Preprocessing**: SelectKBest (top 100 features per modality)
- Decomposes $N$-way tensor into rank-$R$ components

---

## 10. Machine Learning Evaluation Framework

### 10.1 Cross-Validation Evaluation (`evaluate_fusion_models_nfold`)

**Workflow**:
1. For each fused data file in `ChemicalDice_fusedData/`
2. Perform K-Fold CV (default 10 folds)
3. For each fold: Train → Predict → Record metrics
4. Save per-fold metrics to `{folds}_fold_CV_results/`

**Classification models** (11 models):
- Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes, KNN, MLP, QDA, AdaBoost, Extra Trees, XGBoost

**Regression models** (13 models):
- Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, SVR, K Neighbors, MLP, Gaussian Process, Kernel Ridge

### 10.2 Scaffold Split Evaluation (`evaluate_fusion_models_scaffold_split`)

**Three split types**:
1. `random`: Random scaffold split
2. `balanced`: Balanced scaffold split
3. `simple`: Simple scaffold split

**Workflow**:
1. Grid search over model hyperparameters
2. Train/validation split → select best hyperparameters per model
3. Evaluate on held-out test set
4. Results saved to `scaffold_split_results/`

**Hyperparameter tuning** via `sklearn.model_selection.ParameterGrid` — extensive grids defined for each model.

### 10.3 Metrics Collection (`get_accuracy_metrics`)

Aggregates all CSV results, computes top-performing methods, and generates comparison bar plots.

---

## 11. Model Training (Grover-based)

### 11.1 Pretraining (`pretrain.py`, `GroverTask`)

Three self-supervised tasks:
1. **Atom Vocabulary (AV)**: NLLLoss on atom-type prediction from context
2. **Bond Vocabulary (BV)**: NLLLoss on bond-type prediction from context  
3. **Functional Group (FG)**: BCEWithLogitsLoss on 85 functional group labels

**Total loss**:
$$\mathcal{L} = \mathcal{L}_{AV} + \mathcal{L}_{BV} + \mathcal{L}_{FG} + \lambda(\mathcal{L}_{AV}^{dist} + \mathcal{L}_{BV}^{dist}) + \mathcal{L}_{FG}^{dist}$$

Where $\mathcal{L}^{dist}$ are disagreement (MSE) losses between atom-from-atom and atom-from-bond branches.

### 11.2 Finetuning (`train.py`, `GroverFinetuneTask`)

- **Architecture**: Pretrained GROVEREmbedding → two FFN heads (atom-from-atom, atom-from-bond)
- **Loss**: BCEWithLogitsLoss (classification) or MSELoss (regression) + disagreement penalty
- **Optimizer**: Adam with separate learning rates for GROVER backbone ($\text{lr} \times \text{fine\_tune\_coff}$) and FFN head ($\text{lr}$)
- **Scheduler**: NoamLR (warmup + decay)
- **Early stopping**: Based on validation loss or score

### 11.3 CDI Embedding Training (`getEmbeddings.py`)

**`trainAE_8192` function**:
- Optimizer: SGD (lr=0.5)
- Loss: MSE
- Scheduler: ReduceLROnPlateau (patience=5)
- Epochs: Configurable (typical 500-2000)
- Learns six SCA autoencoders + one SEA autoencoder simultaneously

**`finetune_AE` function** for custom dimensions:
- Loads pretrained 8192-D CDI model
- Freezes CDI weights
- Trains additional finetuning autoencoder for target embedding size

---

## 12. Cross-Validation & Scaffold Splitting

### 12.1 K-Fold Cross-Validation

Implemented in `fusionData.evaluate_fusion_models_nfold()`:
- `sklearn.model_selection.KFold(n_splits=folds, shuffle=True, random_state=42)`
- For PLS-DA and tensor decompose methods: re-computes fusion on training fold only to prevent data leakage
- For other methods: pre-computed fused data is split directly

### 12.2 Scaffold Splitting (`splitting.py`)

Three implementations:
1. **Random scaffold split** — shuffles scaffolds randomly
2. **Balanced scaffold split** — ensures similar scaffold sizes across splits
3. **Simple scaffold split** — splits by Bemis-Murcko scaffold

Based on RDKit `MurckoScaffold` computation, ensuring no scaffold overlap between train/val/test sets.

---

## 13. Metrics & Evaluation

### Classification Metrics (from `metrics.py`)

| Metric | Implementation | Formula |
|--------|---------------|---------|
| AUC-ROC | `sklearn.metrics.roc_auc_score` | Area under ROC curve |
| Accuracy | Custom (`accuracy`) | $\frac{TP + TN}{TP + TN + FP + FN}$ |
| Precision | `sklearn.metrics.precision_score` | $\frac{TP}{TP + FP}$ |
| Recall / Sensitivity | Custom (`recall`, `sensitivity`) | $\frac{TP}{TP + FN}$ |
| Specificity | Custom (`specificity`) | $\frac{TN}{TN + FP}$ |
| F1 Score | `sklearn.metrics.f1_score` | $2 \times \frac{precision \times recall}{precision + recall}$ |
| Balanced Accuracy | `sklearn.metrics.balanced_accuracy_score` | $\frac{sensitivity + specificity}{2}$ |
| MCC | `sklearn.metrics.matthews_corrcoef` | Matthews Correlation Coefficient |
| Cohen's Kappa | `sklearn.metrics.cohen_kappa_score` | $\kappa = \frac{p_o - p_e}{1 - p_e}$ |
| PRC-AUC | Custom (`prc_auc`) | Area under Precision-Recall curve |

### Regression Metrics

| Metric | Implementation | Formula |
|--------|---------------|---------|
| R² Score | `sklearn.metrics.r2_score` | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ |
| MSE | `sklearn.metrics.mean_squared_error` | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ |
| RMSE | Custom (`rmse`) | $\sqrt{\text{MSE}}$ |
| MAE | `sklearn.metrics.mean_absolute_error` | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ |

### Default Selection
- **Classification**: Models ranked by AUC-ROC
- **Regression**: Models ranked by R² Score

---

## 14. Analysis Utilities

### 14.1 Normality Check (`barplot_normality_check`)
- Uses D'Agostino's K² test (scipy.stats.normaltest)
- Plots p-values per column with α=0.05 significance line

### 14.2 Missing Value Visualization (`plot_missing_values`)
- Stacked bar charts showing missing vs non-missing values per column

### 14.3 Tensor Decomposition (`apply_analysis_nonlinear4`)
- CP decomposition via TensorLy `parafac`
- Treats modalities as tensor dimensions, extracts latent factors

### 14.4 CCA Fusion (`ccafuse`)
- Two-stage: PCA down-projection → CCA → concatenation or summation
- Applied to all pairwise modality combinations

### 14.5 Dynamic Autoencoder (`DynamicAutoencoder`)
- Flexible architecture: input → [hidden_sizes] → latent → [hidden_sizes reversed] → input
- Used for non-linear feature fusion in `apply_analysis_nonlinear3`

---

## 15. Fine-Tuning & Downstream Tasks

### 15.1 FineTuneChemicalDiceIntegrator

**Architecture** (`arch.py` / `architecture.py`):
```python
class FineTuneChemicalDiceIntegrator(nn.Module):
    def __init__(self, CDI, user_embed_dim=128, default_embed_dim=8000):
        # Freeze CDI
        for param in CDI.parameters():
            param.requires_grad = False
        # Add finetuning autoencoder: 8000 → ... → user_embed_dim → ... → 8000
```

**Pipeline**:
1. CDI-Basic produces 8192-D embedding
2. Finetuning autoencoder compresses to `user_embed_dim` (e.g., 128)
3. Optionally, different finetuning autoencoders for different tasks

### 15.2 Classifier Module

```python
class Classifier(nn.Module):
    def __init__(self):
        self.classifier = nn.Sequential(
            Linear(2048, 512) → ReLU → 
            Linear(512, 128) → ReLU → 
            Linear(128, 32) → ReLU → 
            Linear(32, 2)
        )
```

### 15.3 Downstream Embedding Extraction

Three embedding extraction modes (from `getEmbeddings.py`):

| Mode | Function | Output | Description |
|------|----------|--------|-------------|
| 8192-D | `AutoencoderReconstructor_training_8192` | 8192-D | Full CDI super-embedding |
| Custom D | `AutoencoderReconstructor_training_other` | user-specified D | CDI → finetune AE → user dim |
| Single D | `AutoencoderReconstructor_training_single` | user-specified D | CDI trained directly to target dim |

---

## 16. R and Python Package Interfaces

### 16.1 Python Package (`ChemicalDice`)

**Installation**:
```bash
pip install ChemicalDice
```

**Usage**:
```python
from ChemicalDice import smiles_to_embeddings
embeddings = smiles_to_embeddings.collect_features_from_csv(
    filepath="smiles.csv",
    convert_to_canonical=False
)
```

**Architecture**:
- `python-package/ChemicalDice/smiles_to_embeddings.py` — main API
- `python-package/ChemicalDice/core/api_client.py` — API client
- `python-package/ChemicalDice/descriptors/` — local descriptor computation (mirrors training scripts)
- `python-package/ChemicalDice/experiments/` — experiment pipelines
- `python-package/ChemicalDice/sota_pipeline/` — state-of-the-art comparison pipeline

### 16.2 R Package

**Installation**:
```r
remotes::install_github("the-ahuja-lab/ChemicalDice", subdir = "R-package")
```

**Usage**:
```r
library(ChemicalDice)
embeddings <- collect_features_from_csv(filepath="smiles.csv")
```

### 16.3 CDI Bot

A containerized, LLM-powered web application providing natural-language interface to CDI, available on Docker.

---

## 17. Summary of Technical Specifications

### CDI-Basic

| Parameter | Value |
|-----------|-------|
| Architecture | Two-Tiered Hierarchical Autoencoder |
| Tier 1 | 6 Semantic Commonality Autoencoders (SCA) |
| Tier 2 | 1 Super-Embedding Autoencoder (SEA) |
| Default embedding dimension | 8192 |
| Learnable parameters | ~1.09B (total framework) |
| Compression factor (k) | 7 (per `architecture.py`) or 3 (per `arch.py`) |
| Training optimizer | SGD (lr=0.5) |
| Loss function | Weighted MSE (encoding + reconstruction + concat) |
| Learning weights | Per-dimension learnable scaling |

### CDI-Generalised

| Parameter | Value |
|-----------|-------|
| Architecture | Mamba State-Space Model (SMI-SSED) |
| Input | Raw SMILES strings |
| Output dimension | 8192 |
| Training objective | MSE + angular alignment |
| Inference type | Single model pass (no descriptor pre-computation) |

### Modality Dimensionalities

| Modality | Raw Dimension | After Preprocessing |
|----------|--------------|-------------------|
| MOPAC | ~2,966 | ~2,966 |
| GROVER | 3,072 | 3,072 |
| ChemBERTa | 768 | 768 |
| Signaturizer | 10,000 | 10,000 |
| ImageMol | 512 | 512 |
| Mordred | 1,613 | variable (~700 after filtering) |

### Supported Fusion Methods (14 total)

| Method | Type | Supervised |
|--------|------|------------|
| PCA | Linear | No |
| ICA | Linear | No |
| IPCA | Linear | No |
| CCA | Linear | No |
| PLS-DA | Linear | Yes |
| t-SNE | Non-linear | No |
| Kernel PCA | Non-linear | No |
| Isomap | Non-linear | No |
| LLE | Non-linear | No |
| Spectral Embedding | Non-linear | No |
| RKS | Non-linear | No |
| Autoencoder (simple) | Non-linear | No |
| **CDI** | **Non-linear (hierarchical AE)** | **No** |
| Tensor Decomposition (CP) | Non-linear | No |

### Data Preprocessing Methods

| Category | Number of Methods | Options |
|----------|-----------------|---------|
| Scaling | 4 | standardize, minmax, robust, pareto |
| Normalization | 4 | constant_sum, L1, L2, max |
| Transformation | 7 | cubicroot, log10, log, log2, sqrt, powertransformer, quantiletransformer |
| Imputation | 5 | knn, mean, median, most_frequent, interpolate |

### Evaluation Models

| Task Type | Number of Models | Algorithms |
|-----------|----------------|------------|
| Classification | 11 | Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes, KNN, MLP, QDA, AdaBoost, Extra Trees, XGBoost |
| Regression | 13 | Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, SVR, K Neighbors, MLP, Gaussian Process, Kernel Ridge |

---

## 18. Key Design Highlights (Paper-Ready)

> These are the strongest, most defensible design points in the CDI codebase — each verified against the source with file:line references. They are written in paper-style language so they can be lifted directly into a methods section, talk, or README.

### 18.1 Learnable Per-Modality Importance Weighting

**Source**: `arch.py:86-89`, `architecture.py` (class `ChemicalDiceIntegrator`)

Rather than blindly concatenating six descriptor sets with wildly different scales and dimensionalities (10,000-dim Signaturizer vs. 512-dim ImageMol), CDI assigns each modality a **learned, per-dimension scaling weight**:

```python
# choice=1: per-dimension learnable scaling for each modality
self.weights = nn.ParameterList([
    nn.Parameter(torch.ones(1, latent_space_dims[i]))
    for i in range(len(latent_space_dims))
])
```

**Paper sentence**: *"Each modality is scaled by a learnable, dimension-wise importance vector before fusion, allowing the model to determine the relative contribution of each representation source during training."*

**Formally**: For modality $i$ with input $\mathbf{x}_i \in \mathbb{R}^{d_i}$, the weighted input is:

$$\tilde{\mathbf{x}}_i = \mathbf{x}_i \odot \mathbf{w}_i, \quad \mathbf{w}_i \in \mathbb{R}^{d_i} \text{ learnable}$$

where $\odot$ denotes element-wise multiplication.

---

### 18.2 Leave-One-Out Semantic Commonality (Core Novelty)

**Source**: `getEmbeddings.py:441-446`, `arch.py` (`forward`), `docs/architecture.md`

Each of the six Tier-1 Semantic Commonality Autoencoders (SCAs) is trained to **reconstruct one modality from the concatenation of the other five** (Leave-One-Out / LOO objective):

```python
key_1 = torch.cat([k2, k3, k4, k5, k6], dim=1)   # all modalities except 1
key_2 = torch.cat([k1, k3, k4, k5, k6], dim=1)   # all modalities except 2
# ... six LOO views in total
```

**Paper sentence**: *"Six semantic-commonality autoencoders are trained under a leave-one-out reconstruction objective: each autoencoder must reconstruct a held-out modality from the remaining five, forcing its latent subspace to capture only the shared (consensus) information across feature domains."*

**Formally**: For modality $j$:

$$\mathbf{z}_j = f_{\text{enc},j}\left(\bigoplus_{i \neq j} \tilde{\mathbf{x}}_i\right), \qquad \mathcal{L}_{RE,j} = \left\| \hat{\mathbf{x}}_j - \mathbf{x}_j \right\|_2^2$$

where $\bigoplus$ denotes concatenation, $\mathbf{z}_j$ is the $j$-th semantic-commonality latent, and $\hat{\mathbf{x}}_j$ is the reconstructed held-out modality. This enforces cross-modal semantic consensus — the defining novelty of CDI.

---

### 18.3 Balanced Three-Component Objective

**Source**: `getEmbeddings.py:421-471`

CDI-Basic is trained with a **balanced, three-component loss** ($\alpha = \beta = \gamma = 0.33$):

```python
alpha, beta, gamma = 0.33, 0.33, 0.33
total_loss = (α * total_encoding_loss / 6) \
           + (β * total_reconstruction_loss / 6) \
           + (γ * reconstruction_loss_concat)
```

**Paper sentence**: *"The fusion engine is optimized with a weighted combination of (i) encoding fidelity — each semantic-commonality latent must reconstruct its own target modality; (ii) leave-one-out reconstruction fidelity; and (iii) super-embedding coherence — the Tier-2 autoencoder must reconstruct its concatenated input."*

**Formally**:

$$\mathcal{L}_{total} = \frac{\alpha}{6}\sum_{j=1}^{6} \text{MSE}(\mathbf{z}_j, \mathbf{x}_j) + \frac{\beta}{6}\sum_{j=1}^{6} \text{MSE}\left(\hat{\mathbf{c}}_j, \mathbf{c}_j\right) + \gamma \, \text{MSE}\left(\hat{\mathbf{c}}, \mathbf{c}\right)$$

with $\mathbf{c}_j = \bigoplus_{i \neq j} \tilde{\mathbf{x}}_i$ the LOO concatenation, $\mathbf{c} = \bigoplus_{j=1}^{6} \mathbf{z}_j$ the concatenated latent, and $\alpha = \beta = \gamma = 1/3$.

---

### 18.4 Leakage-Free Supervised Fusion Evaluation

**Source**: `fusionData.py:795-798`, `fusionData.py:806-810` (`evaluate_fusion_models_nfold`)

Supervised fusion methods (PLS-DA; tensor decomposition, which uses label-dependent `SelectKBest`) are **re-fit inside each cross-validation fold on the training split only**, then applied to the held-out test split:

```python
# per fold:
train_dataframes, test_dataframes, train_label, test_label = \
    save_train_test_data_n_fold(self.dataframes, self.prediction_label, train_index, test_index, ...)
X_train = self.fuseFeaturesTrain_plsda(n_components=..., train_dataframes=train_dataframes, train_label=train_label)
X_test  = self.fuseFeaturesTest_plsda(n_components=..., test_dataframes=test_dataframes)
```

**Paper sentence**: *"To prevent information leakage, all supervised fusion transforms (PLS-DA and tensor decomposition) are fitted exclusively on the training fold and subsequently applied to the test fold, ensuring that reported performance reflects genuine generalization."*

This is a methodological rigor point that reviewers explicitly check for in representation-learning benchmarks.

---

### 18.5 End-to-End Reproducibility (Fixed-Seed Determinism)

**Source**: `getEmbeddings.py:12-25`

```python
seed_value = 42
np.random.seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
```

**Paper sentence**: *"All experiments use a fixed random seed (42) with deterministic CuDNN settings and per-worker DataLoader seeding, making CDI training fully reproducible."*

---

### 18.6 One-Line Summary (for abstracts/talks)

> *"CDI learns per-modality importance via trainable dimension-wise weights while enforcing cross-modal semantic consensus through leave-one-out autoencoder reconstruction, trained under a balanced three-part objective with leakage-free evaluation and full reproducibility."*

---

## References

1. **CDI Framework**: [github.com/the-ahuja-lab/ChemicalDice](https://github.com/the-ahuja-lab/ChemicalDice)
2. **GROVER**: Rong et al., "Self-Supervised Graph Transformer on Large-Scale Molecular Data", NeurIPS 2020. [arXiv:2007.02835](https://arxiv.org/abs/2007.02835)
3. **ChemBERTa**: Chithrananda et al., "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction", 2020. [arXiv:2010.09885](https://arxiv.org/abs/2010.09885)
4. **ImageMol**: Zeng et al., "ImageMol: A Self-Supervised Representation Learning Framework for Molecular Property Prediction", 2021.
5. **Signaturizer**: [github.com/ersilia-os/signaturizer](https://github.com/ersilia-os/signaturizer)
6. **Mordred**: Moriwaki et al., "Mordred: a molecular descriptor calculator", J. Cheminf. 2018.
7. **MOPAC**: Stewart, "MOPAC: A semiempirical molecular orbital program", J. Comput.-Aided Mol. Des. 1990.
8. **Mamba SSM**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
