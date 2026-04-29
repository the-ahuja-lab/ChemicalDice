# SOTA Benchmark Analysis

This document outlines the end-to-end pipeline for benchmarking molecular embedding models (e.g., UniMol, MolT5, Atomas, MolCA, CDI) on binary classification datasets. 

The pipeline handles the entire Machine Learning lifecycle: from raw dataset inspection and generic labeling, through high-dimensional embedding extraction, rigorous data quality analysis, advanced model training (incorporating under-sampling).

---

## 1. Environment Setup

Combining SOTA models can be challenging because older models like MolCA have strict legacy dependencies (Python 3.8), while standard `pip` tries to pull newer packages (like Spacy 3.8 + Numpy 2.0) that fundamentally conflict. We use `uv` to perfectly backtrack and solve these conflicts.

**1. Create the base Conda environment:**
```bash
conda create -n mol_embeddings python=3.8
conda activate mol_embeddings
```

**2. Install PyTorch & PyTorch Geometric (PyG):**
*(These must be installed first to match MolCA's core requirements).*
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
```

**3. Install uv and pin the problematic Spacy package:**
```bash
pip install uv
uv pip install "spacy<3.8"
```

**4. Install all remaining dependencies:**
```bash
uv pip install rouge_score nltk ogb peft rdkit salesforce-lavis deepspeed transformers pytorch-lightning unimol_tools huggingface_hub pandas torch_geometric Levenshtein
```

---

## 2. Directory Structure & Repository Cloning

Atomas and MolCA are not installable via `pip`. You must clone their source code locally so our script can import their modules dynamically.

Create a main project folder and clone the repositories inside it:
```bash
mkdir mol_project
cd mol_project

# Clone the research repos 
git clone https://github.com/yikunpku/Atomas.git
git clone https://github.com/thunlp/MolCA.git 

# Create directories for weights and outputs
mkdir checkpoints
mkdir data
mkdir csv_embeddings
```

Your structure should look exactly like this:
```text
mol_project/
├── Atomas/                 # Cloned repo
├── MolCA/                  # Cloned repo
├── checkpoints/            # Custom weights go here
├── data/                   # Hardcoded Atomas weights go here
├── csv_embeddings/         # Output folder for CSVs
└── ChemicalDice.sota_pipeline.extract_embeddings   # Our master script
```

---

## 3. Downloading the Weights

While Uni-Mol and MolT5 dynamically download weights, **Atomas** and **MolCA** have hardcoded paths in their source code that we must satisfy manually.

**1. Satisfy Atomas's Hardcoded MolT5 Paths:**
Atomas specifically looks for local folders named `molt5decoder-base` and `molt5decoder-large`. Use the HuggingFace CLI to download them directly into those folders:
```bash
hf download laituan245/molt5-base --local-dir ./data/pretrained/molt5decoder-base
hf download laituan245/molt5-large --local-dir ./data/pretrained/molt5decoder-large
```

**2. Place the Main Checkpoints:**
Download `atomas_pretrained.ckpt` (Atomas Large version) and `stage1.ckpt` (MolCA) from their respective release links and place them inside the `checkpoints/` folder.

---

## 4. Patching the MolCA Source Code

MolCA's initialization code tries to load a pre-trained Graph Neural Network (`graphcl_80.pth`) before applying your custom checkpoint. Since you only need to extract features using your *already trained* `stage1.ckpt`, this step crashes looking for a file you don't have.

We must comment it out. Open `MolCA/model/blip2.py` and scroll to **line ~81** inside the `init_graph_encoder` function. Comment out the loading block:

```python
def init_graph_encoder(self, gin_num_layers, gin_hidden_dim, gin_drop_ratio):
    graph_encoder = GNN(num_layer=gin_num_layers, emb_dim=gin_hidden_dim, drop_ratio=gin_drop_ratio)
    
    # --- WE COMMENTED THIS ENTIRE CHUNK OUT ---
    # ckpt = torch.load('gin_pretrained/graphcl_80.pth', map_location=torch.device('cpu'))
    # missing_keys, unexpected_keys = graph_encoder.load_state_dict(ckpt, strict=False)
    # if len(missing_keys) or len(unexpected_keys):
    #     print(...)
    # ------------------------------------------
    
    ln_graph = nn.LayerNorm(graph_encoder.num_features)
    return graph_encoder, ln_graph
```

## 5. Extract Embeddings

After setting up your environment and weights, use the extraction script to generate embeddings for your dataset using all the downloaded models. 

Run the following command, adjusting the paths to match your inputs:

```bash
python -m ChemicalDice.sota_pipeline.extract_embeddings \
    --input_csv ./datasets/mydata.csv \
    --output_dir ./parquets \
    --atomas_ckpt checkpoints/atomas_pretrained.ckpt \
    --molca_ckpt checkpoints/stage1.ckpt \
    --batch_size 64 \
    --gpu_fraction 0.6
```

---

## 6. Evaluation using Evaluate.py

After extracting your molecular embeddings, you can run the benchmarking suite to evaluate classification performance across multiple SOTA models (AdaBoost, XGBoost, LightGBM, ExtraTrees, GradientBoosting).

### **Directory Structure Requirements**

The `Evaluation` script relies on a organized base directory (`BASE_DIR`) containing the following structure:

```text
BASE_DIR/
├── embeddings/             # Place all extracted embedding CSVs here
├── labels/                 # Place your label CSVs here (e.g., tox21.csv)
└── results_cv/             # (Auto-generated) Output directory for CV results
```

### **Execution**

Run the evaluation module from the command line by passing the path to your base directory, the name of your dataset (without the `.csv` extension), and the name of the target column you want to classify.

```bash
python -m ChemicalDice.sota_pipeline.Evaluate <BASE_DIR> <DATASET_NAME> <TARGET_COLUMN>
```

**Example:**
If your base directory is `./benchmark_workspace`, your labels are in `benchmark_workspace/labels/tox21.csv`, and your target column is **`NR-ER`**:
```bash
python -m ChemicalDice.sota_pipeline.Evaluate ./benchmark_workspace tox21 NR-ER
```

### **Expected Outputs**

The script runs a 5-Fold Stratified Cross-Validation across 3 random seeds. It automatically handles severe class imbalances using Random Under-Sampling (RUS) on the training folds. 

Once finished, the raw metrics are saved as individual CSVs inside the dynamically created `results_cv/` folder:
- `<BASE_DIR>/results_cv/<DATASET_NAME>_<DESCRIPTOR>_cv/`

Each CSV contains rigorous evaluation metrics including **ROC_AUC, Accuracy, Balanced_Acc, F1, Precision, Recall, and Kappa**.
