# Quickstart Guide

ChemicalDice integrates flawlessly from the Command Line or directly within Python modules. Here is a 5-minute initialization sequence mapping you from zero to full deployment routing.

---

## 🗂 Input Data Requirements

For both inference and descriptor calculation, ChemicalDice expects your input CSV files to follow a specific structure:

* **Mandatory Column:** The file **must** contain a column named exactly `SMILES`.
* **Example `my_smiles.csv`:**
  ```csv
  SMILES,Compound_ID
  CCO,Ethanol
  Cc1ccccc1,Toluene
  C1CCCCC1,Cyclohexane
  ```

---

## 1. Remote API Inference (Minimal)
Assuming you invoked a minimal install (`pip install ChemicalDice`), you can compute massive 8192-D features instantly using our backend cluster.

*   **Input**: A `.csv` file containing a `SMILES` column.
*   **Output**: A `.csv` file containing the original SMILES plus 8192 integrated feature columns.

**Via CLI:**
```bash
# Execute remote feature extraction
cdi fetch --input my_smiles.csv --output final_embeddings.csv --canonicalize
```

**Via Python:**
```python
from ChemicalDice.core.api_client import collect_features_from_csv

df = collect_features_from_csv("my_smiles.csv", convert_to_canonical=True)
df.to_csv("final_embeddings.csv", index=False)
```

---

## 2. Descriptor Calculation

Calculate massive external modalities (Quantum, Physicochemical, Linguistic, etc.) locally before converting them to HDF5 sets for autoencoder training.

*   **Input**: A `.csv` file with a `SMILES` column.
*   **Output**: 
    1.  **Stage 1 (Compute)**: Individual `.csv` files for each descriptor in the `output_dir`.
    2.  **Stage 2 (Convert)**: Merged and imputed `.h5` files ready for multi-modal training.

**Via CLI:**
```bash
# Calculate 6 main descriptors from a SMILES CSV
cdi compute --input data.csv --descriptors mordred grover chemberta signaturizer mopac imagemol --output_dir data

# Convert raw CSVs to aligned HDF5 manifolds
cdi convert --input_dir data --output_dir data_h5
```

**Via Python:**
```python
from ChemicalDice.descriptors.calculate import calculate_descriptors
from ChemicalDice.descriptors.convert import convert_to_hdf5

# Calculate the 6 main modalities
calculate_descriptors(input_file="data.csv", descriptors=["all"])

# Process CSVs into aligned HDF5 files for training
convert_to_hdf5(input_dir="Chemicaldice_data", output_dir="data_h5")
```

---

## 3. Model Training (Training Setup)

CDI supports two primary training modes. For full details, see the **[Training Specifics](training.md)**.

### **A. Unsupervised CDI-Basic**
Integrates multiple modalities into a unified latent space via leave-one-out training.

*   **Input**: A list of aligned `.h5` descriptor files.
*   **Output**: A trained multi-modal autoencoder model and checkpoint.

**Via CLI:**
```bash
# Train on all 6 modalities
cdi train-basic --data-files data_h5/mordred.h5 data_h5/Grover.h5 data_h5/Chemberta.h5 data_h5/Signaturizer.h5 data_h5/mopac.h5 data_h5/ImageMol.h5 --epochs 50
```

**Via Python:**
```python
from ChemicalDice.training.basic_model import train_basic_cdi

# Training with the full 6-modality ensemble
model = train_basic_cdi(
    h5_file_paths=[
        "data_h5/mordred.h5", "data_h5/Grover.h5", "data_h5/Chemberta.h5",
        "data_h5/Signaturizer.h5", "data_h5/mopac.h5", "data_h5/ImageMol.h5"
    ],
    num_epochs=50,
    model_path="cdi_model.pt",
    embedding_path="cdi_embeddings.h5"
)
```

### **B. Supervised CDI-Generalised (Mamba)**
Learns to predict 8192-D CDI embeddings directly from SMILES sequences.

*   **Input**: A `SMILES` CSV and a target `.h5` file (the 8192-D embeddings from CDI-Basic).
*   **Output**: A fine-tuned Mamba model for sequence-to-embedding prediction.

**Via CLI:**
```bash
# Fine-tune state-space network against 8192-D targets
cdi train-gen --smiles-csv smiles.csv --target-h5 cdi_embeddings.h5 --mamba-dir ./cdi_generalised_model
```

**Via Python:**
```python
from ChemicalDice.training.gen_model import train_generalised_cdi

model = train_generalised_cdi(
    csv_path="smiles.csv",
    target_h5_file="cdi_embeddings.h5"
)
```

---

## 4. Deployment

Launch asynchronous embedding streaming servers for massive internal data.

*   **Input**: Trained model artifacts and server configuration.
*   **Output**: A live FastAPI/REST service providing `/predict` and `/stream-features-from-csv` endpoints.

### **Required Artifacts:**
To launch the server locally, you must have the following files available:
1.  **`generalised_smi_ssed_model.pth`**: The fine-tuned weights for the CDI regression head.
2.  **`smi_ssed_model/`**: The directory containing the base Mamba model and tokenizer.

### **Configuration (Environment Variables):**
You can configure the server using the following environment variables:
*   `MAMBA_DIR`: Path to the base Mamba model directory.
*   `MODEL_WEIGHTS`: Path to the fine-tuned `.pth` weights file.
*   `API_KEY`: Secret key required for authentication.

**Via CLI:**
```bash
# Set environment variables and launch the server
export MAMBA_DIR="./models/smi_ssed"
export MODEL_WEIGHTS="./models/weights.pth"

cdi serve --host 0.0.0.0 --port 8000
```
