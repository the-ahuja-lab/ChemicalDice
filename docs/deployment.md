# **Deployment**

The Chemical Dice Integrator (CDI) is designed for flexible deployment, allowing you to run the heavy deep learning backend in a containerized environment while interacting with it through lightweight Python or R clients.

---

## 📚 Contents

* [Prerequisites](#🛠-prerequisites--system-requirements)
* [Docker & API Setup](#⚡-docker--api-setup)
* [Python Implementation](#🐍-python-implementation)
* [R Implementation](#📊-r-implementation)

---

## 🛠 Prerequisites & System Requirements

Before installing the Chemical Dice Integrator ecosystem, ensure your system meets the following requirements. The framework is designed to run within a **Docker** container to handle deep learning dependencies, with **Python** or **R** acting as the client interface.

### 1. Hardware Requirements

* **GPU (Recommended):** NVIDIA GPU with CUDA support for high-throughput embedding generation.
* **Memory:** Minimum 8GB RAM (16GB+ recommended for large-scale CSV processing).
* **Disk Space:** ~10GB for the Docker image and model weights.

### 2. Core Environments

| Component | Required Version | Purpose |
| --- | --- | --- |
| **Docker** | 20.10+ | Runs the CDI API and deep learning backend. |
| **NVIDIA Container Toolkit** | Latest | Enables GPU acceleration inside Docker. |
| **Python** | 3.8 — 3.11 | Required for the Python client and RDKit integration. |
| **R** | 4.0.0+ | Required for the R interface users. |

### 3. API & Model Access

* **Hugging Face Account:** You will need access to the [the-ahuja-lab/ChemicalDice](https://huggingface.co/the-ahuja-lab/ChemicalDice) repository to pull the necessary model files.
* **Network Access:** Ensure your firewall allows communication on port `8002` (or your chosen local port) for the REST API.

---

## ⚡ Docker & API Setup

The recommended way to use ChemicalDiceIntegrator model is through the provided **Docker environment**.

### 1. Build and Run the Docker Environment

The Docker build creates an image that exposes a REST API for generating embeddings via HTTP requests.

```bash
# Build the image
docker build -t chemicaldice-api .

# Run the container with GPU support (port 8002)
docker run -d --gpus all -p 8002:8000 --name chemicaldice-container chemicaldice-api
```

### 2. Access Documentation & Test

* **Swagger UI:** [http://localhost:8002/docs](http://localhost:8002/docs)
* **Test with Curl:**
```bash
curl -X 'POST' 'http://localhost:8002/predict-single-smile' \
  -H 'Content-Type: application/json' \
  -d '{"smiles": "CCO"}'
```

---

## 🐍 Python Implementation

### Installation

```bash
pip install numpy pandas rdkit tqdm requests
pip install -i https://test.pypi.org/simple/ ChemicalDice
```

### Usage

```python
from ChemicalDice import smiles_to_embeddings
import pandas as pd

# Load or create your data
df = pd.DataFrame({'SMILES': ['CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O']})
df.to_csv("smiles.csv", index=False)

# Generate embeddings via local API
CDI_embeddings = smiles_to_embeddings.collect_features_from_csv(
    filepath="smiles.csv",
    convert_to_canonical=False,
    URL="http://localhost:8002/"
)

print(CDI_embeddings.head())
```

---

## 📊 R Implementation

### 1. Installation

```r
# Install CRAN dependencies
install.packages(c("httr", "data.table", "progress", "jsonlite", "reticulate", "curl", "remotes"))

# Install ChemicalDice R package from GitHub
remotes::install_github("the-ahuja-lab/ChemicalDice", subdir = "R-package")
```

### 2. Configuration & Setup

Before use, configure `reticulate` to point to a Python environment containing **RDKit**.

```r
library(ChemicalDice)
library(reticulate)

# Configure your Python/Conda environment
use_condaenv("my_rdkit_env", required = TRUE)

# Ensure RDKit is available
py_require("rdkit")
rdkit <- import("rdkit.Chem", convert = TRUE)
```

### 3. Usage in R

The R interface allows you to process SMILES data frames and interface seamlessly with the local Docker API.

```r
# Define your SMILES data
smiles_data <- data.frame(
  SMILES = c("CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O")
)

# Save the SMILES data to a CSV file
write.csv(smiles_data, file = "smiles.csv", row.names = FALSE)

# Generate integrated embeddings
# Ensure the Docker container is running on port 8002
embeddings <- collect_features_from_df(
  filepath="smiles.csv",
  convert_to_canonical = FALSE,
  URL = "http://localhost:8002/"
)

# View results
print(head(embeddings))
```

> **Note:** For the R implementation, ensure your Python environment is set up with `conda install -c conda-forge rdkit`.
