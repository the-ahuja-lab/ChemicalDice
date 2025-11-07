# **ChemicalDice Integrator (CDI)**  
**CDI (ChemicalDice Integrator)** is a high-performance deep learning framework designed to unify heterogeneous chemical representations into a single, high information rich latent space. By fusing six complementary molecular embeddings, CDI produces a consolidated vector optimized for large-scale cheminformatics, bioinformatics, and AI-driven molecular discovery tasks.

---

<div align="center">
  <img src="Images/CDI.png" alt="ChemicalDice Integrator Overview" width="750">
</div>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg">
  <img src="https://img.shields.io/badge/docs-passing-green">
  <img src="https://img.shields.io/badge/python-3.9+-blue">
  <a href="https://github.com/the-ahuja-lab/ChemicalDiceIntegrator">
    <img src="https://img.shields.io/badge/Code-Source-black">
  </a>
</p>

---

##  **Overview**

CDI extends the **ChemicalDice** featurization ecosystem by performing unsupervised integration of **six distinct molecular embeddings**:

- 🧬 **Quantum Descriptors**  
- ⚗️ **Bioactivity Signatures**  
- 💬 **Language Model Embeddings**  
- 🌐 **Graph-Derived Representations**  
- ⚖️ **Physicochemical Profiles**  
- 🖼️ **2D Molecular Image Features**  

Each compound’s six feature types are combined to create a **single latent embedding** that captures chemical, structural, and biological semantics. These embeddings can be directly used for tasks such as **QSAR modeling**, **virtual screening**, **drug-target interaction prediction**, and **bioactivity clustering**.

---

## ⚙️ **Implementation Details**

| Parameter | Description |
|------------|--------------|
| Framework | PyTorch / TensorFlow |
| Loss Function | Mean Squared Error + Reconstruction Error |
| Activation | ReLU |
| Latent Dimension | 8192 |
| Input  | SMILES |
| Output | Unified latent embedding (`.npy` or `.csv`) |

---

## 📦 **Installation**

```bash
git clone https://github.com/the-ahuja-lab/ChemicalDiceIntegrator.git
cd ChemicalDiceIntegrator
pip install -r requirements.txt
```

---

## 🚀 **Usage Example**

```python
from CDI import ChemicalDiceIntegrator

# Load six input embeddings for each molecule
# Example: quantum, bioactivity, language, graph, physicochemical, and 2D features

integrator = ChemicalDiceIntegrator()
super_embeddings = integrator.fit_transform(six_feature_matrix)
```

**Output:**
```
Molecule_ID | Super_Embedding_Vector (8192 dims)
-----------------------------------------------
MOL_001     | [0.0123, 0.4421, 0.2235, ...]
MOL_002     | [0.1032, 0.5124, 0.1346, ...]
```

Install packages
----------------

To use the **ChemicalDice** package, you need to install it along with
its dependencies. You can install ChemicalDice and its dependencies
using the following commands:

.. code:: bash

   pip install numpy pandas tqdm rdkit 
   pip install -i https://test.pypi.org/simple/ ChemicalDice


Calculation of Embeddings
--------------------------

.. code:: python

   # SMILES column should be present in the CSV file.
   #example CSV file:
   # SMILES,other_column1,other_column2
   # CC(=O)OC1=CC=CC=C1C(=O)O,1,2
   # C1=CC=CC=C1,3,4
   # C1=CC=C(C=C1)C(=O)O,1,2
   from ChemicalDice import  smiles_to_embeddings
   embeddings = smiles_to_embeddings.collect_features_from_csv(
      filepath="smiles.csv",
      key = "API_KEY",  # Replace with your actual API key
   )






---
## 🚀 **R Installation*
## Overview
This package provides an R interface to validate, canonicalize, and make embeddings from SMILES using ChemicalDice API. It uses RDKit (via reticulate) for SMILES validation and canonicalization, and streams CSV files for feature extraction.

## Installation

### System Requirements
- R (>= 4.0)
- Python (with RDKit installed)
- The following R packages: `httr`, `data.table`, `progress`, `jsonlite`, `reticulate`, `curl`

### Install R dependencies
```r
install.packages(c("httr", "data.table", "progress", "jsonlite", "reticulate","curl"))
remotes::install_github("the-ahuja-lab/ChemicalDice@main", subdir = "R-package")




```

### Python & RDKit setup
You must have Python and RDKit installed. The easiest way is via conda:
```sh
conda create -n chemicaldice python=3.9 rdkit -c conda-forge
```

## Usage

Load libraries, point reticulate to your conda environment and import rdkit:
```r
library(ChemicalDice)
library(reticulate)
use_condaenv("chemicaldice", required = TRUE)
py_require("rdkit")
rdkit <- import("rdkit.Chem", convert = TRUE)
```

### Feature Extraction from CSV
Your CSV must have a column named `SMILES`.
```r
features <- collect_features_from_csv("smiles.csv",key = "API_KEY")
```

- The function will validate all SMILES, overwrite the CSV with canonical SMILES, and stream the file to the server.
- Returns a numeric matrix of features (rows = molecules, columns = features).



## 📊 **Applications**

- 🧩 Unified embedding generation for **QSAR / virtual screening**  
- 🧠 Latent-space mapping for **deorphanization** and **bioactivity clustering**  
- ⚗️ Foundation for **Chemical Foundation Models**  
- 🔬 Enables **cross-modal integration** of text, graph, and physicochemical data**  

---

## 🧠 **Citation**

If you use **ChemicalDice Integrator (CDI)** in your research, please cite:

> *ChemicalDice Integrator (CDI): An Evolutionary-Guided Deep Learning Framework for Unified Molecular Embedding Integration*  
> The Ahuja Lab, 2025.  
> [GitHub Repository](https://github.com/the-ahuja-lab/ChemicalDiceIntegrator)
