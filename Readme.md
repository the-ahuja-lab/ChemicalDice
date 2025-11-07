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

## License Key

**CDI is free for academic institutions, however, for commercial utilization a commercial license key is required. Users (academic/commercial) may apply for a valid "License Key" [here](https://forms.gle/2GjV3hUwzF7efVbC8).**

You can also generate your own predictions using CDI’s [Colab notebook](https://colab.research.google.com/drive/)

## Environment Setup (done using requirement.txt)

**Major dependencies**
1. [RDKit (v2022.3.1)](https://www.rdkit.org/)
2. Python (v3.8)

**Minor dependencies**
1. os
2. [scikit-learn (v1.2.1)](https://scikit-learn.org/stable/whats_new/v1.0.html)
3. [pandas (v1.4.3)](https://pandas.pydata.org/)
4. [numpy (v>=1.20.3)](https://numpy.org)
5. [tqdm](https://tqdm.github.io)
6. [joblib (v1.1.1)](https://pypi.org/project/joblib/)
7. [importlib ](https://pypi.org/project/importlib/)
8. [importlib-resources (v5.7.1)](https://github.com/python/importlib_resources)



## How to use CDI?

## 📦 **Installation**

```bash
git clone https://github.com/the-ahuja-lab/ChemicalDiceIntegrator.git
cd ChemicalDiceIntegrator
pip install -r requirements.txt
```

---

### Installation using pip 
```
$ pip install -i https://test.pypi.org/simple/CDI
```

### License activation (One time)
To apply for the license [click here](https://forms.gle/2GjV3hUwzF7efVbC8)




## 🚀 **Usage Example**

```python
from CDI import ChemicalDiceIntegrator

# Load SMILES

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

> *ChemicalDice Integrator (CDI):Chemical Dice: A Scalable Framework for Multimodal Molecular Representation Learning*  
> The Ahuja Lab, 2025.  
> [GitHub Repository](https://github.com/the-ahuja-lab/ChemicalDiceIntegrator)
