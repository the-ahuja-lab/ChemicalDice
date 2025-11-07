# 🧪 ChemicalDice

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![R Version](https://img.shields.io/badge/R-%3E%3D%204.0-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Build](https://img.shields.io/badge/build-passing-success)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey)

---

### 🌟 A Multi-Modal Molecular Featurizer for Deep Learning and Cheminformatics

**ChemicalDice** is a deep learning featurizer built through **unsupervised learning** on the **ChEMBL** database.  
It captures **six distinct molecular representations**, providing a unified embedding space for molecules.

### 🔬 Features

ChemicalDice integrates six complementary molecular representations:

| Representation Type | Description |
|----------------------|-------------|
| 🧭 **Quantum descriptors** | Captures electronic and quantum-level properties |
| 💊 **Bioactivity profiles** | Encodes pharmacological behavior from ChEMBL |
| 🧠 **Language model embeddings** | Learns SMILES syntax via transformer models |
| 🔗 **Graph-based features** | Encodes molecular topology via GNNs |
| ⚗️ **Physicochemical properties** | Classical descriptors like logP, MW, TPSA |
| 🧬 **2D image-based features** | CNN embeddings of molecular depictions |

ChemicalDice takes **SMILES strings** as input and generates **comprehensive embeddings**, enabling robust and versatile characterization for:
- QSAR modeling  
- Virtual screening  
- Drug discovery pipelines  
- Bioinformatics & cheminformatics research  

---

## 📦 Overview

This repository includes an **R interface** for:
- Validating and canonicalizing SMILES (via RDKit)
- Streaming SMILES to the **ChemicalDice API** for embedding generation
- Returning a clean, feature-rich numeric matrix

Under the hood, it uses:
- **RDKit** (via `reticulate`) for validation and standardization  
- **HTTR** for API interaction  
- **data.table** for efficient CSV streaming  

---

## ⚙️ Installation

### 🧰 System Requirements
- **R** ≥ 4.0  
- **Python** ≥ 3.9 (with RDKit installed)
- The following R packages:  
  `httr`, `data.table`, `progress`, `jsonlite`, `reticulate`, `curl`

---

### 🪄 Step 1 — Install R Dependencies
```r
install.packages(c("httr", "data.table", "progress", "jsonlite", "reticulate", "curl"))

# Optionally, install directly from GitHub
remotes::install_github("the-ahuja-lab/ChemicalDice@main", subdir = "R-package")
