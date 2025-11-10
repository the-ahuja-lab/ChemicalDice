# **Chemical Dice Integrator (CDI)**  
**CDI (Chemical Dice Integrator)** is a high-performance deep learning framework designed to unify heterogeneous chemical representations into a single, high information rich latent space. By fusing six complementary molecular embeddings, CDI produces a consolidated vector optimized for large-scale cheminformatics, bioinformatics, and AI-driven molecular discovery tasks.

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

CDI extends the **Chemical Dice Integrator** featurization ecosystem by performing unsupervised integration of **six distinct molecular embeddings**:

- 🧬 **Quantum Descriptors**  
- ⚗️ **Bioactivity Signatures**  
- 💬 **Language Model Embeddings**  
- 🌐 **Graph-Derived Representations**  
- ⚖️ **Physicochemical Profiles**  
- 🖼️ **2D Molecular Image Features**  

Each compound’s six feature types are combined to create a **single latent embedding** that captures chemical, structural, and biological semantics. These embeddings can be directly used for tasks such as **QSAR modeling**, **virtual screening**, **drug-target interaction prediction**, and **bioactivity clustering**.

## License Key

**CDI is free for academic institutions, however, for commercial utilization a commercial license key is required. Users (academic/commercial) may apply for a valid "License Key" [here](https://forms.gle/2GjV3hUwzF7efVbC8).**

You can also generate your own predictions using CDI’s [ipynb notebook](https://github.com/the-ahuja-lab/ChemicalDice/blob/main/demo/CDI_demo.ipynb)

## Environment Setup 

**Dependencies**
1. Python (v3.8)
2. [RDKit (v2022.3.1)](https://www.rdkit.org/)
3. [pandas (v1.4.3)](https://pandas.pydata.org/)
4. [numpy (v>=1.20.3)](https://numpy.org)
5. [tqdm](https://tqdm.github.io)

##  Quick Start: Get Featurizer Like a Pro - R or Python, We’ve Got You Covered
Whether you’re an R wizard or a Python powerhouse, Chemical Dice Integrator(CDI) has you covered.
If you’re diving into machine learning for chemistry or bioinformatics, you don’t need to worry about choosing the right featurizer - we’ve already done the hard work for you.

With just two pip commands, you’re ready to generate rich, unified molecular embeddings for your giant ML workflows - no confusion, no setup hassle. 🚀


### Installation using pip 
```
$ pip install numpy pandas tqdm rdkit
$ pip install -i https://test.pypi.org/simple/ ChemicalDice
```

---

### License activation (One time)
To apply for the license [click here](https://forms.gle/2GjV3hUwzF7efVbC8)

To compute molecular embeddings from SMILES strings stored in a CSV file, ensure that your dataset includes a column named **`SMILES`**.

---

## Example CSV File

```csv
SrNum,SMILES
1,OCC/C=C\CC
2,OCC/C=C\CC
3,CCCC(=O)O
4,CCCCCCCC=O
5,CC(=O)C(=O)C
```

---


## 🚀 **Usage Example**

```python
# SMILES column should be present in the CSV file
# Example usage of Chemical Dice Integrator to compute embeddings

from ChemicalDice import smiles_to_embeddings
import pandas as pd

embeddings = smiles_to_embeddings.collect_features_from_csv(
    filepath="smiles.csv",
    key="API_KEY"  # Replace with your actual API key
)
CDI_emb = pd.DataFrame(embeddings)
CDI_emb.to_csv("CDI_features.csv",index=False)

```

---

**Output:**
```
Super_Embedding_Vector (8192 dims)
-----------------------------------------------
[0.0123, 0.4421, 0.2235, ...] 
[0.1032, 0.5124, 0.1346, ...] 
```

---
## 🚀 **R Installation*
## Overview
This package provides an R interface to validate, canonicalize, and make embeddings from SMILES using ChemicalDice API. It uses RDKit (via reticulate) for SMILES validation and canonicalization, and streams CSV files for feature extraction.

## Installation

### System Requirements
- R (>= 4.0)
- Python (with RDKit installed)
- The following R packages: `httr`, `data.table`, `progress`, `jsonlite`, `reticulate`, `curl`, `remotes`

### Install R dependencies
```r
install.packages(c("httr", "data.table", "progress", "jsonlite", "reticulate", "curl", "remotes"))
remotes::install_github("the-ahuja-lab/ChemicalDice@main", subdir = "R-package")
```


## Usage

Load libraries, point reticulate to your conda environment and import rdkit:
```r
library(ChemicalDice)
library(reticulate)
py_require("rdkit")
rdkit <- import("rdkit.Chem", convert = TRUE)
```

### Feature Extraction from CSV
Your CSV must have a column named `SMILES`.
```r
features <- collect_features_from_csv("smiles.csv",key = "API_KEY")
features_df= data.frame(features)
features_df
write.csv(features_df,"CDI_features.csv")
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
