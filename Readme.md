# ChemicalDice

**ChemicalDice** is a deep learning–based molecular featurizer developed using unsupervised learning on the ChEMBL database.
It captures six complementary molecular representations, providing a unified, information-rich embedding for each compound:

🧬 Quantum descriptors

⚗️ Bioactivity profiles

💬 Language model embeddings

🌐 Graph-based features

⚖️ Physicochemical properties

🖼️ 2D image-based representations

ChemicalDice takes SMILES strings as input and produces comprehensive embeddings suitable for a wide range of cheminformatics and bioinformatics tasks — including QSAR modeling, virtual screening, and molecular property prediction.

🧠 Available as both Python and R packages for seamless integration into diverse computational workflows.

<div align="center"> <img src="Images/CDI.png" alt="ChemicalDice Overview" width="750"> </div> <p align="center"> <img src="https://img.shields.io/badge/License-MIT-blue.svg"> <img src="https://img.shields.io/badge/docs-passing-green"> <img src="https://img.shields.io/badge/python-3.9+-blue"> <a href="https://github.com/the-ahuja-lab/inertrope"> <img src="https://img.shields.io/badge/Code-Source-black"> </a> </p>
⚙️ Installation
🐍 Python Package

📘 Get Started with ChemicalDice (Python)

<details> <summary>▶️ <b>Installation Commands</b></summary>
pip install numpy pandas tqdm rdkit
pip install -i https://test.pypi.org/simple/ ChemicalDice

</details> <details> <summary>💡 <b>Example: Generate Molecular Embeddings</b></summary>
from ChemicalDice import smiles_to_embeddings

# Example CSV (smiles.csv)
# SMILES,other_column1,other_column2
# CC(=O)OC1=CC=CC=C1C(=O)O,1,2
# C1=CC=CC=C1,3,4
# C1=CC=C(C=C1)C(=O)O,1,2

embeddings = smiles_to_embeddings.collect_features_from_csv(
    filepath="smiles.csv",
    key="API_KEY"  # Replace with your actual API key
)

</details>
📊 R Package

📗 Get Started with ChemicalDice (R)

🧩 Overview

The R package provides a native interface to the ChemicalDice API for validating, canonicalizing, and featurizing SMILES.
It uses RDKit (via reticulate) for SMILES handling and supports streaming of large CSV files for efficient feature extraction.

💻 System Requirements

R ≥ 4.0

Python (with RDKit installed)

R packages: httr, data.table, progress, jsonlite, reticulate, curl

<details> <summary>▶️ <b>Install Dependencies</b></summary>
install.packages(c("httr", "data.table", "progress", "jsonlite", "reticulate", "curl"))
remotes::install_github("the-ahuja-lab/ChemicalDice@main", subdir = "R-package")

</details> <details> <summary>🐍 <b>Set Up Python & RDKit Environment</b></summary>
conda create -n chemicaldice python=3.9 rdkit -c conda-forge

</details> <details> <summary>💡 <b>Example: Extract Features from CSV</b></summary>
library(ChemicalDice)
library(reticulate)

use_condaenv("chemicaldice", required = TRUE)
py_require("rdkit")
rdkit <- import("rdkit.Chem", convert = TRUE)

# Extract features from CSV containing a 'SMILES' column
features <- collect_features_from_csv("smiles.csv", key = "API_KEY")


✅ The function validates and canonicalizes all SMILES, overwrites the CSV with canonical forms, and streams it to the server.
Returns a numeric matrix of features (rows = molecules, columns = features).

</details>
🚀 Key Features

🔗 Unified featurization integrating six molecular representations

⚙️ Cross-platform support (Python & R)

🧩 API-based architecture for scalable batch processing

📦 Containerized (Docker-ready) for reproducible deployments

📊 Seamless integration with ML pipelines for QSAR, ADMET, and property prediction
