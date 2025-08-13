# ChemicalDice

ChemicalDice is a deep learning featurizer developed using unsupervised learning on the ChEMBL database. It captures six distinct molecular 
representations: quantum descriptors, bioactivity profiles, language model embeddings, graph-based features, physicochemical properties, and 
2D image-based features. ChemicalDice takes SMILES strings as input and generates comprehensive embeddings for each molecule, enabling robust and 
versatile molecular characterization for downstream cheminformatics and bioinformatics applications.

## Overview
This package provides an R interface to validate, canonicalize, and batch-embed SMILES strings using a remote ChemicalDice server. It uses RDKit (via reticulate) for SMILES validation and canonicalization, and streams CSV files to a server for feature extraction.

## Installation

### System Requirements
- R (>= 4.0)
- Python (with RDKit installed)
- The following R packages: `httr`, `data.table`, `progress`, `jsonlite`, `reticulate`

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

In R, point reticulate to your conda environment:
```r
library(reticulate)
use_condaenv("chemicaldice", required = TRUE)
py_require("rdkit")
rdkit <- import("rdkit.Chem", convert = TRUE)
```

## Usage

### Batch Feature Extraction from CSV
Your CSV must have a column named `SMILES`.
```r
library(ChemicalDice)
features <- collect_features_from_csv("smiles.csv",key = "API_KEY")
```

- The function will validate all SMILES, overwrite the CSV with canonical SMILES, and stream the file to the server.
- Returns a numeric matrix of features (rows = molecules, columns = features).









