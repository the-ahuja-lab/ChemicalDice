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
## 📚 Contents

- [Overview](#overview)
  - [Colab Demo](#colab-demo)
- [ChemicalDice Python Package](#chemicaldice-python-package-python-interface-to-the-chemical-dice-integrator-api)
  - [Installation](#installation)
  - [Usage](#usage)
- [ChemicalDice R Package](#chemicaldice-r-r-interface-to-the-chemicaldice-api)
  - [Installation](#installation-1)
  - [Configuration & Setup](#configuration--setup)
  - [Usage](#usage-1)
- [Troubleshooting & Notes](#troubleshooting--notes)

##  **Overview**

CDI extends the **Chemical Dice Integrator** featurization ecosystem by performing unsupervised integration of **six distinct molecular embeddings**:

-  **Quantum Descriptors**  
-  **Bioactivity Signatures**  
-  **Language Model Embeddings**  
-  **Graph-Derived Representations**  
-  **Physicochemical Profiles**  
-  **2D Molecular Image Features**  

Each compound’s six feature types are combined to create a **single latent embedding** that captures chemical, structural, and biological semantics. These embeddings can be directly used for tasks such as **QSAR modeling**, **virtual screening**, **drug-target interaction prediction**, and **bioactivity clustering**.

### Colab Demo

| Python Users | R Users |
| :--- | :--- |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I6vQ_7SlhagbnXVlg4btWoYal_NcCElt?usp=sharing) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DpHSlauU-Z-Xj3b3dycSgGImFA4ua4g9?usp=sharing) |

## **ChemicalDice Python Package: Python Interface to the Chemical Dice Integrator API**

### **Installation**

#### **1. Prerequisites & System Requirements**
*   **Python** (version 3.8 or higher)
*   **RDKit** (v2022.3.1 or higher) — [https://www.rdkit.org/](https://www.rdkit.org/)
*   **pandas** (v1.4.3 or higher) — [https://pandas.pydata.org/](https://pandas.pydata.org/)
*   **numpy** (v1.20.3 or higher) — [https://numpy.org](https://numpy.org)
*   **tqdm** (v4.65 or higher) - [https://pypi.org/project/tqdm/](https://pypi.org/project/tqdm/)
* **requests** (2.32.4 or higher)-[https://pypi.org/project/requests/](https://pypi.org/project/requests/)

#### **2. Install Python Dependencies**

Open terminal or jupyter notebook run the following command to install all required python packages.

```bash
pip install numpy pandas rdkit tqdm requests
```

#### **3. Install the ChemicalDice Python Package**

```bash
pip install -i https://test.pypi.org/simple/ ChemicalDice
```

To use the ChemicalDice service, you need a free API key.

1. Fill out the API access request form with your details:
    **[https://forms.gle/gPtd8Wqw4akd9Awt5](https://forms.gle/gPtd8Wqw4akd9Awt5)**

2.  You will receive your `API_KEY` via email after your request is approved.

### **Usage**

#### **Feature Extraction from a CSV File**

The primary function, `smiles_to_embeddings`, processes a CSV file containing SMILES strings, validates and canonicalizes them, and streams the data to the ChemicalDice API to generate molecular embeddings.



**Step 1: Prepare Your Input CSV**

Your input file must meet the following requirements:

* **Column Name:** The file **must** contain a column named exactly `SMILES`.
* **File Size:** The input file size must not exceed **20 MB**.

**Example `smiles.csv`**:
```csv
SMILES,Compound_ID
CCO,Ethanol
Cc1ccccc1,Toluene
C1CCCCC1,Cyclohexane
```

**Step 2: Run the Feature Extraction**

Replace `"API_KEY"` with the API key you received from ChemicalDice.

```python
from ChemicalDice import smiles_to_embeddings

# Generate embeddings from CSV 
CDI_embeddings = smiles_to_embeddings.collect_features_from_csv(
    filepath="smiles.csv",
    key="API_KEY",
    convert_to_canonical=True
)

# CDI_embeddings is a pandas.DataFrame;
# Save to CSV
CDI_embeddings.to_csv("CDI_embeddings.csv", index=False)
```

#### **Function Details: `smiles_to_embeddings.collect_features_from_csv`**

*   **Purpose**: Processes a CSV file to generate molecular feature embeddings.
*   **Input**: Path to a CSV file with a `SMILES` column.
*   **Process**:
    1.  **Validation**: Uses RDKit to validate each SMILES string. Invalid entries are flagged and skipped.
    2.  **Canonicalization(Optional)**: The original `SMILES` column in your input CSV is converted to canonical SMILES. In case you do not want canonicalization you can set convert_to_canonical argument to False.
    3.  **Feature Extraction**: The CSV is streamed to the ChemicalDice API, which returns a data frame of molecular features.
*   **Output**: A dataframe where the first column contains the input **SMILES**, other columns correspond to the extracted features, and rows correspond to successfully processed molecules.  
This standardized output can be used directly for downstream tasks such as QSAR modeling, clustering, virtual screening, or integration into machine learning pipelines.

### **Troubleshooting & Notes**

*   **API Key**: A valid API key is required to authenticate your requests.
*   **Backup Your Data**: The input CSV file is modified in-place. Always work on a copy of your **original data** to prevent data loss.
*   **Invalid SMILES**: Molecules with invalid SMILES will be skipped during processing and will not appear in the output feature dataframe. Check the function's messages or your overwritten CSV for details on which entries were invalid in column `is_valid`.
*   **Network Connection**: A stable internet connection is required to communicate with the ChemicalDice API.

For technical issues, please ensure all prerequisites are met and your configuration is correct. For API-related problems, contact the ChemicalDice service administrators.


## **ChemicalDice R: R Interface to the ChemicalDice API**

This package provides a robust R interface to the ChemicalDice API for computational chemistry and cheminformatics. It facilitates the validation and canonicalization of SMILES strings using RDKit and enables large-scale feature extraction (molecular embeddings) via a streamlined CSV-based pipeline.

---

### **Installation**

#### **1. Prerequisites & System Requirements**

*   **R** (version 4.0.0 or higher)
*   **Python** (version 3.7 or higher) with the `rdkit` package installed.
*   **R Packages**: `httr`, `data.table`, `progress`, `jsonlite`, `reticulate`, `curl`, `remotes`.

#### **2. Install R Dependencies**

Open R or RStudio and run the following command to install all required R packages from CRAN:

```r
install.packages(c("httr", "data.table", "progress", "jsonlite", "reticulate", "curl", "remotes"))
```

#### **3. Install the ChemicalDice R Package**

Install the package directly from the GitHub repository:

```r
remotes::install_github("the-ahuja-lab/ChemicalDice", subdir = "R-package")
```

---

### **Configuration & Setup**

#### **A. Configure Python and RDKit**

Before using the package, you must configure the `reticulate` package to use Python environment that has RDKit installed.

```r
# Load the necessary R libraries
library(reticulate)
library(httr)
library(data.table)
library(progress)
library(jsonlite)
library(curl)
library(ChemicalDice)

# Point reticulate to Conda environment (replace 'my_rdkit_env' with your environment name)
use_condaenv("my_rdkit_env", required = TRUE)

#py_require tells reticulate your R session needs RDKit, checks for it
# In case Rdkit is missing creates a Python environment to install it so code runs seamlessly.
py_require("rdkit") 

# Alternatively, point to a specific Python executable
# use_python("/path/to/your/python", required = TRUE)

# Import RDKit
rdkit <- import("rdkit.Chem", convert = TRUE)
```

> **Important Note**: Ensure your Python environment has `rdkit` installed. You can install it via Conda with: `conda install -c conda-forge rdkit`.

#### **B. Obtain an API Key**

To use the ChemicalDice service, you need a free API key.

1.  Fill out the API access request form with your details.
    **[https://forms.gle/gPtd8Wqw4akd9Awt5](https://forms.gle/gPtd8Wqw4akd9Awt5)**

2.  You will receive your `API_KEY` via email after your request is approved.

---

### **Usage**

#### **Feature Extraction from a CSV File**

The primary function, `collect_features_from_csv`, processes a CSV file containing SMILES strings, validates and canonicalizes them, and streams the data to the ChemicalDice API to generate molecular embeddings.

**Step 1: Prepare Your Input CSV**

Your input file must meet the following requirements:

* **Column Name:** The file **must** contain a column named exactly `SMILES`.
* **File Size:** The input file size must not exceed **20 MB**.

**Example `smiles.csv`**:
```csv
SMILES,Compound_ID
CCO,Ethanol
Cc1ccccc1,Toluene
C1CCCCC1,Cyclohexane
```

**Step 2: Run the Feature Extraction**

Replace `"API_KEY"` with the API key you received from ChemicalDice.

```r
# Load the library (if not already loaded at the top of your script)
library(ChemicalDice)

# Extract features
CDI_embeddings <- collect_features_from_csv(
    filepath="smiles.csv",
    key="API_KEY",
    convert_to_canonical=TRUE
)

#check CDI_embeddings data frame
head(CDI_embeddings[,1:10])

# Save the features to a new CSV file
write.csv(CDI_embeddings, "CDI_embeddings.csv", row.names = FALSE)
```

#### **Function Details: `collect_features_from_csv`**

*   **Purpose**: Processes a CSV file to generate molecular feature embeddings.
*   **Input**: Path to a CSV file with a `SMILES` column, Chemical Dice API key.
*   **Process**:
    1.  **Validation**: Uses RDKit to validate each SMILES string. Invalid entries are flagged and skipped.
    2.  **Canonicalization(Optional)**: The original `SMILES` column in your input CSV is converted to canonical SMILES. In case you do not want canonicalization you can set convert_to_canonical argument to False.
    3.  **Feature Extraction**: The CSV is streamed to the ChemicalDice API, which returns a data frame of molecular features.
*   **Output**: A data frame where the first column contains the input **SMILES**, other columns correspond to the extracted features, and rows correspond to successfully processed molecules.  
This standardized output can be used directly for downstream tasks such as QSAR modeling, clustering, virtual screening, or integration into machine learning pipelines.

---

### **Troubleshooting & Notes**

*   **API Key**: A valid API key is required to authenticate your requests.
*   **Backup Your Data**: The input CSV file is modified in-place. Always work on a copy of your **original data** to prevent data loss.
*   **Invalid SMILES**: Molecules with invalid SMILES will be skipped during processing and will not appear in the output feature dataframe. Check the function's messages or your overwritten CSV for details on which entries were invalid in column `is_valid`.
*   **Network Connection**: A stable internet connection is required to communicate with the ChemicalDice API.

For technical issues, please ensure all prerequisites are met and your configuration is correct. For API-related problems, contact the ChemicalDice service administrators.
