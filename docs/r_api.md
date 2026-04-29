# R Package API

This package provides a robust R interface to the ChemicalDice API for computational chemistry and cheminformatics. It facilitates the validation and canonicalization of SMILES strings using RDKit and enables large-scale feature extraction (molecular embeddings) via a streamlined CSV-based pipeline.

---

## Installation

### **1. Prerequisites & System Requirements**

*   **R** (version 4.0.0 or higher)
*   **Python** (version 3.7 or higher) with the `rdkit` package installed.
*   **R Packages**: `httr`, `data.table`, `progress`, `jsonlite`, `reticulate`, `curl`, `remotes`.

### **2. Install R Dependencies**

Open R or RStudio and run the following command to install all required R packages from CRAN:

```r
install.packages(c("httr", "data.table", "progress", "jsonlite", "reticulate", "curl", "remotes"))
```

### **3. Install the ChemicalDice R Package**

Install the package directly from the GitHub repository:

```r
remotes::install_github("the-ahuja-lab/ChemicalDice", subdir = "R-package")
```

---

## Configuration & Setup

### **A. Configure Python and RDKit**

Before using the package, you must configure the `reticulate` package to use a Python environment that has RDKit installed.

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

# py_require tells reticulate your R session needs RDKit, checks for it
# In case Rdkit is missing creates a Python environment to install it so code runs seamlessly.
py_require("rdkit") 

# Alternatively, point to a specific Python executable
# use_python("/path/to/your/python", required = TRUE)

# Import RDKit
rdkit <- import("rdkit.Chem", convert = TRUE)
```

> **Important Note**: Ensure your Python environment has `rdkit` installed. You can install it via Conda with: `conda install -c conda-forge rdkit`.

---

## Usage

### **Feature Extraction from a CSV File**

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

```r
# Load the library (if not already loaded at the top of your script)
library(ChemicalDice)

# Extract features
CDI_embeddings <- collect_features_from_csv(
    filepath="smiles.csv",
    convert_to_canonical=FALSE
)

# check CDI_embeddings data frame
head(CDI_embeddings[,1:10])

# Save the features to a new CSV file
write.csv(CDI_embeddings, "CDI_embeddings.csv", row.names = FALSE)
```

### **Function Details: `collect_features_from_csv`**

*   **Purpose**: Processes a CSV file to generate molecular feature embeddings.
*   **Input**: Path to a CSV file with a `SMILES` column.
*   **Process**:
    1.  **Validation**: Uses RDKit to validate each SMILES string. Invalid entries are flagged and skipped.
    2.  **Canonicalization(Optional)**: The original `SMILES` column in your input CSV is converted to canonical SMILES. In case you want canonicalization you can set `convert_to_canonical` argument to `TRUE`.
    3.  **Feature Extraction**: The CSV is streamed to the ChemicalDice API, which returns a data frame of molecular features.
*   **Output**: A data frame where the first column contains the input **SMILES**, other columns correspond to the extracted features, and rows correspond to successfully processed molecules.  
This standardized output can be used directly for downstream tasks such as QSAR modeling, clustering, virtual screening, or integration into machine learning pipelines.

---

## Troubleshooting & Notes

*   **Backup Your Data**: The input CSV file is modified in-place. Always work on a copy of your **original data** to prevent data loss.
*   **Invalid SMILES**: Molecules with invalid SMILES will be skipped during processing and will not appear in the output feature dataframe. Check the function's messages or your overwritten CSV for details on which entries were invalid in column `is_valid`.
*   **Network Connection**: A stable internet connection is required to communicate with the ChemicalDice API.

For technical issues, please ensure all prerequisites are met and your configuration is correct. For API-related problems, contact the ChemicalDice service administrators.
