

install.packages(c("httr", "data.table", "progress", "jsonlite", "reticulate", "curl", "remotes"))
remotes::install_github("the-ahuja-lab/ChemicalDice", subdir = "R-package")

# Load the necessary R libraries
library(reticulate)
library(httr)
library(data.table)
library(progress)
library(jsonlite)
library(curl)
library(ChemicalDice)



#py_require tells reticulate your R session needs RDKit, checks for it
# In case Rdkit is missing creates a Python environment to install it so code runs seamlessly.
py_require("rdkit")

### Input Prepration

data <- data.frame(
  SMILES = c(
    'CCO',                          # Ethanol
    'c1ccccc1',                     # Benzene
    'CC(=O)Oc1ccccc1C(=O)O',        # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine
    'C(C1C(C(C(C(O1)O)O)O)O)O'      # Glucose
  )
)


filename <- "smiles.csv"


write.csv(data, filename, row.names = FALSE, quote = FALSE)

### Running CDI

library(ChemicalDice)

# Extract features
CDI_embeddings <- collect_features_from_csv(
    filepath="smiles.csv",
    key="ajci8JYskz5FulkeXaczeQmVTYF1cABnP7pdfUFDBgjuCVJZ6R7YjA",#"API_KEY",
    convert_to_canonical=TRUE
)


head(CDI_embeddings[,1:10])

### Saving Results

write.csv(CDI_embeddings, "CDI_embeddings.csv", row.names = FALSE)