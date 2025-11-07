ChemicalDice R PackageChemicalDice is an R package that provides an interface to the ChemicalDice API, a powerful deep learning featurizer developed using unsupervised learning on the ChEMBL database.This package enables R users to seamlessly validate, canonicalize, and generate comprehensive molecular embeddings from SMILES strings.Key Features of ChemicalDice Embeddings:The API captures six distinct molecular representations, offering a robust and versatile characterization platform for cheminformatics and bioinformatics:Quantum DescriptorsBioactivity ProfilesLanguage Model EmbeddingsGraph-Based FeaturesPhysicochemical Properties2D Image-Based FeaturesInstallation1. System RequirementsTo ensure proper functionality, you must meet the following requirements:R (version 4.0 or higher)Python (with RDKit installed, as this package uses reticulate for validation/canonicalization).2. Install R DependenciesThe core R package dependencies can be installed from CRAN, while the main ChemicalDice package is installed directly from GitHub (assuming your package resides in the R-package subdirectory).install.packages(c("httr", "data.table", "progress", "jsonlite", "reticulate", "curl"))
# Install the main package from GitHub
remotes::install_github("the-ahuja-lab/ChemicalDice@main", subdir = "R-package")
3. Python & RDKit SetupThe SMILES validation and canonicalization step relies on RDKit via the R reticulate package. The recommended approach is to use a Conda environment:conda create -n chemicaldice python=3.9 rdkit -c conda-forge
UsageInitialization and RDKit ImportBefore using the main functions, you must load the necessary libraries and point reticulate to the correct Python environment:library(ChemicalDice)
library(reticulate)

# Point to the Conda environment created above
use_condaenv("chemicaldice", required = TRUE) 

# Import RDKit for canonicalization and validation
py_require("rdkit")
rdkit <- import("rdkit.Chem", convert = TRUE)
Feature Extraction from CSVThe primary function, collect_features_from_csv(), handles the entire workflow: SMILES validation, canonicalization, and streamed feature extraction via the ChemicalDice API.Note: Your input CSV file must contain a column named SMILES.# Example usage: Replace "smiles.csv" with your file path
# Replace "API_KEY" with your actual ChemicalDice API key
features <- collect_features_from_csv(
    file_path = "smiles.csv",
    key = "YOUR_API_KEY"
)
ArgumentDescriptionfile_pathPath to the input CSV containing the SMILES column.keyYour required API_KEY for accessing the ChemicalDice service.OutputThe function returns a numeric matrix of features where:Rows correspond to the input molecules (in the order they were processed).Columns correspond to the generated ChemicalDice embeddings (features).Important: The function will automatically overwrite the original CSV file with the canonicalized SMILES strings after validation.Benchmarking & PerformanceFor performance details on the underlying ChemicalDice API, including resource consumption and time complexity related to molecule size, please refer to the [Benchmarking section on our main documentation page]. (Link to external documentation if available).
