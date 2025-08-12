\name{collect_features_from_csv}
\alias{collect_features_from_csv}
\title{Collect Features from SMILES in a CSV File}
\description{
  Reads a CSV file containing SMILES strings, validates and canonicalizes them using RDKit (via reticulate), and streams the file to a remote server to receive molecular feature embeddings in batches.
}
\usage{
collect_features_from_csv(filepath, key = NULL)
}
\arguments{
  \item{filepath}{Path to the CSV file containing a 'SMILES' column.}
  \item{key}{(Optional) A base64-encoded string for the server URL.}
}
\details{
  The function reads the CSV, validates and canonicalizes SMILES using RDKit via reticulate, overwrites the file with canonical SMILES, and streams the file to a server endpoint. The server is expected to return feature vectors as a binary stream, which are collected and assembled into a matrix.
}
\value{
  A numeric matrix of molecular features (rows = molecules, columns = features), or NULL if an error occurs or no batches are received.
}
\examples{
# collect_features_from_csv("molecules.csv")
}
\seealso{
  \code{process_smiles} for SMILES validation and canonicalization.
}
