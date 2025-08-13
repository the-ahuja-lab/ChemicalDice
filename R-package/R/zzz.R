# This file ensures required packages and Python modules are loaded when the package is attached

.onAttach <- function(libname, pkgname) {
  suppressPackageStartupMessages({
    library(httr)
    library(progress)
    library(jsonlite)
    library(data.table)
    library(reticulate)
  })
  py_require("rdkit")
  assign("rdkit", import("rdkit.Chem", convert = TRUE), envir = parent.env(environment()))
}
