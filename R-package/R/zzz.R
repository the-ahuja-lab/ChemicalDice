# This file ensures required packages and Python modules are loaded when the package is attached

.onAttach <- function(libname, pkgname) {
  suppressPackageStartupMessages({
    library(httr)
    library(progress)
    library(jsonlite)
    library(data.table)
    library(reticulate)
  })
}

