# client.R

# Load required libraries
library(httr)
library(progress)
library(jsonlite)
library(data.table)
library(reticulate)

# --- Configuration (must match the server) ---
BATCH_SIZE <- 32
NUM_FEATURES <- 8192 # Assuming this is your feature size
# In R, we specify the size in bytes. float32 is 4 bytes.
FLOAT_SIZE_BYTES <- 4
DTYPE_R <- "numeric" # The 'what' argument for readBin


# Before you run the function, you might need to tell reticulate which
# Python environment to use (the one where RDKit is installed).
# Import the necessary RDKit Python module into an R object
# rdkit <- import("rdkit.Chem", convert = TRUE)

#' Validates and canonicalizes a SMILES string using RDKit
#'
#' This function uses the Python RDKit library via reticulate to parse
#' a SMILES string. If the string is valid, it returns the canonical
#' SMILES. If it is invalid, it returns NA.
#'
#' @param smiles_string A character string representing a molecule.
#' @return The canonical SMILES string or NA_character_ if invalid.

process_smiles <- function(smiles_string) {
  rdkit <- import("rdkit.Chem", convert = TRUE)
  # Use tryCatch to handle errors from RDKit (for invalid SMILES)
  tryCatch({
    # 1. Convert the SMILES string to an RDKit molecule object
    mol <- rdkit$MolFromSmiles(smiles_string)
    
    # This check is crucial because MolFromSmiles returns NULL for invalid SMILES
    if (is.null(mol)) {
      stop("Invalid SMILES string") # Force an error to be caught
    }
    
    # 2. Convert the molecule object back to a canonical SMILES string
    canonical_smiles <- rdkit$MolToSmiles(mol)
    
    return(canonical_smiles)
    
  }, error = function(e) {
    # 3. If any step above fails, catch the error and return NA
    # message(paste("Invalid SMILES:", smiles_string, "->", e$message))
    return(NA_character_)
  })
}

is_valid_smiles <- function(smiles_string) {
  rdkit <- import("rdkit.Chem", convert = TRUE)
  tryCatch({
    mol <- rdkit$MolFromSmiles(smiles_string)
    if (is.null(mol)) {
      return(FALSE)
    }
    return(TRUE)
  }, error = function(e) {
    return(FALSE)
  })
}

# --- Main Function ---
collect_features_from_csv <- function(filepath, key , convert_to_canonical = TRUE) {
  
  # --- 1. Pre-process and Validate the Input CSV ---
  library(data.table)
  rdkit <- import("rdkit.Chem", convert = TRUE)
  message("Reading and validating CSV...")

  URL <- "https://chemicaldice.ahujalab.iiitd.edu.in/stream-features-from-csv"
  
  # Use data.table's fread for fast reading
  df_data <- fread(filepath)
  
  if (!"SMILES" %in% names(df_data)) {
    stop("CSV must contain a 'SMILES' column.")
  }
  
  # Apply the processing function to the SMILES column
  # Note: `lapply` returns a list, so we unlist it back to a vector
  #   df_data[, SMILES := unlist(lapply(SMILES, process_smiles))]

  df_data$is_valid <- unlist(lapply(df_data$SMILES, is_valid_smiles))

  num_invalid <- sum(!df_data$is_valid)


  if (num_invalid > 0) {
    # Print invalid rows
    print(df_data[!df_data$is_valid, ])
    
    cat(sprintf("Found %d invalid SMILES. See above for details.\n", num_invalid))
    # cat("There are invalid SMILES in the input CSV. Please fix or remove them before proceeding.\n")
    
    # Save dataframe to CSV
    write.csv(df_data, filepath, row.names = FALSE)
    
    # Keep only valid SMILES and reset rownames (like reset_index in pandas)
    df_data <- df_data[df_data$is_valid, , drop = FALSE]
    rownames(df_data) <- NULL
    
    cat(sprintf("Proceeding with %d valid SMILES.\n", nrow(df_data)))
    
    } else {
    cat("All SMILES are valid.\n")
    }


  
  # Assume df_data is a data.frame with a column "SMILES"
  # Assume process_smiles() is already defined (e.g., via reticulate with RDKit)

  if (convert_to_canonical) {
    cat("Converting SMILES to canonical form...\n")
    
    # Apply process_smiles to each SMILES string
    df_data$SMILES <- unlist(lapply(df_data$SMILES, process_smiles))
    
    # Save canonicalized dataframe to a temporary CSV file
    filepath <- tempfile(fileext = ".csv")
    write.csv(df_data, filepath, row.names = FALSE)

    #cat(sprintf("Coverted SMILES to  Canonical SMILES \n"))
    # cat(sprintf("Saved canonical SMILES to temp file: %s\n", filepath))
    
  } else {
    # Save dataframe to a temporary CSV file without canonicalization
    filepath <- tempfile(fileext = ".csv")
    write.csv(df_data[,c("SMILES"),drop=FALSE], filepath, row.names = FALSE)
    
    cat(sprintf("Saved SMILES to temp file: %s\n", filepath))
  }

  
  # Overwrite the original file with the processed data
  fwrite(df_data, filepath)
  
  # --- 2. Prepare for Streaming Request ---
  received_batches <- list()
  NUM_ROWS <- nrow(df_data)
  batch_byte_size <- as.integer(BATCH_SIZE * NUM_FEATURES * FLOAT_SIZE_BYTES)
  total_batches <- ceiling(NUM_ROWS / BATCH_SIZE)

  headers <- c()
  if (!is.null(key)) {
    headers <- add_headers(`X-API-Key` = key)
  }

  resp <- POST(
    URL,
    body = list(file = upload_file(filepath, type = "text/csv")),
    encode = "multipart",
    headers
  )

  if (http_error(resp)) {
    stop(sprintf("HTTP request failed: %s", status_code(resp)))
  }

  message(sprintf("Sent %s. Receiving stream...", filepath))
  raw_content <- content(resp, "raw")
  if (length(raw_content) == 0L) {
    message("No data received.")
    return(NULL)
  }

  buffer <- raw()
  pb <- progress_bar$new(format = "  [:bar] :percent (:current/:total) batches", total = total_batches, clear = FALSE, width = 60)

  byte_index <- 1L
  while (byte_index <= length(raw_content)) {
    # Append next chunk to buffer
    buffer <- c(buffer, raw_content[byte_index:min(length(raw_content), byte_index + batch_byte_size - 1L)])
    byte_index <- byte_index + batch_byte_size
    
    # Process all complete batches in buffer
    while (length(buffer) >= batch_byte_size) {
      chunk_raw <- buffer[1:batch_byte_size]
      buffer <- buffer[-(1:batch_byte_size)]
      
      con <- rawConnection(chunk_raw, open = "rb")
      values <- readBin(con, what = "numeric", n = BATCH_SIZE * NUM_FEATURES, size = 4, endian = "little")
      close(con)
      
      mat <- matrix(values, nrow = BATCH_SIZE, ncol = NUM_FEATURES, byrow = TRUE)
      received_batches[[length(received_batches) + 1]] <- mat
      pb$tick()
    }
  }
  pb$terminate()

  # Handle leftover partial batch
  if (length(buffer) > 0) {
    con <- rawConnection(buffer, open = "rb")
    values <- readBin(con, what = "numeric", n = length(buffer) / FLOAT_SIZE_BYTES, size = 4, endian = "little")
    close(con)
    rows <- length(values) / NUM_FEATURES
    if (rows > 0) {
      mat <- matrix(values, nrow = rows, ncol = NUM_FEATURES, byrow = TRUE)
      received_batches[[length(received_batches) + 1]] <- mat
    }
  }

  # Concatenate batches and trim padding to NUM_ROWS
  all_rows <- do.call(rbind, received_batches)
  final_array <- all_rows[seq_len(NUM_ROWS), , drop = FALSE]

  feature_cols <- paste0("CDI", seq_len(ncol(final_array)))
  df_features <- as.data.frame(final_array)
  colnames(df_features) <- feature_cols
  df_features <- cbind(SMILES = df_data$SMILES, df_features)

  message("\nStream finished. Concatenating batches...")
  if (num_invalid > 0) {
    cat("Invalid SMILES were skipped. Check your input file where is_valid == FALSE.\n")
  }
  return(df_features)
}



