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
# use_condaenv("your_conda_env_name", required = TRUE)

# Import the necessary RDKit Python module into an R object


#' Validates and canonicalizes a SMILES string using RDKit
#'
#' This function uses the Python RDKit library via reticulate to parse
#' a SMILES string. If the string is valid, it returns the canonical
#' SMILES. If it is invalid, it returns NA.
#'
#' @param smiles_string A character string representing a molecule.
#' @return The canonical SMILES string or NA_character_ if invalid.

process_smiles <- function(smiles_string) {
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



# --- Main Function ---
collect_features_from_csv <- function(filepath, key = NULL) {
  
  # --- 1. Pre-process and Validate the Input CSV ---
  library(data.table)
  message("Reading and validating CSV...")
  
  # Use data.table's fread for fast reading
  df_data <- fread(filepath)
  
  if (!"SMILES" %in% names(df_data)) {
    stop("CSV must contain a 'SMILES' column.")
  }
  
  # Apply the processing function to the SMILES column
  # Note: `lapply` returns a list, so we unlist it back to a vector
  #   df_data[, SMILES := unlist(lapply(SMILES, process_smiles))]
  
  df_data$SMILES <- unlist(lapply(df_data$SMILES, process_smiles))
  
  # Check for invalid entries (which we defined as NA in our function)
  num_invalid <- sum(is.na(df_data$SMILES))
  
  if (num_invalid > 0) {
    stop(paste("Found", num_invalid, "invalid SMILES. Please fix or remove them before proceeding."))
  } else {
    message("All SMILES are valid.")
  }
  
  # Overwrite the original file with the processed data
  fwrite(df_data, filepath)
  
  # --- 2. Prepare for Streaming Request ---
  
  # Decode the key (assuming base64 for this example)
  # The Python `decode` function is not standard, so this is an assumption.
  if (!is.null(key)) {
    URL <- rawToChar(base64_dec(key))
  } else {
    # Provide a default URL if no key is given
    URL <- "http://127.0.0.1:8000/stream-features-from-csv"
  }
  
  received_batches <- list()
  byte_buffer <- raw() # Buffer for incoming raw bytes
  
  NUM_ROWS <- nrow(df_data)
  batch_byte_size <- BATCH_SIZE * NUM_FEATURES * FLOAT_SIZE_BYTES
  total_batches <- ceiling(NUM_ROWS / BATCH_SIZE)
  
  # Initialize progress bar
  pb <- progress_bar$new(
    format = "[:bar] :percent | Batches: :current/:total | ETA: :eta",
    total = total_batches,
    width = 80
  )
  
  # --- 3. Send Request and Collect Streamed Data ---
  
  message(paste("Sent", basename(filepath), "to server. Receiving stream..."))
  
  tryCatch({
    res <- POST(
      url = URL,
      body = list(file = upload_file(filepath, type = 'text/csv')),
      write_stream(function(chunk) {
        byte_buffer <<- c(byte_buffer, chunk)
        while (length(byte_buffer) >= batch_byte_size) {
          batch_bytes <- byte_buffer[1:batch_byte_size]
          
          # Convert raw bytes to a numeric vector
          float_vector <- readBin(batch_bytes, what = DTYPE_R, size = FLOAT_SIZE_BYTES, n = BATCH_SIZE * NUM_FEATURES)
          
          # Reshape vector into a matrix (byrow=TRUE matches NumPy's default)
          batch_matrix <- matrix(float_vector, nrow = BATCH_SIZE, ncol = NUM_FEATURES, byrow = TRUE)
          
          received_batches[[length(received_batches) + 1]] <<- batch_matrix
          byte_buffer <<- byte_buffer[-(1:batch_byte_size)]
          pb$tick()
        }
        return(TRUE) # Keep connection open
      })
    )
    stop_for_status(res) # Check for HTTP errors after stream ends
  }, error = function(e) {
    message(paste("Error during request:", e$message))
    return(NULL)
  })
  
  if (length(received_batches) == 0) {
    message("No batches were received.")
    return(NULL)
  }
  
  # --- 4. Assemble Final Matrix ---
  
  message("\nStream finished. Concatenating batches...")
  final_array_with_padding <- do.call(rbind, received_batches)
  
  # Trim padding from the last batch
  final_array <- final_array_with_padding[1:NUM_ROWS, ]
  
  message("Assembly complete!")
  message("Done")
  return(final_array)
}




