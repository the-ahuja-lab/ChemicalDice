import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math
import hashlib
import base64
from rdkit import Chem
import sys
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math
import tempfile



# --- Configuration (must match the server) ---
# URL = "https://chemicaldice.ahujalab.iiitd.edu.in/stream-features-from-csv"
URL = "http://chemicaldice.ahujalab.iiitd.edu.in:8001/stream-features-from-csv"
BATCH_SIZE = 32
NUM_FEATURES = 8192
DTYPE = np.float32




def collect_features_from_csv(filepath: str,key: str ,convert_to_canonical: bool = True):
    """
    Collect feature embeddings from a CSV of SMILES strings by sending the file to a CDI
    API service and streaming back binary batches of numeric features.
    This function:
    - Loads a CSV file and requires a column named 'SMILES'.
    - Validates each SMILES string 
    - Optionally canonicalizes SMILES using process_smiles() and writes a temporary CSV
        containing the (possibly canonicalized) SMILES prior to upload.
    - Posts the CSV as multipart/form-data to a CDI API service endpoint (module-level
        constant URL), passing the provided API key via the "X-API-Key" header.
        returns a pandas.DataFrame where the first column is
        the SMILES strings and the remaining columns are features named "CDI1", "CDI2", ...
    Args:
            filepath (str): Path to the input CSV file. The CSV must contain a column named 'SMILES'.
            key (str): API key value to send in the "X-API-Key" request header.
            convert_to_canonical (bool, optional): If True (default), canonicalize SMILES using
                    process_smiles() before upload. Canonicalization results are written to a temporary
                    CSV which is submitted to the server; the original file is left untouched.
    Returns:
            pandas.DataFrame or None:
                    - On success: a DataFrame with shape (N, M+1), where N is the number of SMILES rows
                        in the input CSV and M == NUM_FEATURES is the number of numeric features per row.
                        The first column is 'SMILES' and the remaining columns are labeled 'CDI1'..'CDIM'.
                    - On network/request failure or if no batches are received: None is returned (after
                        printing an error or warning message).
                    - On invalid input (see Raises) an exception is raised instead of returning None.
    """

    df_data = pd.read_csv(filepath)
    if 'SMILES' not in df_data.columns:
        raise ValueError("CSV must contain a 'SMILES' column.")

    df_data['is_valid'] = df_data['SMILES'].apply(is_valid_smiles)
    
    num_invalid = (~df_data['is_valid']).sum()
    if num_invalid > 0:
        print(df_data[~df_data['is_valid']])
        print(f"Found {num_invalid} invalid SMILES. See above for details.")
        # print("There are invalid SMILES in the input CSV. Please fix or remove them before proceeding.")
        # Ask whether to continue with only valid SMILES or exit
        df_data.to_csv(filepath, index=False)
        df_data = df_data[df_data['is_valid']].reset_index(drop=True)
        print(f"Proceeding with {len(df_data)} valid SMILES.")
    else:
        print("All SMILES are valid.")


    if convert_to_canonical:
        print("Converting SMILES to canonical form...")
        df_data['SMILES'] = df_data['SMILES'].apply(process_smiles)
        # Save canonicalized dataframe to a temporary CSV file and update filepath to point to it
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
        tmp.close()
        df_data.to_csv(tmp.name, index=False)
        filepath = tmp.name
        print(f"Saved canonical SMILES to temp file: {filepath}")
    else:
        # Save canonicalized dataframe to a temporary CSV file and update filepath to point to it
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
        tmp.close()
        df_data.to_csv(tmp.name, index=False)
        filepath = tmp.name
        print(f"Saved canonical SMILES to temp file: {filepath}")

    NUM_ROWS = df_data.shape[0]
    # Calculate the size of one complete batch in bytes
    batch_byte_size = BATCH_SIZE * NUM_FEATURES * np.dtype(DTYPE).itemsize

    # Calculate expected number of batches for the progress bar
    total_batches = math.ceil(NUM_ROWS / BATCH_SIZE)
    # Calculate the size of one complete batch in bytes
    batch_byte_size = BATCH_SIZE * NUM_FEATURES * np.dtype(DTYPE).itemsize
    
    # Calculate expected number of batches for the progress bar
    total_batches = math.ceil(NUM_ROWS / BATCH_SIZE)
    headers = {"X-API-Key": key}
    received_batches = []
    try:
        # Open the local CSV file to be sent in the request
        with open(filepath, 'rb') as csv_file:
            # The 'files' dict tells requests to send a multipart/form-data POST
            # The key 'file' must match the argument name in the FastAPI endpoint
            files = {'file': (os.path.basename(filepath), csv_file, 'text/csv')}
            
            with requests.post(URL, files=files,headers=headers, stream=True) as response:
                response.raise_for_status()
                print(f"Sent {filepath}. Receiving stream...")

                progress_bar = tqdm(total=total_batches, unit="batch")

                for chunk in response.iter_content(chunk_size=batch_byte_size):
                    if chunk:
                        batch = np.frombuffer(chunk, dtype=DTYPE).reshape(BATCH_SIZE, NUM_FEATURES)
                        received_batches.append(batch)
                        progress_bar.update(1)
                
                progress_bar.close()

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return

    if not received_batches:
        print("No batches were received.")
        return
    # Remove temporary file if it was created in the system temp directory
    try:
        tmpdir = os.path.abspath(tempfile.gettempdir())
        file_abspath = os.path.abspath(filepath)
        if file_abspath.startswith(tmpdir + os.sep) or file_abspath == tmpdir:
            os.remove(file_abspath)
    except Exception as e:
        print(f"Warning: could not remove temporary file {filepath}: {e}")
    # Assemble the final array
    print("\nStream finished. Concatenating batches...")

    if num_invalid > 0:
        print("Invalid SMILES were skipped. Check your input file which is_valid column where False indicates invalid SMILES.")
    final_array_with_padding = np.vstack(received_batches)
    
    # Trim any padding added to the last batch
    final_array = final_array_with_padding[:NUM_ROWS]

    # Convert the NumPy array to a DataFrame and prepend the SMILES column
    feature_cols = [f'CDI{i+1}' for i in range(final_array.shape[1])]
    df_features = pd.DataFrame(final_array, columns=feature_cols)
    df_features.insert(0, 'SMILES', df_data['SMILES'].values)

    return df_features




def process_smiles(s):
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        print(f"Invalid SMILES: {s}")
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None
