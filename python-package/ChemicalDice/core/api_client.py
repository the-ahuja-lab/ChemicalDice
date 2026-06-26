"""
API Client for ChemicalDice Integrator Services.
Handles streaming high-dimensional molecular embeddings from a backend ASGI server.
"""

import os
import math
import tempfile
import logging
from typing import Optional

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_URL = "http://chemicaldice.ahujalab.iiitd.edu.in:8001/stream-features-from-csv"
BATCH_SIZE = 32
NUM_FEATURES = 8192
DTYPE = np.float32
DEFAULT_KEY = "ajci8JYskz5FulkeXaczeQmVTYF1cABnP7pdfUFDBgjuCVJZ6R7YjA" 


def is_valid_smiles(smiles: str) -> bool:
    """Validate a single SMILES string using RDKit."""
    if not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def process_smiles(s: str) -> Optional[str]:
    """Convert a SMILES string to its canonical form using RDKit."""   
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        logger.warning(f"Invalid SMILES encountered during canonicalization: {s}")
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def collect_features_from_csv(
    filepath: str,
    convert_to_canonical: bool = False,
    key: str = DEFAULT_KEY,
    url: str = DEFAULT_URL
) -> Optional[pd.DataFrame]:
    """
    Collect feature embeddings from a CSV containing a 'SMILES' column by querying the CDI API.
    Uses chunked binary streams to prevent memory exhaustion and large payload overhead.

    Args:
        filepath: Path to the input CSV file. Must contain a 'SMILES' column.
        convert_to_canonical: If True, computationally canonicalizes SMILES before upload.
        key: The authentication key sent in the "X-API-Key" request header.
        url: The absolute HTTP endpoint corresponding to the active deployment ASGI server.

    Returns:
        pd.DataFrame: A populated dataframe containing the input SMILES alongside the 8192-D CDI vectors.
                      Returns None on network failure.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")   

    df_data = pd.read_csv(filepath)
    if 'SMILES' not in df_data.columns:
        raise ValueError("CSV must contain exactly one 'SMILES' column.")

    # 1. Validation Phase
    df_data['is_valid'] = df_data['SMILES'].apply(is_valid_smiles)     
    num_invalid = (~df_data['is_valid']).sum()

    if num_invalid > 0:
        logger.warning(f"Found {num_invalid} structurally invalid SMILES. They will be removed from processing.")
        # Store metadata state in original file per legacy pattern requirements
        df_data.to_csv(filepath, index=False)
        df_data = df_data[df_data['is_valid']].reset_index(drop=True)  
        logger.info(f"Proceeding strictly with {len(df_data)} valid SMILES.")
    else:
        logger.info("All SMILES structures verified as valid.")        

    # 2. Pre-Processing Phase
    if convert_to_canonical:
        logger.info("Converting SMILES to canonical graph representation...")
        df_data['SMILES'] = df_data['SMILES'].apply(process_smiles)    

    # 3. Secure File Handover
    # Write sanitized inputs to a protected temporary CSV mapped for upload
    fd, tmp_path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    df_data.to_csv(tmp_path, index=False)
    logger.debug(f"Temporary file instantiated for upload routing: {tmp_path}")

    num_rows = df_data.shape[0]
    total_batches = math.ceil(num_rows / BATCH_SIZE)
    batch_byte_size = BATCH_SIZE * NUM_FEATURES * np.dtype(DTYPE).itemsize

    headers = {"X-API-Key": key}
    received_batches = []

    # 4. Asynchronous Network Stream
    try:
        with open(tmp_path, 'rb') as csv_file:
            files = {'file': (os.path.basename(tmp_path), csv_file, 'text/csv')}

            logger.info(f"Establishing stream connection to {url}")    
            with requests.post(url, files=files, headers=headers, stream=True) as response:
                response.raise_for_status()

                # TQDM visually tracks byte-packet reconstruction      
                progress_bar = tqdm(total=total_batches, unit="batch", desc="Receiving Embeddings")
                for chunk in response.iter_content(chunk_size=batch_byte_size):
                    if chunk:
                        # Reconstruct tensor batch from direct byte translation
                        batch = np.frombuffer(chunk, dtype=DTYPE).reshape(BATCH_SIZE, NUM_FEATURES)
                        received_batches.append(batch)
                        progress_bar.update(1)
                progress_bar.close()

    except requests.exceptions.RequestException as e:
        logger.error(f"Critical stream failure: {e}")
        return None
    finally:
        # Guarantee strict cleanup of intermediate artifacts
        try:
            os.remove(tmp_path)
        except OSError as e:
            logger.warning(f"Unable to cleanly unlink temporary file {tmp_path}: {e}")

    if not received_batches:
        logger.error("Empty stream generated by backend. No batches captured.")
        return None

    logger.info("Merging binary batches into dense index...")

    # Trim terminal padding from strict-batch processing lengths       
    final_array_with_padding = np.vstack(received_batches)
    final_array = final_array_with_padding[:num_rows]

    # Map raw dimensions back to discrete tabular semantics
    feature_cols = [f'CDI{i+1}' for i in range(final_array.shape[1])]  
    df_features = pd.DataFrame(final_array, columns=feature_cols)      
    df_features.insert(0, 'SMILES', df_data['SMILES'].values)

    return df_features