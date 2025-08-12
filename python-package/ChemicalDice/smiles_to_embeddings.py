import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math
import hashlib
import base64
from rdkit import Chem
# --- Configuration (must match the server) ---

BATCH_SIZE = 32
NUM_FEATURES = 8192
DTYPE = np.float32




def decode(encoded_text):
    # Decode base64 to bytes, then to string
    return base64.b64decode(encoded_text.encode()).decode()


# encoded=
# print("Decoded:", decode(encoded))

def collect_features_from_csv(filepath: str,key: str = None):
    """
    Sends a CSV file to the server and collects the streamed feature
    batches into a final NumPy array.
    """
    received_batches = []
    key = decode(key)

    df_data = pd.read_csv(filepath)




    df_data = pd.read_csv(filepath)
    if 'SMILES' not in df_data.columns:
        raise ValueError("CSV must contain a 'SMILES' column.")



    df_data['SMILES'] = df_data['SMILES'].apply(process_smiles)
    df_data['is_valid'] = df_data['SMILES'].notnull()

    # Print summary
    num_invalid = (~df_data['is_valid']).sum()
    if num_invalid > 0:
        print(f"Found {num_invalid} invalid SMILES. See above for details.")
        raise ValueError("There are invalid SMILES in the input CSV. Please fix or remove them before proceeding.")
    else:
        print("All SMILES are valid.")

    # Overwrite the SMILES column with canonical SMILES for valid entries
    # df_data.loc[df_data['is_valid'], 'SMILES'] = df_data.loc[df_data['is_valid'], 'canonical_SMILES']
    

    # Save back to the same file
    # filepath = os.path.basename(filepath)
    
    df_data.to_csv(filepath, index=False)

    NUM_ROWS = df_data.shape[0]
    # Calculate the size of one complete batch in bytes
    batch_byte_size = BATCH_SIZE * NUM_FEATURES * np.dtype(DTYPE).itemsize

    # Calculate expected number of batches for the progress bar
    total_batches = math.ceil(NUM_ROWS / BATCH_SIZE)

    try:
        # Open the local CSV file to be sent in the request
        with open(filepath, 'rb') as csv_file:
            # The 'files' dict tells requests to send a multipart/form-data POST
            # The key 'file' must match the argument name in the FastAPI endpoint
            files = {'file': (os.path.basename(filepath), csv_file, 'text/csv')}

            with requests.post(key, files=files, stream=True) as response:
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

    # Assemble the final array
    print("\nStream finished. Concatenating batches...")
    final_array_with_padding = np.vstack(received_batches)

    # Trim any padding added to the last batch
    final_array = final_array_with_padding[:NUM_ROWS]

    print("Assembly complete!")
    print("Done")
    return final_array


def process_smiles(s):
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        print(f"Invalid SMILES: {s}")
        return None
    return Chem.MolToSmiles(mol, canonical=True)



