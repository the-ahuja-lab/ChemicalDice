"""
Data Preparation Utilities for CDI Model Training.
Handles external descriptor extraction triggers and efficient big-data conversion (CSV -> HDF5).
"""

import os
import logging
from typing import List, Dict

import pandas as pd
import numpy as np
import h5py
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)

# --- Descriptor Generation ---

def generate_multi_descriptors(input_csv: str, output_dir: str = "Chemicaldice_data"):
    """
    Calculates six separate structural/bioactivity descriptors utilizing the legacy ChemicalDice logic.
    Assumes `ChemicalDice` is pip-installed from Test.PyPI containing the generation engines.
    """
    import ChemicalDice.smiles_preprocess as pre
    import ChemicalDice.bioactivity as bio
    import ChemicalDice.chemberta as chb
    import ChemicalDice.Grover as grv
    import ChemicalDice.ImageMol as imm
    import ChemicalDice.chemical as chm
    import ChemicalDice.quantum as qtm

    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Bootstrapping advanced descriptor calculation engines...")
    qtm.get_mopac_prerequisites()

    logger.info("Executing SMILES canonization & spatial configurations mapping (MOL2/SDF)...")
    pre.add_canonical_smiles(input_csv)
    pre.create_mol2_files(input_csv)
    pre.create_sdf_files(input_csv)

    logger.info("Calculating independent modalities...")
    qtm.descriptor_calculator(input_csv, output_file=os.path.join(output_dir, "mopac.csv"))
    grv.get_embeddings(input_csv, output_file_name=os.path.join(output_dir, "Grover.csv"))
    imm.image_to_embeddings(input_csv, output_file_name=os.path.join(output_dir, "ImageMol.csv"))
    chb.smiles_to_embeddings(input_csv, output_file=os.path.join(output_dir, "Chemberta.csv"))
    bio.calculate_descriptors(input_csv, output_file=os.path.join(output_dir, "Signaturizer.csv"))
    chm.descriptor_calculator(input_csv, output_file=os.path.join(output_dir, "mordred.csv"))
    logger.info("Descriptor generation complete.")


# --- HDF5 Conversion & Optimization ---

def get_common_ids(csv_files: List[str], chunksize: int = 10000) -> set:
    """Intersects entity IDs across highly dimensional chunks to guarantee pairwise representation."""
    common_ids = None

    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing required descriptor payload: {csv_path}")

        ids_in_file = set()
        sep = '\t' if 'imagemol' in os.path.basename(csv_path).lower() else ','

        for chunk in pd.read_csv(csv_path, usecols=["id"], chunksize=chunksize, sep=sep):
            ids_in_file.update(chunk["id"].dropna().values)

        if common_ids is None:
            common_ids = ids_in_file
        else:
            common_ids &= ids_in_file  # set intersection

        logger.debug(f"{os.path.basename(csv_path)} -> Intersection yields {len(common_ids)} valid anchors.")

    return common_ids


def convert_csv_to_hdf5(csv_path: str, h5_path: str, common_ids: set, chunksize: int = 10000) -> None:
    """Chunks structured text matrices directly into compressed out-of-core compatible format."""
    with h5py.File(h5_path, 'w') as h5_file:
        first_chunk = True
        sep = '\t' if 'imagemol' in os.path.basename(csv_path).lower() else ','

        for chunk in pd.read_csv(csv_path, chunksize=chunksize, sep=sep):
            # Strict spatial intersection
            chunk = chunk[chunk["id"].isin(common_ids)]
            if chunk.empty:
                continue

            # Expunge metadata labels prior to matrix formatting
            chunk = chunk.drop(columns=[col for col in ["id", "SMILES"] if col in chunk.columns])

            if first_chunk:
                h5_file.create_dataset("data", data=chunk.values, maxshape=(None, chunk.shape[1]), chunks=True)
                first_chunk = False
            else:
                original_shape = h5_file["data"].shape[0]
                h5_file["data"].resize(original_shape + chunk.shape[0], axis=0)
                h5_file["data"][-chunk.shape[0]:] = chunk.values


def process_mordred_with_imputation(csv_path: str, h5_path: str, common_ids: set, chunksize: int = 10000, n_neighbors: int = 5) -> None:
    """Distinct conversion handler for noisy Mordred schemas utilizing real-time KNN imputation."""
    
    # Static fallback subset representation of stable properties
    SUBSET_COLUMNS = [
        "ABC", "ABCGG", "SlogP_VSA9", "ATS7are", "n8HRing", "n5Ring", "FNSA4", "ECIndex", "NsssssP", "ATSC1se", 
        "SMR", "AATS3p", "ATS5m", "SssGeH2", "ATSC5i", "AMID", "Sm", "nG12AHRing", "SssssN", "n9aHRing"
        # Truncated subset logic handled cleanly
    ]

    imputer = KNNImputer(n_neighbors=n_neighbors)

    with h5py.File(h5_path, 'w') as h5_file:
        first_chunk = True

        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            chunk = chunk[chunk["id"].isin(common_ids)]
            if chunk.empty:
                continue

            # Enforce rigid dimensionality across volatile samples
            valid_cols = [col for col in SUBSET_COLUMNS if col in chunk.columns]
            chunk = chunk[valid_cols]
            chunk = chunk.apply(pd.to_numeric, errors="coerce")

            imputed = imputer.fit_transform(chunk)
            imputed_df = pd.DataFrame(imputed, columns=valid_cols)

            if first_chunk:
                h5_file.create_dataset("data", data=imputed_df.values, maxshape=(None, imputed_df.shape[1]), chunks=True)
                first_chunk = False
            else:
                original_shape = h5_file["data"].shape[0]
                h5_file["data"].resize(original_shape + imputed_df.shape[0], axis=0)
                h5_file["data"][-imputed_df.shape[0]:] = imputed_df.values


def format_dataset_pipeline(input_dir: str = "Chemicaldice_data", output_dir: str = "Chemicaldice_data") -> None:
    """Orchestrates end-to-end HDF5 formatting schema against exactly 6 baseline files."""
    csv_files = [
        os.path.join(input_dir, "mopac.csv"),
        os.path.join(input_dir, "Grover.csv"),
        os.path.join(input_dir, "ImageMol.csv"),
        os.path.join(input_dir, "Chemberta.csv"),
        os.path.join(input_dir, "Signaturizer.csv"),
        os.path.join(input_dir, "mordred.csv"),
    ]

    logger.info("Computing global ID intersection maps...")
    common_ids = get_common_ids(csv_files)
    logger.info(f"Targeting {len(common_ids)} intersecting instances globally.")

    standard_files = {
        "mopac.csv": "mopac.h5",
        "Grover.csv": "Grover.h5",
        "ImageMol.csv": "ImageMol.h5",
        "Chemberta.csv": "Chemberta.h5",
        "Signaturizer.csv": "Signaturizer.h5",
    }

    os.makedirs(output_dir, exist_ok=True)
    
    for csv_name, h5_name in standard_files.items():
        logger.info(f"Restructuring {csv_name} -> {h5_name}")
        convert_csv_to_hdf5(
            os.path.join(input_dir, csv_name),
            os.path.join(output_dir, h5_name),
            common_ids
        )

    logger.info("Restructuring mordred.csv with spatial imputation...")
    process_mordred_with_imputation(
        os.path.join(input_dir, "mordred.csv"),
        os.path.join(output_dir, "mordred.h5"),
        common_ids
    )

    logger.info("✅ Dataset universally optimized into chunked native HDF5 layouts.")
