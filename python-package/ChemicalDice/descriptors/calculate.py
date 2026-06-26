import argparse
import os
import sys
from ChemicalDice.descriptors import (
    smiles_preprocess, bioactivity, chemberta, Grover, ImageMol, chemical, quantum
)

def calculate_descriptors(input_file, output_dir="Chemicaldice_data", descriptors=None):
    """
    Calculate molecular descriptors from SMILES.
    
    Args:
        input_file (str): Path to input CSV file.
        output_dir (str): Directory to store output CSVs.
        descriptors (list): List of descriptors to calculate. Defaults to all.
    """
    if descriptors is None or "all" in descriptors:
        descriptors = ["mopac", "grover", "imagemol", "chemberta", "signaturizer", "mordred"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # download prerequisites for quantum, grover and ImageMol
    quantum.get_mopac_prerequisites()
    
    # preprocessing of smiles to different formats
    print(f"Preprocessing {input_file}...")
    smiles_preprocess.add_canonical_smiles(input_file)
    smiles_preprocess.create_mol2_files(input_file)
    smiles_preprocess.create_sdf_files(input_file)
    
    # Map of descriptor names to their calculation functions
    calc_map = {
        "mopac": lambda: quantum.descriptor_calculator(input_file, output_file=os.path.join(output_dir, "mopac.csv")),
        "grover": lambda: Grover.get_embeddings(input_file, output_file_name=os.path.join(output_dir, "Grover.csv")),
        "imagemol": lambda: ImageMol.image_to_embeddings(input_file, output_file_name=os.path.join(output_dir, "ImageMol.csv")),
        "chemberta": lambda: chemberta.smiles_to_embeddings(input_file, output_file=os.path.join(output_dir, "Chemberta.csv")),
        "signaturizer": lambda: bioactivity.calculate_descriptors(input_file, output_file=os.path.join(output_dir, "Signaturizer.csv")),
        "mordred": lambda: chemical.descriptor_calculator(input_file, output_file=os.path.join(output_dir, "mordred.csv")),
    }

    for desc in descriptors:
        if desc in calc_map:
            print(f"Calculating {desc}...")
            try:
                calc_map[desc]()
            except Exception as e:
                print(f"Error calculating {desc}: {e}")
        else:
            print(f"Warning: Descriptor {desc} not recognized.")

def main():
    parser = argparse.ArgumentParser(description="Calculate molecular descriptors from SMILES.")
    parser.add_argument("--input", required=True, help="Input CSV file containing SMILES.")
    parser.add_argument("--output_dir", default="Chemicaldice_data", help="Directory to store output CSVs.")
    parser.add_argument("--descriptors", nargs="+", 
                        choices=["mopac", "grover", "imagemol", "chemberta", "signaturizer", "mordred", "all"],
                        default=["all"],
                        help="List of descriptors to calculate.")
    
    args = parser.parse_args()
    calculate_descriptors(args.input, args.output_dir, args.descriptors)

if __name__ == "__main__":
    main()