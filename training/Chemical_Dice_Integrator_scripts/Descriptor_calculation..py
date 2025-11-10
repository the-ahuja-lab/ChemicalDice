# create a directory for storing descriptors filefrom ChemicalDice
from ChemicalDice import smiles_preprocess, bioactivity, chemberta, Grover, ImageMol, chemical, quantum
import os
os.mkdir("Chemicaldice_data")
# download prerequisites for quantum, grover and ImageMol
quantum.get_mopac_prerequisites()
# input file containing SMILES and labels
input_file = "Chembl_35_training_smiles.csv"
# preprocessing of smiles to different formats
smiles_preprocess.add_canonical_smiles(input_file)
smiles_preprocess.create_mol2_files(input_file)
smiles_preprocess.create_sdf_files(input_file)
# calculation of all descriptors
quantum.descriptor_calculator(input_file, output_file="Chemicaldice_data/mopac.csv")
Grover.get_embeddings(input_file,  output_file_name="Chemicaldice_data/Grover.csv")
ImageMol.image_to_embeddings(input_file, output_file_name="Chemicaldice_data/ImageMol.csv")
chemberta.smiles_to_embeddings(input_file, output_file = "Chemicaldice_data/Chemberta.csv")
bioactivity.calculate_descriptors(input_file, output_file = "Chemicaldice_data/Signaturizer.csv")
chemical.descriptor_calculator(input_file, output_file="Chemicaldice_data/mordred.csv")