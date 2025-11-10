import os
import shutil
import sys
from multiprocessing import Pool
from typing import List, Tuple

from tqdm import tqdm
import subprocess
import pandas as pd
import os
# Arguments to be passed to the called script
from ChemicalDice.utils import get_data, makedirs, load_features, save_features
from ChemicalDice.molfeaturegenerator import get_available_features_generators, \
    get_features_generator
from ChemicalDice.task_labels import rdkit_functional_group_label_features_generator


import random

import numpy as np
import torch
from rdkit import RDLogger
import argparse

from ChemicalDice.parsing import parse_args, get_newest_train_args
from ChemicalDice.utils import create_logger
from ChemicalDice.cross_validate import cross_validate
from ChemicalDice.fingerprint import generate_fingerprints
from ChemicalDice.predict import make_predictions, write_prediction
from ChemicalDice.pretrain import pretrain_model
from ChemicalDice.torchvocab import MolVocab

from ChemicalDice.grover_need import *

import requests
from rdkit import Chem
import tarfile

def get_grover_prerequisites(path):
    """
    Ensure that prerequisites for the Grover model are available in the specified directory.

    This function checks if the Grover model file ('grover_large.pt') exists in the given path. If not,
    it downloads the model from a URL, extracts it from a tar.gz file, and places it in the specified directory.

    Parameters
    ----------
    path : str
        Directory path where the Grover model and its prerequisites will be stored or are already located.

    Returns
    -------
    str
        Path to the Grover model file ('grover_large.pt') within the specified directory.
    """
    if os.path.exists(os.path.join(path,"grover_large.pt")):
        pass
    else:
        # URL of the file to be downloaded
        url = "https://huggingface.co/SuvenduK/grover/resolve/main/grover_large.tar.gz"
        # Name of the file to save as
        filename = os.path.join(path,"grover_large.tar.gz")
        download_file(url, filename)
        print("Grover model is downloaded")
        # Path to the tar.gz file
        extract_tar_gz(filename,path)
        print("Grover model is extracted")
    return os.path.join(path,"grover_large.pt")

def get_embeddings(input_file, output_file_name,output_dir = "temp_data"):
    """
    Generate molecular embeddings using the Grover model.

    This function prepares input data, runs Grover to generate molecular embeddings,
    and saves the embeddings to the specified output file.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file containing Canonical SMILES.
    output_file_name : str
        Name of the file where the molecular embeddings will be saved.
    output_dir : str, optional
        Directory where intermediate and output files will be stored (default is "temp_data").

    Returns
    -------
    None

    """
    checkpoint_path_grover = get_grover_prerequisites(output_dir)
    smiles_df = pd.read_csv(input_file)
    if not os.path.exists(output_dir):
        print("making directory ", output_dir)
        os.makedirs(output_dir)
    smiles_list = smiles_df["Canonical_SMILES"]
    if "id" in smiles_df.columns:
        smiles_id_list = smiles_df["id"]
    else:
        smiles_df["id"] = [ "C"+str(id) for id in range(len(smiles_list))]
        smiles_id_list = smiles_df["id"]

    smiles_list_valid = []
    smiles_id_list_valid = []
    for smiles,id in zip(smiles_list,smiles_id_list):
        if is_valid_smiles(smiles):
            smiles_list_valid.append(smiles)
            smiles_id_list_valid.append(id)
        else:
            print("This is a invalid smiles: ", smiles)
    
    smiles_list = smiles_list_valid
    smiles_id_list = smiles_id_list_valid

    smile_to_graph(smiles_list, output_dir)

    grover_input_file = os.path.join(output_dir,"grover_input.csv")
    features_file = os.path.join(output_dir,"graph_features.npz")
    grover_output_model= os.path.join(output_dir,"graph_features.npz")
    smiles_id_list = "___".join(smiles_id_list)
    # Run the called script with arguments
    # subprocess.run(["python", "grovermain.py", "fingerprint", "--data_path", grover_input_file, "--features_path", features_file , "--checkpoint_path", "ckpts/grover_large.pt",
    #             "--fingerprint_source", "both", "--output", grover_output_model, "--dropout", ".2","--grover_output",output_file_name,"--id_list",smiles_id_list])
    

    # Create a namespace object
    args = argparse.Namespace()

    # Now add attributes to the namespace object
    args.data_path = grover_input_file
    args.features_path = features_file
    args.checkpoint_path = checkpoint_path_grover 
    args.fingerprint_source = "both"
    args.output = grover_output_model
    args.dropout = 0.2
    args.grover_output = output_file_name
    args.id_list = smiles_id_list
    args.parser_name = "fingerprint"
    args.output_path = output_dir
    args.no_cuda = True
    args.no_cache = True

    setup(seed=42)
    # Avoid the pylint warning.
    a = MolVocab
    # supress rdkit logger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # Initialize MolVocab
    mol_vocab = MolVocab

    args = parse_args(args)
    # Now args is an object that has the same attributes as if you had parsed the command line arguments



    train_args = get_newest_train_args()
    logger = create_logger(name='fingerprint', save_dir=None, quiet=False)

    
    feas = generate_fingerprints(args, logger)
    #np.savez_compressed(args.output_path, fps=feas)

# import subprocess
# grover_input_file = "/storage2/suvendu/chemdice/Benchmark_data_descriptors/chemdice_descriptors/tox21data/graphfiles/grover_input.csv"
# features_file = "/storage2/suvendu/chemdice/Benchmark_data_descriptors/chemdice_descriptors/tox21data/graphfiles/graph_features.npz"
# output_file_name = "a.csv"
# grover_output_model = "fp.npz"
# subprocess.run(["python", "grovermain.py", "fingerprint", "--data_path", grover_input_file, "--features_path", features_file , "--checkpoint_path", "ckpts/grover_large.pt",
#                 "--fingerprint_source", "both", "--output", grover_output_model, "--dropout", ".2","--grover_output",output_file_name])












