from ChemicalDice.molfeaturegenerator import get_available_features_generators, get_features_generator
from ChemicalDice.molgraph import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from ChemicalDice.molgraph import MolGraph, BatchMolGraph, MolCollator
from ChemicalDice.moldataset import MoleculeDataset, MoleculeDatapoint
from ChemicalDice.scaler import StandardScaler

# import requests
# import pkg_resources
# import tqdm
# import os

# def download_file(url, filename):
#     response = requests.get(url, stream=True)

#     # Get the total file size
#     file_size = int(response.headers.get("Content-Length", 0))

#     # Initialize the progress bar
#     progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)

#     with open(filename, 'wb') as file:
#         for data in progress.iterable:
#             # Write data read to the file
#             file.write(data)

#             # Update the progress bar manually
#             progress.update(len(data))

# import tarfile

# def extract_tar_gz(file_path,path):
#     with tarfile.open(file_path, 'r:gz') as tar:
#         tar.extractall(path)

# def get_grover_prerequisites(path):
#     # URL of the file to be downloaded
#     url = "https://ai.tencent.com/ailab/ml/ml-data/grover-models/pretrain/grover_large.tar.gz"
#     # Name of the file to save as
#     filename = os.path.join(path,"grover_large.tar.gz")
#     download_file(url, filename)
#     print("Grover model is downloaded")
#     # Path to the tar.gz file
#     extract_tar_gz(filename,path)
#     print("Grover model is extracted")

# checkpoint_path_grover = pkg_resources.resource_filename('ChemicalDice', 'models/grover_large.pt')
# if os.path.exists(checkpoint_path_grover):
#     get_grover_prerequisites(checkpoint_path_grover)
# # from .utils import load_features, save_features
