import pandas as pd
from openbabel import pybel
#import mol2_to_image
import time
import os
from tqdm import tqdm

#from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor



from ChemicalDice.smiles_preprocess_need import *








# def add_mol2_files(input_file,output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     smiles_df = pd.read_csv(input_file)
#     smiles_list = smiles_df['SMILES']
#     if 'id' in smiles_df.columns:
#         smiles_id_list = smiles_df['id']
#     else:
#         smiles_df['id'] = [ "C"+str(id) for id in range(len(smiles_list))]
#         smiles_id_list = smiles_df['id']
#     mol2_file_paths = smile_to_mol2(smiles_list, smiles_id_list, output_dir)
#     smiles_df['mol2_files'] = mol2_file_paths
#     smiles_df.to_csv(input_file,index=False)
    #return(smiles_df)

def add_canonical_smiles(input_file):
    """
    Convert a list of SMILES strings to canonical SMILES strings. Add a column Canonical_SMILES to input file.

    Parameters
    ----------
    input_file : str
        Input file containing SMILES column.

    Returns
    -------
    None
        This function updates the input CSV file in place and does not return any value.
    """
    smiles_df = pd.read_csv(input_file)
    smiles_list = smiles_df['SMILES']
    if 'id' in smiles_df.columns:
        smiles_id_list = smiles_df['id']
    else:
        smiles_df['id'] = [ "C"+str(id) for id in range(len(smiles_list))]
        smiles_id_list = smiles_df['id']
    canonical_smiles_list = smile_to_canonical(smiles_list)
    smiles_df['Canonical_SMILES'] = canonical_smiles_list
    smiles_df.to_csv(input_file,index=False)







def create_sdf_files(input_file, output_dir="temp_data/sdffiles"):
  """
  Convert MOL2 files to SDF files and update the input CSV file.

  This function reads a CSV file containing paths to MOL2 files and converts each MOL2 file to an SDF file.
  The generated SDF files are saved in the specified output directory. The input CSV file is updated with
  paths to the generated SDF files.

  Parameters
  ----------
  input_file : str
    Path to the input CSV file containing MOL2 file paths. The CSV file must have columns 'mol2_files' and 'id'.
  output_dir : str, optional
    Path to the directory where the SDF files will be saved. Default is "temp_data/sdffiles".
  Returns
  -------
  None
    This function updates the input CSV file in place and does not return any value.
  """
  smiles_df = pd.read_csv(input_file)
  if not os.path.exists(output_dir):
    print("making directory ", output_dir)
    os.makedirs(output_dir)
  mol2file_name_list = smiles_df['mol2_files']
  id_list = smiles_df['id']
  sdf_list = []
  for mol2file_name,id in tqdm(zip(mol2file_name_list, id_list)):
    try:
        sdf_name = os.path.join(output_dir,id+".sdf")
        if os.path.exists(sdf_name):
            print(sdf_name," already exist")
        else:
            for mol in pybel.readfile("mol2", mol2file_name):
                mymol = mol
            mymol.write("sdf", sdf_name ,overwrite=True)
        sdf_list.append(sdf_name)
    except:
        print("Error in conversion of ", mol2file_name)
        sdf_list.append("")
  smiles_df['sdf_files'] = sdf_list
  smiles_df.to_csv(input_file,index=False)






import multiprocessing

cpu_to_use = multiprocessing.cpu_count() * 0.5
cpu_to_use = int(cpu_to_use)

def create_mol2_files(input_file, output_dir="temp_data/mol2files", ncpu = cpu_to_use):
    """
    Convert SMILES strings from a CSV file to MOL2 files using multiprocessing.

    This function reads a CSV file containing SMILES strings, generates 3D structures,
    and saves them as MOL2 files in the specified output directory. The conversion is
    performed in parallel using multiple CPU cores.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file containing SMILES strings. The CSV file must have
        a column named 'SMILES' and optionally an 'id' column.
    output_dir : str, optional
        Path to the directory where the MOL2 files will be saved. Default is "temp_data/mol2files".
    ncpu : int, optional
        The number of CPU cores to use for parallel processing. Default is half of total number of cores.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    smiles_df = pd.read_csv(input_file)
    smiles_list = smiles_df['SMILES']
    if 'id' in smiles_df.columns:
        smiles_id_list = smiles_df['id']
    else:
        smiles_df['id'] = ["C" + str(id) for id in range(len(smiles_list))]
        smiles_id_list = smiles_df['id']
    smiles_id_tuples = [(smiles, smiles_id, output_dir) for smiles, smiles_id in zip(smiles_list, smiles_id_list)]

    # with Pool() as pool:
    #     mol2_file_paths = list(tqdm(pool.imap(smile_to_mol2, smiles_id_tuples), total=len(smiles_id_tuples)))
    with ProcessPoolExecutor(max_workers=ncpu) as executor:
        mol2_file_paths = list(tqdm(executor.map(smile_to_mol2, smiles_id_tuples), total=len(smiles_id_tuples)))

    smiles_df['mol2_files'] = mol2_file_paths
    smiles_df.to_csv(input_file, index=False)


# input_file = 'pcbaexample.csv'
# output_dir = 'mol2files4'

# start_time = time.time()
# add_mol2_files(input_file, output_dir, ncpu)
# end_time = time.time()
# execution_time = end_time - start_time
# print("Script execution time: {:.2f} seconds".format(execution_time))



