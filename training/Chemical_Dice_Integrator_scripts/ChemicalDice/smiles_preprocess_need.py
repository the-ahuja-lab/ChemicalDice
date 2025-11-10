import pandas as pd
from openbabel import pybel
#import mol2_to_image
import time
import os
from tqdm import tqdm

#from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor



def smile_to_mol2(smiles_list,smiles_id_list, output_dir):
  mol2_file_paths = []
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  n=0
  for smiles, smiles_id in tqdm(zip(smiles_list,smiles_id_list),total=len(smiles_list)):
    n+=1
    mol2file_name = os.path.join(output_dir,str(smiles_id)+".mol2")
    uff=False
    try:
        if os.path.exists(mol2file_name):
            pass
        else:
            smile_to_mol2_mmff(smiles, steps=5000000, filename=mol2file_name)
        mol2_file_paths.append(mol2file_name)
    except:
        uff=True
    if uff==True:
        try:        
            if os.path.exists(mol2file_name):
                pass
            else:
                smile_to_mol2_uff(smiles, steps=5000000, filename=mol2file_name)
            mol2_file_paths.append(mol2file_name)
        except:
            mol2_file_paths.append("")
  return(mol2_file_paths)



def smile_to_canonical(smiles_list):
    canonical_smiles_list = []
    for smiles in smiles_list:
        try:
            molecule = pybel.readstring("smi", smiles )
            canonical_smiles = molecule.write("can").strip()
            canonical_smiles_list.append(canonical_smiles)
        except:
            canonical_smiles_list.append(smiles)
    return(canonical_smiles_list)


def smile_to_mol2_uff(smile, steps, filename):
    """
    Convert a SMILES string to a MOL2 file using the UFF forcefield.

    This function reads a SMILES string, generates a 3D structure using the Universal Force Field (UFF),
    and writes the resulting structure to a MOL2 file.

    Parameters
    ----------
    smile : str
        The SMILES string representing the molecule.
    steps : int
        The number of steps for the 3D optimization process.
    filename : str
        The name of the output MOL2 file.
    """

    mymol = pybel.readstring("smi", smile)
    #print(mymol.molwt)
    mymol.make3D(steps=steps,forcefield='uff')
    mymol.write(format="mol2",filename=filename,overwrite=True)


def smile_to_mol2_mmff(smile, steps, filename):
    """
    Convert a SMILES string to a MOL2 file using the MMFF forcefield.

    This function reads a SMILES string, generates a 3D structure using the Merck Molecular Force Field (MMFF),
    and writes the resulting structure to a MOL2 file.

    Parameters
    ----------
    smile : str
        The SMILES string representing the molecule.
    steps : int
        The number of steps for the 3D optimization process.
    filename : str
        The name of the output MOL2 file.
    """
        
    mymol = pybel.readstring("smi", smile)
    #print(mymol.molwt)
    mymol.make3D(steps=steps,forcefield='mmff94')
    mymol.write(format="mol2",filename=filename,overwrite=True)

def unique(list1):
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return(unique_list)

def line_prepender(filename, line ):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

def smile_to_mol2(smiles_id_tuple):
    """
    Convert a SMILES string to a MOL2 file using MMFF or UFF forcefields.

    This function takes a tuple containing a SMILES string, an identifier, and an output directory.
    It generates a 3D structure using the MMFF forcefield and saves it as a MOL2 file. If the MMFF
    forcefield fails, it tries to use the UFF forcefield. If both attempts fail, it returns an empty string.

    Parameters
    ----------
    smiles_id_tuple : tuple
        A tuple containing:
        - smiles (str): The SMILES string representing the molecule.
        - smiles_id (str): An identifier for the molecule.
        - output_dir (str): The directory where the MOL2 file will be saved.

    Returns
    -------
    str
        The path to the generated MOL2 file. Returns an empty string if the generation fails.
    """
    smiles, smiles_id, output_dir = smiles_id_tuple
    mol2_file_path = os.path.join(output_dir, str(smiles_id) + ".mol2")
    if not os.path.exists(mol2_file_path):
        try:
            smile_to_mol2_mmff(smiles, steps=500000, filename=mol2_file_path)
        except:
            try:
                smile_to_mol2_uff(smiles, steps=500000, filename=mol2_file_path)
            except:
                return ""
    return mol2_file_path