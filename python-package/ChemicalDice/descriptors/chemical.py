import os
from mordred import Calculator, descriptors
import pandas as pd
from rdkit import Chem
from tqdm import tqdm


def descriptor_calculator(input_file,output_file):
  """
  Calculate molecular descriptors for the molecules in the input file and save the results to the output file.

  This function reads SMILES strings and corresponding SDF file names from an input CSV file, calculates 
  molecular descriptors for each molecule, and writes the results to an output CSV file. The descriptors 
  are calculated using the mordred package.

  Parameters
  ----------
  input_file : str
      Path to the input CSV file containing SMILES strings and SDF file names.
  output_file : str
      Path to the output CSV file where the calculated descriptors will be saved.

  Returns
  -------
  None
  """
  smiles_df = pd.read_csv(input_file)
  sdffile_name_list = smiles_df['sdf_files']
  id_list = smiles_df['id']
  smiles_list = smiles_df['SMILES']
  calc = Calculator(descriptors, ignore_3D=False)
  desc_columns=[str(d) for d in calc.descriptors]
  f = open(output_file,"w")
  f.write("id,SMILES,")
  header = ",".join(desc_columns)
  f.write(header)
  f.write("\n")
  for sdffile_name,id,smile in tqdm(zip(sdffile_name_list, id_list, smiles_list)):
    try:
      suppl = Chem.SDMolSupplier(sdffile_name)
      Des = calc(suppl[0])
      lst = []
      lst.append(id)
      lst.append(smile)
      for i in range(len(Des)):
        myVariable =Des[i]
        if type(myVariable) == int or type(myVariable) == float or str(type(myVariable)) == "<class 'numpy.float64'>":
          lst.append(Des[i])
        else:
          lst.append(None)
      lst = [str(x) for x in lst]
      row_str=",".join(lst)
      f.write(row_str)
      f.write("\n")
      #print("=",end="")
    except Exception as e:
      print(" Error in descriptor calculation",end="\t")
      print(id)
      print(e)
  f.close()

