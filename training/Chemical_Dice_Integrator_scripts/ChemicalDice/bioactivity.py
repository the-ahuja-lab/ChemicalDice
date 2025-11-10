from signaturizer import Signaturizer
import pandas as pd



def calculate_descriptors(input_file, output_file):
  """
  Calculate bioactivity descriptors using signaturizer for a given set of SMILES strings and save the results to a CSV file.

  Parameters
  ----------
  input_file : str
      The path to the input CSV file. The file should contain a column 'Canonical_SMILES' with SMILES strings and a column 'id' with unique identifiers.
  output_file : str
      The path to the output CSV file where the calculated descriptors will be saved.

  Returns
  -------
  None

  Notes
  -----
  The function uses the 'GLOBAL' model of the Signaturizer class to calculate descriptors.
  The resulting DataFrame is saved to a CSV file with the columns 'id', 'SMILES', and descriptor columns.
  """
  smiles_df = pd.read_csv(input_file)
  smiles = smiles_df['Canonical_SMILES']
  id_list = smiles_df['id']
  sign = Signaturizer('GLOBAL')
  results = sign.predict(smiles)
  signaturizer_df = pd.DataFrame(results.signature)
  signaturizer_df = signaturizer_df.add_prefix('Sign_')
  signaturizer_df.insert(loc=0, column='SMILES', value=smiles)
  signaturizer_df.insert(loc=0, column='id', value=id_list)
  signaturizer_df.to_csv(output_file,index=False)
  print("Descictors saved to ", output_file)
