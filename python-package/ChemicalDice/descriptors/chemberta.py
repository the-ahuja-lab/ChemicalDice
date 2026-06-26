from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer
import torch
import pandas as pd
from tqdm import tqdm


# download the pretrained model
model_version = 'DeepChem/ChemBERTa-77M-MLM'

# download and load the tokenizer which is used for pretraining the above model
model = RobertaModel.from_pretrained(model_version, output_attentions=True)
tokenizer = RobertaTokenizer.from_pretrained(model_version)


# load the compound smiles

# smilesdf = pd.read_csv("Metabokiller_data_final.csv")
# smiles = smilesdf["SMILES"].tolist()[1:100]

# print(smiles[0:4])






#descriptor_calculator(smiles)


def smiles_to_embeddings(input_file, output_file):
  """
  Convert SMILES strings to ChemBERTa embeddings(A large language model) and save the results to a CSV file.

  Parameters
  ----------
  input_file : str
      The path to the input CSV file. The file should contain a column 'Canonical_SMILES' with SMILES strings and a column 'id' with unique identifiers.
  output_file : str
      The path to the output CSV file where the calculated descriptor embeddings will be saved.

  Returns
  -------
  None

  Notes
  -----
  The function reads the input CSV file, extracts the SMILES strings and their corresponding identifiers, and then calls the Calculates ChemBERTa function to calculate the embeddings. The resulting embeddings are saved to the output CSV file.
  """
  smiles_df = pd.read_csv(input_file)
  smiles = smiles_df['Canonical_SMILES'].to_list()
  chem_id = smiles_df['id'].to_list()
  # get the ChemBERTa embeddings
  final_df = pd.DataFrame()
  chem_id_final = []
  smiles_final = []
  for i, smi in enumerate(tqdm(smiles)):
    try:
      # print(smi)
      # Tokenize the smiles and obtain the tokens:
      encoded_input = tokenizer(smi, add_special_tokens=True, return_tensors='pt')
      
      # generate the embeddings
      with torch.no_grad():
        model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1)
      
      # convert the emeddings output to a dataframe
      df = pd.DataFrame(embeddings).astype("float")
      final_df = pd.concat([final_df, df])
      chem_id_final.append(chem_id[i])
      smiles_final.append(smi)
    except:
      print("Error for smiles ",smi)

  # add a prefix to all the column names
  final_df = final_df.add_prefix('ChB77MLM_')
  # print(final_df)
  final_df.insert(loc=0, column='SMILES', value=smiles_final)
  final_df.insert(loc=0, column='id', value=chem_id_final)
  final_df.to_csv(output_file,index=False)

