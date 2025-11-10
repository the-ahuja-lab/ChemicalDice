import os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from ChemicalDice.image_dataloader import ImageDataset2, load_filenames_and_labels_multitask2, get_datasets2
from ChemicalDice.cnn_model_utils import load_model, train_one_epoch_multitask, evaluate_on_multitask, save_finetune_ckpt
from ChemicalDice.train_utils import fix_train_random_seed, load_smiles
from ChemicalDice.public_utils import cal_torch_model_params, setup_device, is_left_better_right
from ChemicalDice.splitter import split_train_val_test_idx, split_train_val_test_idx_stratified, scaffold_split_train_val_test, \
    random_scaffold_split_train_val_test, scaffold_split_balanced_train_val_test

import requests

def download_file(url, filename):
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)



#import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd
import csv


def writeEmbeddingsIntoFile(file_path, file_name, ids, embeddings):
    if not os.path.exists(file_path):
        os.makedirs(file_path) 
    with open(file_path + file_name, mode='w', newline='') as file:
        writer = csv.writer(file) 
        for row in range(len(ids)):
            writer.writerow(np.concatenate(([ids[row]], embeddings[row])))
    print(len(ids), embeddings.shape)

def get_filename_without_extension(file_path):
    base_name = os.path.basename(file_path)
    filename_without_extension = os.path.splitext(base_name)[0]
    return filename_without_extension

from rdkit import Chem
from rdkit.Chem import Draw

def smile_to_img(smiles_list,smiles_id_list,output_dir):
    image_file_paths = []
    
    for smis, index in zip(smiles_list,smiles_id_list):
        if is_valid_smiles(smis):
            file_path = os.path.join(output_dir,index + ".png")
            #image = Smiles2Img2(smis, , savePath=file_path)
            size=224
            mol = Chem.MolFromSmiles(smis)
            img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size),returnPNG=False)
            img.save(file_path)
            if img is None:
                print("Error in smile to image for ", index, smis)
                image_file_paths.append('')
            else:
                image_file_paths.append(file_path)
        else:
            image_file_paths.append('')

    return image_file_paths


def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def add_image_files(input_file, output_dir):
    smiles_df = pd.read_csv(input_file)
    if not os.path.exists(output_dir):
        print("making directory ", output_dir)
        os.makedirs(output_dir)
    smiles_list = smiles_df['Canonical_SMILES']
    if 'id' in smiles_df.columns:
        smiles_id_list = smiles_df['id']
    else:
        smiles_df['id'] = [ "C"+str(id) for id in range(len(smiles_list))]
        smiles_id_list = smiles_df['id']
    image_file_paths = smile_to_img(smiles_list, smiles_id_list, output_dir)
    smiles_df['image_files'] = image_file_paths
    smiles_df.to_csv(input_file,index=False)

# import pkg_resources

# Get the absolute path of the checkpoint file
