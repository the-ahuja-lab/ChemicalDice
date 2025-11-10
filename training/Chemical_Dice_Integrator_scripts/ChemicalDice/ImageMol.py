
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

from ChemicalDice.imagemol_need import *





#import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd
import csv



from rdkit import Chem
from rdkit.Chem import Draw


def get_imagemol_prerequisites(path):
    """
    Ensure that prerequisites for the ImageMol model are available in the specified directory.

    This function downloads the ImageMol model file ('ImageMol.pth.tar') from a GitHub repository 
    and saves it in the specified directory.

    Parameters
    ----------
    path : str
        Directory path where the ImageMol model file will be stored.

    Returns
    -------
    str
        Path to the downloaded ImageMol model file ('ImageMol.pth.tar').
    
    
    """

    # URL of the file to be downloaded
    url = "https://raw.githubusercontent.com/suvendu-kumar/ImageMol_model/main/ImageMol.pth.tar"
    # Name of the file to save as
    filename = os.path.join(path,"ImageMol.pth.tar")
    download_file(url, filename)
    print("ImageMol model is downloaded")
    return filename

# Get the absolute path of the checkpoint file
# def image_to_embeddings(input_file, output_file_name):
#     """
#     Convert images referenced in an input CSV file to embeddings using the ImageMol model and save them to a CSV file.

#     Parameters
#     ----------
#     input_file : str
#         Path to the input CSV file containing references to images.
#     output_file_name : str
#         Path to the output CSV file where the embeddings will be saved.

#     Returns
#     -------
#     None

#     Notes
#     -----
#     This function assumes the existence of pretrained models and required setup for ImageMol. 
#     It processes images from the input CSV file, extracts embeddings using a pretrained ResNet18 model, 
#     and saves the embeddings to the specified output CSV file.

#     Raises
#     ------
#     FileNotFoundError
#         If the input CSV file (`input_file`) does not exist.

#     IOError
#         If there is an issue with reading the input CSV file or writing the output CSV file.

#     """
#     #csv_filename = input_file
#     #image_folder_224 = input_dir
#     add_image_files(input_file, output_dir = "temp_data/images/")
#     checkpoint_path = get_imagemol_prerequisites("temp_data/")
#     resume = checkpoint_path
#     image_model = "ResNet18"
#     imageSize = 224
#     ngpu = 1
#     runseed=2021
#     workers = 2

#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     #image_folder, txt_file = get_datasets2(dataset=csv_filename, dataroot=image_folder_224, data_type="processed")
#     verbose = False

#     device, device_ids = setup_device(ngpu)

#     # fix random seeds
#     fix_train_random_seed()

#     # architecture name
#     if verbose:
#         print('Architecture: {}'.format(image_model))
#     num_tasks = 10000
#     model = load_model(image_model, imageSize=imageSize, num_classes=num_tasks)

#     #print("++++++++++++++++++++++++++")
#     if resume:
#         if os.path.isfile(resume):  # only support ResNet18 when loading resume
#             #print("=> loading checkpoint '{}'".format(resume))
#             if torch.cuda.is_available():
#               checkpoint = torch.load(resume)
#             else:
#               checkpoint = torch.load(resume, map_location=torch.device('cpu'))
#             ckp_keys = list(checkpoint['state_dict'])
#             cur_keys = list(model.state_dict())
#             model_sd = model.state_dict()
#             if image_model == "ResNet18":
#                 ckp_keys = ckp_keys[:120]
#                 cur_keys = cur_keys[:120]

#             for ckp_key, cur_key in zip(ckp_keys, cur_keys):
#                 model_sd[cur_key] = checkpoint['state_dict'][ckp_key]

#             model.load_state_dict(model_sd)
#             arch = checkpoint['arch']
#             #print("resume model info: arch: {}".format(arch))
#         else:
#             print("=> no checkpoint found at '{}'".format(resume))
    
#     if torch.cuda.is_available():
#       model = model.cuda()
#     else:
#       model = model

#     if len(device_ids) > 1:
#         model = torch.nn.DataParallel(model, device_ids=device_ids)

#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     img_transformer_test = [transforms.CenterCrop(imageSize), transforms.ToTensor()]

#     names = load_filenames_and_labels_multitask2(txt_file=input_file)

#     names = np.array(names)
#     num_tasks = len(names)
#     test_dataset = ImageDataset2(names, img_transformer=transforms.Compose(img_transformer_test),
#                                 normalize=normalize, ret_index=False, args=None)
#     test_dataloader = torch.utils.data.DataLoader(test_dataset,
#                                                   batch_size=1,
#                                                   shuffle=False,
#                                                   num_workers=workers,
#                                                   pin_memory=True)


#     # Extract embeddings from the last layer of predictions
#     embeddings = []
#     img_names = []
#     with torch.no_grad():
#         model.eval()
#         for images, img_name in test_dataloader:
#             images = images.to(device)
#             outputs = model(images)
#             embeddings.append(outputs.cpu().numpy())
#             print(len(outputs),img_name
#             if len(outputs) > 0:
#                 img_names.append(img_name)

#     # Concatenate embeddings from all batches
#     embeddings = np.concatenate(embeddings, axis=0)
#     df = pd.DataFrame(embeddings)
#     df = df.add_prefix('ImageMol_')
#     filenames_without_extension = [get_filename_without_extension(path) for path in img_names]


#     # writeEmbeddingsIntoFile(".", f'ImageMol.csv', ids, embeddings)

#     df.index = filenames_without_extension
#     df.index.name = 'id'
#     df.to_csv(output_file_name)


def image_to_embeddings(input_file, output_file_name):
    """
    Convert images referenced in an input CSV file to embeddings using the ImageMol model and save them to a CSV file.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file containing references to images.
    output_file_name : str
        Path to the output CSV file where the embeddings will be saved.

    Returns
    -------
    None

    Notes
    -----
    This function assumes the existence of pretrained models and required setup for ImageMol. 
    It processes images from the input CSV file, extracts embeddings using a pretrained ResNet18 model, 
    and saves the embeddings to the specified output CSV file.

    Raises
    ------
    FileNotFoundError
        If the input CSV file (`input_file`) does not exist.

    IOError
        If there is an issue with reading the input CSV file or writing the output CSV file.

    """
    #csv_filename = input_file
    #image_folder_224 = input_dir
    add_image_files(input_file, output_dir = "temp_data/images/")
    checkpoint_path = get_imagemol_prerequisites("temp_data/")
    resume = checkpoint_path
    image_model = "ResNet18"
    imageSize = 224
    ngpu = 1
    runseed=2021
    workers = 2

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #image_folder, txt_file = get_datasets2(dataset=csv_filename, dataroot=image_folder_224, data_type="processed")
    verbose = False

    device, device_ids = setup_device(ngpu)

    # fix random seeds
    fix_train_random_seed()

    # architecture name
    if verbose:
        print('Architecture: {}'.format(image_model))
    num_tasks = 10000
    model = load_model(image_model, imageSize=imageSize, num_classes=num_tasks)

    #print("++++++++++++++++++++++++++")
    if resume:
        if os.path.isfile(resume):  # only support ResNet18 when loading resume
            #print("=> loading checkpoint '{}'".format(resume))
            if False:
              checkpoint = merged_df = merged_df[merged_df['Task'].isin(task_completed)]
            else:
              checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            ckp_keys = list(checkpoint['state_dict'])
            cur_keys = list(model.state_dict())
            model_sd = model.state_dict()
            if image_model == "ResNet18":
                ckp_keys = ckp_keys[:120]
                cur_keys = cur_keys[:120]

            for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                model_sd[cur_key] = checkpoint['state_dict'][ckp_key]

            model.load_state_dict(model_sd)
            arch = checkpoint['arch']
            #print("resume model info: arch: {}".format(arch))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    
    if torch.cuda.is_available():
      model = model.cuda()
    else:
      model = model

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transformer_test = [transforms.CenterCrop(imageSize), transforms.ToTensor()]

    names = load_filenames_and_labels_multitask2(txt_file=input_file)

    names = np.array(names)
    names = [x for x in names if not x == "nan"]

       
    test_dataset = ImageDataset2(names, img_transformer=transforms.Compose(img_transformer_test),
                                normalize=normalize, ret_index=False, args=None)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=64,
                                                  shuffle=False,
                                                  num_workers=workers,
                                                  pin_memory=True)


    # Extract embeddings from the last layer of predictions
    embeddings = []
    with torch.no_grad():
        model.eval()
        for images in test_dataloader:
            images = images.to(device)
            outputs = model(images)
            embeddings.append(outputs.cpu().numpy())

    # Concatenate embeddings from all batches
    embeddings = np.concatenate(embeddings, axis=0)
    df = pd.DataFrame(embeddings)
    df = df.add_prefix('ImageMol_')
    filenames_without_extension = [get_filename_without_extension(path) for path in names]



    df.index = filenames_without_extension
    df.index.name = 'id'
    # df.to_csv(output_file_name)
    df.to_csv(output_file_name, sep="\t",encoding="utf-8")
    
    df.to_pickle(output_file_name.replace(".csv",".pkl"))




