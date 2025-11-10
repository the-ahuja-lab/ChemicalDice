




# ChemicalDice Integrator
from tqdm import tqdm
import os
import csv
import copy
import time
import pickle
import random
import numpy as np
import pandas as pd


from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


from os import listdir
from os.path import isfile, join

# from imblearn.over_sampling import SMOTE
# smote = SMOTE()


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
import torch.backends.cudnn
import warnings
warnings.filterwarnings('ignore')
############################################

import pandas as pd
import numpy as np
from pathlib import Path
import time
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import time
import h5py
import numpy as np

import numpy as np
import h5py

import os

import torch.nn.init as init
import math
from sklearn.impute import KNNImputer
# from rdkit import Chem



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')



def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def set_seed(seed_value = 42):
    rs = RandomState(MT19937(SeedSequence(seed_value)))
    np.random.seed(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)



class Autoencoder(nn.Module):
    def __init__(self, dims):
        super(Autoencoder, self).__init__()

        self.dims = dims

        latent_space_ind = 0
        latent_space_dim = 1e5

        for i in range(len(self.dims)):
            if self.dims[i] < latent_space_dim:
                latent_space_dim = self.dims[i]
                latent_space_ind = i

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(self.dims)-1):
            if i < latent_space_ind:
                self.encoder.append(nn.Linear(dims[i], dims[i+1]))
                self.encoder.append(nn.ReLU())
            else:
                self.decoder.append(nn.Linear(dims[i], dims[i+1]))
                self.decoder.append(nn.ReLU())

        # Weight initialization
        self.init_weights()

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        for l in self.encoder:
            x = l(x)
        return x

    def decode(self, x):
        for l in self.decoder:
            x = l(x)
        return x

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

class ChemicalDiceIntegrator(nn.Module):
    def getSum(self, lst, ind):
        res = 0
        for i in range(len(lst)):
            if i != ind:
                res += lst[i]
        return res

    def getAEDimensions(self, inp_dim, latent_space_dim, k=7):
        ae_dim = []
        this_dim = inp_dim
        while this_dim > latent_space_dim:
            ae_dim += [this_dim]
            this_dim = math.ceil(this_dim/k)

        return ae_dim + [latent_space_dim] + ae_dim[::-1]

    def remove_element_at_index(self, lst, index):
        if index < 0 or index >= len(lst):
            raise IndexError("Index out of range")

        return lst[:index] + lst[index + 1:]

    def __init__(self, latent_space_dims, embedding_dim, embd_sizes,embd_sizes_sum, k=[], lr=1e-3,weight_decay=0):
        super(ChemicalDiceIntegrator,self).__init__()

        self.latent_space_dims = latent_space_dims
        self.encoders = nn.ModuleDict({})

        self.choice = 1

        if self.choice == 2:
            self.weights = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(len(latent_space_dims))])
        elif self.choice == 1:
            self.weights = nn.ParameterList([nn.Parameter(torch.ones(1, latent_space_dims[i])) for i in range(len(latent_space_dims))])
        else:
            raise ValueError("Invalid choice value. Choose 1 or 2.")


        for i in range(len(embd_sizes)):
            if k==[]:
                dim = self.getAEDimensions(embd_sizes_sum[i], embd_sizes[i])
            else:
                dim = self.getAEDimensions(embd_sizes_sum[i], embd_sizes[i], k[i])
            print(i,dim)
            self.encoders[f'{i}'] = Autoencoder(dim)


        ae_dim = self.getAEDimensions(self.getSum(self.latent_space_dims, -1), embedding_dim)
        # print(ae_dim)
        self.encoders[f'{6}'] = Autoencoder(ae_dim)

    def forward(self, x):
        # Apply weights based on choice
        x = [x[i] * self.weights[i] for i in range(len(x))]

        # Prepare input by removing elements at each index
        inp = [torch.cat(self.remove_element_at_index(x, i), dim=1) for i in range(len(x))]

        # Encode inputs and store outputs
        enc = [self.encoders[f'{i}'].encode(inp[i]) for i in range(len(x))]
        op = [self.encoders[f'{i}'](inp[i]) for i in range(len(x))]

        # Process concatenated keys
        concat_key = torch.cat(enc, dim=1)
        concat_key_enc = self.encoders['6'].encode(concat_key)
        concat_key_op = self.encoders['6'](concat_key)

        # Return all encoded and output values
        return (*enc, *op, concat_key_enc, concat_key_op)


class MyDataset(Dataset):
    def __init__(self, h5_paths, y, chunk_size=1000):
        self.files = [h5py.File(p, 'r') for p in h5_paths]
        self.datasets = [f[key] for f, key in zip(self.files,
                                                  ['mopac', 'chemberta', 'mordred', 'signaturizer', 'imagemol', 'grover'])]
        self.y = y
        self.n_samples = len(self.y)
        self.chunk_size = chunk_size
        self.current_chunk = -1
        self.cache = None

    def _load_chunk(self, chunk_idx):
        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, self.n_samples)
        self.cache = [
            ds[start:end] for ds in self.datasets
        ] + [self.y[start:end]]
        self.current_chunk = chunk_idx

    def __getitem__(self, index):
        chunk_idx = index // self.chunk_size
        offset = index % self.chunk_size
        if self.current_chunk != chunk_idx:
            self._load_chunk(chunk_idx)

        k1, k2, k3, k4, k5, k6, y = [x[offset] for x in self.cache]
        return (torch.tensor(k1, dtype=torch.float32),
                torch.tensor(k2, dtype=torch.float32),
                torch.tensor(k3, dtype=torch.float32),
                torch.tensor(k4, dtype=torch.float32),
                torch.tensor(k5, dtype=torch.float32),
                torch.tensor(k6, dtype=torch.float32),
                torch.tensor(y))

    def __len__(self):
        return self.n_samples

    def __del__(self):
        for f in self.files:
            f.close()


def ChemicalDiceIntegrator(embed_dim,CDI_epochs,CDI_k,number_of_samples,checkpoint_path=None):
    data = MyDataset(["Chemicaldice_data/mopac.h5","Chemicaldice_data/Chemberta.h5","Chemicaldice_data/mordred.h5", "Chemicaldice_data/Signaturizer.h5", "Chemicaldice_data/ImageMol.h5", "Chemicaldice_data/Grover.h5"],  np.array([-1] * number_of_samples))
    data_loader = DataLoader(data, batch_size = batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=32, worker_init_fn=worker_init_fn)

    X1_dim = 212  # mopac
    X2_dim = 384  # Chemberta
    X3_dim = 1421  # mordred
    X4_dim = 3200  # Signaturizer
    X5_dim = 10000  # ImageMol
    X6_dim = 4800  # Grover


    latent_space_dims = [X1_dim, X2_dim, X3_dim, X4_dim, X5_dim, X6_dim]
    # print('k=', embed_dim)
    embd_sizes =[X1_dim,X2_dim,X3_dim,X4_dim,X5_dim,X6_dim]

    def sum_except_self(nums):
        total_sum = sum(nums)
        return [total_sum - num for num in nums]

    embd_sizes_sum = sum_except_self(embd_sizes)
    net_cdi = ChemicalDiceIntegrator(latent_space_dims=latent_space_dims, embedding_dim=embed_dim, embd_sizes=embd_sizes, embd_sizes_sum=embd_sizes_sum,k=CDI_k).to(device)

    trainAE(net_cdi, "AER_"+str(embed_dim), embed_dim, data_loader, data_loader, CDI_epochs, True,checkpoint_path=checkpoint_path)



number_of_samples = 2231870
seed_value = 42
rs = RandomState(MT19937(SeedSequence(seed_value)))
np.random.seed(seed_value)
batch_size = 64

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def count_items(arr):
    unique_values, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique_values, counts))

def convert_to_list(lst):
    ans = []
    for i in range(len(lst)):
        ans.append(lst[i].item())
    return ans



# Load the model and optimizer state if resuming training
def resume_training(model, optimizer, checkpoint_path):
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        start_epoch = 0
    return start_epoch

# Modify the trainAE function to include checkpoint saving and loading
def trainAE(model, dataset_name, embed_dim, train_loader, val_loader, epochs, verbose=False, checkpoint_path="checkpoint.pth"):
    # These are the parameters
    log_file = open("train_resume.txt", "a")
    NUM_EPOCHS = epochs
    LOSS_CRITERION = nn.MSELoss()
    LEARNING_RATE = 0.05
    WEIGHT_DECAY = 0
    OPTIMIZER = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, patience=5, verbose=True)

    # Resume training if checkpoint exists
    start_epoch = resume_training(model, OPTIMIZER, checkpoint_path)

    loss_train = []
    loss_val = []

    alpha, beta, gamma = 0.33, 0.33, 0.33

    for epoch in range(start_epoch, NUM_EPOCHS):

        st = time.time()
        model.train()
        # print(f'Epoch: {epoch + 1}')

        batch_loss_train = 0
        average_batch_loss_train = 0

        progress_bar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",file=log_file)



        for i, data in progress_bar:
            # print(len(data))
            OPTIMIZER.zero_grad()

            k1, k2, k3, k4, k5, k6, labels = data

            k1, k2, k3, k4, k5, k6= k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), k6.to(device)
            labels = labels.to(device)

            key_1 = torch.cat([k2, k3, k4, k5, k6], dim=1)
            key_2 = torch.cat([k1, k3, k4, k5, k6], dim=1)
            key_3 = torch.cat([k1, k2, k4, k5, k6], dim=1)
            key_4 = torch.cat([k1, k2, k3, k5, k6], dim=1)
            key_5 = torch.cat([k1, k2, k3, k4, k6], dim=1)
            key_6 = torch.cat([k1, k2, k3, k4, k5], dim=1)

            key_1_enc, key_2_enc, key_3_enc, key_4_enc, key_5_enc, key_6_enc, key_1_reconstruction, key_2_reconstruction, key_3_reconstruction, key_4_reconstruction, key_5_reconstruction, key_6_reconstruction, _, concat_reconstruction = model.forward([k1, k2, k3, k4, k5, k6])

            encoding_loss1 = LOSS_CRITERION(key_1_enc, k1)
            encoding_loss2 = LOSS_CRITERION(key_2_enc, k2)
            encoding_loss3 = LOSS_CRITERION(key_3_enc, k3)
            encoding_loss4 = LOSS_CRITERION(key_4_enc, k4)
            encoding_loss5 = LOSS_CRITERION(key_5_enc, k5)
            encoding_loss6 = LOSS_CRITERION(key_6_enc, k6)

            reconstruction_loss1 = LOSS_CRITERION(key_1_reconstruction, key_1)
            reconstruction_loss2 = LOSS_CRITERION(key_2_reconstruction, key_2)
            reconstruction_loss3 = LOSS_CRITERION(key_3_reconstruction, key_3)
            reconstruction_loss4 = LOSS_CRITERION(key_4_reconstruction, key_4)
            reconstruction_loss5 = LOSS_CRITERION(key_5_reconstruction, key_5)
            reconstruction_loss6 = LOSS_CRITERION(key_6_reconstruction, key_6)

            concat_key = torch.cat([key_1_enc, key_2_enc, key_3_enc, key_4_enc, key_5_enc, key_6_enc], dim=1)
            reconstruction_loss_concat = LOSS_CRITERION(concat_key, concat_reconstruction)

            total_encoding_loss = encoding_loss1 + encoding_loss2 + encoding_loss3 + encoding_loss4 + encoding_loss5 + encoding_loss6
            total_reconstruction_loss = reconstruction_loss1 + reconstruction_loss2 + reconstruction_loss3 + reconstruction_loss4 + reconstruction_loss5 + reconstruction_loss6
            # print('train',total_encoding_loss, total_reconstruction_loss, reconstruction_loss_concat)
            total_loss_encoder = (alpha * total_encoding_loss / 6) + (beta * total_reconstruction_loss / 6) + (gamma * reconstruction_loss_concat)

            # print(device)
            total_loss_encoder.backward()
            OPTIMIZER.step()

            _loss = total_loss_encoder.item()

            batch_loss_train += _loss

            average_batch_loss_train = batch_loss_train / (i+1)
            log_file.flush()


        loss_train.append(average_batch_loss_train)

        SCHEDULER.step(average_batch_loss_train)


        ## Validation
        batch_loss_val = 0
        avg_loss_val = 0


        if verbose:
            log_file.write(f"Epoch: {epoch + 1} Train loss: {average_batch_loss_train} time: {time.time() - st}\n")
            log_file.write(f"{epoch+1},{alpha * total_encoding_loss.item() / 6},{beta * total_reconstruction_loss.item() / 6},{gamma * reconstruction_loss_concat.item()}\n")
            log_file.flush()
            # print(f'Epoch: {epoch + 1} Train loss: {average_batch_loss_train} val loss: {avg_loss_val}')
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"{dataset_name}_checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': OPTIMIZER.state_dict()
            }, checkpoint_path)
            if verbose:
                print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")
    log_file.close()
    torch.save(model.state_dict(), f"{dataset_name}_cdi.pt")
    embeddings = None



# order to check CDI_k value mopac(213) Chemberta(384) Signaturizer(3200) ImageMol(10000) Grover(4800)
ChemicalDiceIntegrator(embed_dim=8192,CDI_epochs=600,CDI_k=[7,8,8,10,10,12],number_of_samples=2231870,checkpoint_path=None)