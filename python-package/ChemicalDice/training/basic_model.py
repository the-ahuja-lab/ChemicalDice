#srun -A gauravahuja -p short -q short --ntasks=1 --cpus-per-task=8 --mem=20G --gres=shard:1 --time=5:50:00 --pty /bin/bash
#conda activate chemicaldice
# ChemicalDice Integrator
from tqdm import tqdm
import os
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

NUM_DESCRIPTORS = 2 # it should be changed if the number of descriptors is changed

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

        # FIX: The bottleneck (latent space) is ALWAYS exactly in the middle
        # of the symmetric dims list. We no longer rely on finding the minimum.
        latent_space_ind = len(self.dims) // 2

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
# class Autoencoder(nn.Module):
#     def __init__(self, dims):
#         super(Autoencoder, self).__init__()

#         self.dims = dims

#         latent_space_ind = 0
#         latent_space_dim = 1e5

#         for i in range(len(self.dims)):
#             if self.dims[i] < latent_space_dim:
#                 latent_space_dim = self.dims[i]
#                 latent_space_ind = i

#         self.encoder = nn.ModuleList()
#         self.decoder = nn.ModuleList()

#         for i in range(len(self.dims)-1):
#             if i < latent_space_ind:
#                 self.encoder.append(nn.Linear(dims[i], dims[i+1]))
#                 self.encoder.append(nn.ReLU())
#             else:
#                 self.decoder.append(nn.Linear(dims[i], dims[i+1]))
#                 self.decoder.append(nn.ReLU())

#         # Weight initialization
#         self.init_weights()

#     def forward(self, x):
#         x = self.encode(x)
#         x = self.decode(x)
#         return x

#     def encode(self, x):
#         for l in self.encoder:
#             x = l(x)
#         return x

#     def decode(self, x):
#         for l in self.decoder:
#             x = l(x)
#         return x

#     def init_weights(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     init.constant_(module.bias, 0)

class ChemicalDiceIntegrator(nn.Module):
    def getSum(self, lst, ind):
        res = 0
        for i in range(len(lst)):
            if i != ind:
                res += lst[i]
        return res

    # def getAEDimensions(self, inp_dim, latent_space_dim, k=7):
    #     ae_dim = []
    #     this_dim = inp_dim
    #     while this_dim > latent_space_dim:
    #         ae_dim += [this_dim]
    #         this_dim = math.ceil(this_dim/k)

    #     return ae_dim + [latent_space_dim] + ae_dim[::-1]

    def getAEDimensions(self, inp_dim, latent_space_dim, k=7):
            # FIX: If input is smaller than latent, create a direct mapping layer
            if inp_dim <= latent_space_dim:
                return [inp_dim, latent_space_dim, inp_dim]

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

    def __init__(self, latent_space_dims, bottleneck, embd_sizes, embd_sizes_sum, k=[], lr=1e-3, weight_decay=0):
        super(ChemicalDiceIntegrator, self).__init__()

        self.latent_space_dims = latent_space_dims
        self.encoders = nn.ModuleDict({})

        self.choice = 1

        if self.choice == 2:
            self.weights = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(len(latent_space_dims))])
        elif self.choice == 1:
            self.weights = nn.ParameterList([nn.Parameter(torch.ones(1, latent_space_dims[i])) for i in range(len(latent_space_dims))])
        else:
            raise ValueError("Invalid choice value. Choose 1 or 2.")

        self.count_sca_blocks = len(embd_sizes)
        for i in range(self.count_sca_blocks):
            if k == []:
                dim = self.getAEDimensions(embd_sizes_sum[i], embd_sizes[i])
            else:
                dim = self.getAEDimensions(embd_sizes_sum[i], embd_sizes[i], k[i])
            print(i, dim)
            self.encoders[f'{i}'] = Autoencoder(dim)

        ae_dim = self.getAEDimensions(self.getSum(self.latent_space_dims, -1), bottleneck)
        # print(ae_dim)
        self.encoders[f'{self.count_sca_blocks + 1}'] = Autoencoder(ae_dim)

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
        concat_key_enc = self.encoders[f'{self.count_sca_blocks+1}'].encode(concat_key)
        concat_key_op = self.encoders[f'{self.count_sca_blocks+1}'](concat_key)

        # Return all encoded and output values
        return (*enc, *op, concat_key_enc, concat_key_op)


class MyDataset(Dataset):
    def __init__(self, h5_file_paths, y, chunk_size=1000):
        self.files = [h5py.File(p, 'r') for p in h5_file_paths]
        self.datasets = [f[list(f.keys())[0]] for f in self.files]
        # self.datasets = [f[key] for f, key in zip(self.files,
        #                                           [ 'chemberta', 'imagemol'])]
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

        # k1, k2, y = [x[offset] for x in self.cache]#### it should be changed
        # return (torch.tensor(k1, dtype=torch.float32),
        #         torch.tensor(k2, dtype=torch.float32),
        #         torch.tensor(y))
        *features, y = [x[offset] for x in self.cache]

        features = [torch.tensor(f, dtype=torch.float32) for f in features]
        y = torch.tensor(y)

        return (*features, y)

    def __len__(self):
        return self.n_samples

    def __del__(self):
        for f in self.files:
            f.close()


# def train_basic_cdi(embed_dim,num_epochs,k_values,number_of_samples,checkpoint_path=None):
#     self.datasets = [f[list(f.keys())[0]] for f in self.files]
#     #data = MyDataset(["../Chemicaldice_data/Chemberta_scaled.h5", "../Chemicaldice_data/ImageMol_scaled.h5"],  np.array([-1] * number_of_samples))
#     data_loader = DataLoader(data, batch_size = batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=32, worker_init_fn=worker_init_fn)

#     # print('k=', embed_dim)
#     embd_sizes = [ds.shape[1] for ds in data.datasets]
#     latent_space_dims = embd_sizes
#     NUM_DESCRIPTORS = len(embd_sizes)


#     def sum_except_self(nums):
#         total_sum = sum(nums)
#         return [total_sum - num for num in nums]

#     embd_sizes_sum = sum_except_self(embd_sizes)
#     net_cdi = ChemicalDiceIntegrator(latent_space_dims=latent_space_dims, embedding_dim=embed_dim, embd_sizes=embd_sizes, embd_sizes_sum=embd_sizes_sum,k=k_values).to(device)

#     trainAE(net_cdi, "AER_"+str(embed_dim), embed_dim, data_loader, data_loader, num_epochs, True,checkpoint_path=checkpoint_path)

def train_basic_cdi(
    h5_file_paths,
    num_epochs=15,
    learning_rate=0.001,
    model_path="cdi_model.pt",
    embedding_path="cdi_embeddings.h5",
    bottleneck=8192,
    k_values=None,
    number_of_samples=None,
    checkpoint_path=None
):

    global NUM_DESCRIPTORS

    # -----------------------------
    # Dataset
    # -----------------------------
    if number_of_samples is None:
        # auto-detect from first file
        with h5py.File(h5_file_paths[0], "r") as f:
            number_of_samples = f[list(f.keys())[0]].shape[0]

    y_dummy = np.array([-1] * number_of_samples)

    data = MyDataset(h5_file_paths, y_dummy)

    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=32,
        worker_init_fn=worker_init_fn
    )

    # -----------------------------
    # Descriptor dimensions (AUTO)
    # -----------------------------
    embd_sizes = [ds.shape[1] for ds in data.datasets]
    latent_space_dims = embd_sizes

    NUM_DESCRIPTORS = len(embd_sizes)

    print("🔹 Number of descriptors:", NUM_DESCRIPTORS)
    print("🔹 Descriptor dims:", embd_sizes)

    # -----------------------------
    # Sum of others (for AE input)
    # -----------------------------
    def sum_except_self(nums):
        total = sum(nums)
        return [total - n for n in nums]

    embd_sizes_sum = sum_except_self(embd_sizes)

    # -----------------------------
    # Model
    # -----------------------------
    net_cdi = ChemicalDiceIntegrator(
        latent_space_dims=latent_space_dims,
        bottleneck=bottleneck,
        embd_sizes=embd_sizes,
        embd_sizes_sum=embd_sizes_sum,
        k=k_values if k_values is not None else NUM_DESCRIPTORS*[8]
    ).to(device)

    # -----------------------------
    # Train
    # -----------------------------
    trainAE(
        net_cdi,
        model_path,
        embedding_path,
        data_loader,
        data_loader,
        num_epochs,
        learning_rate,
        True,
        checkpoint_path=checkpoint_path
    )



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

# # Modify the trainAE function to include checkpoint saving and loading
# def trainAE(model, dataset_name, embed_dim, train_loader, val_loader, epochs, verbose=False, checkpoint_path="checkpoint.pth"):
#     # These are the parameters
#     log_file = open("train_model_all.txt", "a")
#     progress_file = open("progress.txt", "a")
#     NUM_EPOCHS = epochs
#     LOSS_CRITERION = nn.MSELoss()
#     LEARNING_RATE = 0.05
#     WEIGHT_DECAY = 0
#     OPTIMIZER = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, patience=5)

#     # Resume training if checkpoint exists
#     start_epoch = resume_training(model, OPTIMIZER, checkpoint_path)

#     loss_train = []
#     loss_val = []

#     alpha, beta, gamma = 0.33, 0.33, 0.33
#     # columns = [
#     #     "epoch",
#     #     "encoding_loss1", "encoding_loss2", "encoding_loss3",
#     #     "reconstruction_loss1", "reconstruction_loss2", "reconstruction_loss3",
#     #     "reconstruction_loss_concat",
#     #     "total_encoding_loss",
#     #     "total_reconstruction_loss",
#     #     "total_loss_encoder"
#     # ]
#     columns = ["epoch"]

#     columns += [f"encoding_loss_{i}" for i in range(NUM_DESCRIPTORS)]
#     columns += [f"reconstruction_loss_{i}" for i in range(NUM_DESCRIPTORS)]

#     columns += [
#         "reconstruction_loss_concat",
#         "total_encoding_loss",
#         "total_reconstruction_loss",
#         "total_loss_encoder"
#     ]
#     # Write header
#     log_file.write(",".join(columns) + "\n")
#     for epoch in range(start_epoch, NUM_EPOCHS):

#         st = time.time()
#         model.train()
#         # print(f'Epoch: {epoch + 1}')

#         batch_loss_train = 0
#         average_batch_loss_train = 0

#         progress_bar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",file=progress_file)



#         for i, data in progress_bar:
#             # print(len(data))
#             OPTIMIZER.zero_grad()

#             # k1, k2, labels = data
#             *features, labels = data
#             features = [f.to(device) for f in features]
#             labels = labels.to(device)
#             # k1, k2= k1.to(device), k2.to(device)
#             # labels = labels.to(device)

#             key_1 = torch.cat([k2], dim=1)
#             key_2 = torch.cat([k1], dim=1)

#             outputs = model.forward(features)
#             #key_1_enc, key_2_enc, key_1_reconstruction, key_2_reconstruction,  _, concat_reconstruction = model.forward([k1, k2])

#             # encoding_loss1 = LOSS_CRITERION(key_1_enc, k1)
#             # encoding_loss2 = LOSS_CRITERION(key_2_enc, k2)



#             # reconstruction_loss1 = LOSS_CRITERION(key_1_reconstruction, key_1)
#             # reconstruction_loss2 = LOSS_CRITERION(key_2_reconstruction, key_2)



#             # concat_key = torch.cat([key_1_enc, key_2_enc], dim=1)
#             # reconstruction_loss_concat = LOSS_CRITERION(concat_key, concat_reconstruction)

#             # total_encoding_loss = encoding_loss1 + encoding_loss2 
#             # total_reconstruction_loss = reconstruction_loss1 + reconstruction_loss2 
#             # # print('train',total_encoding_loss, total_reconstruction_loss, reconstruction_loss_concat)
#             # total_loss_encoder = (alpha * total_encoding_loss / NUM_DESCRIPTORS) + (beta * total_reconstruction_loss / NUM_DESCRIPTORS) + (gamma * reconstruction_loss_concat)

#             # # print(device)
#             # total_loss_encoder.backward()
#             # OPTIMIZER.step()

#             # _loss = total_loss_encoder.item()

#             # batch_loss_train += _loss

#             # average_batch_loss_train = batch_loss_train / (i+1)
#             enc_outputs = outputs[:NUM_DESCRIPTORS]
#             recon_outputs = outputs[NUM_DESCRIPTORS:2*NUM_DESCRIPTORS]
#             concat_reconstruction = outputs[-1]

#             encoding_losses = []
#             reconstruction_losses = []

#             for i in range(NUM_DESCRIPTORS):
#                 loss_enc = LOSS_CRITERION(enc_outputs[i], target)
#                 loss_rec = LOSS_CRITERION(recon_outputs[i], input_concat)

#                 encoding_losses.append(loss_enc)
#                 reconstruction_losses.append(loss_rec)

#                 target = features[i]

#                 encoding_loss += LOSS_CRITERION(enc_outputs[i], target)

#                 others = [features[j] for j in range(NUM_DESCRIPTORS) if j != i]
#                 input_concat = torch.cat(others, dim=1)

#                 reconstruction_loss += LOSS_CRITERION(recon_outputs[i], input_concat)


#             total_encoding_loss = sum(encoding_losses)
#             total_reconstruction_loss = sum(reconstruction_losses)
#             concat_key = torch.cat(enc_outputs, dim=1)
#             reconstruction_loss_concat = LOSS_CRITERION(concat_key, concat_reconstruction)
#             total_loss_encoder = (
#                 alpha * encoding_loss / NUM_DESCRIPTORS +
#                 beta * reconstruction_loss / NUM_DESCRIPTORS +
#                 gamma * reconstruction_loss_concat
#             )
            
#             progress_file.flush()


#         # row_data = [
#         #     epoch + 1,
#         #     encoding_loss1.item(), encoding_loss2.item(),
#         #     reconstruction_loss1.item(), reconstruction_loss2.item(),
#         #     reconstruction_loss_concat.item(),
#         #     total_encoding_loss.item(),
#         #     total_reconstruction_loss.item(),
#         #     total_loss_encoder.item()
#         # ]
#         row_data = [epoch + 1]

#         # per-descriptor encoding loss
#         for i in range(NUM_DESCRIPTORS):
#             row_data.append(encoding_losses[i].item())

#         # per-descriptor reconstruction loss
#         for i in range(NUM_DESCRIPTORS):
#             row_data.append(reconstruction_losses[i].item())

#         # fusion + totals
#         row_data.extend([
#             reconstruction_loss_concat.item(),
#             total_encoding_loss.item(),
#             total_reconstruction_loss.item(),
#             total_loss_encoder.item()
#         ])
#         # Convert all items to strings and join with a comma, then add the newline
#         log_file.write(",".join(map(str, row_data)) + "\n")
#         log_file.flush()
#         loss_train.append(average_batch_loss_train)

#         SCHEDULER.step(average_batch_loss_train)
#         # if epoch + 1 == 200:
#         #     print("🔹 Extracting embeddings at epoch 200...")
#         #     model.eval()
#         #     embeddings = None

#         #     with torch.no_grad():
#         #         for data in train_loader:
#         #             k1, k2, _ = data
#         #             k1, k2= (
#         #                 k1.to(device),
#         #                 k2.to(device)
#         #             )

#         #             *_, output, _ = model.forward([k1, k2])

#         #             out_np = output.cpu().numpy()
#         #             embeddings = (
#         #                 out_np
#         #                 if embeddings is None
#         #                 else np.concatenate(
#         #                     (embeddings, out_np), axis=0
#         #                 )
#         #             )

#         #     with h5py.File(f"AER_8192_embeddings.h5", "w") as hf:
#         #         hf.create_dataset(f"embeddings", data=embeddings)

#         model.eval()
#         embeddings = None

#         with torch.no_grad():
#             for data in train_loader:

#                 # -----------------------------
#                 # Dynamic unpacking
#                 # -----------------------------
#                 *features, _ = data
#                 features = [f.to(device) for f in features]

#                 # -----------------------------
#                 # Forward
#                 # -----------------------------
#                 outputs = model.forward(features)

#                 # -----------------------------
#                 # Extract fused embedding
#                 # -----------------------------
#                 concat_key_enc = outputs[-2]   # second last is fused embedding

#                 out_np = concat_key_enc.cpu().numpy()

#                 embeddings = (
#                     out_np
#                     if embeddings is None
#                     else np.concatenate((embeddings, out_np), axis=0)
#                 )

#         if verbose:
#             log_file.write(f"Epoch: {epoch + 1} Train loss: {average_batch_loss_train} time: {time.time() - st}\n")
#             log_file.write(f"{epoch+1},{alpha * total_encoding_loss.item() / NUM_DESCRIPTORS},{beta * total_reconstruction_loss.item() / NUM_DESCRIPTORS},{gamma * reconstruction_loss_concat.item()}\n")
#             log_file.flush()
#             # print(f'Epoch: {epoch + 1} Train loss: {average_batch_loss_train} val loss: {avg_loss_val}')
#         if (epoch + 1) % 10 == 0:
#             checkpoint_path = f"{dataset_name}_checkpoint_epoch_{epoch+1}.pt"
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': OPTIMIZER.state_dict()
#             }, checkpoint_path)
#             if verbose:
#                 print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")
#     log_file.close()
#     torch.save(model.state_dict(), f"{dataset_name}_cdi.pt")
#     embeddings = None


def trainAE(model, model_path, embedding_path, train_loader, val_loader, epochs, learning_rate, verbose=False, checkpoint_path="checkpoint.pth"):

    log_file = open("train_model_all.txt", "a")
    progress_file = open("progress.txt", "a")

    NUM_EPOCHS = epochs
    LOSS_CRITERION = nn.MSELoss()
    LEARNING_RATE = learning_rate
    WEIGHT_DECAY = 0

    OPTIMIZER = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, patience=5)

    # Resume training
    start_epoch = resume_training(model, OPTIMIZER, checkpoint_path)

    loss_train = []

    alpha, beta, gamma = 0.33, 0.33, 0.33

    # -----------------------------
    # Dynamic header
    # -----------------------------
    columns = ["epoch"]
    columns += [f"encoding_loss_{i}" for i in range(NUM_DESCRIPTORS)]
    columns += [f"reconstruction_loss_{i}" for i in range(NUM_DESCRIPTORS)]
    columns += [
        "reconstruction_loss_concat",
        "total_encoding_loss",
        "total_reconstruction_loss",
        "total_loss_encoder"
    ]

    log_file.write(",".join(columns) + "\n")
    log_file.flush()

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(start_epoch, NUM_EPOCHS):

        st = time.time()
        model.train()

        batch_loss_train = 0

        progress_bar = tqdm(
            enumerate(train_loader, 0),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            file=progress_file
        )

        for i, data in progress_bar:

            OPTIMIZER.zero_grad()

            # -----------------------------
            # Dynamic unpacking
            # -----------------------------
            *features, labels = data
            features = [f.to(device) for f in features]
            labels = labels.to(device)

            # -----------------------------
            # Forward
            # -----------------------------
            outputs = model.forward(features)

            enc_outputs = outputs[:NUM_DESCRIPTORS]
            recon_outputs = outputs[NUM_DESCRIPTORS:2*NUM_DESCRIPTORS]
            concat_reconstruction = outputs[-1]

            # -----------------------------
            # Loss computation
            # -----------------------------
            encoding_losses = []
            reconstruction_losses = []

            for j in range(NUM_DESCRIPTORS):

                target = features[j]

                loss_enc = LOSS_CRITERION(enc_outputs[j], target)
                encoding_losses.append(loss_enc)

                others = [features[k] for k in range(NUM_DESCRIPTORS) if k != j]
                input_concat = torch.cat(others, dim=1)

                loss_rec = LOSS_CRITERION(recon_outputs[j], input_concat)
                reconstruction_losses.append(loss_rec)

            total_encoding_loss = sum(encoding_losses)
            total_reconstruction_loss = sum(reconstruction_losses)

            concat_key = torch.cat(enc_outputs, dim=1)
            reconstruction_loss_concat = LOSS_CRITERION(concat_key, concat_reconstruction)

            total_loss_encoder = (
                alpha * total_encoding_loss / NUM_DESCRIPTORS +
                beta * total_reconstruction_loss / NUM_DESCRIPTORS +
                gamma * reconstruction_loss_concat
            )

            total_loss_encoder.backward()
            OPTIMIZER.step()

            batch_loss_train += total_loss_encoder.item()

        avg_loss = batch_loss_train / len(train_loader)
        loss_train.append(avg_loss)

        # -----------------------------
        # Logging (dynamic)
        # -----------------------------
        row_data = [epoch + 1]

        for loss in encoding_losses:
            row_data.append(loss.item())

        for loss in reconstruction_losses:
            row_data.append(loss.item())

        row_data.extend([
            reconstruction_loss_concat.item(),
            total_encoding_loss.item(),
            total_reconstruction_loss.item(),
            total_loss_encoder.item()
        ])

        log_file.write(",".join(map(str, row_data)) + "\n")
        log_file.flush()

        SCHEDULER.step(avg_loss)

        # -----------------------------
        # Embedding extraction (FIXED)
        # -----------------------------
        if epoch + 1 == epochs:
            print(f"🔹 Extracting embeddings at epoch {epochs}...")
            model.eval()

            embeddings = None

            with torch.no_grad():
                for data in train_loader:

                    *features, _ = data
                    features = [f.to(device) for f in features]

                    outputs = model.forward(features)
                    concat_key_enc = outputs[-2]   # fused embedding

                    out_np = concat_key_enc.cpu().numpy()

                    embeddings = (
                        out_np
                        if embeddings is None
                        else np.concatenate((embeddings, out_np), axis=0)
                    )

            with h5py.File(f"{embedding_path}", "w") as hf:
                hf.create_dataset("embeddings", data=embeddings)

        # -----------------------------
        # Checkpoint
        # -----------------------------
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"{model_path.replace('.pt', '')}_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': OPTIMIZER.state_dict()
            }, checkpoint_path)

            if verbose:
                print(f"Checkpoint saved: {checkpoint_path}")

        if verbose:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | Time: {time.time()-st:.2f}s")

    log_file.close()

    torch.save(model.state_dict(), f"{model_path}")  

