# Generelised ChemicalDice
import os
os.chdir("/workspace/materials.smi_ssed/smi_ssed/notebooks")
import sys
sys.path.append('../inference')
# materials.smi-ssed (smi-ssed)
from smi_ssed.load import load_smi_ssed

# Data
import pandas as pd
import numpy as np
import torch

# Chemistry
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
from typing import List, Union
# smi seed
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cosine_similarity
import pandas as pd
import h5py
import numpy as np
import os
import sys
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr



def train_test_split_numpy(
    *arrays: Union[np.ndarray, List],
    test_size: float = 0.25,
    random_state: int = None,
    shuffle: bool = True
) -> List[Union[np.ndarray, List]]:
    """
    Splits arrays or lists into random train and test subsets.

    This function mimics the basic functionality of scikit-learn's train_test_split
    using only NumPy.

    Args:
        *arrays: A sequence of indexable objects (e.g., numpy arrays, lists)
                 to split. All arrays must have the same length.
        test_size: The proportion of the dataset to include in the test split.
                   Should be between 0.0 and 1.0.
        random_state: An integer to seed the random number generator for
                      reproducible shuffling.
        shuffle: Whether or not to shuffle the data before splitting.

    Returns:
        A list containing train-test split of input arrays.
        (e.g., X_train, X_test, y_train, y_test)
    """
    # --- Input Validation ---
    if not arrays:
        raise ValueError("At least one array must be provided to split.")

    # Check that all arrays have the same length
    first_array_len = len(arrays[0])
    for arr in arrays:
        if len(arr) != first_array_len:
            raise ValueError("All input arrays must have the same length.")

    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be a float between 0.0 and 1.0.")

    # --- Index Generation and Shuffling ---
    num_samples = first_array_len
    indices = np.arange(num_samples)

    if shuffle:
        # Use a RandomState object for reproducibility if random_state is set
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)

    # --- Splitting Logic ---
    split_point = int(num_samples * (1 - test_size))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    # --- Create the final split arrays ---
    result = []
    for arr in arrays:
        # Convert to numpy array if it's a list for easier indexing
        if isinstance(arr, list):
            arr = np.array(arr)

        # Append train and test sets for the current array
        result.append(arr[train_indices])
        result.append(arr[test_indices])

    return result




SMILES_FILE = '/workspace/smiles.csv'
SMILES_COLUMN_NAME = 'Canonical_SMILES'
EMBEDDING_FILE = '/workspace/AER_8192_embeddings.h5'
EMBEDDING_KEY = 'embeddings'
SMI_SSED_MODEL_FOLDER = '../models/smi_ssed_base' 
MODEL_SAVE_DIR = '/workspace/smi_ssed_finetuned_model'
YOUR_EMBEDDING_DIM = 8192


MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- 1. Custom Dataset ---
class SmilesEmbeddingDataset(Dataset):
    def __init__(self, smiles_list, embeddings_array, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.smiles = smiles_list
        self.embeddings = embeddings_array
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smiles_str = self.smiles[index]
        embedding_vector = self.embeddings[index]
        target_tensor = torch.tensor(embedding_vector, dtype=torch.float32)

        # The smi_ssed tokenizer is non-standard but returns what we need
        encoding = self.tokenizer.encode_plus(
            smiles_str,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',  # Use the string 'max_length' here
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids']

        # Create an attention mask (1 for real tokens, 0 for padding)
        # Assumes the padding token ID is 0, which is standard.
        attention_mask = (input_ids != 0).float()

        return {
            'input_ids': input_ids.flatten(),
            'attention_mask': attention_mask.flatten(),
            'targets': target_tensor
        }

# --- 2. Model Definition  ---
class SmiSsedPredictor(nn.Module):
    def __init__(self, smi_ssed_model, embedding_dim):
        super(SmiSsedPredictor, self).__init__()
        self.base_model_encoder = smi_ssed_model.encoder
        self.regressor = nn.Linear(self.base_model_encoder.config['n_embd'], embedding_dim)

    def forward(self, input_ids, attention_mask):
        # Get hidden states from the base Mamba encoder.
        outputs = self.base_model_encoder(input_ids, mask=attention_mask)
        hidden_states = outputs[0]

        # masked pooling
        if hidden_states.dim() == 3:
            # Output is 3D (batch, seq_len, dim), likely in eval mode. Perform pooling.
            expanded_mask = attention_mask.unsqueeze(-1)

            # Handle potential mismatch between mask length and output sequence length
            if expanded_mask.shape[1] != hidden_states.shape[1]:
                mask_for_pooling = torch.zeros_like(hidden_states)
                # Use the shorter of the two lengths to be safe
                slice_len = min(expanded_mask.shape[1], hidden_states.shape[1])
                mask_for_pooling[:, :slice_len, :] = expanded_mask[:, :slice_len, :]
            else:
                mask_for_pooling = expanded_mask

            sum_hidden_states = torch.sum(hidden_states * mask_for_pooling, 1)
            sum_mask = torch.clamp(mask_for_pooling.sum(1), min=1e-9)
            molecular_representation = sum_hidden_states / sum_mask
        else:
            # Output is 2D (seq_len, dim), likely in train mode.
            # We will take the representation of the last token.
            molecular_representation = hidden_states[-1, :]
            # Since this is for a single sample, we need to add a batch dimension
            # for the regressor, but the loss function will handle the batch.
            # Let's reshape it to be safe for the regressor.
            if molecular_representation.dim() == 1:
                 molecular_representation = molecular_representation.unsqueeze(0)

        # Pass the single molecular representation through the regressor
        predicted_embedding = self.regressor(molecular_representation)
        return predicted_embedding

# --- 3. Training and Evaluation Loop  ---
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        # Since the model returns a 2D tensor in train mode, we must process
        # the batch one sample at a time.
        optimizer.zero_grad()
        batch_outputs = []
        for i in range(input_ids.size(0)):
            # Process one sample at a time by adding a temporary batch dimension
            single_output = model(input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0))
            batch_outputs.append(single_output)

        # Reconstruct the batch of outputs
        outputs = torch.cat(batch_outputs, dim=0)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_cos_sims = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            # In eval mode, the model handles batches correctly
            outputs = model(input_ids, attention_mask)

            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            cos_sim = cosine_similarity(outputs, targets, dim=1)
            all_cos_sims.extend(cos_sim.cpu().numpy())

    avg_cos_sim = sum(all_cos_sims) / len(all_cos_sims) if all_cos_sims else 0
    return total_loss / len(data_loader), avg_cos_sim

# --- Main Execution ---
if __name__ == '__main__':
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print("Loading data...")
    smiles_df = pd.read_csv(SMILES_FILE)
    smiles_list = smiles_df[SMILES_COLUMN_NAME].tolist()
    with h5py.File(EMBEDDING_FILE, 'r') as hf:
        embeddings_array = hf[EMBEDDING_KEY][:]
    print(f"Loaded {len(smiles_list)} SMILES and {embeddings_array.shape[0]} embeddings.")

    indices = list(range(len(smiles_list)))

    np.random.seed(42) # for reproducibility
    np.random.shuffle(indices)
    split_point = int(0.9 * len(indices))
    train_indices, val_indices = indices[:split_point], indices[split_point:]

    train_smiles = [smiles_list[i] for i in train_indices]
    val_smiles = [smiles_list[i] for i in val_indices]
    train_embeddings = embeddings_array[train_indices]
    val_embeddings = embeddings_array[val_indices]
    print(f"Training set size: {len(train_smiles)}, Validation set size: {len(val_smiles)}")

    # Load the pre-trained smi_ssed model and get its tokenizer
    print(f"Loading smi_ssed base model from: {SMI_SSED_MODEL_FOLDER}")
    smi_ssed_base_model = load_smi_ssed(
        folder="/workspace/materials.smi_ssed/smi_ssed/inference/smi_ssed",
        ckpt_filename='smi_ssed_130.pt'
    )
    tokenizer = smi_ssed_base_model.tokenizer

    train_dataset = SmilesEmbeddingDataset(train_smiles, train_embeddings, tokenizer, MAX_LENGTH)
    val_dataset = SmilesEmbeddingDataset(val_smiles, val_embeddings, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # Initialize the new custom model
    model = SmiSsedPredictor(smi_ssed_base_model, YOUR_EMBEDDING_DIM).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Model Architecture ---")
    print(model)
    print("--------------------------\n")
    for epoch in range(EPOCHS):
        current_epoch = epoch + 1
        print(f'--- Epoch {current_epoch}/{EPOCHS} ---')

        # Training loop with tqdm for batch progress
        model.train()
        total_loss = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {current_epoch} [Train]", leave=False)
        for batch in train_iter:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()
            batch_outputs = []
            for i in range(input_ids.size(0)):
                single_output = model(input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0))
                batch_outputs.append(single_output)
            outputs = torch.cat(batch_outputs, dim=0)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_iter.set_postfix(loss=loss.item())
        train_loss = total_loss / len(train_loader)
        print(f'Training Loss: {train_loss:.4f}')

        # Validation loop with tqdm for batch progress
        model.eval()
        total_loss = 0
        all_cos_sims = []
        val_iter = tqdm(val_loader, desc=f"Epoch {current_epoch} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_iter:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)

                batch_outputs = []
                for i in range(input_ids.size(0)):
                    single_output = model(input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0))
                    batch_outputs.append(single_output)
                outputs = torch.cat(batch_outputs, dim=0)

                loss = loss_fn(outputs, targets)
                total_loss += loss.item()
                cos_sim = cosine_similarity(outputs, targets, dim=1)
                all_cos_sims.extend(cos_sim.cpu().numpy())
                val_iter.set_postfix(loss=loss.item())
        val_loss = total_loss / len(val_loader)
        val_cos_sim = sum(all_cos_sims) / len(all_cos_sims) if all_cos_sims else 0
        print(f'Validation Loss: {val_loss:.4f} | Validation Cosine Similarity: {val_cos_sim:.4f}')

        # Save checkpoint at the end of each epoch
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f'checkpoint_epoch_{current_epoch}.pth')
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')
        print('-' * 20)

    print("Training complete.")