"""
CDI-Generalised Supervised SMILES -> Embeddings Module.
Wraps standard Mamba block embeddings (smi_ssed) to map discrete SMILES topologies straight into 8192-D CDI space.
"""

import os
import sys
import logging
import contextlib
from typing import List

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from rdkit import Chem

logger = logging.getLogger(__name__)

class MambaDataset(Dataset):
    """
    Pairs raw token-sequence input lengths against exact 8192-D numerical labels
    for strictly supervised alignment mechanics.
    """
    def __init__(self, raw_smiles: List[str], target_h5_file: str):       
        self.raw_smiles = raw_smiles
        self.target_h5_file = target_h5_file

        if not os.path.exists(target_h5_file):
            raise FileNotFoundError(f"Missing essential regression targets: {target_h5_file}")

        with h5py.File(target_h5_file, 'r') as h5_file:
            # Dynamically retrieve the first label key
            self.dataset_key = list(h5_file.keys())[0]
            target_shape = h5_file[self.dataset_key].shape
            if len(self.raw_smiles) != target_shape[0]:
                raise ValueError(
                    f"Dimensional collision! Label rows ({target_shape[0]}) disjoint from SMILES inputs ({len(self.raw_smiles)})."
                )

    def __len__(self) -> int:
        return len(self.raw_smiles)

    def __getitem__(self, idx: int) -> tuple:
        smiles_string = self.raw_smiles[idx]
        with h5py.File(self.target_h5_file, 'r') as h5_file:
            target_tensor = h5_file[self.dataset_key][idx]
        return smiles_string, torch.tensor(target_tensor, dtype=torch.float32)

class SmiSsedPredictor(nn.Module):
    """
    Top-layer regression module mapping state-space language geometries (from HuggingFace Mamba models)
    into custom ChemicalDice coordinates.
    """
    def __init__(self, mamba_model_dir: str, target_dim: int = 8192):     
        super(SmiSsedPredictor, self).__init__()

        # Graceful load mechanism using internal API paths
        if not os.path.exists(mamba_model_dir):
            raise FileNotFoundError(f"Cannot resolve root foundational model dependencies at: {mamba_model_dir}")
        sys.path.insert(0, mamba_model_dir)

        try:
            from smi_ssed.load import load_smi_ssed
        except ImportError as e:
            raise ImportError(
                "Execution constraint failure: Could not load the required 'smi_ssed' package. "
                "Ensure HuggingFace implementation exists at target payload directory."
            ) from e

        # Redirect massive print outputs inherently logged by external load mechanisms
        logger.info("Initializing HuggingFace Smi-SSED Model Runtime Core (This dictates high latency).")
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                self.smi_ssed_model, self.smi_ssed_tokenizer = load_smi_ssed()

        # Isolate semantic execution maps statically vs trainable linear layer weights
        self._freeze_mamba()
        self.embedding_dim = self.smi_ssed_model.config.d_model
        self.regressor = nn.Linear(self.embedding_dim, target_dim)        

    def _freeze_mamba(self):
        """Secures state-space layers rigidly."""
        for param in self.smi_ssed_model.parameters():
            param.requires_grad = False
        self.smi_ssed_model.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Executes frozen attention sweeps projecting the mean sequence dimension through the trained 8192-D classifier.
        """
        # Execute dynamically ignoring backpropagation scope across Mamba matrices
        with torch.no_grad():
            outputs = self.smi_ssed_model(input_ids=input_ids)
            # Extrapolate pooled dimension maps
            hidden_states = outputs.hidden_states[-1]

            # Mask handling equivalent to native transformer logic        
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.sum(mask_expanded, dim=1)
                pooled_output = sum_hidden / sum_mask
            else:
                pooled_output = hidden_states.mean(dim=1)

        # Supervised linear regression against target vectors
        return self.regressor(pooled_output)


def _cosine_similarity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes pure distance offset optimization maps (1 - CosineSimilarity)"""
    cos_sim = F.cosine_similarity(pred, target, dim=1)
    return 1 - cos_sim.mean()


def train_generalised_cdi(
    csv_path: str,
    target_h5_file: str,
    mamba_model_dir: str = None,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    target_dim: int = 8192,
    save_path: str = "generalised_smi_ssed_model.pth"
) -> SmiSsedPredictor:
    """
    Executes deep MSE + Euclidean regression tracking aligning external state-space networks into target HDF5 tensors.
    """
    if mamba_model_dir is None:
        from pathlib import Path
        mamba_model_dir = str(Path.home() / ".chemicaldice" / "materials.smi_ssed")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    logger.info(f"Establishing primary compute matrix on: {device}")      
    print(csv_path)
    # Process source logic
    df = pd.read_csv(csv_path)
    if "SMILES" not in df.columns:
        raise ValueError("Strict sequence validation failed. Dataframe missing target 'SMILES' column.")
    raw_smiles = df["SMILES"].tolist()

    logger.info("Initializing supervised dataset architecture.")
    dataset = MambaDataset(raw_smiles, target_h5_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    # Resolving SmiSsed
    model = SmiSsedPredictor(mamba_model_dir, target_dim=target_dim).to(device)
    optimizer = optim.Adam(model.regressor.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    logger.info("Executing Phase-2 Regression Adjustments...")

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_cos_sim = 0.0, 0.0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch_smiles, batch_targets in progress:
            batch_targets = batch_targets.to(device)

            # Execution string alignments enforcing static 128 context windows natively deployed
            inputs = model.smi_ssed_tokenizer(
                list(batch_smiles),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)

            # Loss tracking maps
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            # Metric extraction purely for real-time visualization        
            cos_sim_loss = _cosine_similarity_loss(predictions, batch_targets).item()
            total_loss += loss.item()
            total_cos_sim += cos_sim_loss

            progress.set_postfix(
                mse=(total_loss / (progress.n + 1)),
                cos_dist=(total_cos_sim / (progress.n + 1))
            )

    # Secure blueprint arrays exclusively maintaining linear layer modifications tracking
    torch.save(model.state_dict(), save_path)
    logger.info(f"Target Regression Network Blueprint successfully serialized to: {save_path}")

    return model