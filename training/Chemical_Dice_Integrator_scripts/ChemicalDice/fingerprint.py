"""
The fingerprint generation function.
"""
from argparse import Namespace
from logging import Logger
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ChemicalDice.molgraph import MolCollator
from ChemicalDice.moldataset import MoleculeDataset
from ChemicalDice.utils import get_data, create_logger, load_checkpoint


def do_generate(model: nn.Module, 
                data: MoleculeDataset,
                args: Namespace,
                ) -> List[List[float]]:
    """
    Do the fingerprint generation on a dataset using the pre-trained models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param args: A Namespace object with necessary arguments.
    :return: A list of fingerprints.
    """
    # Set the model to evaluation mode
    model.eval()
    args.bond_drop_rate = 0
    preds = []

    # Check dataset size
    if len(data) == 0:
        raise ValueError("The MoleculeDataset is empty. Please provide valid data.")
    print(f"Dataset size: {len(data)}")

    mol_collator = MolCollator(args=args, shared_dict={})

    # Diagnostic for dataset samples
    print("Inspecting first few dataset items:")
    for i in range(min(3, len(data))):  # Inspect up to 3 items
        print(f"Item {i}: {data[i]}")

    # DataLoader setup
    num_workers = 4
    mol_loader = DataLoader(data,
                            batch_size=32,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=mol_collator)
    
    print("Checking DataLoader batches:")
    try:
        for batch_idx, item in enumerate(mol_loader):
            print(f"Processing batch {batch_idx}")
            if len(item) != 5:
                raise ValueError(f"Unexpected batch structure: {item}")
            
            _, batch, features_batch, _, _ = item
            print(f"Batch {batch_idx}: batch={batch}, features_batch={features_batch}")

            # Perform prediction
            with torch.no_grad():
                batch_preds = model(batch, features_batch)
                preds.extend(batch_preds.data.cpu().numpy())
    except Exception as e:
        print(f"Error encountered in DataLoader: {e}")
        raise

    print("Fingerprint generation completed successfully.")
    return preds

import pandas as pd
import numpy as np
def generate_fingerprints(args: dict, logger: Logger = None) -> List[List[float]]:
    """
    Generate the fingerprints.

    :param logger:
    :param args: Arguments.
    :return: A list of lists of target fingerprints.
    """

    checkpoint_path = args.checkpoint_path
    #print(args)
    if logger is None:
        logger = create_logger('fingerprints', quiet=False)
    print('Loading data')
    test_data = get_data(path=args.data_path,
                         args=args,
                         use_compound_names=False,
                         max_data_size=float("inf"),
                         skip_invalid_smiles=False)
    test_data = MoleculeDataset(test_data)

    logger.info(f'Total size = {len(test_data):,}')
    logger.info(f'Generating...')
    # Load model
    model = load_checkpoint(checkpoint_path, cuda=args.cuda, current_args=args, logger=logger)
    model_preds = do_generate(
        model=model,
        data=test_data,
        args=args
    )
    #print(len(model_preds))
    df_raveled = pd.DataFrame(list(map(np.ravel, model_preds)))
    df_raveled = df_raveled.add_prefix('Grover_')
    smiles_id_list = args.id_list
    smiles_id_list = smiles_id_list.split("___")
    df_raveled.index = smiles_id_list
    df_raveled.index.name = 'id'
    #print("gro", args.grover_output)
    df_raveled.to_csv(args.grover_output)


    return model_preds
