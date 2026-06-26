"""
Training Package: Models, Descriptors, and Utilities (Tier 2)

Dependencies:
    - torch
    - h5py
    - scikit-learn
    - smi_ssed
"""

import sys
import logging

# Ensure optional Tier 2 dependencies are present gracefully
try:
    import torch
except ImportError:
    logging.warning(
        "PyTorch is missing! The 'ChemicalDice.training' module requires the [training] extra. "
        "Run `pip install ChemicalDice[training]`"
    )
    # Raising the error is deferred to actual usage calls, allowing __init__ parsing to safely pass.
