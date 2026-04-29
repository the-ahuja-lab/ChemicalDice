# Basic Usage Tutorial

If you want to rapidly translate an internal library of SMILES structures into the CDI embedding maps, utilizing the basic Python API is easiest.

Ensure you've installed `ChemicalDice` (Tier 1).

## 1. Local Initialization
```python
import pandas as pd
from ChemicalDice.core.api_client import collect_features_from_csv
```

## 2. Setting Up Data
Your CSV requires exactly one column titled `SMILES`. ChemicalDice uses this structure strictly.

```csv
SMILES,Assay_Target,Assay_Result
CC(=O)Oc1ccccc1C(=O)O,Aspirin,Active
```
*Any other columns (like `Assay_Target`) will be ignored during the actual extraction iteration securely.*

## 3. Invoking the Extractor
```python
df_results = collect_features_from_csv(
    filepath="compounds.csv",
    convert_to_canonical=True # Enables safety RDKit filtering
)

print(df_results.head())
# Output:
# SMILES                    CDI1      CDI2      CDI3  ... CDI8192
# CC(=O)Oc1ccccc1C(=O)O    0.4121   -1.1092    0.052 ....
```
