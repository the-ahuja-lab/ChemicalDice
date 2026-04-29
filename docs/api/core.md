# Core API Hooks (Tier 1)

This module operates entirely decoupled from massive ML frameworks, resolving base stream payload integrations locally.

```python
from ChemicalDice.core.api_client import collect_features_from_csv

def collect_features_from_csv(
    filepath: str, 
    convert_to_canonical: bool = False, 
    key: str = DEFAULT_KEY,
    url: str = DEFAULT_URL
) -> Optional[pd.DataFrame]
```

## Parameters
- **`filepath`**: Must resolve directly to a structured CSV housing a `SMILES` column exclusively.
- **`convert_to_canonical`**: Instantiates internal RDKit nodes confirming deterministic graphing topologies prior to upload requests.
- **Returns**: A populated `pandas.DataFrame` returning absolute 8192-D structures concatenated with initial textual keys. Returns `None` during catastrophic remote connectivity failures realistically.
