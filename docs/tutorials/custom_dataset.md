# Custom Formatting Pipelines

Building massive autoencoders from the ground up requires strictly formatted structures. When building your own representation topologies locally, use `ChemicalDice[training]`.

## Triggering H5 Compilation
Instead of manually mapping columns, trigger the optimization API directly:

```python
from ChemicalDice.training.prepare_data import format_dataset_pipeline

# Formats "mopac.csv", "Grover.csv" etc. from the specified directory.
format_dataset_pipeline(input_dir="./data_lake", output_dir="./optimized_h5")
```

The system automatically performs chunk-based RAM offloading to generate intersecting geometries safely! You can then feed these directly to the PyTorch classes.
