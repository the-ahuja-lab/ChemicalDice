# How to Run

1. Activate the conda environment:
   bash run_alvadesc_new.sh
   (This script sets up everything and runs the pipeline. Output is logged in logs_new_with_nan/)

2. To merge output chunks:
   python merge_alvadesc_chunks.py

3. For post-processing or analysis, use final.py.


## Contents

- **final.py**: Processes AlvaDesc output, extracts failed molecule IDs from logs, and loads/saves descriptor arrays.
- **merge_alvadesc_chunks.py**: Merges multiple chunked `.npz` files of descriptors into a single final `.npz` file.
- **run_alvadesc_new.sh**: Bash script to activate the `alvadesc` conda environment and launch the pipeline.
- **run_alvadesc_pipeline_new.py**: Main Python pipeline for computing descriptors from SMILES using AlvaDesc, with logging and multiprocessing support.
 **embeddings_new_with_nan/**: Output folders for descriptor files.
 **logs_new_with_nan/**: Log output folders.


## Installation

- I install the Python wrapper with pip:
   ```bash
   pip install alvadescpy
   ```
- I already have alvaDesc installed (from https://www.alvascience.com/alvadesc/) and a valid license. I put the license file (e.g., `alvaDesc.lic`) in the alvaDesc installation directory (e.g., `/opt/alvadesc/` or wherever alvaDesc is installed). 

   If I ever move the CLI, I set the path in Python like this:
   ```python
   from alvadescpy import CONFIG
   CONFIG['alvadesc_path'] = '/home/suvenduk/Yasser/alvadesc/alvadesc_extracted/usr/bin/alvaDescCLI'
   ```

   If the license file is not picked up automatically, I can also set the license path explicitly (if supported by alvaDesc):
   ```python
   CONFIG['license_path'] = '/path/to/alvaDesc.lic'
   ```

## Usage


## How I Run the Pipeline

1. I always use the `alvadesc` conda environment (see run_alvadesc_new.sh):
   ```bash
   conda activate alvadesc
   ```
2. To launch the pipeline, I just run:
   ```bash
   bash run_alvadesc_new.sh
   ```
   This script activates the environment and runs `run_alvadesc_pipeline_new.py` in the background, logging output to `logs_new_with_nan/alvadesc_run_with_nan.log`.
3. If I want to merge output chunks:
   ```bash
   python merge_alvadesc_chunks.py
   ```
4. For post-processing or analysis, I use `final.py`.

### Example: Using alvadescpy in Python

```python
from alvadescpy import alvadesc
# Calculate all descriptors for a SMILES string
descriptors = alvadesc(ismiles='CCC', descriptors='ALL')
# Get a dictionary with labels
descriptors = alvadesc(ismiles='CCC', descriptors='ALL', labels=True)
```

## Requirements

- Python 3.x
- Conda environment with AlvaDesc dependencies
- alvadescpy Python package
- Licensed alvaDesc installation

